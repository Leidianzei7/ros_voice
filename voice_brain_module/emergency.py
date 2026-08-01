#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紧急呼叫中止意图识别。

外部紧急呼叫模块开始拨打紧急电话 / 发送紧急短信时，向 /voice/listen_mode 发
`emergency`；此后 voice_node 强制开麦、免唤醒词，brain_node 对听到的每一句话
判断"用户是不是要打断这次紧急求助"，判定成立就往 /command 下发一条
{"actuator":"紧急呼叫","action":"中止紧急情况"}。

判定分两级，**规则优先、LLM 兜底**：

  1. 规则层（本模块，纯离线、微秒级）
     · 求救加强句（"救命"、"快叫救护车"）→ KEEP，对取消句有一票否决权
     · 明确取消句（"不用打了"、"我没事"）→ ABORT，立即下发
  2. LLM 层（llm.classify_emergency_abort，brain_node 在后台线程里调，约 1s）
     规则层拿不准时才走，用来兜住老人松散的口语（"哎呀不至于，别麻烦人家了"）。

规则层必须能独立完成判定，不能只依赖 LLM：紧急场景下网络本身可能就是不通的
（DashScope 调不通时 TTS、LLM 一起哑火），而"取消"这个动作恰恰不能等网络。

安全取向是**不对称**的：
  · 漏判（该中止没中止）＝ 多打一通电话，虚惊一场，代价可控
  · 误判（不该中止却中止）＝ 真实求救被撤销，可能危及生命
所以一切歧义、超时、异常一律判 KEEP；求救加强句与取消句同时出现时不自作主张，
交给 LLM 仲裁，LLM 也拿不准就继续呼叫。
"""
import re

# 判定结果
ABORT   = "ABORT"     # 用户明确要中止
CONFIRM = "CONFIRM"   # 用户明确确认（"需要"/"对"/"好"）—— 即刻发起
KEEP    = "KEEP"      # 用户没有要中止/确认，或反而在强化求救
UNKNOWN = "UNKNOWN"   # 规则层拿不准，交给 LLM 层

# 判定来源，随指令下发给紧急侧留档
DETECTOR_RULE = "rule"
DETECTOR_LLM  = "llm"


# ── 求救加强句：出现即对取消句一票否决 ─────────────────────────
# 只放"痛苦陈述"与"催促动词"，**不放** 120／报警／救护车 这类光杆名词——
# 它们同样出现在"不用打120了"这种最常见的取消句里，放进来会把正常取消
# 全部误判成求救。名词交给下面的 _NEG/_CALL 邻近匹配去处理。
_KEEP_RE = re.compile("|".join([
    r"救命", r"救救", r"帮帮我", r"快来人", r"来人[啊呀]",
    r"我不行了", r"喘不[上过]", r"胸口疼", r"心口疼",
    r"好疼", r"很疼", r"疼得", r"疼死",
    r"流血", r"出血", r"起不来", r"站不起", r"动不了",
    r"摔倒", r"摔了", r"晕[倒过]",
    r"(快点?|赶紧|马上|立刻)[打叫喊来]",
]))

# ── 明确取消句（独立成立，无需搭配"电话/呼叫"等宾语）─────────────
_ABORT_PHRASE_RE = re.compile("|".join([
    r"我(真的?)?没(什么)?事",      # 我没事 / 我真没事 / 我没什么事
    r"没事[儿了啊呀吧]",            # 没事了 / 没事儿
    r"不用[了啦]", r"用不着", r"不至于",
    r"按错", r"点错", r"弄错", r"搞错", r"误触", r"误报", r"误按",
    r"虚惊一场", r"不小心碰",
    r"别麻烦", r"不用麻烦", r"不想麻烦", r"甭麻烦",
    r"取消", r"撤销", r"中止", r"挂断", r"挂[了掉]",
    r"不用发了", r"别发了", r"不需要", r"不要[了啦]?", r"不用[了啦]?",
]))

# ── 否定词 + 求助对象 的邻近匹配 ────────────────────────────────
# "不用打120了"、"电话别打了"、"不要给我女儿发短信" —— 否定词与求助对象
# 之间隔着动词/助词，靠字符距离而不是相邻来判定，前后两个方向都查。
_NEG_RE  = re.compile(r"不用|不要|不需要|不必|无需|别|甭|取消|撤销|中止|停止|算了")
_CALL_RE = re.compile(r"电话|呼叫|报警|求救|急救|救护车|120|110|短信|信息|通知|联系|叫人|喊人|打了")
_NEG_CALL_GAP = 6   # 否定词与求助对象之间允许的最大字符间隔


def _neg_near_call(text: str) -> bool:
    """否定词与"打电话/发短信/报警"之类的求助对象是否挨得足够近。"""
    negs  = [(m.start(), m.end()) for m in _NEG_RE.finditer(text)]
    calls = [(m.start(), m.end()) for m in _CALL_RE.finditer(text)]
    for ns, ne in negs:
        for cs, ce in calls:
            if 0 <= cs - ne <= _NEG_CALL_GAP:      # 否定在前："不用打120"
                return True
            if 0 <= ns - ce <= _NEG_CALL_GAP:      # 对象在前："电话别打了"
                return True
    return False


# ── 确认句：用户主动说要联络 ──────────────────────────────────
# 只放短小、不会歧义的肯定词。不作为一票否决（它和取消句同时出现时交给上层仲裁）。
_CONFIRM_RE = re.compile("|".join([
    r"需要", r"要的?", r"对[啊呀了吧]?", r"嗯+", r"好[啊呀的吧了]?",
    r"是的?", r"可以", r"行[啊呀了吧]?", r"打[啊呀吧了]?", r"发[啊呀吧了]?",
    r"联系[啊呀吧了]?", r"叫[啊呀吧了]?",
]))


def detect_confirm_intent(text: str) -> bool:
    """用户是否在确认需要联络。

    用 fullmatch：只匹配纯肯定词（"好"、"需要"、"嗯"、"对"等），
    不匹配长句里的碎片——"今天天气不错"里的"不"不会误触发确认。
    """
    if not text:
        return False
    # 去标点留核心字
    core = re.sub(r"[^\w一-鿿]", "", text, flags=re.UNICODE)
    if not core:
        return False
    return bool(_CONFIRM_RE.fullmatch(core))


def detect_abort_intent(text: str) -> tuple[str, str]:
    """规则层判定。返回 (ABORT/KEEP/UNKNOWN, 判定依据说明)。

    第二个返回值只用于日志，方便事后复盘"当时凭什么撤的"。
    """
    if not text:
        return UNKNOWN, "空文本"

    has_keep     = bool(_KEEP_RE.search(text))
    has_phrase   = bool(_ABORT_PHRASE_RE.search(text))
    has_neg_call = _neg_near_call(text)
    has_abort    = has_phrase or has_neg_call

    # 自相矛盾（"我没事，你快叫救护车"）：不自作主张，交给 LLM 仲裁。
    # LLM 也拿不准就是 KEEP —— 继续呼叫。
    if has_keep and has_abort:
        return UNKNOWN, "取消句与求救句同时出现，交由大模型仲裁"

    if has_keep:
        return KEEP, "命中求救加强句"
    if has_neg_call:
        return ABORT, "命中「否定词 + 求助对象」"
    if has_phrase:
        return ABORT, "命中明确取消句"
    return UNKNOWN, "规则未命中"
