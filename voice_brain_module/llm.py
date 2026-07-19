#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
from openai import OpenAI
from .config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from .commands import COMMANDS, validate_commands

# ── LLM 提示词构建 ────────────────────────────────────────────

def _compact_param(name: str, p: dict) -> str:
    if p.get("options"):
        opts = p["options"]
        if opts and opts[0] == "_LAZY_":
            from .commands import _resolve_song_options
            opts = _resolve_song_options() or ["(曲库为空)"]
        opts_str = " | ".join(opts)
        return f'{name}=[{opts_str}]'
    if p.get("range"):
        lo, hi = p["range"]
        unit = p.get("unit", "")
        return f"{name}={lo}~{hi}{unit}"
    return name


def build_command_reference() -> str:
    """生成紧凑的指令参考字符串，嵌入 LLM 系统提示词。"""
    lines = []
    for cmd in COMMANDS:
        params = cmd.get("params", {})
        param_str = ", ".join(_compact_param(k, v) for k, v in params.items()) if params else "无"
        lines.append(
            f"- 执行机构：{cmd['actuator']}；"
            f"执行动作：{cmd['action']}；"
            f"参数：{param_str}  ({cmd['desc']})"
        )
    return "\n".join(lines)


def build_system_prompt() -> str:
    ref = build_command_reference()
    return (
        "你是小智，一个专为老年人设计的智能陪护机器人。"
        "你既是用户的聊天伙伴，也能帮助控制机器人执行实体动作，两者同等重要。"
        "用户可能随时切换——聊着家常突然让你去拿东西，或者完成动作后继续聊天，"
        "你要自然地在两种模式之间流转。\n\n"
        "【陪伴原则】\n"
        "- 说话温和、耐心，语气像一个亲切的晚辈或朋友，避免生硬或敷衍\n"
        "- 鼓励用户多说话，可以主动追问（如：\"那后来呢？\"\"您平时喜欢吃什么？\"）\n"
        "- 用户谈及家人、往事、身体状况、兴趣爱好时，认真回应并表达关心\n"
        "- 用户重复说同一件事时，保持耐心，不要表现出不耐烦\n"
        "- 若用户情绪低落、孤独或诉说烦恼，给予真诚的情感回应，不要急着转移话题\n\n"
        "【联网搜索】\n"
        "你具备联网搜索能力。当用户询问实时信息（天气、新闻、电视节目、药品、养生知识等）时，"
        "主动搜索后再回答，不要用过时信息敷衍。\n"
        "⚠️ 联网搜索是你自身的内置能力，由系统自动完成，**不是机器人的执行机构**。"
        "严禁把\"网络搜索\"\"联网搜索\"\"查询\"之类写进 [执行指令]——"
        "那里只能出现【可用指令集】里的物理执行机构。搜索完直接在 [口语回复] 里"
        "说出结果，[执行指令] 该是什么就是什么（通常是 []）。\n\n"
        "【感知能力】\n"
        "感知数据会以【标记】形式出现在用户消息中：\n"
        "- 视觉识别：可识别摄像头视野内的物体，可抓取物体为苹果、香蕉、瓶子、蛋糕；"
        "上下文以「物体×数量」标注（如「苹果×2」）\n"
        "- 情绪识别：可识别用户情绪状态\n\n"
        "⚠️ 若消息中无【当前视野物体】/【用户情绪】标记，说明没有感知数据。"
        "被问到\"看到了什么\"时必须诚实回答，**严禁编造物体名称**。\n\n"
        "⚠️ 视觉数据只有物体和数量，**不包含物体在哪件家具上**。用户问"
        "\"桌子上/椅子上/地上/柜子上有什么\"，问的都是同一件事——你眼前看到了什么。"
        "一律照【当前视野物体】如实回答，**不要因为用户说的家具名称与你的预期不符，"
        "就说看不到或没有数据**。回答时也不必强调物体具体在哪件家具上。\n\n"
        "【回复格式】严格按以下格式，两个标签都必须出现：\n"
        "[口语回复]：\n"
        "<自然口语回答；若有动作请求，用一句话告知用户正要做什么——\n"
        "如\"好的，我这就帮您把苹果拿过来\"。\n"
        "⚠️ **决定了要执行就只说做什么，绝对不要反问。**\n"
        "\"需要我帮你抓吗？\"然后立刻去抓——这非常奇怪，用户还没回答你就动手了。\n"
        "也不要问\"要不要放歌\"\"是不是想听XX\"——直接说\"给您放XX\"然后放。>\n"
        "[执行指令]：\n"
        "<JSON 数组，每项：{\"actuator\": ..., \"action\": ..., \"params\": {...}}>\n\n"
        "示例 1（纯聊天）：\n"
        "用户：我今天有点无聊\n"
        "[口语回复]：\n"
        "哎呀，那我来陪您聊聊天吧！您平时喜欢看什么节目，或者有什么想聊的吗？\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 2（询问实时信息，先联网搜索再回答，回答里不要出现\"搜索后\"之类的字样）：\n"
        "用户：今天天气怎么样\n"
        "[口语回复]：\n"
        "今天多云，气温十度到十八度，出门记得带件外套，注意保暖哦。\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 3（可执行动作）：\n"
        "用户：帮我抓一下苹果\n"
        "[口语回复]：\n"
        "好的，我这就帮您把苹果拿过来。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"机械臂\",\"action\":\"抓取\",\"params\":{\"target\":\"苹果\"}}]\n\n"
        "示例 4（多个指令组合）：\n"
        "用户：前进一米然后抓香蕉\n"
        "[口语回复]：\n"
        "好的，我先前进一米再帮您拿香蕉。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"底盘\",\"action\":\"前进\",\"params\":{\"speed\":0.2,\"distance\":1.0}},{\"actuator\":\"机械臂\",\"action\":\"抓取\",\"params\":{\"target\":\"香蕉\"}}]\n\n"
        "示例 5（有视觉数据）：\n"
        "【当前视野物体】共3个：苹果×2, 杯子\n【可抓取物体】苹果×2\n\n---\n用户说：椅子上有什么\n"
        "[口语回复]：\n"
        "我看到有两个苹果和一个杯子，需要我帮您拿什么吗？\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 6（无视觉数据）：\n"
        "用户：地上有什么\n"
        "[口语回复]：\n"
        "我现在还没看到画面，视觉功能还没有数据。您需要什么，我来帮您想办法。\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 7（播放歌曲）：\n"
        "用户：给我唱首歌\n"
        "[口语回复]：\n"
        "好的，给您放一首歌。\n"
        "[执行指令]：\n"
        "[{\"actuator\":\"音箱\",\"action\":\"播放歌曲\",\"params\":{\"song\":\"test\"}}]\n\n"
        "示例 8（信息不足，只问不做）：\n"
        "用户：放首歌吧\n"
        "（曲库有茉莉花和军港之夜两首，无法确定用户要哪首）\n"
        "[口语回复]：\n"
        "曲库里有茉莉花和军港之夜，您想听哪一首呢？\n"
        "[执行指令]：\n"
        "[]\n\n"
        "示例 9（不在指令集）：\n"
        "用户：打开窗帘\n"
        "[口语回复]：\n"
        "这个动作我暂时还做不到，抱歉。我能帮您控制底盘移动或者用机械臂拿东西，有需要告诉我。\n"
        "[执行指令]：\n"
        "[]\n\n"
        "【执行规则】\n"
        "- 只能从【可用指令集】中选取指令，**禁止编造不存在的执行机构、动作或参数**\n"
        "- 无法执行的请求：[执行指令] 输出 []，[口语回复] 说明原因并告知能做什么\n"
        "- 纯聊天/问询：[执行指令] 输出 []，[口语回复] 认真回答\n"
        "- 只有动作没有问询：[口语回复] 直接告知（如\"好的，马上去\"\"给您放一首茉莉花\"），\n"
        "  **严禁反问**——\"要不要帮你拿\"\"需要我放歌吗\"\"是不是想听XX\"一律禁止\n"
        "- **什么时候只问不做**：当信息不足以确定动作时，可以提问——但此时\n"
        "  [执行指令] 必须输出 []，等用户下一轮回答了再动手。\n"
        "  例如用户说\"放首歌\"但没说是哪首，而且曲库有多首可选：\n"
        "  正确的做法是回复\"曲库里有XX和XX，您想听哪首？\"然后 []。\n"
        "  错误做法是随便挑一首就放，或者反问完立刻去放。\n"
        "- **什么时候直接做**：用户明确说了要什么（\"播放test\"\"帮我抓苹果\"），\n"
        "  就直接告知并执行，不反问。判断标准：你自己是否已经确定了要执行\n"
        "  的动作——确定了就不问，没确定就问完等回复。\n"
        "- 无视觉数据时，回答物体相关问题必须诚实，不编造物体名称\n"
        "- 用户谈及视野内物体但未要求抓取时，描述所见即可，不自动生成抓取指令\n"
        "- 参数值必须严格匹配选项，不做近义替换\n"
        "- 不输出两个标签之外的任何内容\n"
        "- 检测到用户情绪低落或痛苦时，[口语回复] 给予真诚安抚，[执行指令] 输出 []\n\n"
        f"【可用指令集】：\n{ref}"
    )


# ── LLM 客户端 ────────────────────────────────────────────────

# openai 默认 connect=5s / read=600s，且默认重试 2 次 → 最坏要等很久才恢复。
# 语音机器人要快速自恢复：单次 20s 上限、重试 1 次（max_retries=1）。
# 最坏 2 × 20 = 40s 抛异常 → generate_response 兜底 → brain_node 的
# finally 恢复监听。注意：这只是安全网，不解决根因。
llm_client = OpenAI(
    api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
    timeout=20.0, max_retries=1,
)

_SYSTEM_PROMPT = build_system_prompt()
_SPOKEN_PREFIX = "[口语回复]："
_INSTR_PREFIX  = "[执行指令]："

_MAX_RETRIES = 2   # 校验失败时最多让 LLM 重新生成的次数


def generate_response(
    text: str,
    system_prompt: str = _SYSTEM_PROMPT,
    vision_context: str = "",
) -> tuple[str, list[dict]]:
    """
    调用 LLM，返回 (口语回复, 指令列表)。
    若 LLM 输出的指令未通过 commands.validate_commands 校验，会把错误反馈给
    LLM 让其重新生成；重试耗尽后丢弃指令并在口语回复中告知用户。

    vision_context: 可选的视觉/情绪上下文，会注入到用户消息中，
                    供 LLM 回答"桌上有什么"或进行情绪关怀。
    """
    user_content = text
    if vision_context:
        user_content = f"{vision_context}\n\n---\n用户说：{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    spoken, commands, last_err = "", [], ""

    for attempt in range(_MAX_RETRIES + 1):
        try:
            raw = _call_llm(messages)
        except Exception as e:
            # 超时/网络错误：不静默卡住，给用户一句友好的口头反馈
            print(f"[LLM 调用失败] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            return "抱歉，我这会儿网络有点慢，没太跟上，您能再说一遍吗？", []
        spoken   = _parse_spoken(raw)
        commands = _parse_commands(raw)

        ok, last_err = validate_commands(commands)
        if ok:
            if not spoken:
                print(f"[LLM 原始输出]\n{raw}\n", flush=True)
                print("[警告] LLM 未生成口语回复，检查提示词格式", file=sys.stderr)
            return spoken, commands

        if attempt < _MAX_RETRIES:
            print(f"[校验失败,重试 {attempt+1}/{_MAX_RETRIES}] {last_err}",
                  file=sys.stderr, flush=True)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": (
                f"你上次输出的执行指令未通过校验：{last_err}。"
                f"请保持完全相同的两段式输出格式，但执行机构、动作名称和参数值"
                f"必须严格使用【可用指令集】中列出的；若没有任何匹配项，"
                f"[执行指令] 输出空数组 []。"
            )})

    print(f"[警告] 重试 {_MAX_RETRIES} 次后仍校验失败,丢弃 commands: {last_err}",
          file=sys.stderr)
    suffix = "另外，我刚才没能正确生成动作指令，可以再说一遍吗？"
    spoken = f"{spoken}\n{suffix}" if spoken else suffix
    return spoken, []


def _call_llm(messages: list) -> str:
    """单次调用 LLM（流式累积），返回完整原始文本。启用千问联网搜索。"""
    stream = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        extra_body={
            "enable_search": True,
            # forced_search 必须嵌在 search_options 内。放在 extra_body 顶层会被
            # 静默忽略——实测放顶层时的回答与完全不开搜索逐字一致（都答出过期日期）。
            "search_options": {
                "forced_search": True,   # turbo 自行判断偏保守，常漏搜
            },
        },
    )
    raw = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        raw += delta
    return raw


def _call_llm_simple(messages: list) -> str:
    """轻量 LLM 调用（非流式，无搜索），供记忆提取等后台任务使用。"""
    resp = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=False,
    )
    return resp.choices[0].message.content or ""


def _parse_spoken(raw: str) -> str:
    """提取 [口语回复]： 与 [执行指令]： 之间的文本（可多行）。"""
    s_idx = raw.find(_SPOKEN_PREFIX)
    if s_idx == -1:
        return ""
    start = s_idx + len(_SPOKEN_PREFIX)
    c_idx = raw.find(_INSTR_PREFIX, start)
    end   = c_idx if c_idx != -1 else len(raw)
    return raw[start:end].strip()


def is_addressing_robot(user_text: str, last_question: str = "") -> bool:
    """快速判断用户是否在对机器人说话。

    当机器人刚问了一个问题（last_question 非空），而用户的语音转写内容
    看起来不是在回答机器人（比如在跟房间里的其他人说话、自言自语碎片等），
    返回 False，让上游决定是否回一句"请问您在和我讲话吗？"。

    若 last_question 为空（机器人没有在等答案），默认返回 True，
    不做额外判断，避免误拦截正常的主动对话。
    """
    if not last_question:
        return True

    # 构建上下文
    ctx_parts = [f"机器人刚才问：{last_question}"]
    ctx_parts.append(f"用户语音转写：{user_text}")
    ctx = "\n".join(ctx_parts)

    messages = [
        {"role": "system", "content": (
            "你是一个对话分析器。你的任务是判断用户的语音转写内容"
            "是否在对AI机器人助手说话。\n\n"
            "判定为 NO（没有在对机器人说话）的情况：\n"
            "- 用户明显在跟另一个人说话（用了其他人的称呼或名字）\n"
            "- 用户的语音是自言自语、背景噪音转写的零碎文字\n"
            "- 用户说话的内容跟机器人刚问的问题完全无关，"
            "且语气像是在跟另一个人交流\n\n"
            "判定为 YES（在对机器人说话）的情况：\n"
            "- 用户在回答机器人的问题\n"
            "- 用户在向机器人提问或发出指令\n"
            "- 用户提到了机器人的名字（小智）\n\n"
            "拿不准时输出 YES：误判成 NO 会让机器人反问\"请问您在和我讲话吗\"，"
            "打断正常对话，代价远大于误放行。\n"
            "只输出 YES 或 NO，不要任何解释或其他文字。"
        )},
        {"role": "user", "content": ctx},
    ]
    try:
        raw = _call_llm_simple(messages).strip().upper()
        result = raw.startswith("YES")
        if not result:
            print(f"[对话检测] 判定用户不在对机器人说话: {user_text[:80]}",
                  flush=True)
        return result
    except Exception as e:
        print(f"[对话检测] 调用失败，默认放行: {e}", file=__import__('sys').stderr)
        return True   # 出错时默认认为在对话，避免误拒绝


def classify_meta_response(user_text: str) -> str:
    """判断用户对元问题"请问您在和我讲话吗？"的应答意图。

    返回三分类结果：
      "NOT_ADDRESSING" — 用户仍在跟别人说话，不是在回应机器人
      "REJECTING"      — 用户在回应机器人，但明确拒绝（"没有"、"你一边去"等）
      "ADDRESSING"     — 用户在回应机器人，确认在对话

    用一个 LLM 调用同时完成两种判断，避免 is_addressing_robot + is_rejecting_robot
    两次调用的不一致问题（尤其是"没有，你一边去"这种悖论式输入：
    用户在"对机器人说"自己"没在跟机器人说话"）。
    """
    messages = [
        {"role": "system", "content": (
            "你是一个对话分析器。机器人刚问了用户一句：\"请问您在和我讲话吗？\"\n"
            "现在用户说了一句话，你需要判断用户的意图，输出以下三类之一：\n\n"

            "ADDRESSING — 用户在回应机器人，且没有拒绝的意思。包括：\n"
            "  · 任何肯定答复，无论多简短：\"是\"、\"对\"、\"嗯\"、\"是啊\"、"
            "\"当然\"、\"对，我在跟你说话\"\n"
            "  · 重新陈述或补充自己的需求：\"帮我拿个苹果\"、\"我刚说的那个杯子\"。"
            "用户直接说事情，本身就证明他在对机器人讲话——哪怕内容和上面那句"
            "问话毫无关联。\n"
            "  · 向机器人提任何问题或下达任何指令\n\n"

            "REJECTING — 用户在回应机器人，但明确表示不是在和机器人说话、"
            "或者驱赶机器人。例子：\"没有\"、\"不是\"、\"没跟你说话\"、"
            "\"你一边去\"、\"别烦我\"、\"走开\"。\n"
            "  注意：必须出现明确的否定词或驱赶词才算，"
            "不要仅因为回答简短就归入这一类。\n\n"

            "NOT_ADDRESSING — 用户的话明显是说给别人听的，不是在回应机器人。"
            "必须有正面证据，例如：出现了另一个人的名字或称呼（\"小王你过来\"）、"
            "明显在向第三方转述、或纯属背景噪音转写出的无意义碎片。\n"
            "  ⚠️ 不要仅凭\"内容和机器人的问句无关\"就判这一类——"
            "用户重新说自己需求时，内容天然就和问句无关，那属于 ADDRESSING。\n\n"

            "拿不准时一律输出 ADDRESSING：误放行只是多聊一句，"
            "误拦截会直接终止对话，代价大得多。\n"
            "只输出 NOT_ADDRESSING、REJECTING 或 ADDRESSING，不要任何解释。"
        )},
        {"role": "user", "content": f"用户说：{user_text}"},
    ]
    try:
        raw = _call_llm_simple(messages).strip().upper()
        if "NOT_ADDRESSING" in raw:
            result = "NOT_ADDRESSING"
        elif "REJECTING" in raw:
            result = "REJECTING"
        else:
            result = "ADDRESSING"  # 兜底
        print(f"[元对话分类] {result}: {user_text[:80]}", flush=True)
        return result
    except Exception as e:
        print(f"[元对话分类] 调用失败，默认 ADDRESSING: {e}",
              file=__import__('sys').stderr)
        return "ADDRESSING"  # 出错时走正常流程


def _parse_commands(raw: str) -> list[dict]:
    """从 LLM 原始输出中提取 [执行指令] 后的 JSON 数组。"""
    idx = raw.find(_INSTR_PREFIX)
    if idx == -1:
        return []
    after = raw[idx + len(_INSTR_PREFIX):].strip()
    first_line = after.split("\n")[0].strip()
    try:
        result = json.loads(first_line)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        start = after.find("[")
        end   = after.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(after[start: end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        print(f"[警告] 执行指令 JSON 解析失败: {after[:200]}", file=sys.stderr)
    return []
