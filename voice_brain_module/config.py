#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
from pypinyin import lazy_pinyin

# ── 模型路径（SenseVoice ONNX 模型，由环境变量优先）──────────
# 设置 SENSEVOICE_MODEL_DIR 指向模型目录；未设置时自动探测
# 优先级：SENSEVOICE_MODEL_DIR > 包内置 onnx_model > workspace models/
_MODEL_DIR_RAW = os.getenv("SENSEVOICE_MODEL_DIR", "")
if _MODEL_DIR_RAW:
    SENSEVOICE_MODEL_DIR = _MODEL_DIR_RAW
else:
    # 回退：先看包内同级 onnx_model/（兼容旧布局），再看 workspace models/
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    _pkg_onnx = os.path.join(_pkg_dir, "..", "onnx_model")
    _ws_onnx  = os.path.join(_pkg_dir, "..", "..", "..", "models", "sensevoice")
    if os.path.isdir(_pkg_onnx):
        SENSEVOICE_MODEL_DIR = os.path.abspath(_pkg_onnx)
    elif os.path.isdir(_ws_onnx):
        SENSEVOICE_MODEL_DIR = os.path.abspath(_ws_onnx)
    else:
        # 包内无、workspace 无 → 扔运行时错误，好过静默崩溃
        raise FileNotFoundError(
            f"SenseVoice 模型目录未找到，已尝试：\n"
            f"  · {os.path.abspath(_pkg_onnx)}\n"
            f"  · {os.path.abspath(_ws_onnx)}\n"
            f"请设置环境变量 SENSEVOICE_MODEL_DIR 指向模型目录"
        )

# ── 设备配置 ──────────────────────────────────────────────
# 按名称子串匹配设备，优先级高于 DEVICE_INDEX 回退值。
# 运行 `python3 -c "import sounddevice as sd; print(sd.query_devices())"` 查看设备列表
DEVICE_NAME        = "USB PnP Audio Device"   # 麦克风输入设备名（子串匹配）
OUTPUT_DEVICE_NAME = "USB PnP Audio Device"   # 扬声器输出设备名（子串匹配）
DEVICE_INDEX        = 0   # 名称匹配失败时的回退索引
OUTPUT_DEVICE_INDEX = 0

def resolve_device(name: str, kind: str = "input") -> int:
    """按名称子串匹配设备，返回 index；失败时回退到 DEVICE_INDEX/OUTPUT_DEVICE_INDEX。"""
    import sounddevice as sd
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if name and name in d.get("name", ""):
            if kind == "input" and d.get("max_input_channels", 0) > 0:
                return idx
            if kind == "output" and d.get("max_output_channels", 0) > 0:
                return idx
    # 回退
    fallback = DEVICE_INDEX if kind == "input" else OUTPUT_DEVICE_INDEX
    import warnings
    warnings.warn(f"未找到匹配 '{name}' 的{kind}设备，使用回退 index={fallback}")
    return fallback
HW_SAMPLE_RATE      = 48000  # 硬件原生采样率
SAMPLE_RATE         = 16000  # SenseVoiceSmall 要求 16kHz，代码内自动重采样
CHANNELS            = 1
CHUNK               = 1024   # 每次读取帧数，越小延迟越低

# ── VAD 参数 ──────────────────────────────────────────────
SPEECH_HOLD_SEC = 1.5   # 停顿多久后触发识别（秒）；调大可减少把一句话切碎
MIN_SPEECH_SEC  = 0.3   # 有效语音最短时长（秒），低于此值丢弃

# ── VAD 模式选择 ───────────────────────────────────────────
# "energy" : 启动时校准底噪，阈值 = 底噪 + SPEECH_DELTA，动态跟随环境变化
# "webrtc"  : Google WebRTC VAD，无需校准，基于频谱特征，稳态噪声下可能误判
VAD_MODE = "energy"

# 能量阈值 VAD（VAD_MODE = "energy"）
NOISE_INIT_SEC  = 1.5   # 启动校准时长（秒），期间请保持安静
NOISE_ALPHA     = 0.01  # 底噪 EMA 更新速率（越小越平滑）
SPEECH_DELTA    = 5000  # 阈值 = 底噪 + 此值，根据说话音量调整

# WebRTC VAD（VAD_MODE = "webrtc"，无需校准）
VAD_AGGRESSIVENESS = 3    # 0-3，越高对噪声越激进
VAD_FRAME_MS       = 20   # 每帧时长，只能是 10/20/30
VAD_FRAME_SAMPLES  = SAMPLE_RATE * VAD_FRAME_MS // 1000
VAD_SPEECH_TRIGGER = 3    # 连续 N 帧判定为语音才开始录制

# ── 唤醒词 ────────────────────────────────────────────────
WAKE_WORD       = "小智小智"
WAKE_WORD_SHORT = "小智"
_WW_LEN         = len(WAKE_WORD)
_WWS_LEN        = len(WAKE_WORD_SHORT)
_WW_PINYIN      = "".join(lazy_pinyin(WAKE_WORD))       # "xiaozhixiaozhi"
_WWS_PINYIN     = "".join(lazy_pinyin(WAKE_WORD_SHORT)) # "xiaozhi"

# ── 识别文本有效性判断 ─────────────────────────────────────
# 去掉标点/空白后剩下的“有效字符”（中日文字、字母、数字）
_CORE_RE = re.compile(r"[^\w一-鿿]", re.UNICODE)
# VAD 误切常产出单字碎片（如 "我。"）。默认要求 ≥2 个有效字才算有效发言，
# 但下列单字是真实的简短回答/短指令，需放行（尤其持续监听里回答问句时）。
VALID_SHORT = {"好", "嗯", "对", "是", "不", "行", "要", "停", "走", "来", "去"}


def is_meaningful(text: str) -> bool:
    """判断识别文本是否为有效发言（排除纯标点，以及单字误切碎片）。"""
    if not text:
        return False
    core = _CORE_RE.sub("", text)   # 去掉所有标点和空白
    if not core:
        return False                # 纯标点，如 "。"
    if len(core) >= 2:
        return True                 # 两字及以上视为正常发言
    return core in VALID_SHORT      # 单字：只放行白名单里的有效简短回答


# ── LLM 配置 ──────────────────────────────────────────────
# 必须设置环境变量 DASHSCOPE_API_KEY，不提供默认值（防止密钥泄露）
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError(
        "未设置 DASHSCOPE_API_KEY 环境变量。\n"
        "请在终端执行: export DASHSCOPE_API_KEY=your_key_here\n"
        "或在 ~/.bashrc 中永久添加此行。"
    )
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL    = "qwen3.6-flash"

# ── TTS 配置 ──────────────────────────────────────────────
# 可用中文音色（CosyVoice v2）：
#   longxiaochun_v2（男，成熟）longxiaoxia_v2（女，温柔）
#   longxiaobai_v2（男，活泼）longxiaomiao_v2（女，知性）
TTS_VOICE       = "longxiaochun_v2"
TTS_SAMPLE_RATE = 16000

# ── 紧急联络发起（负面情绪触发）─────────────────────────────
# 检测到稳定的负面情绪时，机器人问一句，即刻开确认窗口，15 秒内没人叫停就发
# /emergency/initiate 给紧急方。和外部发起方的流程完全相同。
USER_NAME = "肖奶奶"           # 称呼，用于固定话术

# 情绪 → 联络方式。call = 拨打紧急联系人电话；sms = 发送短信。
# 当前两档都只发短信。`call` 仍是有效取值，将来升级改这里即可。
# **未列出的情绪不触发联络流程**，仍走大模型安抚。
EMERGENCY_CHANNEL_BY_EMOTION = {
    "negative_distress": "sms",
    "low_mood":          "sms",
}

EMERGENCY_ABORT_WINDOW_SEC = 15.0   # 播报后等待叫停的时长（秒）
EMERGENCY_COOLDOWN_SEC     = 30.0   # 两次发起的最小间隔（秒）

# 固定话术（不走大模型：紧急场景要的是确定性）
EMERGENCY_ASK_TEXT = (
    f"{USER_NAME}，你不舒服吗，需要我帮你联系家人或者医生吗"
)
EMERGENCY_CANCELLED_TEXT = "好的，那我就不联系了。您要是不舒服，随时叫我。"

# ── 歌曲播放 ──────────────────────────────────────────────
# 曲库目录：放 mp3/wav/flac/ogg，文件名（不含扩展名）即歌名，
# 会自动成为【可用指令集】里"播放歌曲"的可选值，LLM 提示词与校验一并跟随。
# 曲库为空时，"播放歌曲"指令不会出现，机器人也就不会承诺唱歌。
SONG_DIR     = os.getenv("SONG_DIR", "") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "songs")
SONG_MAX_SEC = 60.0     # 单曲播放上限（秒），超时自动淡出停止
