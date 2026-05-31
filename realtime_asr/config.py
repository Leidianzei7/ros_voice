#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pypinyin import lazy_pinyin

# ── 设备配置 ──────────────────────────────────────────────
# 运行 `python3 -c "import sounddevice as sd; print(sd.query_devices())"` 查看设备列表
DEVICE_INDEX        = 1      # 麦克风输入设备编号（USB PnP Audio Device）
OUTPUT_DEVICE_INDEX = 1      # 扬声器输出设备编号（同一 USB PnP 设备）
HW_SAMPLE_RATE      = 48000  # 硬件原生采样率
SAMPLE_RATE         = 16000  # SenseVoiceSmall 要求 16kHz，代码内自动重采样
CHANNELS            = 1
CHUNK               = 1024   # 每次读取帧数，越小延迟越低

# ── VAD 参数 ──────────────────────────────────────────────
SPEECH_HOLD_SEC = 1.2   # 停顿多久后触发识别（秒）
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

# ── LLM 配置 ──────────────────────────────────────────────
# 推荐通过环境变量 DASHSCOPE_API_KEY 传入，避免明文写在代码里
LLM_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "sk-554e53eda18e429d966101bebe10d492")
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL    = "qwen-turbo"

# ── TTS 配置 ──────────────────────────────────────────────
# 可用中文音色（CosyVoice v2）：
#   longxiaochun_v2（男，成熟）longxiaoxia_v2（女，温柔）
#   longxiaobai_v2（男，活泼）longxiaomiao_v2（女，知性）
TTS_VOICE       = "longxiaochun_v2"
TTS_SAMPLE_RATE = 16000
