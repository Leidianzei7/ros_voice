#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np

# onnxruntime 在无 GPU 机器上 import 时会向 C 层 stderr 打印 WARNING，
# 用 fd 重定向在 C 层屏蔽（Python 的 sys.stderr 重定向对此无效）
def _import_ort_silent():
    _fd = 2  # C 层 stderr fd
    _saved = os.dup(_fd)
    _null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_null, _fd)
    os.close(_null)
    try:
        import onnxruntime as _ort
        return _ort
    finally:
        os.dup2(_saved, _fd)
        os.close(_saved)

ort = _import_ort_silent()
ort.set_default_logger_severity(3)   # 屏蔽 Session 创建时的后续 ORT 日志
import torch
from funasr.frontends.wav_frontend import WavFrontend

_DIR   = os.path.dirname(os.path.abspath(__file__))
_ONNX  = os.path.join(_DIR, '..', 'onnx_model', 'model_quant.onnx')
_CACHE = os.path.expanduser('~/.cache/modelscope/hub/models/iic/SenseVoiceSmall')

# ── 词表 ──────────────────────────────────────────────────────
with open(os.path.join(_CACHE, 'tokens.json')) as _f:
    _VOCAB = json.load(_f)   # list[str], len=25055

# SenseVoice 紧凑嵌入表索引（独立于词表，非 token ID）
# language: 0=auto, 3=zh, 4=en, 7=yue, 14=ja, 17=ko
# textnorm: 14=withitn（启用 ITN）, 15=woitn（不启用 ITN）
_LANG_ZH      = 3
_TEXTNORM_ITN = 14
_BLANK        = 0    # CTC blank

# ── CMVN（Kaldi am.mvn 格式，560-dim LFR 特征）────────────────
def _load_cmvn(path):
    with open(path) as f:
        txt = f.read()
    shift = np.fromstring(
        re.search(r'<AddShift>.*?<LearnRateCoef>\s+\S+\s+\[([^\]]+)\]', txt, re.DOTALL).group(1),
        dtype=np.float32, sep=' ')
    scale = np.fromstring(
        re.search(r'<Rescale>.*?<LearnRateCoef>\s+\S+\s+\[([^\]]+)\]', txt, re.DOTALL).group(1),
        dtype=np.float32, sep=' ')
    return shift, scale   # normalized = (x + shift) * scale

_cmvn_shift, _cmvn_scale = _load_cmvn(os.path.join(_CACHE, 'am.mvn'))

# ── 特征提取前端（FunASR WavFrontend，无 CMVN）───────────────────
_frontend = WavFrontend(
    cmvn_file=None,
    fs=16000,
    window='hamming',
    n_mels=80,
    frame_length=25,
    frame_shift=10,
    lfr_m=7,
    lfr_n=6,
    dither=0.0,
)

def _extract(audio_np: np.ndarray):
    """float32 16kHz → (T, 560) CMVN 归一化特征"""
    t    = torch.from_numpy(audio_np).float().unsqueeze(0)
    tlen = torch.tensor([len(audio_np)])
    with torch.no_grad():
        feats, flen = _frontend(t, tlen)
    feats = feats.numpy()[0]                           # (T, 560)
    feats = (feats + _cmvn_shift) * _cmvn_scale
    return feats.astype(np.float32), int(flen.item())

# ── ONNX 推理会话 ──────────────────────────────────────────────
_ONNX_OPT = os.path.join(_DIR, '..', 'onnx_model', 'model_quant_opt.onnx')

print("正在加载语音识别模型（ONNX INT8）...")
_opts = ort.SessionOptions()
# 默认 execution_mode = SEQUENTIAL，inter_op 不生效，不设；SenseVoice 是串行 encoder 也用不上
_opts.intra_op_num_threads = 4   # 8 核 ARM Cortex-A55，留 4 核给音频/系统/Python 主线程
_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 首次运行生成硬件优化缓存，后续直接加载缓存（~7s 更快）
if os.path.exists(_ONNX_OPT):
    _model_to_load = _ONNX_OPT
else:
    _opts.optimized_model_filepath = _ONNX_OPT
    _model_to_load = _ONNX
# 显式指定 CPU provider，跳过 GPU 设备扫描，消除无 GPU 时的警告
_sess = ort.InferenceSession(_model_to_load, sess_options=_opts,
                             providers=['CPUExecutionProvider'])
print("模型加载完成！\n")

# ── CTC 贪心解码 ───────────────────────────────────────────────
def _ctc_decode(logits: np.ndarray, enc_len: int) -> list:
    ids = np.argmax(logits[:enc_len], axis=-1)
    ids = ids[np.concatenate([[True], ids[1:] != ids[:-1]])]   # 去重复
    return ids[ids != _BLANK].tolist()                          # 去 blank

def _to_text(ids: list) -> str:
    toks = [_VOCAB[i] for i in ids if i < len(_VOCAB)]
    text = ''.join(toks).replace('▁', ' ').strip()
    # 去除所有 <|xxx|> 特殊标记（情感、事件、语言标签等）
    text = re.sub(r'<\|[^|]*\|>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ── 公开接口（与 torch 版签名一致）────────────────────────────
def recognize(audio_np: np.ndarray) -> str:
    if len(audio_np) < 1600:   # < 0.1 s，直接跳过
        return ""
    feats, flen = _extract(audio_np)
    logits, enc_lens = _sess.run(None, {
        'speech':         feats[np.newaxis],
        'speech_lengths': np.array([flen],          dtype=np.int32),
        'language':       np.array([_LANG_ZH],      dtype=np.int32),
        'textnorm':       np.array([_TEXTNORM_ITN], dtype=np.int32),
    })
    ids  = _ctc_decode(logits[0], enc_lens[0])
    return _to_text(ids)
