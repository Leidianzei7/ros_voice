#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import sys
from funasr import AutoModel
import torch

model_dir =  "iic/SenseVoiceSmall" #"QWen/pretrained_models/SenseVoiceSmall"
input_file = ("https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")

model = AutoModel(
    model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_itn=True, disable_pbar=True
)

res = model.generate(
    input=input_file,
    cache={},
    language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=False,
)

print(res)
# import pdb; pdb.set_trace()
print(res[0]['text'].split(">")[-1])
