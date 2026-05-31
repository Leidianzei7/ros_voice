#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import queue
import threading

# TTS 待合成文本队列：generate_response → tts_playback_thread
tts_text_q = queue.Queue()

# 主循环运行标志
running = threading.Event()
running.set()

# 唤醒后等待指令的标志
waiting_for_command = False
