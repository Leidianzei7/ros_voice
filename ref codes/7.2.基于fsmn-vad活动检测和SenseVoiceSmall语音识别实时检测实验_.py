import torch
import os
import pyaudio, queue, threading, time
import soundfile as sf
import numpy as np
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

q = queue.Queue()
running = threading.Event(); running.set()

model = AutoModel(
    model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_itn=True, disable_pbar=True
)

CHUNK, RATE = 1024, 16000
def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    print("Listening...")
    while running.is_set():
        q.put(np.frombuffer(stream.read(CHUNK), dtype=np.int16))
    stream.stop_stream(); stream.close(); p.terminate()

def speech2text(audio_file, language="zh"):
    # 语言可选：zh / en / yue / ja / ko / nospeech
    res = model.generate(
        input=audio_file,
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    text = rich_transcription_postprocess(res[0]["text"])
    return text

def recognize():
    buffer = np.array([], dtype=np.int16)
    while running.is_set():
        chunk = q.get()
        buffer = np.concatenate([buffer, chunk])
        if len(buffer) > RATE * 30:  # 保底 30s 滑动窗
            buffer = buffer[-RATE*30:]
        # 简单能量触发（生产建议用 webrtcvad/fsmn-vad 流式）
        rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
        if rms > 300:  # 阈值按麦克风与环境调优
            print("🎤 Speaking...")
            # 临时写文件触发识别（也可改为内存/临时文件）
            tmp = f"tmp_{int(time.time())}.wav"
            sf.write(tmp, buffer, RATE)
            text = speech2text(tmp, language="zh")
            print("ASR:", text)
            os.remove(tmp)


t1 = threading.Thread(target=record); t2 = threading.Thread(target=recognize)
t1.start(); t2.start()
try: input("按回车停止...\n")
finally:
    running.clear(); t1.join(); t2.join()