from http import HTTPStatus
import matplotlib.pyplot as plt
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from PIL import Image
from io import BytesIO
from dashscope import ImageSynthesis
import matplotlib.pyplot as plt
import dashscope
from clbAImodule import ClbAsrModule
import os
import cv2
import pyaudio
import sounddevice as sd
import soundfile as sf
import wave
import numpy as np
import threading
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import Ollama_Qwen2_5vl
import traitlets
from qwen_vl_utils import process_vision_info
import torch
from funasr import AutoModel
import pygame
import time
import edge_tts
import asyncio
from time import sleep
import ffmpeg

#os.environ["QT_QPA_PLATFORM"] = "xcb"

#=============由于部署多个大模型会导致内存不足，这里采用在线调用qwen-image-plus进行文生图
# 生成自己的key，登陆阿里百炼平台平台注册自己的账号
os.environ['DASHSCOPE_API_KEY'] = 'sk-ba3ab6823ba940838b2bf0aac6912272'
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
api_key = os.getenv("DASHSCOPE_API_KEY")

# ============语音模块初始化==============
tty = ClbAsrModule(port='/dev/ttyUSB0')  # 实例化串口
#========================================
startflag = False # 开始录制标志位
endflag  = True # 结束录制标志位

# --- 播放音频 -
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)  # 等待音频播放结束
        print("播放完成！")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

# 配置音频参数
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# 配置视频参数
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30.0

# 文件保存路径
TEMP_AUDIO_FILE = "temp_audio.wav"
TEMP_VIDEO_FILE = "temp_video.avi"

# 音频录制线程
def record_audio(stop_event):
    # time.sleep(5)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    print("开始录音...")

    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束。")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存音频
    with wave.open(TEMP_AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# 视频录制线程
def record_video(stop_event):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)  # 明确使用 V4L2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(TEMP_VIDEO_FILE, fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT))
    print("开始录像...")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 退出摄像头窗口
                stop_event.set()
        else:
            break

    print("录像结束。")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 合并音视频
def merge_audio_video(audio_file, video_file, output_file):
    print("正在合并音频和视频...")
    ffmpeg.input(video_file).output(audio_file, output_file, vcodec='copy', acodec='aac', strict='experimental').run(overwrite_output=True)
    print(f"合并完成，文件保存为: {output_file}")

#================串口读取函数===================
def on_space_key_press(event):
     global startflag
     global endflag
     startflag = True # 使能开始录制标志位
     endflag = False # 结束录制标志位失能


def on_space_key_release(event):
    global startflag
    global endflag

    startflag = False  # 开始录制标志位失能
    endflag = True  # 结束录制标志位使能


# 串口监听函数
def my_interrupt():
    global dstimer
    on_space_key_release(None)  # 回答
    #dstimer.stop() # 停止定时中断

def read_serial(stop_event):
    global dstimer
    global startflag
    global endflag
    while not stop_event.is_set():
        if tty.wakeup(): # 检测到唤醒词
            data, samplerate = sf.read('wakeup.wav')
            sd.play(data, samplerate) # 播放
            sd.wait()  # 等待播放完成
            dstimer = threading.Timer(8.0, my_interrupt)  # 创建定时中断，时间到呼喊小智
            dstimer.start()  # 开始定时
            on_space_key_press(None) # 调用按键按下记录

        time.sleep(0.01)  # 短暂休眠，减少 CPU 占用


# 主函数
def main():
    global startflag
    global endflag
    stop_event = threading.Event()
    
    # 启动音频和视频录制线程和串口线程
    audio_thread = threading.Thread(target=record_audio, args=(stop_event,))
    video_thread = threading.Thread(target=record_video, args=(stop_event,))
    serial_thread = threading.Thread(target=read_serial, args=(stop_event,))
    serial_thread.start() 


    print("开始录制...")
    while startflag == False:  # 等待唤醒 开始录制 小创小创
        print("⏳ 等待唤醒词！")
        time.sleep(0.1)

    print("🎤 录制中...")
    
    audio_thread.start()
    video_thread.start()
    
    while endflag == False:  # 等待录制定时结束
        pass
        time.sleep(0.1)

    stop_event.set()

    audio_thread.join()
    video_thread.join()
    serial_thread.join()


    print("⏳ 调用大模型开始识别！")

# -------- SenceVoice 语音识别 --模型加载-----
model_dir = "iic/SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True, )

if __name__ == "__main__":
    while 1:
        main()
        folder_path = "./Test_QWen2_VL/"
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, "captured_image.jpg")  # 设置保存路径
        cap = cv2.VideoCapture(TEMP_VIDEO_FILE)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = int(total_frames // 2)
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取帧索引 {frame_index}")
        else:
            # 显示帧
            cv2.imwrite(file_path, frame)
            # cv2.imshow(f"Frame {frame_index}", frame)

        # -------- SenceVoice 推理 ---------
        input_file = (TEMP_AUDIO_FILE)
        res = model_senceVoice.generate(
            input=input_file,
            cache={},
            language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
        )
        prompt = res[0]['text'].split(">")[-1] + ',请用中文回答！'
        # ---------SenceVoice --end----------

        # -------- Ollama_QWen2.5-VL 模型推理 ---------
        results = Ollama_Qwen2_5vl.ask_qwen25vl(f"{file_path}", f"{prompt}")
 
        print(results)

        # 输入文本
        text = results
        # asyncio.run(amain(text, "zh-CN-YunxiaNeural", os.path.join(folder_path,"sft_0.mp3")))
        # play_audio(f'{folder_path}/sft_0.mp3')

        asyncio.run(amain(text, "zh-CN-XiaoyiNeural", os.path.join(folder_path,"sft_0.mp3")))
        play_audio(f'{folder_path}/sft_0.mp3')

        # 播放生成类似图提醒
        play_audio("shengcheng.wav")
        time.sleep(1) # 暂停1s
        print('----同步调用，请等待任务执行----')

        rsp = ImageSynthesis.call(api_key=api_key,
                          model="qwen-image-plus",
                          prompt=text,
                          n=1,
                          size='1328*1328',
                          prompt_extend=True,
                          watermark=True)
        print('response: %s' % rsp)
        if rsp.status_code == HTTPStatus.OK:
            # 在当前目录下保存图片
            for result in rsp.output.results:
                file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                with open('./output/%s' % file_name, 'wb+') as f:
                    f.write(requests.get(result.url).content)
                    img = Image.open(BytesIO(requests.get(result.url).content))
                    img_np = np.array(img)
                    #img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_bgr = cv2.resize(img_np, (800, 600))
                    plt.imshow(img_bgr)                      # 彩色：确保为 RGB；灰度：见下节
                    plt.axis('off')                      # 隐藏坐标轴，画面更整洁
                    plt.show()
        else:
            print('同步调用失败, status_code: %s, code: %s, message: %s' %
                (rsp.status_code, rsp.code, rsp.message))

