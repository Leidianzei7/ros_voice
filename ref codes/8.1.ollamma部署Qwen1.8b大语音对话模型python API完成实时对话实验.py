import time
import asyncio  
import os
import requests
import json
import edge_tts  
import pyaudio
from playsound import playsound
from funasr import AutoModel
import time

# 假设你的麦克风是这个（请根据你的系统替换）
mic_name = "alsa_input.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo"
model = AutoModel(model="paraformer-zh-streaming")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

cache={}
chunk_stride=9600
chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention


def get_llm(text):
    url = 'http://localhost:11434/api/chat'
    headers = {'Content-Type': 'application/json'}

    pload = {
        "model": "qwen:1.8b-chat",
        "messages": [
            {"role": "system", "content": "你叫小创，是一个18岁的女大学生，性格活泼开朗，说话俏皮"},
            {
                "role": "user",
                "content": text
            }
        ]
    }

    response = requests.post(url, headers=headers, json=pload, stream=True)

    return response


VOICE = "zh-CN-XiaoyiNeural"  
OUTPUT_FILE = "/home/pi/Largemodel/edgetts1.mp3"  

async def my_function(TEXT):
    tts = edge_tts.Communicate(text=TEXT, voice=VOICE)
    await tts.save(OUTPUT_FILE)
    # play audio 
    playsound(OUTPUT_FILE)


texts = ""
last_active_time = time.time()
print("begin record ...") # 开始录制声音
os.system(f'pactl set-source-mute {mic_name} 0')  # 取消静音，表示可以正常录制声音
while True:
    buf = stream.read(chunk_stride)
    ## ASR
    res = model.generate(input=buf, cache=cache
                         , is_final=False, chunk_size=chunk_size
                         , encoder_chunk_look_back=encoder_chunk_look_back
                         , decoder_chunk_look_back=decoder_chunk_look_back)
    # print(res)
    text = res[0]['text']  

    if text:
        texts+=text
        last_active_time = time.time()
    else:
        if time.time() - last_active_time > 1.5:
            if texts:
                print("Human:",texts)
                ## 给到大模型LLM
                results = get_llm(texts + "，回答简短一些，保持50字以内！")
                print("AI:", end='')
                last_printed_text = ""
                # 处理流式数据
                for chunk in results.iter_content(chunk_size=1024):
                    if chunk:
                        # print(chunk.decode("utf-8"))
                        # 解析JSON数据
                        response_data = json.loads(chunk.decode("utf-8"))
                        # 打印结果或进行其他处理
                        new_text = response_data["message"]["content"]
                        print(new_text, end='', flush=True)
                        last_printed_text += new_text
				## TTS
                os.system(f'pactl set-source-mute {mic_name} 1') # 静音设备，避免回采
                time.sleep(0.5) # 延时
                asyncio.run(my_function(last_printed_text))
                time.sleep(0.5) # 延时
                os.system(f'pactl set-source-mute {mic_name} 0')  # 取消静音，表示可以正常录制声音

            texts=""
            


  
