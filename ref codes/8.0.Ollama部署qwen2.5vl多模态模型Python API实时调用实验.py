import base64, requests, json
from playsound import playsound
from funasr import AutoModel
import pyaudio
import asyncio 
import edge_tts 


model = AutoModel(model="paraformer-zh-streaming")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

VOICE = "zh-CN-XiaoyiNeural"  
OUTPUT_FILE = "/home/pi/Largemodel/edgetts1.mp3"  

async def my_function(TEXT):
    tts = edge_tts.Communicate(text=TEXT, voice=VOICE)
    await tts.save(OUTPUT_FILE)
    # play audio 
    playsound(OUTPUT_FILE)

def img2base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_qwen25vl(image_path, prompt, model="qwen2.5vl:3b", stream=False):
    b64 = img2base64(image_path)
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt, "images": [b64]}
        ],
        "stream": stream
    }
    resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    resp.raise_for_status()
    if stream:
        return "".join(chunk.get("message", {}).get("content", "") for chunk in resp.iter_lines()
                       if chunk and (line := json.loads(chunk)).get("message"))
    else:
        return resp.json()["message"]["content"]

# 调用大模型
results = ask_qwen25vl("demo.jpg", "请用中文详细描述这张图片。")
print(results) # 打印出来
asyncio.run(my_function(results))

    