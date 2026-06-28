import pyttsx3
# 初始化 TTS 引擎
engine = pyttsx3.init()

voices = engine.getProperty('voices')
# 打印出所有的音色信息
for voice in voices:
    print(f'id = {voice.id}----name = {voice.name}')

# 设置语音属性
engine.setProperty('rate', 150)    # 语速engine.setProperty('volume', 0.9)  # 音量（0.0 到 1.0）

# 优先选择中文语音
for v in voices:
    if 'zh' in v.id or 'cn' in v.id or 'mb' in v.id or 'mandarin' in v.name.lower():
        engine.setProperty('voice', v.id)
        print(f"已选择中文语音：{v.id}")
        break
# 选择语音
#voices = engine.getProperty('voices')
#print(voices)
#engine.setProperty('voice', voices[0].id)  # 使用第一个语音
# 输入文本
text = "你好，今天天气很好，适合爬山"
# 朗读文本
engine.say(text)
# 等待朗读完成
engine.runAndWait()

'''
import pyttsx3

engine = pyttsx3.init()

# 可选：查看可用语音列表，确认存在中文语音（名称通常含 zh、cn、mb/mb-zh 等关键字）
voices = engine.getProperty('voices')
print("可用语音：")
for v in voices:
    print(v)

# 优先选择中文语音
for v in voices:
    if 'zh' in v.id or 'cn' in v.id or 'mb' in v.id or 'mandarin' in v.name.lower():
        engine.setProperty('voice', v.id)
        print(f"已选择中文语音：{v.id}")
        break

# 可选：调节语速与音量
engine.setProperty('rate', 150)   # 语速（默认约 200）
engine.setProperty('volume', 0.9) # 音量 [0.0~1.0]

# 播放中文
engine.say("你好，这是 Jetson Orin Nano 上的中文语音测试。")
engine.runAndWait()
engine.save_to_file("欢迎使用 pyttsx3 中文语音合成。", "output.wav")
engine.runAndWait()
'''