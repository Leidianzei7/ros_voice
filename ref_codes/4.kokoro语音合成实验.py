from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import torch
import pygame
import time

# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
# --- 播放音频 

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


pipeline = KPipeline(lang_code='z') # <= 请务必确保语言代码与语音设置相匹配，参照上述说明。

text =  '湖南创乐博智能科技，专注智能创新，以科技赋能生活，打造智慧未来，引领行业潮流，助力梦想启航！！'

# 离线语言合成及播放
def speech_synthesis(text):
    generator = pipeline(text, voice='zf_xiaoyi', # <= 改变说话角色
    speed=1, split_pattern=r'\n+')
    for i, (gs, ps, audio) in enumerate(generator):
        print(i)  # i => 索引
        print(gs) # gs => 文本
        #print(ps) # ps => 音素
        #display(Audio(data=audio, rate=24000, autoplay=i==0))
        sf.write(f'{i}.wav', audio, 24000) # 保存音频文件
        play_audio(f'{i}.wav')

speech_synthesis(text)