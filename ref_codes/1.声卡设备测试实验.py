import os

# 获取麦克风设备名（通常为 alsa_input.pci-xxxx 或类似）
# 你可以先手动运行 `pactl list sources` 找到你的麦克风名字

# 假设你的麦克风是这个（请根据你的系统替换）
mic_name = "alsa_input.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo"

# 获取当前静音状态
os.system(f'pactl get-source-mute {mic_name}')

# 静音麦克风
#os.system(f'pactl set-source-mute {mic_name} 1')  # 1 表示静音

# 取消静音
os.system(f'pactl set-source-mute {mic_name} 0')  # 0 表示取消静音