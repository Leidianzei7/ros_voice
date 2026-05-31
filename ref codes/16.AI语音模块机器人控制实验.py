# AI 模块切换到离线模式
from clbAImodule import ClbAsrModule
import time

# ============语音模块初始化==============
tty = ClbAsrModule(port='/dev/ttyUSB0')  # 实例化串口
#========================================
# 返回命令
FORWARDCOM =  b'\xaa\x55\x00\x01\xfb'
BACKCOM    =  b'\xaa\x55\x00\x02\xfb'
LEFTCOM    =  b'\xaa\x55\x00\x03\xfb'
RIGHTCOM   =  b'\xaa\x55\x00\x04\xfb'

bobao =  b'\xaa\x55\xff\x01\xfb'


while True:
    '''
    # 发送数据
    tty.writedata(bobao)  
    time.sleep(2)
    '''
    recv_data = tty.detect()
    if(recv_data == FORWARDCOM):
        # 调用机器人控制程序
        #clbrobot.t_up(0.5,1.5)  
        #clbrobot.t_stop(1)
        print("收到前进指令！")
    elif(recv_data == BACKCOM):
        #clbrobot.t_down(0.5,1.5)
        #clbrobot.t_stop(1)
        print("收到后退指令！")
    elif(recv_data == LEFTCOM):
        #clbrobot.turnLeft(0.5,1.5)
        #clbrobot.t_stop(1)
        print("收到左转指令！")
    elif(recv_data == RIGHTCOM):
        #clbrobot.turnRight(0.5,1.5)
        #clbrobot.t_stop(1)
        print("收到右转指令！")