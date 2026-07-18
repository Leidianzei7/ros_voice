# ros_voice

ROS 2 语音交互功能包。麦克风采音 → 离线 ASR → 唤醒词检测 → LLM 理解 → TTS 播报 → 底盘/机械臂控制，全流程以 ROS topic 串联。

## 架构

```
麦克风
  │
  ▼
voice_node  ──/voice/command──▶  brain_node  ──/command──▶  control_node
  │                                  │                            │
采音+VAD+ASR                    LLM推理+TTS播报              /cmd_vel 底盘控制
唤醒词过滤                  视觉/情绪上下文注入              /arm/grasp_command 机械臂
```

### 节点说明

| 节点 | 订阅 | 发布 | 功能 |
|------|------|------|------|
| `voice_node` | `/voice/listen_mode` `/voice/mute` | `/voice/command` (String) | 麦克风采音、VAD 切句、SenseVoice ONNX 识别、唤醒词过滤 |
| `brain_node` | `/voice/command` `/vision/scene_objects` `/vision/emotion_context` | `/command` (String, JSON) `/voice/listen_mode` | Qwen LLM 语义理解、视觉/情绪上下文注入、TTS 口语回复 |
| `control_node` | `/command` | `/cmd_vel` (Twist) `/arm/grasp_command` (String) | 解析 JSON 指令数组，驱动底盘 + 机械臂 |

## 目录结构

```
ros_voice/
├── ros_voice/              # ROS 2 节点（voice_node / brain_node / control_node）
├── voice_brain_module/     # 核心库：ASR / VAD / LLM / TTS / 唤醒词
│   ├── audio.py            #   音频管道（麦克风→重采样→VAD→ASR）
│   ├── asr.py              #   SenseVoice ONNX 语音识别
│   ├── vad.py              #   语音活动检测（energy / WebRTC）
│   ├── wake_word.py        #   唤醒词匹配（中文 + 拼音）
│   ├── llm.py              #   LLM 调用 + 系统提示词构建
│   ├── commands.py         #   指令 schema 定义 + 校验
│   ├── tts.py              #   TTS 语音合成（CosyVoice v2）
│   ├── pipeline.py         #   ROS 端高层管道接口
│   ├── context.py          #   感知上下文去抖管道
│   ├── config.py           #   全部配置
│   └── main.py             #   Standalone 入口（不依赖 ROS）
├── launch/
│   └── voice.launch.py     # 一键启动三节点
├── test/
│   ├── start_voice_brain.py # 无 ROS 独立测试入口
│   └── monitor_rms.py      # 麦克风 RMS 实时监测（校准 VAD 阈值用）
├── ref_codes/              # 历史实验脚本（参考用）
├── ref_docs/               # 参考文档
├── 接口对接文档.md          # 对外接口契约（供其他部门对接用）
├── requirements.txt
├── package.xml
└── setup.py
```

## 模型文件

SenseVoice Small ONNX 模型（~460MB）**不入库**，置于 workspace 级目录：

```
~/ros2_ws/models/sensevoice/
├── model_quant.onnx         # SenseVoice Small INT8 量化模型（必须）
└── model_quant_opt.onnx     # 首次运行时自动生成的优化缓存
```

模型路径优先级：环境变量 `SENSEVOICE_MODEL_DIR` > 自动探测 `~/ros2_ws/models/sensevoice/`。

模型从 ModelScope 获取：`iic/SenseVoiceSmall`。

## 关键配置

配置文件：`voice_brain_module/config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | （必填，无默认值） | 阿里云 API Key，**必须通过环境变量设置** |
| `DEVICE_NAME` | `USB PnP Audio Device` | 麦克风设备名（子串匹配） |
| `OUTPUT_DEVICE_NAME` | `USB PnP Audio Device` | 扬声器设备名 |
| `WAKE_WORD` | `小智小智` | 唤醒词（支持短词「小智」及拼音模糊匹配） |
| `VAD_MODE` | `energy` | VAD 模式：`energy`（动态阈值）或 `webrtc` |
| `SPEECH_DELTA` | 5000 | 能量 VAD 灵敏度，值越小越灵敏 |
| `LLM_MODEL` | `qwen-turbo` | DashScope LLM 模型 |
| `TTS_VOICE` | `longxiaochun_v2` | CosyVoice v2 音色 |

```bash
export DASHSCOPE_API_KEY=your_key_here
```

## 支持的指令

`control_node` 当前支持以下动作（由 LLM 生成 JSON，经 `voice_brain_module/commands.py` 校验后执行）：

| 执行机构 | 动作 | 参数 |
|----------|------|------|
| 底盘 | 前进 | `speed`（m/s）、`distance`（m） |
| 底盘 | 后退 | `speed`（m/s）、`distance`（m） |
| 底盘 | 左转 | `speed`（rad/s）、`angle`（°） |
| 底盘 | 右转 | `speed`（rad/s）、`angle`（°） |
| 底盘 | 停止 | 无 |
| 底盘 | 人体跟踪 | `状态`（开始/停止） |
| 机械臂 | 抓取 | `target`（苹果/香蕉/瓶子/蛋糕/小黄鸭/绿色药盒/大樱桃） |

## 安装与运行

```bash
# 1. 安装 Python 依赖
pip install -r ~/ros2_ws/src/ros_voice/requirements.txt

# 2. 放置模型文件
mkdir -p ~/ros2_ws/models/sensevoice
# 将 model_quant.onnx 放入上述目录

# 3. 编译
cd ~/ros2_ws
colcon build --symlink-install --packages-select ros_voice

# 4. 一键启动
./start_voice.sh

# 或手动启动
source ~/ros2_ws/install/setup.bash
ros2 launch ros_voice voice.launch.py

# 不依赖 ROS，独立测试语音交互
python3 -m voice_brain_module
```

## 依赖

- **ROS 2 Humble**
- **Python 3.10**
- 见 `requirements.txt`（torch 2.4.1 CPU、funasr、onnxruntime、dashscope、openai 等）

## 与旧版的区别

| | 旧版（`Genshin/TTS`） | 当前版 |
|---|---|---|
| 项目定位 | 独立 Python 项目 | 标准 ROS 2 功能包 |
| 核心库名 | `realtime_asr/` | `voice_brain_module/`（含听/想/说全链路） |
| 模型存放 | `onnx_model/` 在源码包内 | `~/ros2_ws/models/sensevoice/` 外置 |
| 编译方式 | 包内误执行 `colcon build` | 在 `ros2_ws/` 根执行，产物在 `build/install/log/` |
| 工作空间 | 软链接挂载 | 直接实体目录 |
| 测试入口 | 项目根 `main.py` | `test/start_voice_brain.py` 或 `python3 -m voice_brain_module` |

## License

MIT
