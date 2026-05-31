# ros_voice

ROS 2 语音交互功能包。麦克风采音 → 离线 ASR → 唤醒词检测 → LLM 理解 → TTS 播报 → 底盘运动控制，全流程以 ROS topic 串联。

## 架构

```
麦克风
  │
  ▼
voice_node  ──/voice/command──▶  brain_node  ──/command──▶  control_node
  │                                  │                            │
采音+VAD+ASR                    LLM推理+TTS播报              /cmd_vel底盘控制
唤醒词过滤
```

### 节点说明

| 节点 | 订阅 | 发布 | 功能 |
|------|------|------|------|
| `voice_node` | — | `/voice/command` (String) | 麦克风采音、VAD 切句、SenseVoice ONNX 识别、唤醒词过滤 |
| `brain_node` | `/voice/command` | `/command` (String, JSON) | Qwen LLM 语义理解、生成执行指令 + TTS 口语回复 |
| `control_node` | `/command` | `/cmd_vel` (Twist) | 解析 JSON 指令数组，驱动底盘执行运动 |

## 目录结构

```
ros_voice/
├── ros_voice/          # ROS 2 节点（voice_node / brain_node / control_node）
├── realtime_asr/       # 核心模块（ASR / VAD / TTS / LLM / 唤醒词）
├── onnx_model/         # SenseVoice Small INT8 量化 ONNX 模型
├── test/               # 独立测试脚本（不依赖 ROS）
│   ├── main.py         # 完整语音交互流程（独立运行）
│   └── monitor_rms.py  # 麦克风 RMS 实时监测（校准 VAD 阈值用）
├── launch/
│   └── voice.launch.py # 一键启动三节点
├── ref codes/          # 历史实验脚本（参考用）
├── ref docs/           # 参考文档
├── package.xml
└── setup.py
```

## 关键配置

配置文件：`realtime_asr/config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEVICE_INDEX` | 1 | 麦克风设备编号 |
| `OUTPUT_DEVICE_INDEX` | 1 | 扬声器设备编号 |
| `WAKE_WORD` | `小智小智` | 唤醒词（支持拼音模糊匹配） |
| `VAD_MODE` | `energy` | VAD 模式：`energy`（动态阈值）或 `webrtc` |
| `SPEECH_DELTA` | 5000 | 能量 VAD 灵敏度，值越小越灵敏 |
| `LLM_MODEL` | `qwen-turbo` | DashScope LLM 模型 |
| `TTS_VOICE` | `longxiaochun_v2` | CosyVoice v2 音色 |

LLM API Key 通过环境变量传入：

```bash
export DASHSCOPE_API_KEY=your_key_here
```

## 支持的底盘指令

`control_node` 当前支持以下动作（由 LLM 生成 JSON 后执行）：

| 动作 | 参数 |
|------|------|
| 前进 | `speed`（m/s）、`distance`（m） |
| 后退 | `speed`（m/s）、`distance`（m） |
| 左转 | `speed`（rad/s）、`angle`（°） |
| 右转 | `speed`（rad/s）、`angle`（°） |
| 停止 | 无 |

## 模型文件

`onnx_model/` 目录中的模型文件体积较大（~460MB），不随仓库分发。克隆后需手动放置：

```
onnx_model/
├── model_quant.onnx        # SenseVoice Small INT8 量化模型（必须）
└── model_quant_opt.onnx    # 首次运行时自动生成，无需手动放置
```

模型可从原始项目或 ModelScope 导出获取：`iic/SenseVoiceSmall`。

## 安装与运行

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt   # 位于 ros2_ws/ 根目录

# 2. 编译
cd ~/ros2_ws
colcon build --symlink-install --packages-select ros_voice

# 3. 启动（自动 source + launch）
./start_voice.sh

# 或手动启动
source ~/ros2_ws/install/setup.bash
ros2 launch ros_voice voice.launch.py

# 不依赖 ROS，直接测试语音交互
python src/ros_voice/test/main.py
```

## 依赖

- **ROS 2 Humble**
- **Python 3.10**
- 见 `requirements.txt`（torch 2.4.1 CPU、funasr、onnxruntime、dashscope、openai 等）

## 与旧版的区别

| | 旧版（`Genshin/TTS`） | 当前版 |
|---|---|---|
| 项目定位 | 独立 Python 项目 | 标准 ROS 2 功能包 |
| 目录结构 | `ros_voice/` 是项目的子目录 | `ros_voice/` 本身即功能包根目录 |
| 编译方式 | 在包内部误执行 `colcon build`，产物散落于源码目录 | 在 `ros2_ws/` 工作空间根执行，产物在 `build/install/log/` |
| 依赖模块 | `realtime_asr/`、`onnx_model/` 游离在功能包之外 | 全部纳入功能包目录，路径自洽 |
| 工作空间挂载 | 通过软链接 `ros2_ws/src/ros_voice → Genshin/TTS/ros_voice` | 直接实体目录，无软链接 |
| 测试入口 | 项目根的 `main.py` | 功能包内 `test/main.py`，与 ROS 层独立 |

## License

MIT
