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
| `voice_node` | `/voice/listen_mode` `/voice/speaking` | `/voice/command` (String) | 麦克风采音、VAD 切句、SenseVoice ONNX 识别、唤醒词过滤 |
| `brain_node` | `/voice/command` `/vision/scene_objects` `/vision/emotion_context` `/voice/listen_mode` | `/command` (String, JSON) `/voice/listen_mode` `/voice/speaking` | Qwen LLM 语义理解、视觉/情绪上下文注入、TTS 口语回复、紧急中止意图判定 |
| `control_node` | `/command` | `/cmd_vel` (Twist) `/arm/grasp_command` (String) `/emergency/abort` (String) | 解析 JSON 指令数组，驱动底盘 + 机械臂 + 转发紧急中止 |

## 麦克风的开与关

麦克风开不开，由**四个条件**决定，逻辑就一行（[voice_node.py:86](ros_voice/voice_node.py#L86)）：

```python
想开麦 = 紧急态 or (轮次闸开 and 未硬静音)
真开麦 = 想开麦 and 未在播报
```

关闭时**从音频源头截断**——直接排空采集队列，VAD 拿不到数据。不是"录了不用"，是根本不录。

| 条件 | 谁控制 | 关 | 开 |
|---|---|---|---|
| 轮次闸 | pipeline 自锁 + brain_node | 识别到一句指令即自锁 | brain 发 `continuous`/`command` |
| 硬静音 | 外部节点（机械臂/底盘） | 发 `mute` | 发 `unmute`（**只有它能解**） |
| 播报闸 | brain_node | 发 `/voice/speaking` = `start` | 发 `end` |
| 紧急旁路 | 紧急呼叫模块 | 发 `emergency_end` | 发 `emergency` |

### 优先级

```
播报闸   ＞   紧急旁路   ＞   硬静音   ＞   轮次闸
强制闭麦      强制开麦       闭麦        开/关
```

- **播报闸最高**：机器人出声时一律闭麦，紧急态也不例外。TTS 与放歌都在 brain_node 进程内，voice_node 无从感知；尤其情绪干预不经过 voice_node，没有这个信号机器人必然把自己的声音收回来。
- **紧急旁路次之**：压过硬静音和轮次闸——机械臂抓取途中 `mute` 着，老人摔倒了照样听得见"不用打了"。被旁路的状态原样记着，`emergency_end` 后完整恢复。
- **硬静音是粘性的**：`continuous`/`command` 打不开它，只有 `unmute` 能解。

> ⚠️ 三对信号都必须**成对发送**：`mute`/`unmute`、`start`/`end`、`emergency`/`emergency_end`。
> 漏发后一半就是麦克风卡死。brain_node 用 `try/finally` 保证 `end` 必发，外部节点自己负责。

### 职责边界

`/voice/listen_mode` 有两个订阅者，各自解读、互不通信：

- **voice_node** —— 只管麦克风开不开、要不要唤醒词。它把听到的话转成文字发到 `/voice/command`，**从不判断这句话是什么意思**。
- **brain_node** —— 用它决定自己的工作状态：常规对话（大模型 + TTS），还是紧急中止判定。

所以同一句"不用打了"，voice_node 的行为完全一样；是 brain_node 决定送去聊天还是送去中止判定。

| msg.data | 谁能发 | 作用 |
|---|---|---|
| `continuous` / `command` | brain_node | 轮次闸开，免/需唤醒词 |
| `mute` / `unmute` | 外部节点 | 硬静音开关 |
| `emergency` / `emergency_end` | 紧急呼叫模块 | 紧急态开关（brain_node 中止成功后也补发一次 `emergency_end`） |

未知值按 `command` 处理（fail-open）：宁可多要一次唤醒词，也不把麦克风锁死。

> ⚠️ **brain_node 永远不可以发 `mute`**——静音后用户没法说话，它也就永远等不到下一条指令去发 `unmute`，直接死锁。

### 命名说明：`listen_mode` 是个误称

这个话题名现在名不副实，读代码时容易被误导，记在这里备查。三个轴里**只有紧急态越了界**：

| 轴 | 谁读 | 影响什么 |
|---|---|---|
| `continuous` / `command` | voice_node | 怎么**听** |
| `mute` / `unmute` | voice_node | 怎么**听** |
| `emergency` / `emergency_end` | voice_node **+ brain_node** | 怎么**听** + 怎么**想** |

前两个轴名副其实；是紧急态这一对值把 brain_node 也拉了进来，话题才从"怎么听"变成了"怎么听 + 怎么想"。

**更合理的做法是拆开**而不是改名：`listen_mode` 只留前两个轴，紧急态另开一个 `/emergency/state`（值 `active`/`inactive`，紧急侧发布，两个节点各自订阅）。紧急态本来就不是"麦克风控制指令"，而是**系统级情境广播**——麦克风只是受影响方之一，brain_node 是另一个，以后可能还有表情屏。改成状态型取值后，紧急侧周期性重发也变得天然合理，顺带能修掉"brain_node 中途重启会漏掉一次性 `emergency`"这个隐患。

**为什么没改**：下游团队已按现有话题名开始对接，改名是破坏性变更。将来若有窗口期，机械臂/底盘不受影响（只用 `mute`/`unmute`，话题名不变），只有紧急侧要改。

### 唤醒词是另一根轴

`wake_required` 决定"收到音之后要不要先说唤醒词"，**和收不收音无关**。默认需要说"小智小智"，`continuous` 免唤醒词，紧急态强制免唤醒词。

紧急期间收到的 `continuous`/`command` **只暂存不生效**，`emergency_end` 后才恢复——否则 brain 每播完一句发的那个 `command` 会让用户第二次喊"不用打了"时还得先报唤醒词。

### 兜底看门狗

| 看门狗 | 阈值 | 触发后 |
|---|---|---|
| 麦克风卡死 | 150s | 强制松开轮次闸 + 播报闸，**不碰硬静音** |
| 紧急态滞留 | 180s | 自动退出紧急态 |

方向相反：前者超时要**放**（brain 崩了不能让整机哑掉），后者要**收**（紧急态是强制开麦，滞留会让麦克风失控）。硬静音永远不兜底——那是外部节点有意为之的无限期静音。

### 已知限制

TTS / 放歌期间收到 `emergency`，麦克风**不会立刻开**，要等播报放完（普通回复几秒，放歌最长 60 秒，`play_song` 阻塞且不支持打断）。播报闸压着紧急旁路是为了防回声，代价就是这个延迟。

## 紧急联络

三方协作，各司其职（模型见 `接口对接文档.md` §3.4）：

| 角色 | 当前是谁 | 发 | 收 |
|------|---------|----|----|
| **发起方** | brain_node（情绪触发） | `/emergency/initiate`、`listen_mode`=`emergency`/`emergency_end` | `/emergency/abort` |
| **语音方** | brain_node | `/emergency/abort`（经 control_node 转发） | `listen_mode`=`emergency`/`emergency_end` |
| **紧急方** | 紧急呼叫模块 | — | `/emergency/initiate`、`/emergency/abort` |

- **发起方**启动 + 授权：谁决定该联络，谁就发 initiate 和 `emergency`；发起方自己开关中止窗口——不依赖紧急侧
- **语音方**判决中止：监听 `emergency` 开窗信号，做规则+大模型两级判定（[emergency.py](voice_brain_module/emergency.py)）
- **紧急方**执行：只管拨号/发短信/挂断，不碰 `listen_mode`

**发起路径**（[brain_node.py](ros_voice/brain_node.py) `_run_emergency_ask`）：稳定负面情绪 → 话术询问 → 等 10 秒 → 无人拒绝就发 `/emergency/initiate` + 开 60 秒中止窗口。拒绝则作罢。无应答照发——老人可能已经说不出话。情绪→渠道映射与话术在 [config.py](voice_brain_module/config.py) 的「紧急呼叫发起」段。

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
│   ├── emergency.py        #   紧急呼叫中止意图识别（规则层）
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

**系统级指令**（`SYSTEM_COMMANDS`，由代码直接下发，**不进 LLM 提示词也不过 `validate_commands`**）：

| 执行机构 | 动作 | 参数 | 触发 |
|----------|------|------|------|
| 紧急呼叫 | 中止紧急情况 | `reason`、`detector`、`utterance` | 紧急态下判定用户要中止求助 |

> 故意对大模型隐藏：聊天中幻觉出一条"中止紧急情况"，紧急侧无从分辨真假。

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
