<div align="right">
  <a href="README.md">English</a>
</div>

# 🎚️ LingoEQ

> **"让底鼓更有力，人声更通透"** — 用自然语言描述音色，让 AI 精准调节 24 频段参数均衡器

<div align="center">

![LingoEQ Demo](docs/demo.png)

*AI 实时理解混音意图，自动调整 EQ 参数，并通过声学反馈自我验证*

</div>

---

## ✨ 核心特性

### 🗣️ 自然语言调参
告别手动拖拽 24 个频段。用口语化描述你想要的音色，LingoEQ 自动翻译为精确的频率、增益、Q 值参数。

**传统方式 vs LingoEQ**
```
❌ 手动操作：在 800Hz 衰减 4dB，Q 值 2.5，滤波器类型 Peaking...
✅ LingoEQ：  "人声有点鼻音，清理一下"
```

### 🧠 Neuro-Symbolic 混合架构
将 LLM 的语义理解能力与硬编码的 DSP 物理约束结合：
- **神经层**：理解"温暖"、"刺耳"、"浑浊"等主观听感词汇
- **符号层**：强制执行声学安全边界（频率 20-20kHz，增益 -24 到 +18dB）
- **认知同步**：用户手动操作的频段被永久锁定，AI 自动避让

### 🎯 DSP 感知的干涉削减算法
基于滤波器带宽（Q 值）计算频域重叠度，当 AI 操作靠近用户手动调节的频段时：
- 自动计算两个频点的八度距离和影响半径
- 按重叠比例削减 AI 的增益强度
- 避免 AI 操作与用户意图相互干扰

**技术细节**：[state_manager.py#L59-L86](app/services/state_manager.py#L59-L86)

---

## 🏗️ 系统架构

LingoEQ 采用 **"AI 大脑 + DSP 执行器"** 分离架构，Python 后端负责理解和决策，C++/JUCE 插件负责实时音频处理。

```
┌─────────────────────────────────────────────────────────────┐
│  C++ Plugin (JUCE)                                          │
│  • 24-band parametric EQ                                    │
│  • Real-time audio processing                               │
│  • Spectrum analyzer & reflexion recorder                   │
└────────────────┬────────────────────────────────────────────┘
                 │ TCP JSON-RPC (port 8888)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Python Backend (LingoEQ)                                   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MCPHandler（中央调度器）                            │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  第一步：资源聚合（ContextManager）                 │  │
│  │  ├─ RouterAgent    → 提取 {乐器, 关键词}            │  │
│  │  ├─ RagEngine      → 检索混音知识                   │  │
│  │  ├─ StateManager   → 获取 EQ 状态 + 锁定频段        │  │
│  │  ├─ SensoryManager → 音频快照分析                   │  │
│  │  └─ MemoryManager  → 列出可用预设                   │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  第二步：Prompt 渲染（PromptEngineer）              │  │
│  │  将上下文注入 System Prompt 模板                    │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  第三步：LLM 推理（LLMClient）                     │  │
│  │  • 提取 <think> 标签                                │  │
│  │  • 解析 JSON 响应                                   │  │
│  │  • 校验工具调用参数                                 │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  第四步：护栏执行（StateManager）                   │  │
│  │  • 锁定频段重定向                                   │  │
│  │  • 计算干涉削减量                                   │  │
│  │  • 参数钳位到安全范围                               │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  第五步：响应序列化                                 │  │
│  │  构建 content_list → 发送给 C++                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  Optional: Reflexion Closed-Loop (max 2 iterations)        │
│  C++ captures pre/post audio → Python analyzes diff →      │
│  LLM self-corrects if new peaks detected                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 前置要求

- **Python 3.10+**
- **OpenAI 兼容的 LLM API**（默认使用 [SiliconFlow](https://siliconflow.cn) + DeepSeek V3）
- **C++/JUCE 插件**（另一个仓库，监听 8888 端口）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/yuzh16/LingoEQ-agent.git
cd LingoEQ-agent

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 LLM_API_KEY
```

### 运行

```bash
python main.py
```

服务启动后监听 `127.0.0.1:8888`，等待 C++ 插件连接。

---

## 📂 项目结构

```
LingoEQ_Project/
├── main.py                        # 入口：初始化 RAG，启动 TCP server
├── config.py                      # 全局配置单例（从 .env 读取）
├── requirements.txt
├── lingoeq_bench_server.py        # Benchmark HTTP 接口（POST /benchmark，供 LingoEQ-Bench 评测使用）
├── .env.example                   # 配置模板
│
├── app/
│   ├── api/
│   │   └── socket_transport.py    # 异步 TCP 传输层
│   │
│   ├── mcp_protocol/
│   │   └── server.py              # 中央调度器（处理所有 JSON-RPC 方法）
│   │
│   ├── core/
│   │   ├── context_manager.py     # 资源聚合层（RAG + 状态 + 感知 + 预设）
│   │   ├── prompt_engineer.py     # Prompt 渲染层
│   │   ├── llm_client.py          # LLM 执行层（调 API、提取思维链、验证 JSON）
│   │   └── router_agent.py        # 意图路由（提取 instruments + keywords）
│   │
│   ├── services/
│   │   ├── state_manager.py       # 影子状态（24 频段镜像 + 护栏逻辑）
│   │   ├── memory_manager.py      # 预设管理（ChromaDB CRUD）
│   │   ├── rag_engine.py          # 知识检索（mixing_tips.json → ChromaDB）
│   │   └── sensory_manager.py     # 音频感知（心理声学计权、掩蔽阈值、reflexion diff）
│   │
│   └── schemas/
│       ├── eq_protocol.py         # Pydantic 模型：EQBand、ToolCall、AgentResponse
│       └── tools_schema.py        # LLM 工具定义列表
│
└── data/
    ├── knowledge_base/
    │   └── mixing_tips.json       # RAG 种子数据（结构化混音知识）
    ├── factory_presets/           # 14 个出厂预设 JSON
    └── vector_db/                 # ChromaDB 持久化目录（gitignore）
```

---

## 🔧 核心功能详解

### 1️⃣ 双阶 Agentic RAG

**传统 RAG 问题**：用户说"底鼓太糊"，直接检索会匹配到无关的"吉他"、"人声"知识。

**LingoEQ 方案**：
1. **RouterAgent**（轻量级 LLM）先提取意图：
   ```json
   {
     "instruments": ["drum"],
     "keywords": ["浑浊", "底鼓", "低频"]
   }
   ```
2. **RagEngine** 基于意图在 ChromaDB 中精确检索：
   - 硬过滤：只在 `drum` 分类中搜索
   - 软匹配：用关键词向量检索相关知识
   - 阈值拦截：L2 距离 > 1.125 的结果直接丢弃

**效果**：有效避免无关知识干扰 LLM 决策，在多乐器场景下尤为显著。

---

### 2️⃣ 影子状态与护栏系统

Python 后端维护 24 个频段的 in-memory 镜像，与 C++ 插件实时同步。

#### 🔒 用户意图保护
```python
# 用户手动操作 Band 5 → 永久锁定
manual_touch_registry[5] = True

# AI 想操作 Band 5 → 自动重定向到最近的空闲频段
if target_band in manual_touch_registry:
    target_band = get_nearest_free_band(target_band)
```

#### 📉 干涉削减算法
```python
# 计算两个频点的八度距离
d_oct = abs(log2(ai_freq / user_freq))

# 计算影响半径（基于 Q 值）
threshold = (1/(2*ai_q) + 1/(2*user_q)) * SAFETY_MARGIN

# 重叠率越高，AI 增益削减越多
if d_oct < threshold:
    overlap_ratio = 1.0 - (d_oct / threshold)
    ai_gain *= (1.0 - overlap_ratio)
```

#### 🛡️ 物理钳位
```python
# 所有参数写入前强制钳位
gain = clamp(gain, -24.0, +18.0)
freq = clamp(freq, 20.0, 20000.0)
q = clamp(q, 0.1, 10.0)
```

---

### 3️⃣ Reflexion 闭环验证

AI 调整参数后，系统自动验证效果：

```
1. C++ 插件录制调整前后的频谱快照（各 20 帧）
2. Python 后端计算差异：
   - 应用 A 加权（等响度补偿）
   - 计算掩蔽阈值（过滤被掩蔽的频段）
   - 提取显著峰值（可能的刺耳点）
3. 如果检测到新增问题 → LLM 二次微调
4. 如果验证通过 → 返回隐藏状态码（不打扰用户）
```

**深度限制**：最多 2 轮，防止死循环。超过后强制清空 tool_calls。

---

### 4️⃣ 预设系统

支持**出厂预设**和**用户预设**两种类型，均通过 ChromaDB 语义检索：

```python
# 用户说："加载人声增亮预设"
# 系统自动匹配最相似的预设（L2 距离 < 1.3）
preset = memory_manager.load_preset("人声增亮")

# 加载时同样遵守护栏规则：
# - 锁定频段自动重定向
# - 参数安全钳位
```

---

## 🌐 JSON-RPC 协议

C++ 插件通过 TCP Socket 与 Python 后端通信，使用换行符分隔的 JSON 消息。

### 支持的方法

| Method | 方向 | 说明 |
|--------|------|------|
| `chat/send` | Plugin → Python | 发送用户消息，返回思维链 + 回复 + 工具调用 |
| `eq/sync` | Plugin → Python | 同步单个参数（触发频段锁定） |
| `audio/snapshot_upload` | Plugin → Python | 上传频谱快照到指定槽位 |
| `audio/snapshot_delete` | Plugin → Python | 删除指定槽位快照 |
| `audio/reflexion_diff_upload` | Plugin → Python | 上传前后对比帧，触发闭环验证 |
| `ping` | Plugin → Python | 心跳检测 |

### 示例：chat/send

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "chat/send",
  "params": {
    "content": "人声有点鼻音，清理一下",
    "auto_check": true
  }
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "role": "thought",
        "text": "鼻音通常集中在 800Hz-1.2kHz，需要窄 Q 值衰减..."
      },
      {
        "type": "text",
        "role": "assistant",
        "text": "已为您清理 1kHz 附近的鼻音频段。"
      },
      {
        "type": "tool_use",
        "tool": "adjust_eq_band",
        "args": {
          "band_index": 7,
          "freq": 1000.0,
          "gain": -4.0,
          "q": 2.5,
          "filter_type": "Peaking",
          "enabled": true
        }
      }
    ]
  }
}
```

---

## ⚙️ 配置说明

所有超参数集中在 `config.py`，通过 `.env` 覆盖。

### 核心配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API 密钥 | *必填* |
| `LLM_BASE_URL` | API 基础 URL | `https://api.siliconflow.cn/v1` |
| `LLM_MODEL_NAME` | 模型名称 | `deepseek-ai/DeepSeek-V3` |
| `LLM_TEMPERATURE` | 推理温度 | `0.3` |
| `MAX_REFLEXION_DEPTH` | 闭环验证最大深度 | `2` |
| `WS_PORT` | TCP 端口（需与 C++ 对齐） | `8888` |

### RAG 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `RAG_TOP_K` | 每次注入的知识条数 | `5` |
| `RAG_DISTANCE_THRESHOLD` | L2 距离阈值 | `1.125` |
| `PRESET_DISTANCE_THRESHOLD` | 预设匹配阈值 | `1.3` |
| `MAX_CONVERSATION_HISTORY` | 携带的最大历史对话轮数 | `5` |

### 心理声学与感知

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PSYCHOACOUSTIC_MODEL` | 计权模型：`a_weighting` / `k_weighting` / `iso226_80phon` | `a_weighting` |
| `MAX_SENSORY_SLOTS` | 允许缓存的音频快照槽位数量 | `5` |
| `SENSORY_TOP_PEAKS` | 每次汇报的最显著峰值数量 | `3` |
| `SENSORY_TOP_VALLEYS` | 每次汇报的最显著谷值数量 | `2` |

### 物理护栏

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `GAIN_LIMIT_MIN` / `MAX` | 增益范围 (dB) | `-24.0` / `+18.0` |
| `FREQ_LIMIT_MIN` / `MAX` | 频率范围 (Hz) | `20.0` / `20000.0` |
| `Q_LIMIT_MIN` / `MAX` | Q 值范围 | `0.1` / `10.0` |
| `SAFETY_MARGIN` | 干涉计算安全系数 | `1.2` |

完整配置列表见 [.env.example](.env.example)。

---

## 🔄 切换 LLM 提供商

LingoEQ 使用 OpenAI SDK，支持任何兼容 OpenAI API 的提供商：

```bash
# 使用 OpenAI 官方
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o
LLM_API_KEY=sk-...

# 使用本地 Ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=llama3

# 使用 Azure OpenAI
LLM_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
LLM_MODEL_NAME=gpt-4
```

**要求**：模型必须支持 **JSON Mode**（`response_format: json_object`）。

---

## 🎓 技术栈

| 层级 | 技术 |
|------|------|
| 运行时 | Python 3.10+, asyncio (Windows-first) |
| LLM 接口 | OpenAI SDK（兼容接口） |
| 数据校验 | Pydantic v2 |
| 向量数据库 | ChromaDB（本地持久化） |
| 嵌入模型 | SentenceTransformer `all-MiniLM-L6-v2` |
| 传输协议 | 自定义 JSON-RPC over TCP（换行符分隔） |
| 配置管理 | `python-dotenv` + 单例模式 |

---

## 🤝 相关项目

- **C++/JUCE 插件端**：[LingoEQ-Plugin](https://github.com/yuzh16/LingoEQ-plugin)

---

## 📄 License

*待定*

---

<div align="center">

**如果这个项目对你有帮助，欢迎 Star ⭐**

</div>
