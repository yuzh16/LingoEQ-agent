# LingoEQ

**LingoEQ** is an AI-powered parametric EQ assistant backend for DAW plugin workflows. Describe what you want to hear in plain language — LingoEQ understands your intent, retrieves domain-specific mixing knowledge, and sends precise EQ parameter commands to your C++/JUCE plugin in real time.

> "Make the vocals warmer" → LingoEQ analyzes intent → retrieves mixing knowledge → adjusts 24-band EQ → validates the result acoustically → sends commands to your plugin.

---

## Features

- **Natural language EQ control** — describe tone in plain language, get precise frequency adjustments
- **RAG-backed mixing knowledge** — retrieves relevant techniques from a structured knowledge base before every response
- **Dual-LLM pipeline** — a lightweight intent router extracts instruments and keywords; the main model generates structured EQ tool calls
- **24-band shadow state** — Python maintains a mirror of the plugin's EQ state with full guardrail logic
- **User intent protection** — bands you touch manually are permanently locked; AI operations are redirected to nearby free bands and gain is attenuated near your manual adjustments via DSP-aware interference calculation
- **Physical safety clamping** — all parameters are hard-clamped before reaching the plugin (gain: −24 to +18 dB, frequency: 20–20000 Hz, Q: 0.1–10)
- **Closed-loop reflexion** — the system captures pre/post audio snapshots, runs psychoacoustic analysis, and optionally triggers a second LLM pass to self-correct; depth is capped to prevent infinite loops
- **Preset memory** — semantic search over factory and user presets via ChromaDB; preset loading also respects user-locked bands
- **Provider-agnostic LLM** — uses the OpenAI-compatible API interface; swap to any provider by changing two environment variables

---

## Architecture

```
C++ Plugin (JUCE)
      │  TCP JSON-RPC  (port 8888)
      ▼
socket_transport.py       async TCP, newline-delimited JSON
      ▼
MCPHandler                central dispatcher (mcp_protocol/server.py)
      ├── RouterAgent     intent extraction → {instruments, keywords}
      ├── RagEngine       knowledge retrieval from mixing_tips.json
      ├── StateManager    24-band shadow state, guardrails, history
      ├── SensoryManager  audio snapshot analysis, reflexion diff
      └── MemoryManager   preset CRUD via ChromaDB
      ▼
PromptEngineer            assembles system prompt + context blocks
      ▼
LLM (DeepSeek / any OpenAI-compatible provider)
      ▼
AgentResponse             thought_process + reply_text + tool_calls
      ▼
C++ Plugin                applies EQ parameter changes
```

---

## Project Structure

```
LingoEQ_Project/
├── main.py                        # Entry point — starts TCP server on 127.0.0.1:8888
├── config.py                      # Global settings singleton (loaded from .env)
├── requirements.txt
│
├── app/
│   ├── api/
│   │   └── socket_transport.py    # Async TCP transport layer
│   ├── mcp_protocol/
│   │   └── server.py              # Central dispatcher — handles all JSON-RPC methods
│   ├── core/
│   │   ├── context_manager.py     # Aggregates RAG + state + sensory + preset context
│   │   ├── prompt_engineer.py     # Renders final system prompt and message list
│   │   ├── llm_client.py          # LLM call, <think> extraction, JSON validation
│   │   └── router_agent.py        # Intent router (instruments + keywords for RAG)
│   ├── services/
│   │   ├── state_manager.py       # Shadow EQ state, guardrails, interference attenuation
│   │   ├── memory_manager.py      # Preset CRUD via ChromaDB
│   │   ├── rag_engine.py          # Knowledge retrieval from mixing_tips.json
│   │   └── sensory_manager.py     # Psychoacoustic snapshot analysis and reflexion diff
│   └── schemas/
│       ├── eq_protocol.py         # Pydantic models: EQBand, ToolCall, AgentResponse
│       └── tools_schema.py        # LLM tool definitions and prompt serializer
│
└── data/
    ├── knowledge_base/
    │   └── mixing_tips.json       # Structured mixing heuristics (RAG seed data)
    └── factory_presets/           # 14 factory preset JSON files
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI-compatible LLM API key (default: [SiliconFlow](https://siliconflow.cn) with DeepSeek V3)
- A compatible C++/JUCE EQ plugin client listening on port 8888 (the plugin side is a separate project)

### Installation

```bash
git clone https://github.com/your-username/LingoEQ.git
cd LingoEQ

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

`.env` variables:

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | API key for your LLM provider | *(required)* |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.siliconflow.cn/v1` |
| `LLM_MODEL_NAME` | Model identifier | `deepseek-ai/DeepSeek-V3` |
| `LLM_TEMPERATURE` | Inference temperature | `0.3` |
| `MAX_REFLEXION_DEPTH` | Max closed-loop correction passes | `2` |
| `WS_PORT` | TCP port (must match C++ plugin) | `8888` |

All tunable parameters (RAG thresholds, guardrail limits, sensory settings, etc.) are also overridable via `.env`. See `.env.example` for the full list.

### Run

```bash
python main.py
```

The server starts on `127.0.0.1:8888` and waits for a plugin client connection.

---

## JSON-RPC Protocol

The plugin communicates with LingoEQ over a TCP socket using newline-delimited JSON messages.

| Method | Direction | Description |
|---|---|---|
| `chat/send` | Plugin → Python | Send a user message; returns `text`, `tool_use`, and `status` items |
| `eq/sync` | Plugin → Python | Sync a single manually-changed parameter (triggers band lock) |
| `audio/snapshot_upload` | Plugin → Python | Upload a frequency spectrum snapshot to a named slot |
| `audio/snapshot_delete` | Plugin → Python | Delete a snapshot slot |
| `audio/reflexion_diff_upload` | Plugin → Python | Upload pre/post frames to trigger closed-loop self-correction |
| `ping` | Plugin → Python | Heartbeat |

**Example request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "chat/send",
  "params": {
    "content": "The vocals sound nasal, clean it up",
    "auto_check": true
  }
}
```

**Example response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "role": "assistant",
    "content": [
      { "type": "text", "role": "thought", "text": "Nasal tone typically sits around 800Hz–1.2kHz..." },
      { "type": "text", "role": "assistant", "text": "Done — I've attenuated the nasal 1kHz range." },
      { "type": "tool_use", "tool": "adjust_eq_band", "args": { "band_index": 7, "freq": 1000.0, "gain": -4.0, "q": 2.5, "filter_type": "Peaking", "enabled": true } }
    ]
  }
}
```

---

## Guardrail System

LingoEQ separates **cognitive guidance** (prompt-level) from **physical enforcement** (code-level) to ensure the AI cannot override your manual decisions or produce unsafe parameters.

| Layer | Mechanism |
|---|---|
| Band locking | Any band you touch via `eq/sync` is permanently added to `manual_touch_registry` |
| Redirection | AI operations targeting a locked band are silently redirected to the nearest free band |
| Interference attenuation | If an AI band overlaps a user band in frequency (Q-aware), its gain is proportionally reduced |
| Physical clamping | All parameter writes are hard-clamped regardless of source |
| Reflexion depth cap | Closed-loop self-correction is limited to `MAX_REFLEXION_DEPTH` iterations (default: 2) |

---

## Switching LLM Providers

LingoEQ uses the OpenAI SDK with a configurable base URL. Any OpenAI-compatible provider works:

```bash
# Example: use a local Ollama instance
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=llama3

# Example: use OpenAI directly
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o
```

The model must support **JSON mode** (`response_format: json_object`) and perform well on structured output tasks.

---

## License

MIT
