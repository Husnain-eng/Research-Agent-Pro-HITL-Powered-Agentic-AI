# Research Agent Pro (HITL Edition)

A production-quality, Human-in-the-Loop research agent built with **LangGraph**, **Ollama**, and modular tool integrations.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Research Agent Pro                         │
│                                                                │
│  User Query                                                    │
│      │                                                         │
│      ▼                                                         │
│  ┌──────────────┐     No pending tool     ┌──────────────────┐ │
│  │  agent_node  │────────────────────────►│  response_node   │ │
│  │  (LLM / ReAct│◄──────────────────┐    └────────┬─────────┘ │
│  └──────┬───────┘                   │             │           │
│         │ pending tool              │             ▼           │
│         ▼                           │           END           │
│  ┌──────────────┐  NodeInterrupt    │                         │
│  │interrupt_node│──────────────────►│ [HITL Prompt]           │
│  │  (checkpoint)│   pauses here     │   y = approve           │
│  └──────────────┘                   │   e = edit              │
│         │ human resumes             │   n = reject            │
│         ▼                           │                         │
│  ┌──────────────┐                   │                         │
│  │  tool_node   │───────────────────┘                         │
│  │ (executes or │  result injected                            │
│  │  injects     │  into state                                 │
│  │  rejection)  │                                             │
│  └──────────────┘                                             │
│                                                               │
│  Checkpoint: SqliteSaver (checkpoints.db)                     │
└───────────────────────────────────────────────────────────────┘
```

### Node Responsibilities

| Node | Role |
|------|------|
| `agent_node` | LLM (Ollama) reasons via ReAct, selects next tool or produces final answer |
| `interrupt_node` | Raises `NodeInterrupt`, pauses graph, checkpoints state |
| `tool_node` | Executes approved/edited tool or injects rejection message |
| `response_node` | Synthesises final answer from all tool results |

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com) installed and running
- [uv](https://docs.astral.sh/uv/) package manager
- Python 3.11+

### 1. Install Ollama & pull a model

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the default model
ollama pull llama3.2
```

### 2. Clone / create project

```bash
git clone <repo-url>
cd research_agent_pro
```

### 3. Set up environment with uv

```bash
# Initialise uv project
uv init --no-workspace

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
uv pip install -r requirements.txt

# Note: langgraph-checkpoint-sqlite is a separate package required for
# persistent checkpointing. It is listed in requirements.txt.
```

### 4. (Optional) Configure settings

Edit `config/settings.py` to change:
- `ollama_model` — e.g., `"mistral"`, `"llama3.2"`, `"qwen2.5"`
- `ollama_base_url` — if Ollama runs on a remote host
- `max_iterations` — max ReAct loops per query
- `checkpoint_db` — SQLite file path

---

## Running

```bash
python main.py
```

---

## How HITL Works

1. **Agent reasons** → decides to call a tool (e.g., `arxiv_search`)
2. **Graph checkpoints** the full state to SQLite **before** the interrupt
3. **`interrupt_node`** raises `NodeInterrupt` — execution **pauses**
4. **CLI displays** the tool name + arguments
5. **Human decides:**
   - `y` → tool runs with original args
   - `e` → human provides new JSON args, then tool runs
   - `n` → tool is skipped; LLM sees a rejection message and re-reasons
6. **Graph resumes** from checkpoint — no context is lost
7. Cycle repeats until LLM produces a final answer

---

## Example Run

```
╔══════════════════════════════════════════════════════════════════╗
║  Research Agent Pro  ·  HITL Edition                            ║
╚══════════════════════════════════════════════════════════════════╝

❯ What are the latest advances in RISC-V low-power embedded systems?

──────────────────────────────────────────────────────────────────
  🔍 Research Query: What are the latest advances in RISC-V...
──────────────────────────────────────────────────────────────────

  ⚙  Agent reasoning…

══════════════════════════════════════════════════════════════════
  ⚡  HUMAN-IN-THE-LOOP INTERRUPT
══════════════════════════════════════════════════════════════════

  Tool: arxiv_search
  Input Arguments:
    {
        "query": "RISC-V low power embedded systems 2024"
    }

  [y] Approve and run
  [e] Edit the input before running
  [n] Reject this tool call

  Decision (y/e/n): e
  Current args: {"query": "RISC-V low power embedded systems 2024"}
  Enter new JSON args:
  ❯ {"query": "RISC-V ultra-low power IoT 2024 2025"}
  ✔ Edited args accepted.

  ⚙  Resuming agent…

  ✔ arxiv_search result:
    Arxiv Papers for: 'RISC-V ultra-low power IoT 2024 2025'
    1. Ultra-Low-Power RISC-V Core for Edge AI...
    ...

══════════════════════════════════════════════════════════════════
  ✅  FINAL ANSWER
══════════════════════════════════════════════════════════════════

  Recent advances in RISC-V low-power embedded systems include...
```

---

## Project Structure

```
research_agent_pro/
│
├── main.py                  # Entry point
│
├── agent/
│   ├── __init__.py
│   ├── graph.py             # LangGraph workflow definition
│   ├── nodes.py             # Node implementations (agent, interrupt, tool, response)
│   ├── state.py             # AgentState TypedDict
│   │
│   ├── tools/
│   │   ├── __init__.py      # Tool registry (ALL_TOOLS, TOOL_MAP)
│   │   ├── duckduckgo.py    # DuckDuckGo web search
│   │   ├── arxiv_tool.py    # Arxiv paper search
│   │   └── wikipedia_tool.py# Wikipedia summary
│   │
│   └── llm/
│       ├── __init__.py
│       └── ollama.py        # ChatOllama setup with tool binding
│
├── cli/
│   ├── __init__.py
│   └── interface.py         # CLI banner, HITL prompts, streaming output
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Global settings dataclass
│
├── requirements.txt
└── README.md
```

---

## Extending the Agent

### Add a new tool

1. Create `agent/tools/my_tool.py` with a `@tool`-decorated function
2. Import and add it to `ALL_TOOLS` in `agent/tools/__init__.py`
3. Done — HITL interrupt works automatically for new tools

### Switch LLM model

```python
# config/settings.py
ollama_model: str = "mistral"   # or "qwen2.5", "deepseek-r1", etc.
```

### Adjust HITL behaviour

Modify `cli/interface.py → prompt_hitl()` for custom approval workflows (e.g., auto-approve certain tools, add timeout, log decisions to file).
# Research-Agent-Pro-HITL-Powered-Agentic-AI
