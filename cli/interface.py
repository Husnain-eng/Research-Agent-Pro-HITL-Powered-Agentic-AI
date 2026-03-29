"""
CLI Interface for Research Agent Pro (HITL Edition).

Responsibilities
----------------
- Display startup banner
- Accept user research queries
- Drive the LangGraph execution loop
- Handle HITL interrupt prompts using Command(resume=...)
- Stream and format all output
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
import uuid
from typing import Any, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.state import AgentState
from config.settings import settings

# ─────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"


def c(text: str, *codes: str) -> str:
    """Wrap text in ANSI colour codes."""
    return "".join(codes) + text + RESET


def divider(char: str = "─", width: int = 70, color: str = DIM) -> str:
    return c(char * width, color)


# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────

BANNER = f"""
{c('╔══════════════════════════════════════════════════════════════════╗', CYAN, BOLD)}
{c('║', CYAN, BOLD)}  {c('Research Agent Pro', WHITE, BOLD)}  {c('·', DIM)}  {c('HITL Edition', YELLOW, BOLD)}                          {c('║', CYAN, BOLD)}
{c('║', CYAN, BOLD)}  {c('Powered by LangGraph + Ollama + DuckDuckGo / Arxiv / Wikipedia', DIM)}  {c('║', CYAN, BOLD)}
{c('╚══════════════════════════════════════════════════════════════════╝', CYAN, BOLD)}

  {c('Model :', DIM)} {c(settings.ollama_model, MAGENTA)}   {c('Checkpoint DB:', DIM)} {c(settings.checkpoint_db, MAGENTA)}
  {c('Type your research query and press Enter.', DIM)}
  {c('Commands:', DIM)} {c('exit', YELLOW)} {c('/', DIM)} {c('quit', YELLOW)} {c('to leave.', DIM)}
"""


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────


def stream_print(text: str, delay: float = settings.stream_delay) -> None:
    """Print text character by character for a streaming effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def print_section(label: str, content: str, label_color: str = CYAN) -> None:
    """Print a labelled section block."""
    print(f"\n{c('▶', label_color, BOLD)} {c(label, label_color, BOLD)}")
    wrapped = textwrap.fill(
        content, width=80, initial_indent="  ", subsequent_indent="  "
    )
    print(wrapped)


def show_tool_result(result: dict[str, Any]) -> None:
    """Pretty-print a tool result."""
    status_sym = {
        "success": c("✔", GREEN),
        "rejected": c("✘", RED),
        "error": c("!", RED),
    }.get(result.get("status", ""), "?")
    print(f"\n  {status_sym} {c(result['name'], MAGENTA)} result:")
    content = result.get("content", "")
    preview = content[:600] + ("…" if len(content) > 600 else "")
    for line in preview.splitlines():
        print(f"    {DIM}{line}{RESET}")


def show_final_answer(answer: str) -> None:
    """Render the final answer prominently."""
    print(f"\n{divider('═', 70, GREEN)}")
    print(c("  ✅  FINAL ANSWER", GREEN, BOLD))
    print(divider('═', 70, GREEN))
    print()
    wrapped = textwrap.fill(
        answer, width=80, initial_indent="  ", subsequent_indent="  "
    )
    print(wrapped)
    print(f"\n{divider('─', 70)}\n")


# ─────────────────────────────────────────────
# HITL prompt
# ─────────────────────────────────────────────


def prompt_hitl(
    tool_name: str, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """
    Display the HITL interrupt panel and collect the human decision.

    Returns
    -------
    dict with keys: decision, edited_args, reject_reason
    Suitable for passing directly to Command(resume=<dict>).
    """
    args_str = json.dumps(tool_args, indent=4, ensure_ascii=False)

    print(f"\n{divider('═', 70, YELLOW)}")
    print(c("  ⚡  HUMAN-IN-THE-LOOP INTERRUPT", YELLOW, BOLD))
    print(divider('═', 70, YELLOW))
    print(f"\n  {c('Tool  :', DIM)} {c(tool_name, MAGENTA, BOLD)}")
    print(f"  {c('Input :', DIM)}")
    for line in args_str.splitlines():
        print(f"    {c(line, WHITE)}")
    print()
    print(f"  {c('[y]', GREEN, BOLD)} Approve and run")
    print(f"  {c('[e]', CYAN, BOLD)} Edit the input before running")
    print(f"  {c('[n]', RED, BOLD)} Reject this tool call")
    print()

    while True:
        try:
            raw = input(f"  {c('Decision', YELLOW, BOLD)} (y/e/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Interrupted — rejecting tool call.", RED))
            return {
                "decision": "reject",
                "edited_args": None,
                "reject_reason": "Session interrupted by user.",
            }

        if raw in ("y", "yes", ""):
            print(c("  ✔ Approved.", GREEN, BOLD))
            return {"decision": "approve", "edited_args": None, "reject_reason": None}

        elif raw in ("e", "edit"):
            print(f"\n  Current args: {c(json.dumps(tool_args), DIM)}")
            print(f"  Enter new JSON args (or press Enter to keep current):")
            try:
                new_raw = input("  ❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                return {"decision": "approve", "edited_args": None, "reject_reason": None}

            if not new_raw:
                print(c("  ✔ Keeping original args. Approved.", GREEN))
                return {"decision": "approve", "edited_args": None, "reject_reason": None}

            try:
                new_args = json.loads(new_raw)
                if not isinstance(new_args, dict):
                    raise ValueError("Args must be a JSON object.")
                print(c("  ✔ Edited args accepted.", GREEN, BOLD))
                return {"decision": "edit", "edited_args": new_args, "reject_reason": None}
            except (json.JSONDecodeError, ValueError) as exc:
                print(c(f"  ✘ Invalid JSON: {exc}. Falling back to approve.", RED))
                return {"decision": "approve", "edited_args": None, "reject_reason": None}

        elif raw in ("n", "no", "reject"):
            try:
                reason = input(f"  {c('Rejection reason (optional):', DIM)} ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = ""
            reason = reason or "Rejected by human operator."
            print(c(f"  ✘ Rejected: {reason}", RED))
            return {
                "decision": "reject",
                "edited_args": None,
                "reject_reason": reason,
            }

        else:
            print(c("  Please enter y, e, or n.", YELLOW))


# ─────────────────────────────────────────────
# State display
# ─────────────────────────────────────────────

# Module-level sets to avoid printing the same item twice across events
_displayed_results: set[str] = set()
_displayed_thoughts: set[str] = set()


def _display_state_update(state: dict) -> None:
    """Print meaningful updates as graph state changes."""
    # Show latest tool result (once per tool_call_id)
    for result in state.get("tool_results", []):
        key = result.get("tool_call_id", "")
        if key and key not in _displayed_results:
            _displayed_results.add(key)
            show_tool_result(result)

    # Show latest AI thought (last non-tool-call AI message)
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "ai" and not msg.get("tool_calls"):
            content = msg.get("content", "").strip()
            if content and content not in _displayed_thoughts:
                _displayed_thoughts.add(content)
                print_section("Agent Thought", content, BLUE)
            break


# ─────────────────────────────────────────────
# Interrupt handling
# ─────────────────────────────────────────────


def _extract_interrupt_payload(snap) -> Optional[dict[str, Any]]:
    """
    Extract the interrupt payload from a graph snapshot's pending tasks.

    Returns the dict passed to interrupt() in interrupt_node, or None.
    """
    for task in getattr(snap, "tasks", []):
        for intr in getattr(task, "interrupts", []):
            val = getattr(intr, "value", None)
            if isinstance(val, dict):
                return val
    return None


# ─────────────────────────────────────────────
# Main agent runner
# ─────────────────────────────────────────────


def run_agent(query: str, graph: Any, thread_id: str) -> None:
    """
    Execute the full research agent loop for a single query.

    Flow
    ----
    1. Stream graph from initial state
    2. When graph pauses at interrupt_node, extract payload from snapshot
    3. Prompt human for decision
    4. Resume with Command(resume=<decision_dict>)
    5. Repeat until graph reaches END
    6. Display final answer
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "query": query,
        "messages": [],
        "pending_tool": None,
        "tool_results": [],
        "human_decision": None,
        "edited_args": None,
        "reject_reason": None,
        "iteration": 0,
        "final_answer": None,
        "error": None,
    }

    print(f"\n{divider()}")
    print(c(f"  🔍 Research Query: {query}", CYAN, BOLD))
    print(f"{divider()}\n")

    print(c("  ⚙  Agent reasoning…", DIM))

    # ── Drive the graph, handling interrupts in a loop ─────────────────
    input_value: Any = initial_state

    while True:
        # Stream until graph pauses or finishes
        last_state: dict = {}
        for event in graph.stream(input_value, config=config, stream_mode="values"):
            last_state = event
            _display_state_update(event)

        # Check if the graph is paused at an interrupt
        snap = graph.get_state(config)
        payload = _extract_interrupt_payload(snap)

        if payload is None:
            # Graph finished cleanly
            break

        # ── HITL prompt ────────────────────────────────────────────────
        tool_name = payload.get("tool_name", "unknown")
        tool_args = payload.get("tool_args", {})

        decision_dict = prompt_hitl(tool_name, tool_args)

        # Resume graph with the human's decision
        print(c("\n  ⚙  Resuming agent…", DIM))
        input_value = Command(resume=decision_dict)

    # ── Display final answer ───────────────────────────────────────────
    final_snap = graph.get_state(config)
    final_answer = final_snap.values.get("final_answer") if final_snap else None

    if not final_answer:
        final_answer = last_state.get("final_answer", "No answer produced.")

    show_final_answer(final_answer)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────


def main() -> None:
    """Entry point: display banner and run the interactive query loop."""
    print(BANNER)

    with SqliteSaver.from_conn_string(settings.checkpoint_db) as checkpointer:
        graph = build_graph(checkpointer)

        while True:
            try:
                query = input(f"{c('❯', CYAN, BOLD)} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(c("\n  Goodbye!", CYAN))
                break

            if not query:
                continue

            if query.lower() in ("exit", "quit", "q"):
                print(c("  Goodbye!", CYAN))
                break

            # Fresh thread per query (isolated checkpoint)
            _displayed_results.clear()
            _displayed_thoughts.clear()
            thread_id = str(uuid.uuid4())

            try:
                run_agent(query, graph, thread_id)
            except KeyboardInterrupt:
                print(c("\n  Query interrupted. Starting fresh.\n", YELLOW))
                continue
            except Exception as exc:  # noqa: BLE001
                print(c(f"\n  ✘ Unexpected error: {exc}", RED))
                import traceback
                traceback.print_exc()
