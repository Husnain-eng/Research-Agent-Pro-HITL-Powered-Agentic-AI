"""
LangGraph node implementations for Research Agent Pro.

Each function is a pure node: receives AgentState, returns a partial update.

Nodes
-----
agent_node      — LLM reasoning / ReAct step
interrupt_node  — HITL pause using langgraph.types.interrupt (v1.0 API)
tool_node       — Execute approved/edited/rejected tool call
response_node   — Synthesize final answer

LangGraph v1.0 HITL pattern
----------------------------
interrupt() inside a node:
  1. Pauses graph execution and checkpoints state
  2. Returns the value passed to Command(resume=...) when the graph is resumed
  3. Enables clean HITL without deprecated NodeInterrupt
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.types import interrupt

from agent.state import AgentState, ToolCall
from agent.tools import TOOL_MAP
from config.settings import settings

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert research assistant. Use the available tools to gather
information and answer the user's research question thoroughly.

Follow the ReAct pattern:
1. Think about what information you need
2. Select the most appropriate tool
3. Analyze the result
4. Repeat until you have enough information
5. Provide a comprehensive final answer

When you have enough information to answer the question, respond with your final answer
directly WITHOUT calling any tools. Start your final response with "FINAL ANSWER:".

Available tools:
- duckduckgo_search: Search the web for current information
- arxiv_search: Find academic research papers
- wikipedia_search: Get encyclopedic background information

Be strategic: use Wikipedia for background, Arxiv for academic depth, DuckDuckGo for
current/specific information."""


# ─────────────────────────────────────────────
# agent_node
# ─────────────────────────────────────────────


def agent_node(state: AgentState) -> dict[str, Any]:
    """
    LLM reasoning node (ReAct step).

    Builds the message history, calls the LLM, and either:
    - Extracts the first tool call → sets pending_tool
    - Detects a final answer     → sets final_answer
    """
    from agent.llm import get_llm

    llm = get_llm()

    # Build LangChain message list
    lc_messages: list[Any] = [SystemMessage(content=SYSTEM_PROMPT)]

    if state["iteration"] == 0:
        lc_messages.append(HumanMessage(content=state["query"]))
    else:
        for msg in state.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
            name = msg.get("name")

            if role == "human":
                lc_messages.append(HumanMessage(content=content))
            elif role == "ai":
                if tool_calls:
                    lc_messages.append(AIMessage(content=content or "", tool_calls=tool_calls))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                lc_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id or "",
                        name=name or "",
                    )
                )

    response = llm.invoke(lc_messages)

    ai_msg_dict: dict[str, Any] = {
        "role": "ai",
        "content": response.content or "",
    }

    tool_calls = getattr(response, "tool_calls", None) or []

    updated_messages = list(state.get("messages", []))
    if state["iteration"] == 0:
        updated_messages.insert(0, {"role": "human", "content": state["query"]})

    if tool_calls:
        raw_tc = tool_calls[0]  # one tool at a time for HITL
        tc_id = raw_tc.get("id") or str(uuid.uuid4())

        pending: ToolCall = {
            "id": tc_id,
            "name": raw_tc["name"],
            "args": raw_tc.get("args", {}),
        }

        ai_msg_dict["tool_calls"] = [
            {
                "id": tc_id,
                "name": pending["name"],
                "args": pending["args"],
                "type": "tool_call",
            }
        ]
        updated_messages.append(ai_msg_dict)

        return {
            "messages": updated_messages,
            "pending_tool": pending,
            "iteration": state["iteration"] + 1,
            "human_decision": None,
            "edited_args": None,
            "reject_reason": None,
            "final_answer": None,
        }

    # No tool call — treat response as final answer
    content = response.content or ""
    if "FINAL ANSWER:" in content:
        content = content.split("FINAL ANSWER:", 1)[1].strip()

    updated_messages.append(ai_msg_dict)

    return {
        "messages": updated_messages,
        "pending_tool": None,
        "final_answer": content,
        "iteration": state["iteration"] + 1,
    }


# ─────────────────────────────────────────────
# interrupt_node  (LangGraph v1.0 API)
# ─────────────────────────────────────────────


def interrupt_node(state: AgentState) -> dict[str, Any]:
    """
    HITL interrupt node using langgraph.types.interrupt (v1.0).

    Calls interrupt() which:
    1. Checkpoints the full graph state to SQLite
    2. Pauses execution — control returns to the caller
    3. Returns the value from Command(resume=...) when the graph is resumed

    The CLI reads the interrupt payload (tool name + args), prompts the
    human, and resumes via graph.stream(Command(resume={...}), config).

    The returned decision dict is written back to AgentState so
    tool_node can act on it immediately after.
    """
    pending = state.get("pending_tool")
    if pending is None:
        return {}

    # Pause here — CLI displays this payload to the human operator
    human_response: dict[str, Any] = interrupt(
        {
            "tool_name": pending["name"],
            "tool_args": pending["args"],
            "tool_id": pending["id"],
        }
    )

    # human_response is whatever the CLI passed into Command(resume=...)
    # Expected shape: {"decision": "approve"|"edit"|"reject",
    #                  "edited_args": dict|None, "reject_reason": str|None}
    decision = human_response.get("decision", "approve")
    edited_args = human_response.get("edited_args")
    reject_reason = human_response.get("reject_reason")

    return {
        "human_decision": decision,
        "edited_args": edited_args,
        "reject_reason": reject_reason,
    }


# ─────────────────────────────────────────────
# tool_node
# ─────────────────────────────────────────────


def tool_node(state: AgentState) -> dict[str, Any]:
    """
    Tool execution node.

    Reads human_decision from state and either:
    - Runs the tool with original or edited args (approve / edit)
    - Injects a rejection message so the LLM can re-reason (reject)

    Appends a ToolMessage-compatible dict to message history.
    """
    pending = state.get("pending_tool")
    if pending is None:
        return {}

    decision = state.get("human_decision", "approve")
    tool_id = pending["id"]
    tool_name = pending["name"]

    tool_results = list(state.get("tool_results", []))
    updated_messages = list(state.get("messages", []))

    if decision == "reject":
        reason = state.get("reject_reason") or "Tool call rejected by human operator."
        result_content = f"[REJECTED] {reason}"
        tool_results.append(
            {
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result_content,
                "status": "rejected",
            }
        )
        updated_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_id,
                "name": tool_name,
                "content": result_content,
            }
        )
        return {
            "tool_results": tool_results,
            "messages": updated_messages,
            "pending_tool": None,
            "human_decision": None,
        }

    # approve or edit
    args = state.get("edited_args") if decision == "edit" else pending["args"]
    args = args or {}

    tool_fn = TOOL_MAP.get(tool_name)
    if tool_fn is None:
        result_content = f"[ERROR] Unknown tool: {tool_name}"
        status = "error"
    else:
        try:
            result_content = tool_fn.invoke(args)
            status = "success"
        except Exception as exc:  # noqa: BLE001
            result_content = f"[ERROR] Tool execution failed: {exc}"
            status = "error"

    tool_results.append(
        {
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": result_content,
            "status": status,
        }
    )
    updated_messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_id,
            "name": tool_name,
            "content": result_content,
        }
    )

    return {
        "tool_results": tool_results,
        "messages": updated_messages,
        "pending_tool": None,
        "human_decision": None,
        "edited_args": None,
    }


# ─────────────────────────────────────────────
# response_node
# ─────────────────────────────────────────────


def response_node(state: AgentState) -> dict[str, Any]:
    """
    Final answer synthesis node.

    If agent_node already produced a final_answer, this is a no-op.
    Otherwise synthesises an answer from accumulated tool results.
    """
    if state.get("final_answer"):
        return {}

    from agent.llm import get_llm_no_tools

    llm = get_llm_no_tools()

    tool_results = state.get("tool_results", [])
    if not tool_results:
        return {
            "final_answer": (
                "I was unable to gather enough information to answer your question."
            )
        }

    results_text = "\n\n---\n\n".join(
        f"[{r['name']}]\n{r['content']}"
        for r in tool_results
        if r.get("status") == "success"
    )

    if not results_text.strip():
        return {
            "final_answer": (
                "All tool calls were rejected or failed. Cannot produce an answer."
            )
        }

    synthesis_prompt = (
        f"Based on the following research results, provide a comprehensive answer to: "
        f'"{state["query"]}"\n\n'
        f"Research Results:\n{results_text}\n\n"
        f"Provide a well-structured, accurate, and thorough answer."
    )

    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    return {"final_answer": response.content or "Unable to synthesise answer."}
