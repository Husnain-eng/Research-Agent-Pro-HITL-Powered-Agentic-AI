"""
LangGraph workflow definition for Research Agent Pro.

Graph topology
--------------

    ┌──────────────┐
    │  agent_node  │◄──────────────────────────┐
    └──────┬───────┘                           │
           │                                   │
     has pending_tool?                         │
           │                                   │
    ┌──────▼───────┐                           │
    │interrupt_node│  interrupt() pauses here  │
    │              │◄── Command(resume=...)     │
    └──────┬───────┘  human decision injected  │
           │                                   │
    ┌──────▼───────┐                           │
    │  tool_node   │                           │
    └──────┬───────┘                           │
           │                                   │
    max iterations?──No────────────────────────┘
           │Yes
    ┌──────▼───────┐
    │response_node │
    └──────┬───────┘
           │
          END

Checkpointing
-------------
SqliteSaver automatically saves state at every node boundary.
interrupt() inside interrupt_node creates a named breakpoint that
the caller can resume by streaming Command(resume=<decision_dict>).
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent.nodes import agent_node, interrupt_node, response_node, tool_node
from agent.state import AgentState
from config.settings import settings


# ─────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────


def route_after_agent(state: AgentState) -> Literal["interrupt_node", "response_node"]:
    """Route to interrupt (HITL) if a tool is pending, else to final response."""
    if state.get("pending_tool") is not None:
        return "interrupt_node"
    return "response_node"


def route_after_tool(state: AgentState) -> Literal["agent_node", "response_node"]:
    """After tool execution, loop back to agent unless max iterations reached."""
    if state.get("iteration", 0) >= settings.max_iterations:
        return "response_node"
    return "agent_node"


# ─────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────


def build_graph(checkpointer: SqliteSaver):
    """
    Build and compile the LangGraph state machine.

    Parameters
    ----------
    checkpointer:
        A SqliteSaver instance for persistent state storage and resume.

    Returns
    -------
    Compiled LangGraph ready for invocation.
    """
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("interrupt_node", interrupt_node)
    workflow.add_node("tool_node", tool_node)
    workflow.add_node("response_node", response_node)

    # Entry point
    workflow.set_entry_point("agent_node")

    # Conditional routing after agent reasons
    workflow.add_conditional_edges(
        "agent_node",
        route_after_agent,
        {
            "interrupt_node": "interrupt_node",
            "response_node": "response_node",
        },
    )

    # After HITL decision is collected, execute the tool
    workflow.add_edge("interrupt_node", "tool_node")

    # After tool runs, either loop back to agent or finish
    workflow.add_conditional_edges(
        "tool_node",
        route_after_tool,
        {
            "agent_node": "agent_node",
            "response_node": "response_node",
        },
    )

    workflow.add_edge("response_node", END)

    # Note: no interrupt_before needed — interrupt() inside interrupt_node
    # handles checkpointing and pausing natively (LangGraph v1.0 API).
    return workflow.compile(checkpointer=checkpointer)
