"""
State definition for the Research Agent graph.

Uses TypedDict for LangGraph compatibility.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class ToolCall(TypedDict):
    """Represents a single pending tool invocation."""

    id: str
    name: str
    args: dict[str, Any]


class ToolResult(TypedDict):
    """Result from a tool execution."""

    tool_call_id: str
    name: str
    content: str
    status: Literal["success", "rejected", "error"]


class AgentState(TypedDict):
    """
    Full mutable state threaded through every graph node.

    Fields
    ------
    query          : Original user research question.
    messages       : Full conversation history (LangChain message dicts).
    pending_tool   : Tool call waiting for human approval (None if none).
    tool_results   : Accumulated tool results this session.
    human_decision : Latest HITL decision ("approve" | "edit" | "reject").
    edited_args    : Replacement args when human chose "edit".
    reject_reason  : Reason string when human chose "reject".
    iteration      : Current ReAct iteration count.
    final_answer   : Populated when the agent produces its conclusion.
    error          : Non-empty when an unrecoverable error occurred.
    """

    query: str
    messages: list[dict[str, Any]]
    pending_tool: Optional[ToolCall]
    tool_results: list[ToolResult]
    human_decision: Optional[Literal["approve", "edit", "reject"]]
    edited_args: Optional[dict[str, Any]]
    reject_reason: Optional[str]
    iteration: int
    final_answer: Optional[str]
    error: Optional[str]
