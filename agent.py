"""
agent.py
--------
This file defines the AI agent's "brain" using LangGraph: a small state
machine that decides, on every turn, whether to (a) call the LLM, or
(b) run a tool the LLM asked for, and loops between the two until the
LLM produces a final answer with no more tool calls.

Graph shape:

        [entry] --> agent --(tool call?)--> tools --> agent --> ... --> END
                       |
                       +--(no tool call)--> END

This file is UI-agnostic -- it doesn't know anything about Streamlit or
Instagram. Any interface (web app, chat widget, messaging webhook) can
reuse `get_runnable()` and `prompt_ai()` as-is.
"""

from dotenv import load_dotenv
import os
import json
import asyncio

from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import InMemorySaver

from typing import Annotated, Literal, Dict
from typing_extensions import TypedDict

from tools import available_functions

load_dotenv()

# --- Model setup ---
model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
chatbot = ChatGroq(
    model=model,
    api_key=os.getenv("GROQ_API_KEY"),
    streaming=True,
)

# Bind every tool from tools.py to the model, so the LLM knows what
# functions it's allowed to call and what arguments each one expects.
tools = [tool for _, tool in available_functions.items()]
chatbot_with_tools = chatbot.bind_tools(tools)


class GraphState(TypedDict):
    """
    The data that flows through every node in the graph.

    - messages: the running conversation history. `add_messages` tells
      LangGraph to APPEND new messages instead of overwriting the list
      on every node's return value.
    - tool_call_count: a simple safety counter (see should_continue)
      to guarantee the graph can never loop forever.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: int


async def call_model(state: GraphState, config: RunnableConfig) -> Dict[str, AnyMessage]:
    """
    The "agent" node: sends the conversation so far to the LLM and lets
    it decide whether to reply directly or call one or more tools.
    Resets tool_call_count to 0 whenever a fresh human turn starts
    (i.e. the previous message wasn't a tool result).
    """
    messages = state["messages"]

    if isinstance(messages[-1], ToolMessage):
        tool_call_count = state.get("tool_call_count", 0)
    else:
        tool_call_count = 0

    response = await chatbot_with_tools.ainvoke(messages, config)

    return {"messages": response, "tool_call_count": tool_call_count}


def tool_node(state: GraphState) -> Dict[str, AnyMessage]:
    """
    The "tools" node: executes every tool call the LLM requested in its
    last message, and packages each result back into a ToolMessage so
    the LLM can read the outcome on its next turn.
    """
    messages = state["messages"]
    tool_call_count = state.get("tool_call_count", 0)
    last_message = messages[-1] if messages else None

    outputs = []

    if last_message and last_message.tool_calls:
        tool_call_count += 1

        for call in last_message.tool_calls:
            tool = available_functions.get(call["name"])
            if tool is None:
                raise Exception(f"Tool '{call['name']}' not found.")

            output = tool.invoke(call["args"])

            # Tool outputs are dicts -- serialize to JSON so ToolMessage
            # (which requires a string) can carry them. ensure_ascii=False
            # keeps Arabic text human-readable instead of \u-escaped.
            content = (
                output if isinstance(output, str)
                else json.dumps(output, ensure_ascii=False)
            )
            outputs.append(ToolMessage(content, tool_call_id=call["id"]))

    return {"messages": outputs, "tool_call_count": tool_call_count}


def should_continue(state: GraphState) -> Literal["__end__", "tools"]:
    """
    The conditional edge after the "agent" node. Decides where the
    graph goes next:
      - "tools" if the LLM's last message requested a tool call
      - END otherwise, or if we've hit the safety cap on tool calls
        (protects against the LLM getting stuck in a call/retry loop)
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    tool_call_count = state.get("tool_call_count", 0)

    if tool_call_count >= 11:
        return END

    if not last_message or not last_message.tool_calls:
        return END

    return "tools"


async def get_runnable():
    """
    Builds and compiles the LangGraph state machine.

    Uses InMemorySaver as the checkpointer -- this keeps per-conversation
    memory in RAM, keyed by thread_id. It's simple and has no external
    dependencies, which makes it ideal for development. For production
    persistence across restarts, swap this for a database-backed
    checkpointer (e.g. Postgres or SQLite).
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


async def prompt_ai(messages, chatbot, thread_id):
    """
    Streams the AI's reply token-by-token for a given conversation.

    thread_id scopes the conversation memory to a single user/session --
    each user gets their own isolated history inside the same running
    app, which is essential once more than one person can chat at once.
    """
    config = {"configurable": {"thread_id": thread_id}}

    async for event in chatbot.astream_events(
        {"messages": messages}, config, version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            yield event["data"]["chunk"].content