"""The ReAct agent: the heart of the framework.

ReAct (Reason + Act) is the loop every modern tool-using agent is built on:

    1. send the conversation history + tool specs to the model
    2. if the model returns tool calls, execute them and append the results
       as "observations" for the model to reason over
    3. repeat until the model answers directly (or the budget is exhausted)

The whole loop is ~50 lines; that's the point — this is what frameworks like
LangChain wrap in layers of abstraction.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from ..llm.base import BaseLLM, Message
from ..memory.buffer import ConversationBufferMemory
from ..prompts.templates import DEFAULT_REACT_SYSTEM
from ..tools.registry import ToolRegistry
from .base import AgentEvent, AgentResult, BaseAgent


class ReActAgent(BaseAgent):
    name = "react"

    def __init__(
        self,
        llm: BaseLLM,
        tools: Sequence | ToolRegistry | None = None,
        memory: ConversationBufferMemory | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 8,
        temperature: float = 0.0,
    ):
        self.llm = llm
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools or [])
        self.memory = memory
        self.system_prompt = system_prompt or DEFAULT_REACT_SYSTEM
        self.max_iterations = max_iterations
        self.temperature = temperature

    def run(self, task: str) -> AgentResult:
        events = list(self.run_stream(task))
        answer = ""
        tool_calls = 0
        iterations = 0
        for event in events:
            if event.type in ("answer", "error") and not answer:
                answer = event.content
            elif event.type == "tool_result":
                tool_calls += 1
            iterations = max(iterations, int(event.data.get("iterations", 0)))
        return AgentResult(
            answer=answer,
            events=events,
            iterations=iterations,
            tool_calls=tool_calls,
        )

    def run_stream(self, task: str) -> Iterator[AgentEvent]:
        # Note: use `is not None`, not `or` — an empty ConversationBufferMemory
        # is falsy (it defines __len__), which would silently create a fresh
        # throwaway buffer every call and break multi-turn memory.
        memory = self.memory if self.memory is not None else ConversationBufferMemory()

        # Persist the user turn into memory *before* building the prompt, so
        # multi-turn chat remembers what was asked earlier — not just the
        # assistant's previous answers.
        user_message = Message.user(task)
        memory.add(user_message)

        messages: list[Message] = [Message.system(self.system_prompt)]
        messages.extend(memory.messages())

        for i in range(self.max_iterations):
            response = self.llm.chat(
                messages,
                tools=self.tools.specs() or None,
                temperature=self.temperature,
            )

            if response.tool_calls:
                assistant = Message.assistant(response.content, response.tool_calls)
                messages.append(assistant)
                memory.add(assistant)
                yield AgentEvent(
                    type="tool_call",
                    content=response.content or "",
                    data={
                        "iterations": i + 1,
                        "calls": [
                            {"name": tc.name, "arguments": tc.arguments}
                            for tc in response.tool_calls
                        ],
                    },
                )
                for tc in response.tool_calls:
                    result = self.tools.execute(tc.name, tc.arguments)
                    tool_message = Message.tool(result, tc.id, tc.name)
                    messages.append(tool_message)
                    memory.add(tool_message)
                    yield AgentEvent(
                        type="tool_result",
                        content=result,
                        data={"iterations": i + 1, "name": tc.name, "id": tc.id},
                    )
            else:
                content = response.content or ""
                assistant = Message.assistant(content)
                messages.append(assistant)
                memory.add(assistant)
                yield AgentEvent(
                    type="answer", content=content, data={"iterations": i + 1}
                )
                return

        yield AgentEvent(
            type="error",
            content=f"stopped after {self.max_iterations} iterations without a final answer",
            data={"iterations": self.max_iterations},
        )

    async def arun(self, task: str) -> AgentResult:
        """Async version of :meth:`run` — the same loop, but awaits the model.

        Tool execution stays synchronous (tools are plain Python callables); the
        async gain is in overlapping many concurrent agent runs and non-blocking
        model I/O.
        """
        memory = self.memory if self.memory is not None else ConversationBufferMemory()

        user_message = Message.user(task)
        memory.add(user_message)

        messages: list[Message] = [Message.system(self.system_prompt)]
        messages.extend(memory.messages())

        events: list[AgentEvent] = []
        tool_calls = 0

        for i in range(self.max_iterations):
            response = await self.llm.achat(
                messages,
                tools=self.tools.specs() or None,
                temperature=self.temperature,
            )

            if response.tool_calls:
                assistant = Message.assistant(response.content, response.tool_calls)
                messages.append(assistant)
                memory.add(assistant)
                events.append(
                    AgentEvent(
                        type="tool_call",
                        content=response.content or "",
                        data={
                            "iterations": i + 1,
                            "calls": [
                                {"name": tc.name, "arguments": tc.arguments}
                                for tc in response.tool_calls
                            ],
                        },
                    )
                )
                for tc in response.tool_calls:
                    result = self.tools.execute(tc.name, tc.arguments)
                    tool_message = Message.tool(result, tc.id, tc.name)
                    messages.append(tool_message)
                    memory.add(tool_message)
                    tool_calls += 1
                    events.append(
                        AgentEvent(
                            type="tool_result",
                            content=result,
                            data={"iterations": i + 1, "name": tc.name, "id": tc.id},
                        )
                    )
            else:
                content = response.content or ""
                memory.add(Message.assistant(content))
                events.append(
                    AgentEvent(type="answer", content=content, data={"iterations": i + 1})
                )
                return AgentResult(
                    answer=content, events=events, iterations=i + 1, tool_calls=tool_calls
                )

        message = f"stopped after {self.max_iterations} iterations without a final answer"
        events.append(
            AgentEvent(type="error", content=message, data={"iterations": self.max_iterations})
        )
        return AgentResult(
            answer=message, events=events, iterations=self.max_iterations, tool_calls=tool_calls
        )
