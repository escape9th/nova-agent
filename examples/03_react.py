"""ReAct agent: the tool-using loop, streamed event by event.

Run:  python examples/03_react.py "27 * 43 等于多少？"
"""

import sys

from nova.agents.react import ReActAgent
from nova.llm.openai_compat import OpenAICompatLLM
from nova.tools.builtin import calculator, fetch_url, get_datetime
from nova.tools.registry import ToolRegistry

task = sys.argv[1] if len(sys.argv) > 1 else "27 * 43 等于多少？"

agent = ReActAgent(
    llm=OpenAICompatLLM(),
    tools=ToolRegistry([calculator, get_datetime, fetch_url]),
)

for event in agent.run_stream(task):
    if event.type == "tool_call":
        for call in event.data["calls"]:
            print(f"→ {call['name']}({call['arguments']})")
    elif event.type == "tool_result":
        print(f"    {event.content[:80]}")
    elif event.type == "answer":
        print(f"\n{event.content}")
