"""Plan-and-execute: break a task into steps, run each, synthesize.

Run:  python examples/05_planner.py "帮我计算 (12+3)*5 和 100/7，并告诉我现在几点"
"""

import sys

from nova.agents.planner import PlanAndExecuteAgent
from nova.llm.openai_compat import OpenAICompatLLM
from nova.tools.builtin import calculator, get_datetime
from nova.tools.registry import ToolRegistry

task = sys.argv[1] if len(sys.argv) > 1 else "帮我计算 (12+3)*5 和 100/7，并告诉我现在几点"

agent = PlanAndExecuteAgent(OpenAICompatLLM(), tools=ToolRegistry([calculator, get_datetime]))

for event in agent.run(task).events:
    if event.type in ("plan", "step"):
        print(f"[{event.type}] {event.content}")
    elif event.type == "tool_call":
        for call in event.data["calls"]:
            print(f"    → {call['name']}({call['arguments']})")
    elif event.type == "answer":
        print(f"\n{event.content}")
