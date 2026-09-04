"""Multi-agent team: a supervisor delegates to specialist agents.

Run:  python examples/06_team.py "写一个关于 Python 的冷笑话，然后算一下 2 的 10 次方"
"""

import sys

from nova.agents.react import ReActAgent
from nova.agents.team import Team, TeamMember
from nova.llm.openai_compat import OpenAICompatLLM
from nova.tools.builtin import calculator
from nova.tools.registry import ToolRegistry

task = sys.argv[1] if len(sys.argv) > 1 else "写一个关于 Python 的冷笑话，然后算一下 2 的 10 次方"

llm = OpenAICompatLLM()

joke_writer = ReActAgent(llm, system_prompt="You are a comedian. Write short, funny jokes.")
math_agent = ReActAgent(
    llm,
    tools=ToolRegistry([calculator]),
    system_prompt="You are a mathematician. Use the calculator tool for arithmetic.",
)

team = Team(
    members=[
        TeamMember("joker", "writes jokes and humorous content", joke_writer),
        TeamMember("math", "does arithmetic and math", math_agent),
    ],
    llm=llm,
)

for event in team.run(task).events:
    if event.type == "tool_call":
        for call in event.data["calls"]:
            print(f"→ {call['name']}: {call['arguments']}")
    elif event.type == "answer":
        print(f"\n{event.content}")
