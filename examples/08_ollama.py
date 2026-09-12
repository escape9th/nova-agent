"""Run an agent with a local Ollama model — no cloud API key needed.

Prerequisites:
    ollama pull qwen2.5     # or llama3.2, etc.
    ollama serve            # listens on http://localhost:11434

Run:  python examples/08_ollama.py "27 * 43 等于多少？"
"""

import sys

from nova.agents.react import ReActAgent
from nova.llm.ollama import OllamaLLM
from nova.tools.builtin import calculator, get_datetime
from nova.tools.registry import ToolRegistry

task = sys.argv[1] if len(sys.argv) > 1 else "27 * 43 等于多少？"

agent = ReActAgent(
    llm=OllamaLLM(model="qwen2.5"),
    tools=ToolRegistry([calculator, get_datetime]),
)

for event in agent.run_stream(task):
    if event.type == "tool_call":
        for call in event.data["calls"]:
            print(f"→ {call['name']}({call['arguments']})")
    elif event.type == "answer":
        print(f"\n{event.content}")
