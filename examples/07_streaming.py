"""Streaming: print tokens as the model generates them.

Run:  python examples/07_streaming.py "给我写一首关于秋天的短诗"
"""

import sys

from nova.llm.base import Message
from nova.llm.openai_compat import OpenAICompatLLM

prompt = sys.argv[1] if len(sys.argv) > 1 else "给我写一首关于秋天的短诗"

llm = OpenAICompatLLM()
for chunk in llm.stream([Message.user(prompt)]):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
