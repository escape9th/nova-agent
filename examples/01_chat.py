"""Basic chat through the OpenAI-compatible layer.

Run:  python examples/01_chat.py
"""

from nova.llm.base import Message
from nova.llm.openai_compat import OpenAICompatLLM

llm = OpenAICompatLLM()

messages = [
    Message.system("You are a concise assistant."),
    Message.user("用一句话解释什么是 AI Agent？"),
]
response = llm.chat(messages)
print(response.content)
