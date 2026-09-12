[English](README.md) · [简体中文](README_zh.md)

# Nova

一个极简、可读、**从零手写**的 AI Agent 框架。

Nova 直接在 OpenAI 兼容协议之上，实现了现代 AI Agent 的核心机制——推理循环、工具调用、记忆、规划与多 Agent 协作——**不依赖任何重量级框架**。整个核心只有几百行纯 Python，每一部分都写得让你能读懂、能改、能讲清楚。

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 为什么选择 Nova？

- **看懂底层。** ReAct 循环、函数调用、向量检索……每个机制都用约 50~100 行代码实现，你真正读得懂。
- **Provider 无关。** 兼容 OpenAI、阿里云百炼（Qwen）、DeepSeek、Moonshot 以及任何 OpenAI 兼容端点。
- **零魔法。** 没有 LangChain、没有 CrewAI。Nova 告诉你一个 agent 框架*做了什么*，而不只是"怎么调用"。

## 特性

- **LLM 抽象** —— 精简的 provider 接口 + OpenAI 兼容适配器 + 离线测试用的确定性 mock。
- **工具系统** —— `@tool` 装饰器从类型注解自动推断 JSON Schema，外加注册表和一组安全的内置工具。
- **记忆** —— 滑动窗口会话缓冲（短期）+ 零依赖的向量库（特征哈希嵌入，长期 RAG）。
- **ReAct agent** —— 经典的"推理 + 行动"循环，带工具调用与迭代上限。
- **规划执行** —— 结构化规划（通过强制工具输出）+ 逐步执行 + 汇总。
- **多 Agent 团队** —— supervisor 通过同一套工具调用机制把任务分派给专家。
- **流式输出** —— token 级流式 + 事件级 agent 追踪。
- **CLI、示例与测试** —— 可运行的 demo，以及完全离线的测试套件。

## 安装

```bash
git clone https://github.com/escape9th/nova-agent.git
cd nova-agent
pip install -e ".[dev]"
```

## 快速开始

```bash
# 1. 配置你的 provider（复制并编辑）
cp .env.example .env

# 2. 跑一个一次性任务
nova run "27 * 43 等于多少？"

# 3. 或进入交互式对话
nova chat
```

```python
from nova.llm.openai_compat import OpenAICompatLLM
from nova.agents.react import ReActAgent
from nova.tools.registry import ToolRegistry
from nova.tools.builtin import calculator, get_datetime

agent = ReActAgent(
    llm=OpenAICompatLLM(),
    tools=ToolRegistry([calculator, get_datetime]),
)

result = agent.run("(12 + 3) * 5 是多少？")
print(result.answer)
```

## 架构

```
                    ┌─────────────────────────────┐
                    │          CLI / API          │
                    └──────────────┬──────────────┘
                                   │
   ┌───────────────────────────────┼───────────────────────────────┐
   │                               │                               │
┌──▼────────────┐       ┌──────────▼──────────┐       ┌────────────▼──┐
│  ReActAgent   │       │ PlanAndExecuteAgent │       │     Team      │
│  reason+act   │       │   plan + execute    │       │  supervisor   │
└──┬────────────┘       └──────────┬──────────┘       └────────────┬──┘
   │           agents              │                              │
   └───────────────────────────────┼──────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
  ┌────▼─────┐              ┌──────▼──────┐             ┌───────▼──────┐
  │  Tools   │              │   Memory    │             │      LLM     │
  │ registry │              │ buffer/vec  │             │ OpenAI-compat│
  └──────────┘              └─────────────┘             └──────────────┘
```

每个组件的逐层讲解见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 示例

| 示例 | 展示内容 |
| --- | --- |
| [examples/01_chat.py](examples/01_chat.py) | 通过 LLM 层进行基本对话 |
| [examples/02_tools.py](examples/02_tools.py) | 注册并调用工具 |
| [examples/03_react.py](examples/03_react.py) | ReAct 工具调用循环 |
| [examples/04_memory.py](examples/04_memory.py) | 向量库 / RAG 检索 |
| [examples/05_planner.py](examples/05_planner.py) | 规划执行 |
| [examples/06_team.py](examples/06_team.py) | 多 Agent 监督协作 |
| [examples/07_streaming.py](examples/07_streaming.py) | token 流式输出 |

## 运行测试

```bash
pytest
```

测试套件完全离线——用 `MockLLM` 驱动预设响应，**不需要 API Key，也不需要联网**。

## 安全说明

内置工具（文件读写、网页抓取）以调用者的权限运行。Nova 有意保持它们的朴素，让信任边界清晰可见；在把 agent 暴露给不可信输入之前，请先做沙箱化。

## 许可证

[MIT](LICENSE)
