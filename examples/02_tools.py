"""Tools: register and call tools directly, without a model.

Run:  python examples/02_tools.py
"""

from nova.tools.builtin import calculator, get_datetime, list_files
from nova.tools.registry import ToolRegistry

registry = ToolRegistry([calculator, get_datetime, list_files])

print("Registered tools:", [t.name for t in registry])
print("calculator('3 * (4 + 5)'):", registry.execute("calculator", {"expression": "3 * (4 + 5)"}))
print("get_datetime():", registry.execute("get_datetime", {}))
print("list_files('.'):", registry.execute("list_files", {"directory": "."}))
