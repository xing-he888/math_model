"""可插拔 skill 注册中心。

新增一个 skill 的步骤:
  1. 在本目录(src/skills/)新建一个 .py 模块;
  2. 用 @tool 定义能力,并在模块底部导出 `tools = [fn1, fn2, ...]`;
  3. 无需改动 agent.py —— 启动时会自动发现并挂载到 agent 上。
"""
import importlib
import pkgutil
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent


def load_skill_tools():
    """遍历 src/skills 下所有模块,收集每个模块导出的 tools 列表。"""
    collected = []
    for mod in pkgutil.iter_modules([str(_SKILLS_DIR)]):
        module = importlib.import_module(f"src.skills.{mod.name}")
        collected.extend(getattr(module, "tools", []))
    return collected
