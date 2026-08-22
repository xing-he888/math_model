"""接入 De-AI Prompt Enhancer 写作技能包:把 test/ 下的中文去 AI 味 SKILL 注入论文生成。

- read_writing_skill():读取技能包 SKILL.md 正文(去掉 YAML 头),供直接注入写 paper 的 prompt;
- get_writing_skill:同一内容的 @tool 封装,供 agent 在写/改中文内容时主动调用。
"""
from langchain_core.tools import tool
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = WORKSPACE_ROOT / "test" / "De-AI-Prompt-Enhancer-Writer-Booster-SKILL"


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip("\n")
    return text


def read_writing_skill(mode: str = "de-AI") -> str:
    """读取写作技能包的规范正文(de-AI 去 AI 味 / good 作者文风),去掉 YAML 头。"""
    sub = "de-AI-writing" if str(mode).strip().lower().startswith("de") else "good-writing"
    skill_md = SKILL_ROOT / sub / "SKILL.md"
    if not skill_md.exists():
        return f"[写作技能包未找到: {skill_md}]"
    text = _strip_frontmatter(skill_md.read_text(encoding="utf-8"))
    idx = SKILL_ROOT / sub / "references" / "ai-trace-index.md"
    if idx.exists():
        text += "\n\n# 附:AI 痕迹索引(精修时参考)\n" + _strip_frontmatter(idx.read_text(encoding="utf-8"))
    return text


@tool
def get_writing_skill(mode: str = "de-AI") -> str:
    """获取中文写作规范(去 AI 味 / 作者文风),用于撰写或润色 paper 目录下的中文论文与思路。
    参数 mode: 'de-AI' 通用去 AI 味(默认,适合论文/综述);'good' 复现作者文风(带第一人称)。
    返回: 该模式的写作规范正文,请严格遵循后再生成/修改中文内容。"""
    return read_writing_skill(mode)


tools = [get_writing_skill]
