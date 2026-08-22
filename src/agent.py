from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from typing import TypedDict,List,Dict,Annotated
from langgraph.graph import MessagesState
from langgraph.types import interrupt, Command
from loguru import logger
from pathlib import Path
import sys
import os

# 无论从根目录还是 src 目录运行，都确保能找到 tool.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage,AIMessage, SystemMessage, ToolMessage, RemoveMessage
from dotenv import  load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from models import get_model, list_models, MODEL_REGISTRY, DEFAULT_MODEL
from pypdf import PdfReader
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Send
from tool import tools as base_tools, write_workspace, WORKSPACE_ROOT
from src.skills import load_skill_tools
from src.skills.writing import read_writing_skill

# 基础工具(tool.py)+ 可插拔 skill(src/skills/):新增 skill 只需在 src/skills/ 放一个模块并导出 tools 列表
tools = base_tools + load_skill_tools()

TOOLS_BY_NAME = {t.name: t for t in tools}
import pandas as pd
import json
import re
import operator
import subprocess
import shutil

# 然后将 search 作为工具传递给 LangGraph 的 Agent
#读取env文件中的apikey
load_dotenv(override=True)

# ---------- 模型（可配置：从注册表选择，支持 deepseek/gpt/glm/qwen/kimi/mimo） ----------
# 通过环境变量 MATH_MODEL 选择要用的模型，例如 deepseek / gpt / glm / qwen / kimi / mimo
# 新增模型：在 src/models.py 的 MODEL_REGISTRY 里加一项即可，无需改这里。
model = get_model()
model_with_tool=model.bind_tools(tools=tools)
tool_node=ToolNode(tools=tools)

# 运行时动态切换全局模型（前端/API 可调用），无需重启进程
def set_model(name: str = None) -> str:
    """重设 model / model_with_tool / model_with_struct 三个全局对象，返回实际生效的 key。"""
    global model, model_with_tool, model_with_struct
    key = (name or os.getenv("MATH_MODEL", DEFAULT_MODEL)).strip().lower()
    if key not in MODEL_REGISTRY:
        key = DEFAULT_MODEL
    model = get_model(key)
    model_with_tool = model.bind_tools(tools=tools)
    model_with_struct = model.with_structured_output(schema=structed_output_state)
    return key

question_path='./question'
dataset_path='./dataset'

#全局状态
class over_all_state(MessagesState):
    input_problem: Annotated[str, '题目原始文本']
    problem_str: Annotated[str, '问题题干']
    problem_index: Annotated[Dict[str, str], '问题索引字典']
    dataset: Annotated[str, '题目给出的数据集']
    modeling_approach: Annotated[str, '建模手自己的思路']
    modeling_analysis: Annotated[str, '模型分析出的基础思路']
    retry_count: Annotated[int, '工具重试计数'] = 0
    tool_rounds: Annotated[int, '工具调用总轮数'] = 0
    review_feedback: Annotated[str, '审核意见'] = ''
    review_result: Annotated[str, '审核结果'] = ''
    human_feedback: Annotated[str, '人工对最终总结的意见'] = ''
    method: Annotated[str, 'worker方法身份'] = ''
    dataset_files: Annotated[str, '数据集文件名清单'] = ''
    answers: Annotated[List[Dict[str, str]], operator.add] = []
    code_files: Annotated[List[str], operator.add] = []
    run_report: List[Dict[str, str]] = []
    failed_qs: Annotated[List[str], operator.add] = []
    done_pairs: Annotated[List[str], operator.add] = []
    # 注意:必须用 add_messages 而非 operator.add——final_analysis 打回时会写入 RemoveMessage 删除指令,
    # 只有 add_messages 能消化它;operator.add 会把指令本身拼进列表,后续 invoke 模型时抛 TypeError
    compare_msgs: Annotated[List, add_messages] = []
    final_summary: Annotated[str, '最终总结'] = ''
    article_chapters: Annotated[List[str], '已生成的论文章节'] = []
    compile_status: Annotated[str, '论文编译验证状态'] = ''

#定义格式化的状态，仅开始时使用
class structed_output_state(TypedDict):
    problem_str:Annotated[str,'问题题干']
    problem_index:Annotated[Dict[str, str], "问题索引字典"]

model_with_struct=model.with_structured_output(schema=structed_output_state)

#用来初始化input_problem
def load_problem(state: over_all_state) -> dict:
    """读取 question 文件夹里的题目文件（.txt/.md 按文本读，.pdf 用 pypdf 提取），合并后写入 input_problem 字段"""
    logger.info('正在运行load_problem节点')
    path = Path(question_path)
    contents = []
    for item in path.iterdir():
        if not item.is_file():
            continue
        try:
            if item.suffix.lower() == ".pdf":
                reader = PdfReader(str(item))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = item.read_text(encoding="utf-8")
        except Exception as e:
            text = f"(读取 {item.name} 失败: {e})"
        contents.append(f"=== {item.name} ===\n{text}")
    return {"input_problem": "\n\n".join(contents) if contents else "(question 文件夹为空)"}

#对文本进行格式化用来初始化problem_str和problem_index
def question_structed(state:over_all_state) ->structed_output_state:
    logger.info('正在运行question_structed节点')
    input_problem=state['input_problem']
    prompt = f"""
    数学建模题目，提取出题目中所有编号问题的完整题干
    （从“问题 X：”开始到下一个“问题”、“问题 X”或“相关说明”之前），
    问题数量不固定，以题目实际编号为准，如“问题1”、“问题2”……
    返回格式必须包含：
    - problem_str: 完整题目全文
    - problem_index: 字典，键为各问题编号（如“问题1”、“问题2”），值为对应问题的完整题干文本。
    
    题目内容：
    {input_problem}
    """
    resp=model_with_struct.invoke(
        [HumanMessage(prompt)]
    )
    return {
        **resp,
        'messages': [AIMessage(content="已成功提取问题索引")]
    }

#建模路由:工具调用未超限才进工具节点;超限后即使模型仍带 tool_calls 也强制进入审核,防止死循环
def modeling_route(state: over_all_state) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if getattr(last, "tool_calls", None) and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
        return "tools"
    return "review"

#此节点为中断节点建模手提供对题目的看法以及让模型给出自己的思路
def modeling(state: over_all_state) -> over_all_state:
    logger.info('正在运行 modeling 节点')
    
    # ----- 1. 重试逻辑处理 -----
    last_message = state["messages"][-1] if state["messages"] else None
    retry_count = state.get("retry_count", 0)
    tool_rounds = state.get("tool_rounds", 0)
    is_tool_return = isinstance(last_message, ToolMessage)
    is_rework = bool(state.get("review_feedback"))

    # 工具每返回一次(无论成败)都计一轮,防止模型反复成功调用工具导致死循环
    if is_tool_return:
        tool_rounds += 1
        content = last_message.content
        if "错误" in content or "失败" in content or "不存在" in content:
            retry_count += 1
            logger.warning(f"工具执行失败，正在进行第 {retry_count} 次重试...")

    # 失败重试超过 3 次或工具总轮数超过上限时,强制禁用工具
    if retry_count >= 3 or tool_rounds >= MAX_TOOL_ROUNDS:
        logger.warning(f"工具调用达上限(失败{retry_count}次/共{tool_rounds}轮)，将禁用工具调用，强制输出结果")
        model_to_use = model_with_tool.bind_tools(tools, tool_choice="none")
        retry_hint = "\n（注意：工具调用次数已达上限，请不要再调用工具，直接根据已有信息给出最终分析。）"
    else:
        model_to_use = model_with_tool
        retry_hint = f"\n（当前工具已调用 {tool_rounds} 轮，失败 {retry_count} 次；如无需更多信息请直接给出最终分析。）"

    # 首次进入节点时中断询问建模手思路；工具调用回退或审核打回时不再中断，直接继续分析
    if is_tool_return or is_rework:
        modeling_approach = ""
    else:
        modeling_approach = interrupt('请简述你的建模思路（可直接回车跳过）') or ""
    problem_str = state["problem_str"]
    problem_index = state["problem_index"]
    dataset = state.get("dataset")

    system_prompt = (
        "你是一名顶级的数学建模手。请严格遵循以下步骤对题目进行深度分析：\n"
        "1. 问题定性（优化/预测/评价/分类）\n"
        "2. 数据洞察与预处理建议\n"
        "3. 合理假设\n"
        "4. 数学模型构建（目标函数+约束条件）\n"
        "5. 求解算法推荐\n"
        "6. 优劣势分析\n"
        "题目与数据已完整提供，通常无需调用工具；如需查资料请用 search。\n"
        "若调用 read_workspace，work_dir 参数必填，只能是 code/paper/photo/dataset 之一，例如 read_workspace(work_dir=\"paper\", rel_path=\"问题1_思路.md\")。\n"
        "若工具连续报错，请根据你的常识完成分析。"
    )

    user_parts = [
        f"每一个小问的题目：{problem_index}",
        f"题目内容：{problem_str}",
        f"数据集位置：{dataset}" if dataset else "（无数据集）"
    ]
    if is_tool_return:
        user_parts.append(f"最近一次工具调用的返回结果（请据此判断上一步是否成功、如何修正参数）：\n{last_message.content}")
    elif modeling_approach and modeling_approach.strip():
        user_parts.append(f"用户提供的思路：{modeling_approach}")
    elif is_rework:
        user_parts.append(f"建模手审核意见（请据此重新分析）：{state.get('review_feedback')}")
    else:
        user_parts.append("（用户未提供思路，请自行构思）")
    user_parts.append(retry_hint)
    user_message = "\n".join(user_parts)

    response = model_to_use.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ])

    return {
        "messages": [response],          # 追加 AI 回复
        "modeling_analysis": response.content,
        "retry_count": retry_count,      # 更新重试计数器
        "tool_rounds": tool_rounds       # 更新工具轮数
    }


#此函数用于转换格式
def _read_file_by_suffix(path: Path) -> str:
    """按文件后缀分发读取：表格类（csv/tsv/xlsx）转 DataFrame 文本、pdf 提取、docx 提取、
    json/jsonl 美化输出，其余按 utf-8 文本读取"""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", encoding="utf-8")
        return df.fillna("").to_string(index=False)
    if suffix in (".xlsx", ".xls"):
        sheets = []
        for sheet_name in pd.ExcelFile(path).sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name)
            sheets.append(f"### 工作表: {sheet_name} ###\n" + df.fillna("").to_string(index=False))
        return "\n\n".join(sheets)
    if suffix == ".json":
        return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2)
    if suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return "\n".join(json.dumps(json.loads(line), ensure_ascii=False) for line in lines)
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            parts.append(pd.DataFrame(rows).fillna("").to_string(index=False))
        return "\n".join(parts)
    if suffix in (".txt", ".md", ".xml", ".html", ".log", ".dat", ".cfg", ".ini"):
        return path.read_text(encoding="utf-8")
    # 未知格式回退 utf-8 文本读取
    return path.read_text(encoding="utf-8")

#此节点用来读取dataset文件夹中文件的数据
def read_dataset(state:over_all_state) -> over_all_state:
    """读取 dataset 文件夹里的数据文件（按后缀分发处理多种格式），合并后写入 dataset 字段"""
    logger.info('正在运行read_dataset节点')
    path = Path(dataset_path)
    contents = []
    for item in sorted(path.iterdir()):
        if not item.is_file():
            continue
        try:
            text = _read_file_by_suffix(item)
        except UnicodeDecodeError:
            text = f"(文件 {item.name} 是二进制格式（{item.suffix}），无法直接读取，请先转为 csv/txt 放入 dataset 文件夹)"
        except Exception as e:
            text = f"(读取 {item.name} 失败: {e})"
        contents.append(f"=== {item.name} ===\n{text}")
    return {"dataset": "\n\n".join(contents) if contents else "(dataset 文件夹为空)"}


#此节点用于人工展示与审核 modeling_analysis：可通过、打回重新建模、或带建议重新建模
def review_modeling_analysis(state: over_all_state) -> Command:
    logger.info('正在运行 review_modeling_analysis 节点')
    analysis = state.get("modeling_analysis") or "(暂无建模分析结果)"
    feedback = interrupt(
        f"建模手请审核以下建模分析结果：\n\n{analysis}\n\n"
        "审核意见（直接回车=通过；输入\"打回\"=不通过重新分析；输入其他内容=作为建议重新分析）："
    )
    feedback = (feedback or "").strip()
    if not feedback or feedback in ("通过", "同意", "ok", "OK", "好", "可以"):
        logger.info('建模手审核通过，进入下一环节')
        return Command(update={"review_feedback": "", "review_result": "passed", "retry_count": 0, "tool_rounds": 0}, goto="send_problem_index")
    logger.info(f"建模手打回，审核意见：{feedback}")
    return Command(update={"review_feedback": feedback, "review_result": "rework", "retry_count": 0, "tool_rounds": 0}, goto="modeling")


#四个工作节点的固定方法分工
METHODS = ["解析法", "数值模拟法", "数据驱动法", "启发式优化法"]

#建模阶段工具调用总轮数上限,防止模型反复调工具死循环
MAX_TOOL_ROUNDS = 6

#执行代码用的 Python 解释器:默认跟随当前运行环境(与 agent 同一解释器),可用环境变量 MATH_PYTHON_EXE 覆盖
PYTHON_EXE = os.environ.get("MATH_PYTHON_EXE") or sys.executable
CODE_DIR = WORKSPACE_ROOT / "code"
PHOTO_DIR = WORKSPACE_ROOT / "photo"

# 出图脚本执行:子进程重试次数 + LLM 自动修复轮数上限(均防死循环)
RUN_RETRY = 2
LLM_MAX_FIX = 2
RUN_TIMEOUT = 30

#从"问题N"键中提取编号用于排序
def _num_key(k: str) -> int:
    m = re.search(r"\d+", k)
    return int(m.group()) if m else 0

#此节点仅作为扇出前的入口(no-op),真正分发由条件边 dispatch_sends 完成
def send_problem_index(state: over_all_state) -> dict:
    logger.info('正在运行 send_problem_index 节点')
    return {}

#屏障节点:四个方法×全部小问都交卷后才放行到 run_solutions,避免汇总读到部分结果
def collect_branches(state: over_all_state) -> dict:
    problem_index = state.get("problem_index") or {}
    qs = sorted(problem_index.keys(), key=_num_key)
    needed = {f"{m}|{k}" for m in METHODS for k in qs}
    got = set(state.get("done_pairs") or [])
    if needed and needed <= got:
        return Command(goto="run_solutions", update={})
    return {}

#条件边路径函数:为每个方法生成一个并行任务,每个任务携带全部问题+数据文件名清单(数据内容由代码自行读取)
def dispatch_sends(state: over_all_state) -> list:
    problem_index = state.get("problem_index") or {}
    if not problem_index:
        return [Send("compare_summarize", {})]
    dataset_dir = Path(dataset_path)
    dataset_files = "\n".join(sorted(f.name for f in dataset_dir.iterdir() if f.is_file())) if dataset_dir.is_dir() else "(dataset 目录为空)"
    return [
        Send("solve_with_method", {
            "method": m,
            "problem_index": problem_index,
            "modeling_analysis": state.get("modeling_analysis") or "",
            "dataset_files": dataset_files,
        })
        for m in METHODS
    ]

#工作节点:单次调用内依次解决所有小问。method 来自 Send 输入全程有效,
#工具循环也在节点内就地完成,避免分支身份经图节点往返丢失(旧实现曾导致 method 变为"未知方法")
def solve_with_method(state: over_all_state) -> dict:
    method = state.get("method") or "未知方法"
    logger.info(f'正在运行 solve_with_method 节点: {method}')
    problem_index = state.get("problem_index") or {}
    results_parts = []
    code_files = []
    failed = []

    for q_key in sorted(problem_index.keys(), key=_num_key):
        q_text = problem_index[q_key]
        prompt = (
            f"你正在用《{method}》解答数学建模问题。\n"
            f"当前小问: {q_key}\n题干:\n{q_text}\n"
            f"整体建模分析:\n{state.get('modeling_analysis') or '(无)'}\n"
            f"数据集文件清单(dataset 目录):\n{state.get('dataset_files') or '(无)'}\n\n"
            "数据说明: 上方清单中的文件位于本机 dataset/ 目录,"
            "代码中应通过 r\"..\\dataset\\文件名\" 读取真实数据,禁止编造数据。\n"
            "如需查资料或查看工作区已有文件,可调用 search / read_workspace 工具(work_dir 参数必填,只能是 code/paper/photo/dataset 之一,例如 read_workspace(work_dir=\"code\", rel_path=\"q1.py\"));\n"
            "不要调用写入类工具,思路与代码由系统自动保存。\n"
            "请给出该问在本方法下的完整求解,输出两部分:\n"
            "1) 思路与公式: Markdown,含关键建模公式(LaTeX 语法 $...$),不写代码;撰写中文表述时,请先调用 get_writing_skill 获取去 AI 味写作规范并严格遵循;\n"
            "2) 求解代码: 用 ```python 代码块包裹的完整可运行 Python 代码。代码规则:\n"
            f"   - 如需画图,必须用 plt.savefig(r\"{method}_{q_key}.png\") 保存到当前目录,文件名以方法名开头避免冲突;\n"
            "   - 画图前必须设置 matplotlib 中文字体: plt.rcParams['font.sans-serif']=['SimHei']; plt.rcParams['axes.unicode_minus']=False。"
        )
        msgs = [HumanMessage(prompt)]
        #节点内工具循环:模型要调工具就在本地执行并继续,最多 MAX_TOOL_ROUNDS 轮
        for _ in range(MAX_TOOL_ROUNDS + 1):
            resp = model_with_tool.invoke(msgs)
            if not getattr(resp, "tool_calls", None):
                break
            tool_msgs = []
            for call in resp.tool_calls:
                tool = TOOLS_BY_NAME.get(call["name"])
                try:
                    result = tool.invoke(call["args"]) if tool else f"未知工具: {call['name']}"
                except Exception as e:
                    result = f"工具执行失败: {e}"
                tool_msgs.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            msgs = msgs + [resp] + tool_msgs
        else:
            #工具轮数超限:改用无工具绑定的模型强制输出最终答案
            resp = model.invoke([*msgs, HumanMessage(
                "工具调用次数已达上限。请不要再调用任何工具，直接给出当前小问的完整求解：思路与公式 + ```python 代码块。"
            )])
            if getattr(resp, "tool_calls", None):
                results_parts.append(f"{q_key}: 求解失败(工具调用超限)")
                failed.append(q_key)
                continue

        #解析最终答案并落盘
        try:
            content = resp.content
            m = re.search(r"```(?:[Pp]ython|[Pp]y3?)?\s*\n?(.*?)```", content, re.S)
            if not m:
                raise ValueError("模型未生成可解析的 Python 代码块")
            code = m.group(1).strip()
            idea = re.sub(r"```(?:[Pp]ython|[Pp]y3?)?\s*\n?.*?```", "", content, flags=re.S).strip()

            write_workspace.invoke({"work_dir": "paper", "rel_path": f"{method}_{q_key}_思路.md", "content": idea})
            code_rel = f"{method}_{q_key}_solution.py"
            write_workspace.invoke({"work_dir": "code", "rel_path": code_rel, "content": code})
            code_files.append(code_rel)
            results_parts.append(f"{q_key}: 完成(思路→paper/{method}_{q_key}_思路.md, 代码→code/{code_rel})")
        except Exception as e:
            logger.error(f"[{method}] {q_key} 求解失败: {e}")
            results_parts.append(f"{q_key}: 求解失败({e})")
            failed.append(q_key)

    update = {"answers": [{method: "\n".join(results_parts)}], "code_files": code_files, "failed_qs": failed}
    update["done_pairs"] = [f"{method}|{k}" for k in sorted(problem_index.keys(), key=_num_key)]
    return update

#把 code/ 下所有 png 移入 photo/(同名冲突自动加 _1/_2 后缀),保证 code/ 不残留图片
def _sweep_pngs_to_photo() -> list:
    moved = []
    for img in list(CODE_DIR.rglob("*.png")):
        dest = PHOTO_DIR / img.name
        i = 1
        while dest.exists():
            dest = PHOTO_DIR / f"{img.stem}_{i}{img.suffix}"
            i += 1
        shutil.move(str(img), str(dest))
        moved.append(dest.name)
    return moved

# ---------- 出图脚本执行与自愈 ----------
# 从脚本文本提取 savefig 指定的输出文件名,用于重入时判断图片是否已生成
def _extract_savefig_paths(text: str) -> List[str]:
    return [Path(m.group(1)).name for m in re.finditer(r"savefig\s*\(\s*[rR]?['\"]([^'\"]+)['\"]", text)]


# 执行单个出图脚本一次,返回结构化结果(不移动图片)
def _run_one_script(script: Path, timeout: int) -> dict:
    before = set(CODE_DIR.rglob("*.png"))
    try:
        proc = subprocess.run(
            [PYTHON_EXE, script.name],
            cwd=str(CODE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "timed_out": False,
            "new_imgs": list(set(CODE_DIR.rglob("*.png")) - before),
            "exc": None,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": "", "stderr": "", "timed_out": True, "new_imgs": [], "exc": None}
    except Exception as e:
        return {"returncode": None, "stdout": "", "stderr": "", "timed_out": False, "new_imgs": [], "exc": e}


# 把新生成的 png 从 code/ 移入 photo/(同名冲突自动加后缀),返回移动后的文件名
def _move_imgs_to_photo(imgs) -> List[str]:
    moved = []
    for img in imgs:
        dest = PHOTO_DIR / img.name
        i = 1
        while dest.exists():
            dest = PHOTO_DIR / f"{img.stem}_{i}{img.suffix}"
            i += 1
        shutil.move(str(img), str(dest))
        moved.append(dest.name)
    return moved


# 把单次执行结果转成人类可读的错误摘要,成功时返回空串
def _run_error_msg(r: dict, timeout: int) -> str:
    if r["timed_out"]:
        return f"运行超时({timeout}秒),已终止"
    if r["exc"] is not None:
        return f"运行出错: {r['exc']}"
    if r["returncode"] != 0:
        return f"运行失败(返回码 {r['returncode']})"
    return ""


# 提交 LLM 诊断并重写脚本:成功写回磁盘返回 True,否则 False
def _llm_fix_script(script: Path, error_log: List[str], expected: List[str]) -> bool:
    try:
        code = script.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"读取待修复脚本失败: {e}")
        return False
    log_text = "\n".join(error_log)[-2500:]
    exp_hint = f"；期望生成的图片文件名: {expected}" if expected else ""
    prompt = (
        "你是一名 Python 绘图排障专家。下面是一段用于数学建模出图的 Python 脚本，运行时失败了。\n"
        "请先诊断失败原因（重点排查：中文字体缺失/乱码、模块未安装、数据读取路径错误、"
        "数据集文件不存在、数组维度/除零、死循环或超时等），然后直接返回修正后的【完整可运行】代码。\n\n"
        f"错误日志（含返回码/stdout/stderr/是否超时）：\n{log_text}\n\n"
        f"原始脚本：\n{code}\n\n"
        "要求：\n"
        "1. 必须用 ```python 代码块包裹完整代码，不要只给片段；\n"
        "2. 必须保留原有的 plt.savefig(...) 调用，且文件名保持不变" + exp_hint + "，否则图片无法被收集；\n"
        "3. 不要改变数据读取路径（数据在 dataset/ 下，使用相对路径 ..\\dataset\\文件名）；\n"
        "4. 不要引入当前环境未安装的额外依赖；\n"
        "5. 中文字体问题可降级为 plt.rcParams['font.sans-serif']=['Microsoft YaHei'] 或改用英文标签。\n"
        "只输出代码块，不要多余解释。"
    )
    try:
        resp = model.invoke([HumanMessage(prompt)])
    except Exception as e:
        logger.error(f"LLM 修复调用失败: {e}")
        return False
    m = re.search(r"```(?:[Pp]ython|[Pp]y3?)?\s*\n?(.*?)```", resp.content or "", re.S)
    if not m:
        return False
    new_code = m.group(1).strip()
    if not new_code:
        return False
    try:
        script.write_text(new_code, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"写回修复脚本失败: {e}")
        return False


# 执行节点:只运行"负责生成图片"的脚本(静态检测 savefig/matplotlib);
# 单个脚本失败自动重试 RUN_RETRY 次,仍失败则提交 LLM 诊断并改写代码(最多 LLM_MAX_FIX 轮),
# 全部失败才中断交由人工处理。重入时跳过已生成图片的脚本,避免重复执行。
def run_solutions(state: over_all_state) -> dict:
    logger.info('正在运行 run_solutions 节点')
    code_files = state.get("code_files") or []
    reports = []
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)

    # 先清理上次运行/中断遗留的 png,避免"非本次新增"的图片永远移不进 photo/
    leftover = _sweep_pngs_to_photo()
    if leftover:
        reports.append({"file": "(清理遗留)", "status": f"移入上次遗留图片: {leftover}"})

    to_run = []
    for rel in code_files:
        script = CODE_DIR / rel
        if not script.is_file():
            reports.append({"file": rel, "status": "文件不存在,跳过"})
            continue
        text = script.read_text(encoding="utf-8", errors="ignore")
        if "savefig" in text or "matplotlib" in text:
            to_run.append((rel, script))
        else:
            reports.append({"file": rel, "status": "未运行(非出图脚本)"})

    failed = []  # (rel, script, error_log)

    for rel, script in to_run:
        expected = _extract_savefig_paths(script.read_text(encoding="utf-8", errors="ignore"))
        # 重入跳过:期望图片已存在说明上一轮已成功
        if expected and all((PHOTO_DIR / n).exists() for n in expected):
            reports.append({"file": rel, "status": f"成功(已有图片,跳过): {expected}"})
            continue

        error_log: List[str] = []
        success = False
        moved: List[str] = []

        # ① 子进程重试:1 次 + RUN_RETRY 次(重试时超时翻倍)
        for attempt in range(1 + RUN_RETRY):
            timeout = RUN_TIMEOUT if attempt == 0 else RUN_TIMEOUT * 2
            r = _run_one_script(script, timeout)
            msg = _run_error_msg(r, timeout)
            if not msg:
                moved = _move_imgs_to_photo(r["new_imgs"])
                if moved:
                    success = True
                    reports.append({"file": rel, "status": f"成功,生成图片: {moved}（第{attempt + 1}次执行）"})
                    break
                msg = "运行成功但未生成任何图片(savefig 未生效?)"
            error_log.append(
                f"[第{attempt + 1}次执行] {msg}"
                + (f" | stdout: {r['stdout'][:400]}" if r["stdout"] else "")
                + (f" | stderr: {r['stderr'][:400]}" if r["stderr"] else "")
            )

        # ② LLM 审查 + 改写代码(最多 LLM_MAX_FIX 轮,每轮重跑一次)
        fix_i = 0
        while not success and fix_i < LLM_MAX_FIX:
            fix_i += 1
            logger.warning(f"[{rel}] 子进程重试失败,第 {fix_i} 次提交 LLM 诊断修复")
            if not _llm_fix_script(script, error_log, expected):
                error_log.append(f"[LLM 修复第{fix_i}轮] 未能解析出可运行代码块,放弃")
                break
            for attempt in range(1 + 1):
                timeout = RUN_TIMEOUT if attempt == 0 else RUN_TIMEOUT * 2
                r = _run_one_script(script, timeout)
                msg = _run_error_msg(r, timeout)
                if not msg:
                    moved = _move_imgs_to_photo(r["new_imgs"])
                    if moved:
                        success = True
                        reports.append({"file": rel, "status": f"成功,生成图片: {moved}（LLM 修复第{fix_i}轮后）"})
                        break
                    msg = "运行成功但未生成图片"
                error_log.append(
                    f"[LLM 修复第{fix_i}轮·第{attempt + 1}次执行] {msg}"
                    + (f" | stderr: {r['stderr'][:400]}" if r["stderr"] else "")
                )
            if success:
                break

        if not success:
            failed.append((rel, script, error_log))
            reports.append({"file": rel, "status": f"失败:经 {RUN_RETRY} 次重试 + {LLM_MAX_FIX} 轮 LLM 修复仍未能出图"})

    # 收尾清扫:即使脚本失败/超时,已 savefig 落盘的 png 也一并移入 photo/
    remaining = _sweep_pngs_to_photo()
    if remaining:
        reports.append({"file": "(收尾清扫)", "status": f"移入图片: {remaining}"})

    # ③ 终极兜底:仍有失败 → 中断交人工,可多轮介入直到成功或放弃。
    # 原理:interrupt 的 resume 会让本节点从头重跑——上方的自动恢复流程(含人工刚改的脚本)会再执行一遍;
    # 若重跑后仍失败,下方 while 会再次 interrupt 询问,人工可继续修改再试,直到成功或输入 skip。
    while failed:
        detail = "\n\n".join(f"脚本: {rel}\n" + "\n".join(log) for rel, _, log in failed)
        feedback = interrupt(
            f"以下出图脚本经 {RUN_RETRY} 次重试 + {LLM_MAX_FIX} 轮 LLM 修复仍失败:\n\n"
            f"{detail}\n\n"
            "请处理:可直接修改 code/ 下对应脚本后回车重试(会带着你的修改重跑,仍失败会再次询问);"
            "或输入 skip 跳过这些失败项继续。"
        )
        fb = (feedback or "").strip()
        if "skip" in fb.lower():
            for rel, _, _ in failed:
                reports.append({"file": rel, "status": "已由人工选择跳过"})
            break
        # 非 skip = 人工要求重试:本轮 resume 的重跑已在上方执行过;若仍失败则回到循环顶部再次询问
        logger.info(f"人工未跳过(反馈: {fb[:50] or '(回车)'}),仍有 {len(failed)} 个失败脚本,继续人工介入")

    return {"run_report": reports}

#路由:有工具调用→执行工具;工具刚返回→回本节点继续对话;否则→结束
def compare_route(state: over_all_state) -> str:
    msgs = state.get("compare_msgs") or []
    last = msgs[-1] if msgs else None
    if getattr(last, "tool_calls", None):
        return "tools"
    return "continue_next" if isinstance(last, ToolMessage) else "end"

#汇总节点:收集四个方法的答卷(可调工具查看工作区),对比分歧并给出每问最终结论,落盘最终总结
def compare_summarize(state: over_all_state) -> dict:
    logger.info('正在运行 compare_summarize 节点')
    merged = {}
    for item in (state.get("answers") or []):
        for k, v in item.items():
            merged.setdefault(k, []).append(v)
    problem_index = state.get("problem_index") or {}
    run_text = "\n".join(
        f"{r.get('file')}: {r.get('status', '')}"
        + (f" | stdout: {r['output']}" if r.get("output") else "")
        + (f" | stderr: {r['error']}" if r.get("error") else "")
        for r in (state.get("run_report") or [])
    )
    msgs = state.get("compare_msgs") or []
    #工具刚返回:延续对话,让模型看到工具结果后继续
    if msgs and isinstance(msgs[-1], ToolMessage):
        resp = model_with_tool.invoke(msgs)
        to_append = []
    else:
        if not merged:
            final = "(四个方法均未产出结果,无法对比总结)"
            write_workspace.invoke({"work_dir": "paper", "rel_path": "最终总结.md", "content": final})
            return {"final_summary": final}
        input_text = "\n\n".join(f"### 方法《{k}》\n" + "\n".join(vs) for k, vs in merged.items())
        fb = (state.get("human_feedback") or "").strip()
        feedback_line = f"人工审核意见(务必据此修改最终总结): {fb}\n\n" if fb else ""
        prompt = (
            "你是数学建模竞赛的总评专家。以下是用四种不同方法解答同一份题目的答卷,以及代码实际运行情况。\n"
            f"题目小问: {list(problem_index.keys())}\n\n"
            f"{input_text}\n\n"
            f"代码运行情况:\n{run_text or '(无运行报告)'}\n\n"
            f"{feedback_line}"
            "如需查看生成的代码或图片,可调用 read_workspace 工具(work_dir 参数必填,只能是 code/paper/photo/dataset 之一)。\n"
            "请输出一份对比总结 Markdown 文档:\n"
            "1. 逐问对比四种方法的结论,标出结论一致与分歧之处;\n"
            "2. 结合代码运行情况(成功/失败/生成的图),说明结果可信度;\n"
            "3. 对每一问给出最可信的最终结论(可综合多种方法),并简要说明理由;\n"
            "4. 整篇文档结构完整,可直接作为《最终总结》;\n"
            "5. 对每一问明确标注:最终采用的方法(从四种中选定)+一句话理由+该问关键数值结果清单"
            "(供论文撰写章节直接引用,数值必须与前述结论一致)。\n\n"
            f"【中文写作规范(去 AI 味,务必严格遵循)】\n{read_writing_skill('de-AI')}"
        )
        resp = model_with_tool.invoke([HumanMessage(prompt)])
        to_append = [HumanMessage(prompt)]

    #模型想调工具:交给工具节点,工具结果会自动回到本节点继续
    if getattr(resp, "tool_calls", None):
        return Command(goto="compare_tool_node", update={"compare_msgs": to_append + [resp]})

    final = resp.content
    write_workspace.invoke({"work_dir": "paper", "rel_path": "最终总结.md", "content": final})
    return {"final_summary": final, "compare_msgs": [resp], "human_feedback": ""}

#此函数用来人工审核最终结果
def final_analysis(state:over_all_state) ->Command:
    final_summary=state['final_summary']
    feedback=interrupt(
        f'请人工最终审查\n'
        f'认可:请输入回车\n'
        f'不认可请输入:打回或提出自己的意见\n'
        f'模型最终结果是:{final_summary}'
    )
    feedback=(feedback or '').strip()
    if not feedback or any(k in feedback for k in ('通过','认可')):
        return Command(goto="write_article")
    return Command(goto='compare_summarize', update={
        'human_feedback': feedback,
        'compare_msgs': [RemoveMessage(id=m.id) for m in (state.get("compare_msgs") or []) if getattr(m, "id", None)],
    })
    

# ============ 论文撰写节点：把最终结论/思路/图片填入 LaTeX 国赛模板的独立副本 ============
# 输出目录：paper/latex/（独立副本，不污染原始 writter_struct 模板）
LATEX_OUT = WORKSPACE_ROOT / "paper" / "latex"
WRITER_TEMPLATE = WORKSPACE_ROOT / "writter_struct"
# 模板里只复制一次、之后不再覆盖的静态文件（类文件/编译脚本/参考文献库等）
LATEX_STATIC = ["cumcmthesis.cls", "book.bib", "build.bat", "clean.bat",
                ".gitignore", "LICENSE", "README.md", "常用LaTex代码指令.txt", "document.tex"]

# 各章节：(文件名, 写作要求)，文件名对应 writter_struct/texfile/<name>.tex
ARTICLE_CHAPTERS = [
    ("1abstract",
     "摘要：300~600字，按 Nature 式'背景→问题→方法→结果→意义'证据链展开。首段简述背景与动机；"
     "随后按'针对问题一/问题二/...'逐问成段，每段必须写明该问最终采用的方法（从解析法/数值模拟法/数据驱动法/"
     "启发式优化法中选定的最适一种）、所用模型与关键定量结果（写具体数字，亮点突出）；末段写优化推广。"
     "结尾用 \\keywords{关键词1\\quad 关键词2...}（3~5个，含模型名与方法名亮点）。只接受文字，不要图表。"),
    ("2ProblemRestatement",
     "问题重述：\\section{问题重述}，含 \\subsection{问题背景}（提炼原题，保持原意）与 "
     "\\subsection{问题提出}（用 enumerate 列出各小问，基本原样复制题目所问）。篇幅不超一页。"),
    ("3ProblemAnalysis",
     "问题分析：\\section{问题分析}，对各小问分别 \\subsection{问题X分析} 写定性、数据洞察、建模思路，"
     "对应 problem_index 的每个小问；并在每问分析末给出'本问拟采用的最适方法及一句话理由'"
     "（从解析法/数值模拟法/数据驱动法/启发式优化法中选定，依据最终总结中的对比），为后文建模章节定主线。"),
    ("4AssumptionAndSign",
     "模型假设与符号说明：\\section{模型的假设} 用 enumerate 列 5 条左右合理假设；"
     "\\section{符号说明} 用 booktabs 三线表（符号|说明）。"),
    ("5MakeModel",
     "模型的建立与求解：\\section{模型的建立与求解}。先写数据预处理，再逐问 "
     "\\subsection{问题X的模型建立与求解}：先一句话声明'本问最终采用的方法及其理由'"
     "（以最终总结中选定的最适方法为主线，其余方法至多一句话对比），再按 模型建立/求解/结果 三子节展开；"
     "结果子节必须给出与最终总结一致的具体数值。可引用'可用图片'中与本问内容匹配的图"
     "\\includegraphics{图片名.png}（graphicspath 已指向 texfile/figures/，直接写裸文件名即可，"
     "不要加目录前缀）并配 \\caption/\\label（每问建议1~2张，图注要说明图中反映了什么结果）；"
     "公式用 $...$ 或 equation 环境，关键公式必须完整可推导；所有数学命令（\\mathrm 等）只能出现在公式内，禁止在正文文本中使用。"),
    ("6ErrorAnalysis",
     "误差分析：\\section{误差分析}，逐问 \\subsection{针对问题X的误差分析} 说明结果检验与误差来源。"),
    ("7ModelEvaluation",
     "模型评价：\\section{模型的评价}，含 优点/缺点/推广 三 subsection，用 itemize 罗列。"),
    ("8Reference",
     "参考文献：用 thebibliography 环境（\\bibitem{ref01}...），按 GB/T 7714 风格列出正文中实际引用文献；"
     "若用到了 AI 工具，按模板注释格式补充 AI 使用声明条目。"),
    ("9Appendix",
     "附录：\\appendix。\\section{详细图表} 索引；\\section{代码程序} 用 "
     "\\lstinputlisting[style=Python, caption={...}]{code/文件名.py} 列出'求解代码文件'中每个 py；"
     "\\section{支撑材料} 枚举。"),
]


def _ensure_latex_skeleton():
    """首次把 writter_struct 模板静态文件复制到 paper/latex 独立副本；已存在则不覆盖。"""
    LATEX_OUT.mkdir(parents=True, exist_ok=True)
    for name in LATEX_STATIC:
        src = WRITER_TEMPLATE / name
        dst = LATEX_OUT / name
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
    (LATEX_OUT / "texfile").mkdir(parents=True, exist_ok=True)
    (LATEX_OUT / "texfile" / "figures").mkdir(parents=True, exist_ok=True)
    (LATEX_OUT / "code").mkdir(parents=True, exist_ok=True)
    # 复制模板自带的 code 示例，避免附录 \lstinputlisting 缺文件
    tcode = WRITER_TEMPLATE / "code"
    if tcode.is_dir():
        for f in tcode.iterdir():
            if f.is_file() and not (LATEX_OUT / "code" / f.name).exists():
                shutil.copy(f, LATEX_OUT / "code" / f.name)


def _strip_tex_fences(text: str) -> str:
    """去掉 LLM 常包裹的 ```latex ... ``` 代码围栏，避免原样写进 .tex 导致编译失败。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[^\n]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def _clean_latex_output():
    """清空上一轮生成的章节/图片/代码，避免换题或重跑时残留旧产物。"""
    for f in (LATEX_OUT / "texfile").glob("*.tex"):
        f.unlink()
    for d in ((LATEX_OUT / "texfile" / "figures"), (LATEX_OUT / "code")):
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()


# ---------- LaTeX 编译验证与自愈：生成后真实编译，报错喂回 LLM 修复，循环到零错误 ----------
LATEX_MAX_FIX = 3    # 编译报错后 LLM 自动修复的最大轮数
LATEX_TIMEOUT = 180  # 单次 xelatex 编译超时(秒)


def _find_xelatex():
    """定位 xelatex：环境变量 MATH_XELATEX > PATH > C:/texlive/*/bin/windows/。找不到返回 None(跳过编译验证)。"""
    cand = os.environ.get("MATH_XELATEX") or shutil.which("xelatex")
    if cand:
        return cand
    for p in Path("C:/texlive").glob("*/bin/windows/xelatex.exe"):
        return str(p)
    return None


def _run_xelatex(xe: str) -> str:
    """在 LATEX_OUT 跑一遍 xelatex，返回 document.log 文本；先清理残留的 synctex 锁文件。"""
    for busy in LATEX_OUT.glob("*.synctex(busy)"):
        try:
            busy.unlink()
        except OSError:
            pass
    try:
        subprocess.run(
            [xe, "-interaction=nonstopmode", "document.tex"],
            cwd=str(LATEX_OUT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=LATEX_TIMEOUT,
        )
    except Exception as e:
        return f"! xelatex 运行失败: {e}"
    log_path = LATEX_OUT / "document.log"
    return log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""


def _extract_latex_errors(log: str) -> List[Dict[str, str]]:
    """从 xelatex 日志提取 '!' 开头的错误块，并按日志中最近打开的 .tex 文件归属错误位置。"""
    lines = log.splitlines()
    opens = [(i, m.group(1)) for i, ln in enumerate(lines)
             for m in re.finditer(r"[ (]\./(\S+\.tex)", ln)]
    errors = []
    for i, ln in enumerate(lines):
        if not ln.startswith("!"):
            continue
        fname = ""
        for oi, f in opens:
            if oi <= i:
                fname = f
            else:
                break
        ctx = " | ".join(x.strip() for x in lines[i:i + 8] if x.strip())
        errors.append({"file": fname, "msg": ctx[:600]})
    return errors


def _fix_latex_with_llm(errors: List[Dict[str, str]]) -> List[str]:
    """把编译错误与相关章节文件内容交给 LLM 修复并写回，返回修复的文件名列表。"""
    files = sorted({e["file"] for e in errors if e["file"].startswith("texfile/")})
    if not files:
        logger.warning(f"编译错误未能定位到章节文件: {[e['file'] for e in errors]}")
        return []
    parts = ["以下 CUMCM 论文的 LaTeX 章节编译报错，请修复。"]
    for e in errors:
        parts.append(f"[错误·{e['file']}] {e['msg']}")
    for f in files:
        path = LATEX_OUT / f
        if path.exists():
            parts.append(f"===== 文件 {f} 完整内容 =====\n" + path.read_text(encoding="utf-8", errors="replace"))
    parts.append(
        "修复要求：只修导致编译错误的语法问题（数学环境缺 $、数学命令用在文本模式、括号/环境不配对、"
        "\\includegraphics 文件名错误、非法字符等），保持文字内容不变、不改写论述。\n"
        "输出格式（可包含多个文件，除此之外不要任何解释文字）：\n"
        "===FILE: texfile/文件名.tex===\n修复后的完整文件内容\n===END==="
    )
    try:
        resp = model.invoke([HumanMessage("\n\n".join(parts))])
    except Exception as e:
        logger.error(f"LLM 修复编译错误调用失败: {e}")
        return []
    fixed = []
    for m in re.finditer(r"===FILE:\s*(\S+)\s*===\n(.*?)\n?===END===", resp.content or "", re.S):
        fname, content = m.group(1).strip(), m.group(2).strip()
        if fname.startswith("texfile/") and content:
            content = _strip_tex_fences(content)
            (LATEX_OUT / fname).write_text(content + "\n", encoding="utf-8")
            fixed.append(fname)
    return fixed


def write_article(state: over_all_state) -> dict:
    logger.info("正在运行 write_article 节点：生成 LaTeX 论文到 paper/latex")
    _ensure_latex_skeleton()
    _clean_latex_output()

    final_summary = state.get("final_summary") or "(无最终总结)"
    approach = (state.get("modeling_approach") or state.get("modeling_analysis") or "(无思路)")
    problem_str = state.get("problem_str") or ""
    problem_index = state.get("problem_index") or {}
    code_files = state.get("code_files") or []

    # 1. 同步图片：photo/*.png -> paper/latex/texfile/figures/
    figs = []
    for img in PHOTO_DIR.glob("*.png"):
        shutil.copy(img, LATEX_OUT / "texfile" / "figures" / img.name)
        figs.append(img.name)
    # 2. 同步求解代码：code/*.py -> paper/latex/code/（供附录 \lstinputlisting 引用）
    for cf in code_files:
        src = CODE_DIR / cf
        if src.exists():
            shutil.copy(src, LATEX_OUT / "code" / cf)

    skill = read_writing_skill("de-AI")
    done = []
    for fname, req in ARTICLE_CHAPTERS:
        prompt = (
            "你是数学建模国赛论文撰写专家，请把以下内容写成符合 CUMCM LaTeX 模板的【单章节】LaTeX 源码。\n"
            "【写作总则·务必遵循】\n"
            "1. 叙事遵循 Nature 式'问题→方法→结果→讨论'证据链：每个论断尽量给出定量结果、公式或图引用作支撑，杜绝空话套话；\n"
            "2. 方法主线：各小问以【最终结论】中选定的最适方法为主线展开（只写一种主线方法，其余方法至多一句话对比），"
            "全文方法口径与最终总结保持一致；\n"
            "3. 图表引用：仅引用文件名与本章内容明显匹配的图，用 \\includegraphics{裸文件名.png} "
            "（graphicspath 已指向 texfile/figures/，禁止加目录前缀如 texfile/figures/ 或 figures/），"
            "并为每张引用的图配 \\caption（说明图反映的结果），宁缺毋滥，禁止为凑数引用无关图片；\n"
            "4. 数值一致性：文中所有关键数值必须与【最终结论】一致，不得自行编造或改写。\n"
            "5. 公式规范：所有数学命令（如 \\mathrm、\\sum、\\frac）必须写在 $...$ 或 equation 环境内，"
            "正文文本中严禁出现数学命令；公式前后括号配对完整，禁止缺 $。\n\n"
            f"【本章要求】{req}\n\n"
            f"【题目全文】{problem_str}\n\n"
            f"【各小问题干】{problem_index}\n\n"
            f"【建模思路】{approach}\n\n"
            f"【最终结论（逐问）】{final_summary}\n\n"
            f"【可用图片（位于 texfile/figures/）】{figs or '(无)'}\n"
            f"【求解代码文件（位于 code/，供附录引用）】{code_files or '(无)'}\n\n"
            f"【中文写作规范（去 AI 味，务必遵循）】\n{skill}\n\n"
            "只输出该章节的完整 LaTeX 正文（不要 \\documentclass、不要 \\begin{document}、不要解释性文字、不要代码围栏），"
            "直接可被 \\input 引用。"
        )
        target = LATEX_OUT / "texfile" / f"{fname}.tex"
        try:
            raw = model.invoke([HumanMessage(prompt)]).content
            latex = _strip_tex_fences(raw)
            if not latex:
                raise ValueError("模型返回为空")
            target.write_text(latex, encoding="utf-8")
            done.append(fname)
        except Exception as e:
            logger.error(f"生成章节 {fname} 失败: {e}")
            # 失败也写占位，保证 \\input 不报缺文件
            target.write_text(f"% 本章（{fname}）生成失败：{e}\n", encoding="utf-8")
            done.append(f"{fname}(失败:{e})")

    # 3. 编译验证与自愈：真实编译一遍，有错误就把日志喂回 LLM 修复，直到零错误或用完修复轮数
    xe = _find_xelatex()
    compile_status = "未检测到 xelatex(可用环境变量 MATH_XELATEX 指定路径)，已跳过编译验证"
    if xe:
        logger.info(f"检测到 xelatex({xe})，开始论文编译验证与自愈")
        log = _run_xelatex(xe)
        errs = _extract_latex_errors(log)
        ok = not errs and "Output written" in log
        for rnd in range(1, LATEX_MAX_FIX + 1):
            if ok:
                break
            logger.warning(f"[编译修复] 第 {rnd}/{LATEX_MAX_FIX} 轮：{len(errs)} 个错误，提交 LLM 修复")
            if not _fix_latex_with_llm(errs):
                break
            log = _run_xelatex(xe)
            errs = _extract_latex_errors(log)
            ok = not errs and "Output written" in log
        if ok:
            _run_xelatex(xe)  # 再跑一遍稳定交叉引用/目录，出最终 PDF
            compile_status = "编译通过(0 错误)，已生成 paper/latex/document.pdf"
        else:
            compile_status = (f"仍有 {len(errs)} 个编译错误(已尝试 {LATEX_MAX_FIX} 轮 LLM 修复)，"
                              f"请人工查看 paper/latex/document.log")
        logger.info(f"论文编译验证结果：{compile_status}")

    return {"article_chapters": done,
            "compile_status": compile_status,
            "messages": [AIMessage(content="已生成 LaTeX 论文章节：" + "、".join(done) + f"；{compile_status}")]}


# 回填 document.tex 封面元信息：标题由 LLM 自动生成，题号/报名号/学校/年份由人工 interrupt 提供
def fill_document_meta(state: over_all_state) -> dict:
    logger.info("正在运行 fill_document_meta 节点：回填 document.tex 元信息")
    doc_path = LATEX_OUT / "document.tex"
    if not doc_path.exists():
        return {"messages": [AIMessage(content="document.tex 不存在，跳过元信息回填")]}

    # 1) 索取需人工确定的元信息（留空=保持模板默认/注释）
    meta = interrupt(
        "请填写论文封面元信息（每行一项，回车留空则保持模板默认）：\n"
        "第1行 题号(如 A/B/C/D)\n"
        "第2行 报名号\n"
        "第3行 学校名称\n"
        "第4行 年份(如 2025)\n"
        "直接回车=全部用模板默认"
    )
    lines = (meta or "").splitlines()
    tihao = lines[0].strip() if len(lines) > 0 else ""
    baoming = lines[1].strip() if len(lines) > 1 else ""
    school = lines[2].strip() if len(lines) > 2 else ""
    year = lines[3].strip() if len(lines) > 3 else ""

    text = doc_path.read_text(encoding="utf-8")

    # 2) 自动生成标题（基于题目）
    try:
        title_prompt = (
            "请用一句话(中文,不超过30字)概括以下数学建模题目的论文标题，"
            "要求准确、学术、体现核心方法或对象，不要带书名号。只输出标题本身。\n"
            f"题目：{state.get('problem_str') or ''}"
        )
        new_title = model.invoke([HumanMessage(title_prompt)]).content.strip().strip("《》").strip()
    except Exception as e:
        logger.warning(f"生成标题失败: {e}")
        new_title = ""
    if new_title:
        # 注意:re.sub 的替换字符串会解析 \t \b \s 等转义,直接传 f"\\title{...}" 会把反斜杠吞掉
        # (\t→制表符)甚至抛 re.error。必须用 lambda 返回字面值,避免转义处理。
        text = re.sub(r"\\title\{[^}]*\}", lambda _: f"\\title{{{new_title}}}", text)

    # 3) 回填人工字段（提供则取消注释并填值，保留行尾 %注释；%? 兼容已启用过的行，保证可重复更新）
    if tihao:
        text = re.sub(r"%?\s*\\tihao\{[^}]*\}", lambda _: f"\\tihao{{{tihao}}}", text, count=1)
    if baoming:
        text = re.sub(r"%?\s*\\baominghao\{[^}]*\}", lambda _: f"\\baominghao{{{baoming}}}", text, count=1)
    if school:
        text = re.sub(r"%?\s*\\schoolname\{[^}]*\}", lambda _: f"\\schoolname{{{school}}}", text, count=1)
    if year:
        text = re.sub(r"%?\s*\\yearinput\{[^}]*\}", lambda _: f"\\yearinput{{{year}}}", text, count=1)

    doc_path.write_text(text, encoding="utf-8")

    # 4) 元信息回填后重新编译，保证最终 PDF 包含封面字段(找不到 xelatex 则跳过)
    xe = _find_xelatex()
    final_build = ""
    if xe:
        log = _run_xelatex(xe)
        errs = _extract_latex_errors(log)
        if not errs:
            errs = _extract_latex_errors(_run_xelatex(xe))  # 第二遍稳定引用
        final_build = ("；最终编译通过，PDF 已更新" if not errs
                       else f"；最终编译仍有 {len(errs)} 个错误，请查看 document.log")

    return {"messages": [AIMessage(
        content=f"已回填元信息：题号={tihao or '默认'} 报名号={baoming or '默认'} "
                f"学校={school or '默认'} 年份={year or '默认'} 标题={new_title or '默认'}{final_build}")]}


#测试时建立检查点暂时用内存存储
checkpointer=InMemorySaver()

#配置线程id
config={
    'configurable':{
        'thread_id':'1111'
    }
}

#Studio/API 模式发起 run 时输入可能为空,单独声明空输入状态避免 EmptyInputError
class input_state(TypedDict):
    pass

builder=StateGraph(state_schema=over_all_state, input=input_state)

builder.add_node('load_problem', load_problem) 
builder.add_node('question_structed',question_structed)
builder.add_node('modeling',modeling)
builder.add_node('read_dataset',read_dataset)
builder.add_node('tool_node',tool_node)
builder.add_node('review_modeling_analysis',review_modeling_analysis)
builder.add_node('send_problem_index',send_problem_index)
builder.add_node('collect_branches',collect_branches)
builder.add_node('solve_with_method',solve_with_method)
builder.add_node('run_solutions',run_solutions)
builder.add_node('compare_summarize',compare_summarize)
builder.add_node('compare_tool_node',ToolNode(tools=tools, messages_key="compare_msgs"))
builder.add_node('final_analysis', final_analysis)
builder.add_node('write_article', write_article)
builder.add_node('fill_document_meta', fill_document_meta)
builder.add_edge("write_article", "fill_document_meta")
builder.add_edge("fill_document_meta", END)

builder.add_edge(START, 'load_problem')  
builder.add_edge('load_problem', 'question_structed') 
builder.add_edge('question_structed','read_dataset')
builder.add_edge('read_dataset',"modeling")
builder.add_conditional_edges(
    "modeling",
    modeling_route,
    {"tools": "tool_node", "review": "review_modeling_analysis"}
)
builder.add_edge("tool_node", "modeling")
builder.add_edge("review_modeling_analysis", "send_problem_index")
builder.add_conditional_edges(
    "send_problem_index",
    dispatch_sends,
    ["solve_with_method", "compare_summarize"]
)
builder.add_edge("solve_with_method", "collect_branches")
builder.add_conditional_edges(
    "compare_summarize",
    compare_route,
    {"tools": "compare_tool_node", "continue_next": "compare_summarize", "end": "final_analysis"}
)
builder.add_edge("compare_tool_node", "compare_summarize")
builder.add_edge("collect_branches", "run_solutions")
builder.add_edge("run_solutions", "compare_summarize")

# 在 LangGraph API/Server 中运行时，持久化由平台自动管理，不传自定义 checkpointer
if os.environ.get("LANGSMITH_LANGGRAPH_API_VARIANT"):
    graph=builder.compile()
else:
    graph=builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    #首次运行：执行到第一个 interrupt 处暂停
    graph.invoke({}, config=config)

    #循环处理各中断节点（modeling 思路输入、审核打回等），直到 END
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        interrupts = getattr(snapshot, "interrupts", None) or []
        prompt = interrupts[0].value if interrupts else "请输入："
        user_input = input(f"{prompt}\n> ")
        graph.invoke(Command(resume=user_input), config=config)

    resp = graph.get_state(config).values
    print(resp)





