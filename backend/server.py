# -*- coding: utf-8 -*-
"""
后端服务: 在不动 src/agent.py 任何逻辑的前提下, 把 LangGraph agent 包装为
FastAPI + SSE 流式接口, 供前端实时展示节点的运行过程与最终结果。

运行方式(项目根目录):
    pip install -r backend/requirements.txt
    python backend/server.py            # 默认 http://127.0.0.1:8000

接口:
    GET  /api/health        健康检查
    GET  /api/state         当前会话状态(最终总结等)
    POST /api/stream        启动一次运行 / 对 interrupt 恢复运行, SSE 流式返回
    POST /api/reset         清空指定会话(重新开始)
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict

# agent 使用 ./question ./dataset 等相对 CWD 的路径, 必须切到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import HTTPException
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel

from src.agent import graph, checkpointer, set_model, list_models
from src.models import MODEL_REGISTRY

DEFAULT_THREAD_ID = os.environ.get("MATH_THREAD_ID", "1111")
DEFAULT_PORT = int(os.environ.get("MATH_BACKEND_PORT", "8000"))

# ---------- 模型选择持久化（配置一次一直可用，类似 opencode 的配置文件） ----------
CONFIG_PATH = PROJECT_ROOT / "model_config.json"

def _load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"model": "deepseek", "keys": {}}

def _load_model_config() -> str:
    return _load_config().get("model", "deepseek")

def _save_config(model: str, keys: dict | None = None) -> None:
    cfg = _load_config()
    cfg["model"] = model
    if keys:
        cfg.setdefault("keys", {})
        cfg["keys"].update({k: v for k, v in keys.items() if v})
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存模型配置失败: {e}")

def _apply_keys(keys: dict | None) -> None:
    """把已保存的 key 写回环境变量，但仅当作兜底：
    若真实环境变量（系统 / .env，已由 src.agent 的 load_dotenv 载入）已存在，则保留真实值，
    避免配置文件中陈旧的/错误的 key 覆盖掉正确的环境变量。"""
    if not keys:
        return
    for k, v in keys.items():
        if v and not os.environ.get(k):
            os.environ[k] = v

# 进程启动：先写入已保存的 key，再切到已保存的模型作为默认
try:
    _cfg = _load_config()
    _apply_keys(_cfg.get("keys"))
    set_model(_cfg.get("model", "deepseek"))
except Exception as e:
    logger.warning(f"初始化模型失败（将使用代码默认）: {e}")

# 前端可读的工作区目录
ALLOWED_WORK_DIRS = {"code", "paper", "photo", "dataset"}

app = FastAPI(title="数学建模 Agent 后端", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 单进程内同一线程的多次运行必须串行, 防止状态交错
_run_lock = asyncio.Lock()


class ModelName(BaseModel):
    """设置默认模型（持久化到 model_config.json），可选附带 API Key"""
    model: str
    api_key: Optional[str] = None

class StreamBody(BaseModel):
    """resume 缺省表示启动新运行; 提供时表示恢复被 interrupt 挂起的运行；model 指定本次运行的模型，api_key 为非 deepseek 模型必填"""
    resume: Optional[str] = None
    thread_id: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None

class KeysBody(BaseModel):
    """批量保存 API Key（持久化到 model_config.json），可选同时设定默认模型；不会切换本次运行。"""
    model: Optional[str] = None
    keys: Dict[str, str] = {}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _msg_summary(msg: dict) -> dict:
    """把 langchain 消息压缩成前端能展示的小结构"""
    mtype = msg.get("type", "")
    if mtype == "ai":
        out = {"role": "ai", "content": str(msg.get("content") or "")[:400]}
        tcs = msg.get("tool_calls") or []
        if tcs:
            out["tool_calls"] = [
                {"name": tc.get("name"), "args": json.dumps(tc.get("args", {}), ensure_ascii=False)[:300]}
                for tc in tcs
            ]
        return out
    if mtype == "tool":
        return {"role": "tool", "name": msg.get("name"), "content": str(msg.get("content") or "")[:300]}
    return {"role": mtype, "content": str(msg.get("content") or "")[:300]}


def _simplify(payload) -> dict:
    """过滤 update 载荷里的巨型字段(完整 messages 等), 只留前端需要的摘要"""
    if not isinstance(payload, dict):
        return {"value": str(payload)[:500]}
    out = {}
    for k, v in payload.items():
        if k == "__interrupt__" or v is None:
            continue
        if k == "messages" and isinstance(v, list):
            out["messages"] = [_msg_summary(m) for m in v if isinstance(m, dict)]
        elif k in ("answers", "run_report", "code_files", "failed_qs", "done_pairs") or (
            isinstance(v, list) and all(isinstance(x, (str, dict)) for x in v) and len(v) <= 8
        ):
            out[k] = v
        elif isinstance(v, (str, int, float, bool)):
            out[k] = str(v)[:400] if isinstance(v, str) else v
    return out


def _extract_interrupts(payload) -> list:
    """从节点 update 中提取 interrupt 载荷"""
    if not isinstance(payload, dict):
        return []
    its = payload.get("__interrupt__") or []
    return [{"value": getattr(it, "value", str(it))} for it in its]


async def _stream_events(resume: Optional[str], thread_id: str, model: Optional[str] = None, api_key: Optional[str] = None):
    runtime_config = {"configurable": {"thread_id": thread_id}}
    try:
        async with _run_lock:
            # 应用本次运行指定的模型（前端下拉框选择）
            if model:
                # 若请求带了 key，先写入对应环境变量
                if api_key:
                    _env = MODEL_REGISTRY.get(model, {}).get("api_key_env")
                    if _env:
                        os.environ[_env] = api_key
                try:
                    set_model(model)
                except Exception as e:
                    logger.warning(f"切换模型失败，沿用当前模型: {e}")
            if resume is not None:
                inputs = Command(resume=resume)
            else:
                # 开始新运行时清空线程残留状态, 保证从头开始
                try:
                    checkpointer.delete_thread(thread_id)
                except Exception:
                    pass
                inputs = {}
            async for update in graph.astream(
                inputs,
                config=runtime_config,
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                for node, payload in update.items():
                    if node == "__interrupt__" and isinstance(payload, (list, tuple)):
                        its = [{"value": getattr(it, "value", str(it))} for it in payload]
                    else:
                        its = _extract_interrupts(payload)
                    if its:
                        for it in its:
                            yield _sse({"type": "interrupt", "node": node, "value": it["value"]})
                        yield _sse({"type": "suspended"})
                        return
                    yield _sse({"type": "update", "node": node, "data": _simplify(payload)})
        yield _sse({"type": "done", "thread_id": thread_id})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("运行异常")
        yield _sse({"type": "error", "error": str(e)})


@app.get("/api/models")
async def get_models():
    """返回所有可选模型（含所需 key 的变量名与提示），供前端下拉框读取"""
    return {"models": list_models()}

@app.get("/api/model")
async def get_model_config():
    """返回当前已保存的默认模型与已保存的 key"""
    _cfg = _load_config()
    return {"model": _cfg.get("model", "deepseek"), "keys": _cfg.get("keys", {})}

@app.post("/api/model")
async def set_model_config(body: ModelName):
    """保存默认模型（持久化），可选附带 API Key，并立即切换"""
    name = (body.model or "deepseek").strip().lower()
    if name not in MODEL_REGISTRY:
        raise HTTPException(400, f"未知模型: {name}（可用: {', '.join(MODEL_REGISTRY.keys())}）")
    keys = None
    if body.api_key:
        _env = MODEL_REGISTRY[name]["api_key_env"]
        os.environ[_env] = body.api_key
        keys = {_env: body.api_key}
    try:
        set_model(name)
    except Exception as e:
        raise HTTPException(400, f"切换模型失败: {e}")
    _save_config(name, keys)
    return {"model": name, "status": "saved"}

@app.post("/api/keys")
async def save_keys(body: KeysBody):
    """批量保存 API Key（持久化到 model_config.json），可选同时设定默认模型；不影响本次运行。
    已存在于系统环境 / .env 的 key 无需重复提交，只提交缺失项即可。"""
    cfg = _load_config()
    cfg.setdefault("keys", {})
    applied = {}
    for env_name, val in (body.keys or {}).items():
        if val and val.strip():
            os.environ[env_name] = val.strip()
            cfg["keys"][env_name] = val.strip()
            applied[env_name] = val.strip()
    if body.model:
        name = body.model.strip().lower()
        if name in MODEL_REGISTRY:
            try:
                set_model(name)
                cfg["model"] = name
            except Exception as e:
                logger.warning(f"切换默认模型失败（沿用当前）: {e}")
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存配置失败: {e}")
    present = {env_name: bool(os.getenv(env_name)) for env_name in set(cfg["keys"].keys()) | {m["api_key_env"] for m in MODEL_REGISTRY.values()}}
    return {"model": cfg.get("model", "deepseek"), "keys": cfg.get("keys", {}), "present": present}

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.get("/api/state")
async def get_state(thread_id: str = DEFAULT_THREAD_ID):
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(cfg)
    return {
        "thread_id": thread_id,
        "next": snap.next,
        "values": _simplify(snap.values),
    }


@app.post("/api/stream")
async def stream_run(body: StreamBody):
    thread_id = body.thread_id or DEFAULT_THREAD_ID
    # 非 deepseek 必须有 API Key（本次请求携带，或已保存）
    if body.model and body.model != "deepseek":
        _env = MODEL_REGISTRY.get(body.model, {}).get("api_key_env")
        _has = bool(body.api_key) or bool(_env and os.environ.get(_env))
        if not _has:
            raise HTTPException(400, f"使用 {body.model} 需要 API Key，请在下拉框填入后保存/运行")
    logger.info(f"线程 {thread_id} 收到运行请求, resume={body.resume is not None}")
    return StreamingResponse(
        _stream_events(body.resume, thread_id, body.model, body.api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_work_path(work_dir: str, rel_path: str = "") -> Path:
    """校验并解析工作区路径, 非法路径抛 400"""
    if work_dir not in ALLOWED_WORK_DIRS:
        raise HTTPException(400, f"work_dir 只能是 {'/'.join(sorted(ALLOWED_WORK_DIRS))}, 收到: {work_dir!r}")
    p = Path(rel_path) if rel_path else Path(".")
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(400, f"rel_path 必须是相对路径且不能包含 '..': {rel_path}")
    target = (PROJECT_ROOT / work_dir / p).resolve()
    if not target.is_relative_to((PROJECT_ROOT / work_dir).resolve()):
        raise HTTPException(400, f"路径超出 {work_dir} 目录: {rel_path}")
    return target


@app.get("/api/files")
async def list_files(work_dir: str):
    """列出工作区目录下的文件与文件夹(名称+大小), 供前端实时刷新"""
    if work_dir not in ALLOWED_WORK_DIRS:
        raise HTTPException(400, f"work_dir 只能是 {'/'.join(sorted(ALLOWED_WORK_DIRS))}")
    root = PROJECT_ROOT / work_dir
    if not root.is_dir():
        return {"work_dir": work_dir, "files": []}
    items = []
    for item in sorted(root.iterdir()):
        if item.is_file():
            items.append({"name": item.name, "size": item.stat().st_size})
        elif item.is_dir():
            items.append({"name": item.name + "/", "size": 0})
    return {"work_dir": work_dir, "files": items}


@app.get("/api/file")
async def read_file(work_dir: str, rel_path: str = ""):
    """读取工作区文本文件(思路/代码/dataset), 返回文本内容"""
    target = _resolve_work_path(work_dir, rel_path)
    if not target.is_file():
        raise HTTPException(404, f"文件不存在: {target}")
    if target.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
        raise HTTPException(400, "图片请用 /api/image 获取")
    try:
        return {"work_dir": work_dir, "rel_path": rel_path, "content": target.read_text(encoding="utf-8")}
    except UnicodeDecodeError:
        raise HTTPException(400, f"{rel_path} 是二进制文件, 无法作为文本读取")


@app.get("/api/image")
async def get_image(work_dir: str, rel_path: str = ""):
    """读取工作区图片(photo 下的 png 等), 二进制返回"""
    target = _resolve_work_path(work_dir, rel_path)
    if not target.is_file():
        raise HTTPException(404, f"图片不存在: {target}")
    return FileResponse(str(target))


@app.post("/api/reset")
async def reset_thread(thread_id: str = DEFAULT_THREAD_ID):
    try:
        checkpointer.delete_thread(thread_id)
    except Exception:
        # 线程不存在时删除会抛错, 忽略即可
        pass
    return {"status": "reset", "thread_id": thread_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)