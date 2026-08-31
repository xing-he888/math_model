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
from dotenv import load_dotenv, set_key

# agent 使用 ./question ./dataset 等相对 CWD 的路径, 必须切到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi import HTTPException
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel
from datetime import datetime
import shutil
import re
import time

from src.agent import graph, checkpointer, set_model, list_models, usage_snapshot, reset_usage, reset_ai_log
from src.models import MODEL_REGISTRY
from src.tool import set_workspace, get_workspace, ws_root

DEFAULT_THREAD_ID = os.environ.get("MATH_THREAD_ID", "1111")
DEFAULT_PORT = int(os.environ.get("MATH_BACKEND_PORT", "8000"))

# ---------- 模型选择持久化（配置一次一直可用，类似 opencode 的配置文件） ----------
CONFIG_PATH = PROJECT_ROOT / "model_config.json"
ENV_PATH = PROJECT_ROOT / ".env"

# ---------- 前端外观设置持久化（液态玻璃主题 + 背景模式，dsh-wallpaper-engine 同思路） ----------
UI_CONFIG_PATH = PROJECT_ROOT / "ui_config.json"

UI_DEFAULTS = {
    "accent": "#4f8cff",       # 强调色（配色预设 + 自定义）
    "glassAlpha": 12,          # 玻璃透明度 0-60，越大越透
    "glassColor": "#ffffff",   # 玻璃基底色
    "textColor": "",           # 文字主色(""=跟随主题默认, 七位hex=全局覆盖 --text)
    "blur": 16,                # 玻璃模糊半径 px（0 = 关闭毛玻璃）
    "scrim": 0.25,             # 背景压暗 0-1
    "border": 0.35,            # 边框强调 0-1
    "wallpaperBlur": 0,        # 背景自身模糊 px
    "objectFit": "cover",      # cover / contain / center / fill
    "flip": False,             # 水平翻转
    "bgMode": "image",         # image / video / carousel
    "bgFile": "bg-miku.jpg",   # 当前背景素材（photo 或 media 目录下的文件名）
    "carouselSecs": 30,        # 轮播间隔秒
    "playbackRate": 1,         # 视频倍速 0.5-2
    "glassWindow": True,       # 弹窗/面板玻璃总开关
    # —— dsh 同款（第二批补齐）——
    "pauseOnHidden": True,     # 页面隐藏（最小化/切标签）时暂停视频
    "pauseOnBlur": False,      # 窗口失焦时暂停视频
    "pauseOnBattery": False,   # 电池供电时暂停视频
    "typeFilter": "all",       # 素材列表过滤：all / image / video
    "hiddenIds": [],           # 隐藏（软删除）的素材文件名
    "shuffle": False,          # 轮播随机顺序
    "mediaDir": "media",       # 背景素材目录（相对项目根；改存储位置=改这个）
    # —— WE 壁纸库（dsh 同款：读 Steam workshop）——
    "bgSource": "local",       # local（media 目录）/ we（Wallpaper Engine 壁纸库）
    "weId": "",                # 当前选中的 WE 壁纸 id
    "contentRatingFilter": "all",  # WE 壁纸分级过滤：all / everyone / pg13 / mature / unrated
}

def _load_ui_config() -> dict:
    cfg = {}
    try:
        if UI_CONFIG_PATH.exists():
            cfg = json.loads(UI_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    out = dict(UI_DEFAULTS)
    for k, v in cfg.items():
        if k not in UI_DEFAULTS:
            continue
        if k in ("accent", "glassColor") and isinstance(v, str) and len(v) == 7 and v.startswith("#"):
            out[k] = v
        elif k == "textColor" and isinstance(v, str) and (v == "" or (len(v) == 7 and v.startswith("#"))):
            out[k] = v
        elif k == "objectFit" and v in ("cover", "contain", "center", "fill"):
            out[k] = v
        elif k == "bgMode" and v in ("image", "video", "carousel"):
            out[k] = v
        elif k == "typeFilter" and v in ("all", "image", "video"):
            out[k] = v
        elif k == "bgSource" and v in ("local", "we"):
            out[k] = v
        elif k == "contentRatingFilter" and v in ("all", "everyone", "pg13", "mature", "unrated"):
            out[k] = v
        elif k == "weId" and isinstance(v, str) and v and "/" not in v and "\\" not in v:
            out[k] = v
        elif k == "bgFile" and isinstance(v, str) and v and "/" not in v and "\\" not in v:
            out[k] = v
        elif k == "mediaDir" and isinstance(v, str) and v and "/" not in v and "\\" not in v:
            out[k] = v
        elif k == "hiddenIds" and isinstance(v, list):
            out[k] = [x for x in v if isinstance(x, str) and x and "/" not in x and "\\" not in x]
        elif k in ("flip", "glassWindow", "pauseOnHidden", "pauseOnBlur", "pauseOnBattery", "shuffle"):
            out[k] = bool(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = v
    return out

def _save_ui_config(patch: dict) -> dict:
    """读-改-写合并保存外观配置；非法字段回退默认值。"""
    cfg = _load_ui_config()
    for k, v in patch.items():
        if k not in UI_DEFAULTS:
            continue
        if k in ("accent", "glassColor"):
            cfg[k] = v if isinstance(v, str) and len(v) == 7 and v.startswith("#") else UI_DEFAULTS[k]
        elif k == "textColor":
            cfg[k] = v if isinstance(v, str) and (v == "" or (len(v) == 7 and v.startswith("#"))) else UI_DEFAULTS[k]
        elif k == "objectFit":
            cfg[k] = v if v in ("cover", "contain", "center", "fill") else UI_DEFAULTS[k]
        elif k == "bgMode":
            cfg[k] = v if v in ("image", "video", "carousel") else UI_DEFAULTS[k]
        elif k == "typeFilter":
            cfg[k] = v if v in ("all", "image", "video") else UI_DEFAULTS[k]
        elif k == "bgSource":
            cfg[k] = v if v in ("local", "we") else UI_DEFAULTS[k]
        elif k == "contentRatingFilter":
            cfg[k] = v if v in ("all", "everyone", "pg13", "mature", "unrated") else UI_DEFAULTS[k]
        elif k == "weId":
            cfg[k] = v if isinstance(v, str) and v and "/" not in v and "\\" not in v else UI_DEFAULTS[k]
        elif k in ("bgFile", "mediaDir"):
            cfg[k] = v if isinstance(v, str) and v and "/" not in v and "\\" not in v else UI_DEFAULTS[k]
        elif k == "hiddenIds":
            cfg[k] = [x for x in v if isinstance(x, str) and x and "/" not in x and "\\" not in x] if isinstance(v, list) else []
        elif k in ("flip", "glassWindow", "pauseOnHidden", "pauseOnBlur", "pauseOnBattery", "shuffle"):
            cfg[k] = bool(v)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            # 各滑杆的合理区间裁剪
            lo, hi = {"glassAlpha": (0, 60), "blur": (0, 60), "scrim": (0, 1),
                      "border": (0, 1), "wallpaperBlur": (0, 60),
                      "carouselSecs": (3, 3600), "playbackRate": (0.5, 2)}.get(k, (None, None))
            if lo is not None:
                cfg[k] = max(lo, min(hi, v))
            else:
                cfg[k] = v
    try:
        UI_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"保存外观配置失败: {e}")
    return cfg

def _load_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"model": "deepseek", "keys": {}}

def _load_model_config() -> str:
    return _load_config().get("model", "deepseek")

def _save_config(model: str) -> None:
    cfg = _load_config()
    cfg["model"] = model
    cfg.pop("keys", None)  # 密钥统一存于 .env，不再写入 model_config.json（避免进入版本库）
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

# ── 按题目(thread_id)粒度加锁: 不同题目可并行跑, 同一题目内必须串行(防止同题交错) ──
# 工作区已按 thread 隔离(见 src/tool._active_ws_id), 不同题互不写错目录;
# 同一题目仍用一把自己的锁串行化若干次请求(含 interrupt 恢复), 保 checkpointer 状态有序。
_thread_run_locks: dict[str, asyncio.Lock] = {}

def _lock_for(thread_id: str) -> asyncio.Lock:
    """取指定题目的运行锁(不存在则创建)"""
    lk = _thread_run_locks.get(thread_id)
    if lk is None:
        lk = asyncio.Lock()
        _thread_run_locks[thread_id] = lk
    return lk

def _thread_is_running(thread_id: str) -> bool:
    """该题目当前是否正在运行"""
    lk = _thread_run_locks.get(thread_id)
    return bool(lk and lk.locked())

def _release_run_lock(thread_id: str) -> None:
    """运行结束后从表里清理空闲锁, 避免历史题目无限堆积"""
    lk = _thread_run_locks.get(thread_id)
    if lk is not None and not lk.locked():
        _thread_run_locks.pop(thread_id, None)


class ModelName(BaseModel):
    """设置默认模型（持久化到 model_config.json），可选附带 API Key"""
    model: str
    api_key: Optional[str] = None

class StreamBody(BaseModel):
    """resume 缺省表示启动新运行; 提供时表示恢复被 interrupt 挂起的运行；
    continue_run=True 表示从最近 checkpoint 续跑(崩溃/中止后, 无挂起中断场景);
    model 指定本次运行的模型，api_key 为非 deepseek 模型必填"""
    resume: Optional[str] = None
    thread_id: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    continue_run: bool = False

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


def _chunk_texts(chunk) -> tuple:
    """从流式 AIMessageChunk 提取 (正文增量, 思考增量)。
    思考链: 各家 OpenAI 兼容推理模型放在 additional_kwargs.reasoning_content
    (ChatDeepSeek 与 models.py 的捕获子类均写入此字段)。"""
    ak = getattr(chunk, "additional_kwargs", None) or {}
    rc = ak.get("reasoning_content")
    think = rc if isinstance(rc, str) else ""
    c = getattr(chunk, "content", None)
    if isinstance(c, str):
        text = c
    elif isinstance(c, list):
        text = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in c
            if isinstance(p, str) or (isinstance(p, dict) and p.get("type") == "text")
        )
    else:
        text = ""
    return text, think


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


async def _stream_events(resume: Optional[str], thread_id: str, model: Optional[str] = None, api_key: Optional[str] = None, continue_run: bool = False):
    runtime_config = {"configurable": {"thread_id": thread_id}}
    loop = asyncio.get_running_loop()
    out_q: asyncio.Queue = asyncio.Queue()  # ("chunk", sse) | ("done", None) | ("stop", None)

    # 把 src.agent 的"LLM调用"日志实时桥接进 SSE:
    # 这些日志在节点执行中途产生,若只随节点 update 推送会滞后到节点结束,故用 sink 直推
    def _llm_log_sink(message) -> None:
        text = str(message).strip()
        if text.startswith("LLM调用"):
            try:
                loop.call_soon_threadsafe(
                    out_q.put_nowait,
                    ("chunk", _sse({"type": "log", "text": text})),
                )
            except RuntimeError:
                pass  # 事件循环已关闭(客户端断开),静默丢弃

    sink_id = logger.add(_llm_log_sink, format="{message}", level="INFO")

    async def _pump(inputs) -> None:
        """跑图并把节点更新/中断转成 SSE 块塞进队列;异常与中断都归为 stop。"""
        is_resume = isinstance(inputs, Command)
        mode = "resume(中断恢复)" if is_resume else ("continue(检查点续跑)" if inputs is None else "新运行")
        logger.info(f"[stream] pump 启动: {mode}")
        agen = graph.astream(
            inputs,
            config=runtime_config,
            stream_mode=["updates", "messages"],  # updates:节点级状态; messages:LLM token 级流式
        )
        try:
            async for smode, item in agen:
                if smode == "messages":
                    # (AIMessageChunk, metadata): 思考/正文增量实时转发,前端打字机展示
                    chunk, meta = item
                    text, think = _chunk_texts(chunk)
                    if not text and not think:
                        continue  # 纯工具调用块等无文本增量
                    node = (meta or {}).get("langgraph_node") or ""
                    mid = getattr(chunk, "id", None) or ""
                    if think:
                        out_q.put_nowait(("chunk", _sse({
                            "type": "reasoning", "text": think, "node": node, "mid": mid,
                        })))
                    if text:
                        out_q.put_nowait(("chunk", _sse({
                            "type": "token", "text": text, "node": node, "mid": mid,
                        })))
                    continue
                if not isinstance(item, dict):
                    continue
                stop = False
                for node, payload in item.items():
                    if node == "__interrupt__" and isinstance(payload, (list, tuple)):
                        its = [{"value": getattr(it, "value", str(it))} for it in payload]
                    else:
                        its = _extract_interrupts(payload)
                    if its:
                        logger.info(f"[stream] 中断 @ {node},挂起等待人工输入")
                        for it in its:
                            out_q.put_nowait(("chunk", _sse({"type": "interrupt", "node": node, "value": it["value"]})))
                        out_q.put_nowait(("chunk", _sse({"type": "suspended"})))
                        out_q.put_nowait(("stop", None))  # 终止信号:主循环 break → 释放本题目锁 → 响应正常结束
                        stop = True
                        break
                    out_q.put_nowait(("chunk", _sse({"type": "update", "node": node, "data": _simplify(payload)})))
                if stop:
                    return
            logger.info("[stream] astream 迭代自然结束(图跑到 END)")
            out_q.put_nowait(("done", None))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("[stream] 运行异常")
            out_q.put_nowait(("chunk", _sse({"type": "error", "error": str(e)})))
            out_q.put_nowait(("stop", None))
        finally:
            # 显式关闭 astream 生成器:中断提前 return 时不留僵尸生成器,
            # 避免其延迟 GC 触发的 GeneratorExit 清理干扰同线程的下一次 resume
            try:
                await agen.aclose()
            except Exception:
                pass

    stopped = False
    pump_task = None
    try:
        run_lock = _lock_for(thread_id)
        async with run_lock:
            # 题目隔离：thread_id 即题目 id。运行内工作区由 agent 按 config.thread_id 自动解析;
            # set_workspace 仅给"运行外浏览文件"的接口当兜底默认。
            set_workspace(thread_id)
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
            if continue_run:
                # 从最近 checkpoint 续跑:
                #   无待续节点(线程为空/已跑完) → 按新运行处理;
                #   有挂起 interrupt(等人工输入) → Command(resume="") 跳过当前提问继续;
                #   中途崩溃/中止(有 next 无 interrupt) → inputs=None 原地继续(langgraph 断点续跑语义)
                try:
                    snap = graph.get_state(runtime_config)
                    has_interrupt = any(getattr(t, "interrupts", None) for t in (snap.tasks or ()))
                except Exception:
                    snap, has_interrupt = None, False
                if snap is None or not snap.next:
                    try:
                        checkpointer.delete_thread(thread_id)
                    except Exception:
                        pass
                    inputs = {}
                    reset_usage()  # 无可续状态按新运行处理, 用量统计一并归零
                    reset_ai_log()  # AI 使用事件同样归零(声明只反映本次运行)
                elif has_interrupt:
                    inputs = Command(resume="")
                else:
                    inputs = None
            elif resume is not None:
                inputs = Command(resume=resume)
            else:
                # 开始新运行时清空线程残留状态, 保证从头开始
                try:
                    checkpointer.delete_thread(thread_id)
                except Exception:
                    pass
                inputs = {}
                reset_usage()  # 新运行用量统计归零, 前端展示本次运行的真实消耗
                reset_ai_log()  # AI 使用事件同样归零(声明只反映本次运行)
            pump_task = asyncio.create_task(_pump(inputs))
            logger.info(f"[stream] 锁已获取,pump 已调度 (thread={thread_id})")
            while True:
                kind, payload = await out_q.get()
                if kind != "chunk":
                    stopped = kind == "stop"
                    break
                yield payload
            logger.info(f"[stream] 主循环结束: {'stop(挂起/异常)' if stopped else 'done(完成)'}")
        if not stopped:
            yield _sse({"type": "done", "thread_id": thread_id})
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("运行异常")
        yield _sse({"type": "error", "error": str(e)})
    finally:
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
        _release_run_lock(thread_id)
        logger.remove(sink_id)


@app.get("/api/models")
async def get_models():
    """返回所有可选模型（含所需 key 的变量名与提示），供前端下拉框读取"""
    return {"models": list_models()}

@app.get("/api/model")
async def get_model_config():
    """返回当前已保存的默认模型；密钥统一从 .env 读取，这里不再返回明文"""
    _cfg = _load_config()
    return {
        "model": _cfg.get("model", "deepseek"),
        "keys": {},
    }

@app.post("/api/model")
async def set_model_config(body: ModelName):
    """保存默认模型（持久化），可选附带 API Key（写入 .env，与系统环境变量同源）"""
    name = (body.model or "deepseek").strip().lower()
    if name not in MODEL_REGISTRY:
        raise HTTPException(400, f"未知模型: {name}（可用: {', '.join(MODEL_REGISTRY.keys())}）")
    if body.api_key:
        _env = MODEL_REGISTRY[name]["api_key_env"]
        set_key(str(ENV_PATH), _env, body.api_key.strip())
        os.environ[_env] = body.api_key.strip()
    try:
        set_model(name)
    except Exception as e:
        raise HTTPException(400, f"切换模型失败: {e}")
    _save_config(name)
    return {"model": name, "status": "saved"}

@app.post("/api/keys")
async def save_keys(body: KeysBody):
    """批量保存 API Key：直接写入 .env（前端配置落盘到 .env，与系统环境变量同源），
    不写入 model_config.json，避免密钥进入版本库。已存在于系统环境 / .env 的 key 无需重复提交。"""
    for env_name, val in (body.keys or {}).items():
        if val and val.strip():
            set_key(str(ENV_PATH), env_name, val.strip())
            os.environ[env_name] = val.strip()
    if body.model:
        name = body.model.strip().lower()
        if name in MODEL_REGISTRY:
            try:
                set_model(name)
            except Exception as e:
                logger.warning(f"切换默认模型失败（沿用当前）: {e}")
    # 仅默认模型持久化到 model_config.json（不再存密钥）
    try:
        _save_config((body.model or _load_config().get("model", "deepseek")).strip().lower())
    except Exception as e:
        logger.warning(f"保存默认模型失败: {e}")
    present = {m["api_key_env"]: bool(os.getenv(m["api_key_env"])) for m in MODEL_REGISTRY.values()}
    return {"model": _load_config().get("model", "deepseek"), "keys": {}, "present": present}

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


@app.get("/api/usage")
async def get_usage():
    """Token 消耗与缓存命中统计(全局累计; /api/reset 与每次新运行时清零)。
    缓存命中率仅 DeepSeek/OpenAI 接口返回, 其余模型前端显示"—"。"""
    return usage_snapshot()


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
        _stream_events(body.resume, thread_id, body.model, body.api_key, body.continue_run),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _browsing_root(thread_id: str = "") -> Path:
    """浏览类接口(files/file/image/pdfs/pdf)的工作区根：
    带 thread_id 时按题查询——多题并行下全局兜底指针(_CURRENT_WS)会被后启动的运行占用,
    前端按题传参才能保证产物面板不串台;不传则沿用全局兜底(旧行为,向后兼容)。"""
    if not thread_id:
        return ws_root()
    tid = thread_id.strip()
    if not tid or "/" in tid or "\\" in tid or ".." in tid:
        raise HTTPException(400, f"非法题目 id: {thread_id!r}")
    return WORKSPACES_ROOT / tid


def _resolve_work_path(work_dir: str, rel_path: str = "", thread_id: str = "") -> Path:
    """校验并解析指定题目工作区路径, 非法路径抛 400"""
    if work_dir not in ALLOWED_WORK_DIRS:
        raise HTTPException(400, f"work_dir 只能是 {'/'.join(sorted(ALLOWED_WORK_DIRS))}, 收到: {work_dir!r}")
    p = Path(rel_path) if rel_path else Path(".")
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(400, f"rel_path 必须是相对路径且不能包含 '..': {rel_path}")
    base = _browsing_root(thread_id)
    target = (base / work_dir / p).resolve()
    if not target.is_relative_to((base / work_dir).resolve()):
        raise HTTPException(400, f"路径超出 {work_dir} 目录: {rel_path}")
    return target


@app.get("/api/files")
async def list_files(work_dir: str, thread_id: str = ""):
    """列出题目工作区目录下的文件与文件夹(名称+大小), 供前端实时刷新;
    thread_id 缺省时用全局兜底工作区(旧行为)"""
    if work_dir not in ALLOWED_WORK_DIRS:
        raise HTTPException(400, f"work_dir 只能是 {'/'.join(sorted(ALLOWED_WORK_DIRS))}")
    root = _browsing_root(thread_id) / work_dir
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
async def read_file(work_dir: str, rel_path: str = "", thread_id: str = ""):
    """读取工作区文本文件(思路/代码/dataset), 返回文本内容"""
    target = _resolve_work_path(work_dir, rel_path, thread_id)
    if not target.is_file():
        raise HTTPException(404, f"文件不存在: {target}")
    if target.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp"):
        raise HTTPException(400, "图片请用 /api/image 获取")
    try:
        return {"work_dir": work_dir, "rel_path": rel_path, "content": target.read_text(encoding="utf-8")}
    except UnicodeDecodeError:
        raise HTTPException(400, f"{rel_path} 是二进制文件, 无法作为文本读取")


@app.get("/api/image")
async def get_image(work_dir: str, rel_path: str = "", thread_id: str = ""):
    """读取工作区图片(photo 下的 png 等), 二进制返回"""
    target = _resolve_work_path(work_dir, rel_path, thread_id)
    if not target.is_file():
        raise HTTPException(404, f"图片不存在: {target}")
    return FileResponse(str(target))


# ---------- PDF 产物展示: 前端内嵌预览生成的论文与合规文档 ----------
_PDF_TARGETS = [
    ("竞赛论文", "paper/latex/document.pdf"),
    ("AI 工具使用详情", "AI 工具使用详情.pdf"),
]


@app.get("/api/pdfs")
async def list_pdfs(thread_id: str = ""):
    """列出当前题目的 PDF 产物(存在才列出), 供前端产物面板展示"""
    root = _browsing_root(thread_id)
    out = []
    for name, rel in _PDF_TARGETS:
        p = root / rel
        if p.is_file():
            out.append({"name": name, "path": rel, "size": p.stat().st_size})
    return {"pdfs": out}


@app.get("/api/pdf")
async def get_pdf(path: str, thread_id: str = ""):
    """返回工作区内 PDF 文件(仅限 .pdf 且禁止路径越界), 供前端内嵌预览"""
    root = _browsing_root(thread_id)
    if not path.lower().endswith(".pdf") or ".." in path.replace("\\", "/").split("/"):
        raise HTTPException(400, "非法 PDF 路径")
    p = (root / path).resolve()
    if not p.is_relative_to(root.resolve()):
        raise HTTPException(400, "路径越界")
    if not p.is_file():
        raise HTTPException(404, "PDF 不存在")
    return FileResponse(str(p), media_type="application/pdf")


@app.get("/api/ui-config")
async def get_ui_config():
    """读取前端外观配置（液态玻璃主题 + 背景模式），不存在时返回默认值"""
    return _load_ui_config()


@app.put("/api/ui-config")
async def put_ui_config(body: dict):
    """合并保存前端外观配置到 ui_config.json（字段白名单 + 区间裁剪）"""
    cfg = _save_ui_config(body if isinstance(body, dict) else {})
    return cfg


def _media_dir() -> Path:
    """背景素材目录（可配置，默认项目根/media；改存储位置 = 改 mediaDir）"""
    rel = _load_ui_config().get("mediaDir") or "media"
    d = PROJECT_ROOT / rel
    d.mkdir(parents=True, exist_ok=True)
    return d

MEDIA_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".mp4", ".webm")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".webm")


def _find_ffmpeg() -> str | None:
    """定位 ffmpeg：环境变量 MATH_FFMPEG → PATH → dsh 插件目录。找不到返回 None（功能自动降级）。"""
    env = os.environ.get("MATH_FFMPEG")
    if env and Path(env).is_file():
        return env
    p = shutil.which("ffmpeg")
    if p:
        return p
    for cand in (PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe",
                 Path.home() / ".dsh-wallpaper-engine" / "ffmpeg" / "ffmpeg.exe"):
        if cand.is_file():
            return str(cand)
    return None


@app.get("/api/media")
async def list_media():
    """列出背景素材目录下的图片/视频，供前端背景选择器使用"""
    root = _media_dir()
    if not root.is_dir():
        return {"files": []}
    items = []
    for f in sorted(root.iterdir()):
        if f.is_file() and f.suffix.lower() in MEDIA_EXTS:
            items.append({"name": f.name, "size": f.stat().st_size})
    return {"files": items}


@app.get("/api/media/file")
async def get_media_file(name: str = ""):
    """按文件名返回背景素材（FileResponse 原生支持 Range，可拖进度）"""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    target = _media_dir() / name
    if not target.is_file():
        raise HTTPException(404, f"素材不存在: {target}")
    return FileResponse(str(target))


@app.post("/api/media/upload")
async def upload_media(file: UploadFile):
    """上传自定义背景素材（jpg/png/webp/mp4/webm），存到背景素材目录"""
    name = file.filename or ""
    if not name or "/" in name or "\\" in name:
        raise HTTPException(400, "非法文件名")
    ext = Path(name).suffix.lower()
    if ext not in MEDIA_EXTS:
        raise HTTPException(400, f"仅支持 {'/'.join(MEDIA_EXTS)}")
    data = await file.read()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, "文件超过 200MB 上限")
    target = _media_dir() / name
    target.write_bytes(data)
    logger.info(f"上传背景素材: {name} ({len(data)} bytes)")
    return {"name": name, "size": len(data)}


_TRANSCODE_STATE: dict = {}  # name -> {phase, percent, error}


@app.get("/api/media/thumb")
async def media_thumb(name: str = ""):
    """视频缩略图：ffmpeg 抽第 1 帧缓存为 jpg。无 ffmpeg / 非视频 → 404（前端用图标兜底）"""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    target = _media_dir() / name
    if not target.is_file() or target.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(404, "仅视频支持抽帧缩略图")
    ff = _find_ffmpeg()
    if not ff:
        raise HTTPException(404, "未找到 ffmpeg（设置 MATH_FFMPEG 或加入 PATH 后可用）")
    cache_dir = PROJECT_ROOT / ".cache" / "thumbs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    thumb = cache_dir / (name + ".jpg")
    if not thumb.exists():
        proc = await asyncio.create_subprocess_exec(
            ff, "-y", "-ss", "0.5", "-i", str(target), "-frames:v", "1",
            "-q:v", "4", "-vf", "scale=480:-2", str(thumb),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        if not thumb.exists():
            raise HTTPException(500, "抽帧失败")
    return FileResponse(str(thumb))


@app.post("/api/media/transcode")
async def transcode_media(body: dict):
    """把视频转码为低帧率版（dsh 同款思路：4K120→24fps 解码占用线性下降）。
    无 ffmpeg 或已在转码 → 返回当前状态；完成后前端改用转码版 URL。"""
    name = body.get("name", "") if isinstance(body, dict) else ""
    fps = int(body.get("fps", 24) or 24) if isinstance(body, dict) else 24
    fps = max(5, min(60, fps))
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    target = _media_dir() / name
    if not target.is_file() or target.suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(404, "仅视频支持转码")
    ff = _find_ffmpeg()
    if not ff:
        return {"name": name, "phase": "skipped", "error": "未找到 ffmpeg"}
    st = _TRANSCODE_STATE.get(name)
    if st and st["phase"] in ("working",):
        return {"name": name, **st}
    out_dir = PROJECT_ROOT / ".cache" / "transcoded"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.{fps}fps.mp4"
    if out.exists():
        _TRANSCODE_STATE[name] = {"phase": "ready", "percent": 100, "error": None, "file": out.name}
        return {"name": name, **_TRANSCODE_STATE[name]}
    _TRANSCODE_STATE[name] = {"phase": "working", "percent": 0, "error": None, "file": out.name}

    async def _run():
        proc = await asyncio.create_subprocess_exec(
            ff, "-y", "-i", str(target), "-vf", f"fps={fps}", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "26", "-an", "-progress", "pipe:1", "-nostats", str(out),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        total_secs = None
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "ignore").strip()
            if text.startswith("out_time_us="):
                try:
                    us = int(text.split("=", 1)[1])
                    if total_secs:
                        _TRANSCODE_STATE[name]["percent"] = min(99, int(us / 1_000_000 / total_secs * 100))
                except Exception:
                    pass
            elif text.startswith("duration=") or text.startswith("out_time_ms="):
                try:
                    ms = int(text.split("=", 1)[1])
                    if total_secs:
                        _TRANSCODE_STATE[name]["percent"] = min(99, int(ms / 1000 / total_secs * 100))
                except Exception:
                    pass
        await proc.wait()
        if out.exists() and out.stat().st_size > 0:
            _TRANSCODE_STATE[name] = {"phase": "ready", "percent": 100, "error": None, "file": out.name}
        else:
            _TRANSCODE_STATE[name] = {"phase": "error", "percent": 0, "error": "转码失败", "file": None}

    asyncio.create_task(_run())
    return {"name": name, **_TRANSCODE_STATE[name]}


@app.get("/api/media/transcode-status")
async def transcode_status(name: str = ""):
    """查询转码进度（phase: idle/working/ready/error/skipped + percent）"""
    if not name:
        return {"phase": "idle", "percent": 0}
    st = _TRANSCODE_STATE.get(name) or {"phase": "idle", "percent": 0, "error": None, "file": None}
    return {"name": name, **st}


@app.get("/api/media/transcoded")
async def get_transcoded(name: str = ""):
    """返回转码产物（.cache/transcoded 下）"""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    target = PROJECT_ROOT / ".cache" / "transcoded" / name
    if not target.is_file():
        raise HTTPException(404, f"转码产物不存在: {target}")
    return FileResponse(str(target))


# ---------- Wallpaper Engine 壁纸库（dsh 同款：直接读 Steam workshop 目录） ----------
WE_APPID = "431960"
WE_INSTALL = Path(r"C:\Program Files (x86)\Steam\steamapps\common\wallpaper_engine")
WE_RATING_ORDER = {"unrated": 0, "everyone": 1, "pg13": 2, "mature": 3}


def _steam_library_dirs() -> list:
    """解析 libraryfolders.vdf 拿所有 Steam 库路径（非默认盘也能找到）。
    默认库 = vdf 所在 steamapps 的上一级；其余库读 "path" 字段。"""
    dirs, seen = [], set()
    vdf_paths = [
        WE_INSTALL.parents[1] / "libraryfolders.vdf",   # C:\...\Steam\steamapps\libraryfolders.vdf
        Path.home() / "Steam" / "steamapps" / "libraryfolders.vdf",
    ]
    env_vdf = os.environ.get("MATH_STEAM_VDF")
    if env_vdf:
        vdf_paths.insert(0, Path(env_vdf))
    for vdf in vdf_paths:
        if not vdf.is_file() or str(vdf) in seen:
            continue
        seen.add(str(vdf))
        root = vdf.parent.parent  # steamapps/.. → Steam 根（默认库）
        if root.is_dir() and str(root) not in seen:
            seen.add(str(root))
            dirs.append(root)
        try:
            text = vdf.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r'"path"\s+"([^"]+)"', text):
                p = Path(m.group(1))
                if p.is_dir() and str(p) not in seen:
                    seen.add(str(p))
                    dirs.append(p)
        except Exception:
            pass
    return dirs


_WE_INV_CACHE = {"t": 0.0, "payload": None}


def _scan_we_wallpapers() -> list:
    """扫描 WE 壁纸库（workshop/content/431960 + WE 自带 projects），读 project.json 出清单。
    短 TTL 缓存（10s），与 dsh 的 inventory 缓存同思路。"""
    now = time.time()
    if _WE_INV_CACHE["payload"] is not None and now - _WE_INV_CACHE["t"] < 10:
        return _WE_INV_CACHE["payload"]
    roots = []
    for lib in _steam_library_dirs():
        ws = lib / "steamapps" / "workshop" / "content" / WE_APPID
        if ws.is_dir():
            roots.append(ws)
    for sub in ("projects", "projects/defaultprojects", "projects/myprojects"):
        p = WE_INSTALL / sub
        if p.is_dir():
            roots.append(p)
    wallpapers, seen = [], set()
    for root in roots:
        try:
            entries = list(root.iterdir())
        except Exception:
            continue
        for d in entries:
            if not d.is_dir():
                continue
            pj = d / "project.json"
            if not pj.is_file():
                continue
            try:
                j = json.loads(pj.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            wid = str(j.get("workshopid") or d.name)
            if wid in seen:
                continue
            seen.add(wid)
            wtype = (j.get("type") or "").lower()
            if wtype not in ("video", "scene", "web", "image"):
                continue  # application 等无法嵌入，跳过
            rating = (j.get("contentrating") or "unrated").lower()
            if rating not in WE_RATING_ORDER:
                rating = "unrated"
            wallpapers.append({
                "id": wid,
                "title": j.get("title") or d.name,
                "type": wtype,
                "rating": rating,
                "preview": j.get("preview") or "preview.jpg",
                "file": j.get("file") or "",
                "dir": str(d),
            })
    wallpapers.sort(key=lambda w: (WE_RATING_ORDER.get(w["rating"], 0), w["title"].lower()))
    _WE_INV_CACHE.update(t=now, payload=wallpapers)
    return wallpapers


def _find_we_wallpaper(wid: str):
    if not wid or "/" in wid or "\\" in wid:
        return None
    for w in _scan_we_wallpapers():
        if w["id"] == wid:
            return w
    return None


@app.get("/api/we/inventory")
async def we_inventory():
    """WE 壁纸库清单（id/title/type/rating/preview/file），供前端壁纸选择器使用"""
    return {"wallpapers": _scan_we_wallpapers()}


@app.get("/api/we/preview")
async def we_preview(id: str = ""):
    """WE 壁纸预览图（preview.jpg），缩略图网格用"""
    wp = _find_we_wallpaper(id)
    if not wp:
        raise HTTPException(404, "壁纸不存在")
    target = Path(wp["dir"]) / wp["preview"]
    if not target.is_file():
        raise HTTPException(404, "预览图不存在")
    return FileResponse(str(target))


@app.get("/api/we/file")
async def we_file(id: str = ""):
    """WE 壁纸主文件：视频返回 mp4 直接播；scene/web/image 返回 preview 静态帧
    （dsh 解析 .pkg 抠主纹理，这里务实用 preview，零解析成本）"""
    wp = _find_we_wallpaper(id)
    if not wp:
        raise HTTPException(404, "壁纸不存在")
    if wp["type"] == "video" and wp["file"]:
        target = Path(wp["dir"]) / wp["file"]
        if target.is_file():
            return FileResponse(str(target))
    target = Path(wp["dir"]) / wp["preview"]
    if not target.is_file():
        raise HTTPException(404, "壁纸文件不存在")
    return FileResponse(str(target))


@app.delete("/api/file")
async def delete_artifact(rel_path: str, thread_id: str = ""):
    """删除题目工作区内单个产物文件(思路/代码/图片/PDF, rel_path 相对工作区根)。
    运行中的题目禁止删除——防止删掉 agent 正在读写的东西(与重置/删题同一保护)。"""
    ws_id = thread_id.strip() or get_workspace()
    _guard_ws_mutation(ws_id)
    if not rel_path or rel_path.strip() in ("", "."):
        raise HTTPException(400, "rel_path 不能为空")
    p = Path(rel_path)
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(400, f"非法路径: {rel_path}")
    root = _browsing_root(thread_id)
    target = (root / p).resolve()
    if not target.is_relative_to(root.resolve()):
        raise HTTPException(400, f"路径越界: {rel_path}")
    if not target.is_file():
        raise HTTPException(404, f"文件不存在: {rel_path}")
    target.unlink()
    logger.info(f"删除产物: {target}")
    return {"status": "deleted", "path": rel_path}


@app.delete("/api/workspaces/{ws_id}/file")
async def delete_ws_file(ws_id: str, target: str = "question", name: str = ""):
    """删除题目工作区里上传的题目/数据文件(target=question|dataset)。
    运行中的题目禁止删除(load_problem/read_dataset 可能还要读它)。"""
    _guard_ws_mutation(ws_id)
    if target not in ("question", "dataset"):
        raise HTTPException(400, "target 只能是 question / dataset")
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "非法文件名")
    f = _ws_dir(ws_id) / target / name
    if not f.is_file():
        raise HTTPException(404, f"文件不存在: {name}")
    f.unlink()
    logger.info(f"删除题目文件: {f}")
    return {"status": "deleted", "target": target, "name": name}


@app.post("/api/reset")
async def reset_thread(thread_id: str = DEFAULT_THREAD_ID):
    _guard_ws_mutation(thread_id)
    set_workspace(thread_id)
    reset_usage()  # 用量统计属于展示层全局态, 与线程一并重置
    reset_ai_log()  # AI 使用事件一并清零
    try:
        checkpointer.delete_thread(thread_id)
    except Exception:
        # 线程不存在时删除会抛错, 忽略即可
        pass
    return {"status": "reset", "thread_id": thread_id}


# ---------- 题目工作区管理（多题隔离：workspaces/{题目id}/ + index.json 注册表） ----------
WORKSPACES_ROOT = PROJECT_ROOT / "workspaces"
WS_INDEX_PATH = WORKSPACES_ROOT / "index.json"
QUESTION_EXTS = (".txt", ".md", ".pdf")
DATASET_EXTS = (".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl")


def _ws_dir(ws_id: str) -> Path:
    """校验题目 id 并返回其工作区目录（不存在则创建；5 个标准子目录一并建齐，
    防止新建题目未上传文件直接运行时 load_problem/read_dataset 抛 FileNotFoundError）"""
    if not ws_id or "/" in ws_id or "\\" in ws_id or ".." in ws_id:
        raise HTTPException(400, "非法题目 id")
    d = WORKSPACES_ROOT / ws_id
    d.mkdir(parents=True, exist_ok=True)
    for sub in ("question", "dataset", "paper", "code", "photo"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _load_workspaces() -> dict:
    try:
        if WS_INDEX_PATH.exists():
            return json.loads(WS_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_workspaces(ws: dict) -> None:
    WORKSPACES_ROOT.mkdir(parents=True, exist_ok=True)
    WS_INDEX_PATH.write_text(json.dumps(ws, ensure_ascii=False, indent=2), encoding="utf-8")


def _thread_has_state(ws_id: str) -> bool:
    """该题是否有未结束的线程状态（有→前端显示「继续运行」而非「开始运行」）"""
    try:
        return checkpointer.get_tuple({"configurable": {"thread_id": ws_id}}) is not None
    except Exception:
        return False


def _ws_info(ws_id: str, meta: dict) -> dict:
    d = WORKSPACES_ROOT / ws_id

    def _names(p: Path) -> list:
        if not p.is_dir():
            return []
        return sorted(f.name for f in p.iterdir() if f.is_file())

    return {
        "id": ws_id,
        "title": meta.get("title") or ws_id,
        "createdAt": meta.get("createdAt", ""),
        "questionFiles": _names(d / "question"),
        "datasetFiles": _names(d / "dataset"),
        "hasState": _thread_has_state(ws_id),
    }


@app.get("/api/workspaces")
async def list_workspaces():
    """题目列表（含每题的 question/dataset 文件与运行状态）；注册表缺失的目录也补列出来"""
    ws = _load_workspaces()
    if WORKSPACES_ROOT.is_dir():
        for d in WORKSPACES_ROOT.iterdir():
            if d.is_dir() and d.name != "index.json" and d.name not in ws:
                ws[d.name] = {"title": d.name, "createdAt": ""}
    return {"workspaces": [_ws_info(i, m) for i, m in ws.items()], "current": get_workspace()}


@app.post("/api/workspaces")
async def create_workspace(body: dict):
    """新建题目：title 必填；id 由 title 安全化生成（同 title 复用）"""
    title = (body.get("title") or "").strip() if isinstance(body, dict) else ""
    if not title:
        raise HTTPException(400, "题目标题不能为空")
    ws_id = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", title)[:60] or "workspace"
    ws = _load_workspaces()
    if ws_id not in ws:
        ws[ws_id] = {"title": title, "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M")}
        _save_workspaces(ws)
        _ws_dir(ws_id)
    return _ws_info(ws_id, ws[ws_id])


def _guard_ws_mutation(ws_id: str) -> None:
    """只阻止对"正在运行的同一题目"的切换/删除/重置：
    运行内工作区已按 thread 隔离(_active_ws_id 读 config.thread_id)，所以并发跑不同题时，
    操作 A 题不会被 B 题在跑而阻塞；但该题自己正在跑时禁止删/重置，避免清空进行中的产物。
    锁在 interrupt 挂起/完成/出错后都会释放，等待人工输入时不受影响。"""
    if _thread_is_running(ws_id):
        raise HTTPException(409, f"题目 {ws_id} 正在运行，请先暂停或等运行结束后再操作")


@app.post("/api/workspaces/{ws_id}/activate")
async def activate_workspace(ws_id: str):
    """切换当前浏览题目（运行内工作区由 agent 按 thread 自动解析，这里只切浏览兜底位置）"""
    if ws_id not in _load_workspaces():
        raise HTTPException(404, "题目不存在")
    set_workspace(ws_id)
    return {"current": ws_id}


@app.delete("/api/workspaces/{ws_id}")
async def delete_workspace(ws_id: str):
    """删除题目：连带工作区目录（含全部产物）+ 线程状态"""
    _guard_ws_mutation(ws_id)
    ws = _load_workspaces()
    if ws_id not in ws:
        raise HTTPException(404, "题目不存在")
    ws.pop(ws_id)
    _save_workspaces(ws)
    d = WORKSPACES_ROOT / ws_id
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
    try:
        checkpointer.delete_thread(ws_id)
    except Exception:
        pass
    if get_workspace() == ws_id:
        set_workspace("default")
    return {"status": "deleted", "id": ws_id}


@app.post("/api/workspaces/{ws_id}/upload")
async def upload_workspace_files(ws_id: str, target: str = "question", files: list[UploadFile] = File(...)):
    """上传文件到指定题的目标目录：target=question（题目文件）| dataset（数据文件）"""
    if target not in ("question", "dataset"):
        raise HTTPException(400, "target 只能是 question / dataset")
    _ws_dir(ws_id)
    d = WORKSPACES_ROOT / ws_id / target
    d.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = f.filename or ""
        if not name or "/" in name or "\\" in name:
            continue
        (d / name).write_bytes(await f.read())
        saved.append(name)
    ws = _load_workspaces()
    if ws_id not in ws:
        ws[ws_id] = {"title": ws_id, "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M")}
        _save_workspaces(ws)
    info = _ws_info(ws_id, ws[ws_id])
    info["saved"] = saved
    return info


def _migrate_legacy_workspace():
    """首次启动：把项目根下的 question/dataset/paper/code/photo 迁入 workspaces/default/，
    保证旧数据在新模型下不丢。只移动文件，目录结构保留。"""
    ws_dir = WORKSPACES_ROOT / "default"
    ws_dir.mkdir(parents=True, exist_ok=True)
    if any(ws_dir.iterdir()):
        return
    moved = []
    for name in ("question", "dataset", "paper", "code", "photo"):
        src = PROJECT_ROOT / name
        if src.is_dir():
            dst = ws_dir / name
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file():
                    try:
                        shutil.move(str(f), str(dst / f.name))
                        moved.append(f"{name}/{f.name}")
                    except Exception:
                        pass
    if moved:
        ws = _load_workspaces()
        ws.setdefault("default", {"title": "默认题目", "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M")})
        _save_workspaces(ws)
        logger.info(f"已迁移旧工作区数据到 workspaces/default/：{len(moved)} 个文件")
    for sub in ("question", "dataset", "paper", "code", "photo"):
        (ws_dir / sub).mkdir(parents=True, exist_ok=True)


_migrate_legacy_workspace()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)