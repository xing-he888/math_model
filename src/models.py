# -*- coding: utf-8 -*-
"""
模型注册表 + 工厂：让 agent 支持在多个大模型之间切换。

所有模型都走 OpenAI 兼容接口（/v1/chat/completions），
因此用同一个 ChatOpenAI 客户端即可对接各家（子类 _ThinkCaptureChatOpenAI 额外捕获
reasoning_content 思考链，行为与父类一致）；DeepSeek 因有 thinking 参数，单独用 ChatDeepSeek；
OpenRouter 是聚合网关，按官方推荐用专用包 langchain-openrouter 的 ChatOpenRouter。
新增模型：在 MODEL_REGISTRY 里加一项即可，无需改动 agent.py。

切换方式：
  - 默认 deepseek；前端下拉框选其它模型时，必须同时填入对应 API Key。
  - 模型与 key 持久化在项目根 model_config.json（配置一次一直可用）。
"""
import os
from langchain_deepseek import ChatDeepSeek

try:
    from langchain_openai import ChatOpenAI
except Exception:  # 未安装时仅 deepseek 可用
    ChatOpenAI = None

if ChatOpenAI is not None:

    class _ThinkCaptureChatOpenAI(ChatOpenAI):
        """带思考链捕获的 ChatOpenAI:GLM/Qwen/Kimi 等 OpenAI 兼容推理模型把思考过程
        放在流式 delta 的 reasoning_content 字段(社区事实标准),langchain-openai 官方
        不提取该字段。这里沿用 langchain-deepseek 同款扩展钩子,只加不改:字段转换、
        请求构造全部走父类,仅把思考增量附加进 additional_kwargs,供后端 SSE 实时转发。
        """

        def _convert_chunk_to_generation_chunk(
            self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
        ):
            generation_chunk = super()._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, base_generation_info
            )
            if generation_chunk is None:
                return None
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            delta = (choices[0].get("delta") if choices else None) or {}
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning")  # OpenRouter 风格字段
            if reasoning is not None:
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
            return generation_chunk

        def _create_chat_result(self, response, generation_info=None):
            rtn = super()._create_chat_result(response, generation_info)
            choices = getattr(response, "choices", None)
            if choices:
                msg = choices[0].message
                rc = getattr(msg, "reasoning_content", None)
                if rc is None and isinstance(getattr(msg, "model_extra", None), dict):
                    rc = msg.model_extra.get("reasoning")
                if rc:
                    rtn.generations[0].message.additional_kwargs["reasoning_content"] = rc
            return rtn

else:
    _ThinkCaptureChatOpenAI = None

try:
    from langchain_openrouter import ChatOpenRouter
except Exception:  # 未安装时 openrouter 不可用
    ChatOpenRouter = None


# thinking 字段：声明该模型思考模式的请求参数格式（on=开/off=关，均为 extra_body 内容）。
# 有此字段 = 模型存在思考开关；没有 = 无思考模式（gpt/kimi/mimo 等，两种实例同配置）。
# 注意：DeepSeek V4 默认即思考模式，且思考模式下拒绝 tool_choice="required"/指定函数名
# （HTTP 400: Thinking mode does not support this tool_choice），故工具/结构化调用必须显式 off。
THINKING_DEEPSEEK = {
    "on": {"thinking": {"type": "enabled"}},
    "off": {"thinking": {"type": "disabled"}},
}
THINKING_ZHIPU = {  # 智谱新版 thinking 参数格式（glm-4-flash 是否支持待实测，报错会自动降级）
    "on": {"thinking": {"type": "enabled"}},
    "off": {"thinking": {"type": "disabled"}},
}
THINKING_QWEN = {  # DashScope 兼容模式 enable_thinking 开关
    "on": {"enable_thinking": True},
    "off": {"enable_thinking": False},
}

MODEL_REGISTRY = {
    "deepseek": {
        "label": "DeepSeek",
        "key_label": "DeepSeek API Key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": THINKING_DEEPSEEK,
    },
    "gpt": {
        "label": "GPT (OpenAI)",
        "key_label": "OpenAI API Key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "glm": {
        "label": "GLM (智谱 Zhipu)",
        "key_label": "智谱 API Key",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4.7",
        "api_key_env": "ZHIPU_API_KEY",
        "thinking": THINKING_ZHIPU,
    },
    "qwen": {
        "label": "通义千问 (Qwen)",
        "key_label": "通义千问 API Key (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "thinking": THINKING_QWEN,
    },
    "kimi": {
        "label": "Kimi",
        "key_label": "Kimi API Key (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "mimo": {
        "label": "MiniMax",
        "key_label": "MiniMax API Key",
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-Text-01",
        "api_key_env": "MINIMAX_API_KEY",
    },
    # OpenRouter 是聚合网关：一个 key 访问 400+ 模型，模型 ID 为 "provider/model" 格式，
    # 端点由专用包 ChatOpenRouter 内置，无需 base_url。key 全部复用 OPENROUTER_API_KEY。
    "openrouter": {
        "label": "OpenRouter",
        "key_label": "OpenRouter API Key",
        "model": "stealth/ox-alpha",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}

DEFAULT_MODEL = "deepseek"

# 模型实例缓存：按 (模型key, 角色, api_key) 复用同一个无状态实例。
# 保证多题并行(共用同一模型)时，set_model 不会反复 new 对象造成模块级绑定竞态；
# api_key 一旦变化(例如请求带新 key)会命中不同缓存项，避免拿到过期 key。
_MODEL_CACHE: dict = {}


def get_model(name: str = None, role: str = "text"):
    """根据名字（或 MATH_MODEL 环境变量）返回一个可用的 ChatModel。

    role:
      "tool" -> 干活实例：思考模式强制关闭（供工具循环/结构化输出等会发强制
                tool_choice 的调用使用，规避 DeepSeek 思考模式的 400 限制）。
      "text" -> 动脑实例：模型支持思考就默认开启（建模分析/诊断/论文写作等节点）。
    """
    key = (name or os.getenv("MATH_MODEL", DEFAULT_MODEL)).strip().lower()
    if key not in MODEL_REGISTRY:
        key = DEFAULT_MODEL
    cfg = MODEL_REGISTRY[key]
    api_key = os.getenv(cfg["api_key_env"], "") or "EMPTY"

    # 决定思考参数：有 thinking 字段才需要传；tool 角色一律 off，text 角色默认 on
    tcfg = cfg.get("thinking")
    extra = tcfg["on"] if (tcfg and role == "text") else (tcfg["off"] if tcfg else None)

    cache_key = (key, role, api_key, bool(extra) if extra is not None else None)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if key == "deepseek":
        instance = ChatDeepSeek(
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=api_key,
            extra_body=extra,
        )
        _MODEL_CACHE[cache_key] = instance
        return instance

    if key == "openrouter":
        if ChatOpenRouter is None:
            raise RuntimeError(
                "使用 OpenRouter 需要安装 langchain-openrouter，请执行：pip install -U langchain-openrouter"
            )
        instance = ChatOpenRouter(
            model=cfg["model"],
            api_key=api_key,
        )
        _MODEL_CACHE[cache_key] = instance
        return instance

    if ChatOpenAI is None:
        raise RuntimeError(
            "使用非 DeepSeek 模型需要安装 langchain-openai，请执行：pip install langchain-openai"
        )
    kwargs = dict(model=cfg["model"], base_url=cfg["base_url"], api_key=api_key)
    if extra is not None:
        kwargs["extra_body"] = extra
    try:
        instance = _ThinkCaptureChatOpenAI(**kwargs)
    except (TypeError, ValueError):
        # 旧版 langchain-openai 不支持 extra_body（pydantic 抛 ValidationError=ValueError 子类）：
        # 退回不传思考参数，等效于各模型默认行为
        kwargs.pop("extra_body", None)
        instance = _ThinkCaptureChatOpenAI(**kwargs)
    _MODEL_CACHE[cache_key] = instance
    return instance


def get_struct_model():
    """格式化专用模型（问题提取/质检等 with_structured_output 步骤）：固定 DeepSeek 非思考实例。

    不随前端/环境变量切换模型变化——with_structured_output 内部会发强制 tool_choice，
    强制思考模型（如 OpenRouter 接入的部分 reasoning 模型）会以 HTTP 400 拒绝，
    因此该步骤钉死在支持显式关闭思考的 DeepSeek 上。需在 .env 配置 DEEPSEEK_API_KEY。
    """
    cfg = MODEL_REGISTRY["deepseek"]
    return ChatDeepSeek(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=os.getenv(cfg["api_key_env"], "") or "EMPTY",
        extra_body=THINKING_DEEPSEEK["off"],
    )


def list_models():
    """返回所有可选模型（含所需 key 的变量名与提示、当前环境是否已配置）。"""
    return [
        {
            "key": k,
            "label": v["label"],
            "model": v["model"],
            "api_key_env": v["api_key_env"],
            "key_label": v["key_label"],
            "key_set": bool(os.getenv(v["api_key_env"])),
        }
        for k, v in MODEL_REGISTRY.items()
    ]
