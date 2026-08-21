# -*- coding: utf-8 -*-
"""
模型注册表 + 工厂：让 agent 支持在多个大模型之间切换。

所有模型都走 OpenAI 兼容接口（/v1/chat/completions），
因此用同一个 ChatOpenAI 客户端即可对接各家；DeepSeek 因有 thinking 参数，单独用 ChatDeepSeek。
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


MODEL_REGISTRY = {
    "deepseek": {
        "label": "DeepSeek",
        "key_label": "DeepSeek API Key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": "disabled",   # DeepSeek 专属：disabled / enabled
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
        "model": "glm-4-flash",
        "api_key_env": "ZHIPU_API_KEY",
    },
    "qwen": {
        "label": "通义千问 (Qwen)",
        "key_label": "通义千问 API Key (DashScope)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "kimi": {
        "label": "Kimi (Moonshot)",
        "key_label": "Kimi API Key (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "api_key_env": "MOONSHOT_API_KEY",
    },
    "mimo": {
        "label": "MiniMax (Mimo)",
        "key_label": "MiniMax API Key",
        "base_url": "https://api.minimax.io/v1",
        "model": "MiniMax-Text-01",
        "api_key_env": "MINIMAX_API_KEY",
    },
}

DEFAULT_MODEL = "deepseek"


def get_model(name: str = None):
    """根据名字（或 MATH_MODEL 环境变量）返回一个可用的 ChatModel。"""
    key = (name or os.getenv("MATH_MODEL", DEFAULT_MODEL)).strip().lower()
    if key not in MODEL_REGISTRY:
        key = DEFAULT_MODEL
    cfg = MODEL_REGISTRY[key]
    api_key = os.getenv(cfg["api_key_env"], "") or "EMPTY"

    if key == "deepseek":
        extra = {"thinking": {"type": cfg.get("thinking", "disabled")}}
        return ChatDeepSeek(
            model=cfg["model"],
            base_url=cfg["base_url"],
            api_key=api_key,
            extra_body=extra,
        )

    if ChatOpenAI is None:
        raise RuntimeError(
            "使用非 DeepSeek 模型需要安装 langchain-openai，请执行：pip install langchain-openai"
        )
    return ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=api_key,
    )


def list_models():
    """返回所有可选模型（含所需 key 的变量名与提示、以及当前环境是否已配置）。"""
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
