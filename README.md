# 数学建模 Agent

# 数学建模 Agent 系统 · 技术栈总结

> **一句话定位**：以 LangGraph 状态机为编排内核、FastAPI 为展示适配层、Electron + React 为交互壳的
> 竞赛级数学建模自动化系统——从题目解析、多方法并行求解、质检打回，到 LaTeX 论文成稿与 AI 使用合规留痕的全流程闭环。

---

## 总体架构

系统采用**内核与展示严格分层**的结构：`src/agent.py` 的图逻辑与展示层零耦合，后端只做协议适配（SSE 化），前端只做渲染与交互。

```
┌─────────────────────────────────────────────────────────────┐
│  Electron 壳 (concurrently 驱动 vite + electron)             │
│  React 18 + TS + Vite ── 液态玻璃 UI / 多题会话桶 / SSE 消费  │
└───────────────▲─────────────────────────┬───────────────────┘
                │ SSE (token/reasoning/    │ POST /api/stream
                │ update/interrupt/…)      │ (resume / continue_run)
┌───────────────┴─────────────────────────▼───────────────────┐
│  FastAPI 适配层 (server.py)                                  │
│  SSE 管道 · 多题运行锁 · 工作区浏览接口 · 外观/产物静态服务     │
└───────────────▲─────────────────────────┬───────────────────┘
                │ graph.astream            │ checkpointer
┌───────────────┴─────────────────────────▼───────────────────┐
│  LangGraph 编排内核 (agent.py)                               │
│  StateGraph · Send 扇出 · 超步屏障 · interrupt 人机协同       │
│  ┌──────────┐   ┌─────────────────────────────┐             │
│  │ 建模/质检 │──►│ solve_with_method × N(并行)  │──► 汇总对比  │
│  └──────────┘   └─────────────────────────────┘   ──► 论文   │
└───────────────▲─────────────────────────────────────────────┘
                │ OpenAI 兼容协议 (ChatDeepSeek / 捕获子类 ChatOpenAI)
        ┌───────┴────────┐
        │ DeepSeek/GLM/  │  ← .env 密钥, thinking 参数矩阵
        │ Qwen/Kimi/…    │
        └────────────────┘
```


启动后端
```powershell
python backend\server.py
```

保持这个窗口开着。再新开一个窗口起前端：

```powershell
cd frontend
npm install   # 第一次
npm start
```

桌面弹出控制台窗口，右上角「后端在线」变绿后点「开始运行」。


## 技术栈清单

| 分层 | 选型 | 版本 | 职责 |
|---|---|---|---|
| 编排内核 | LangGraph | 1.x | 状态机编排、Send 并行扇出、interrupt 人机协同、checkpointer 持久化 |
| LLM 接入 | langchain (core/openai/deepseek/community) | 1.x | OpenAI 兼容协议统一接入、工具绑定、结构化输出 |
| 服务层 | FastAPI + uvicorn | 0.1xx | SSE 流式接口、题目/产物 REST、外观配置持久化 |
| 前端框架 | React 18 + TypeScript + Vite | 18 | 会话 UI、状态桶、SSE 消费（fetch 流式读取器） |
| 桌面壳 | Electron | — | 桌面窗口承载；concurrently 一键并行拉起 vite + electron |
| 文档处理 | pypdf / python-docx / pandas | — | 题目/数据集多格式解析，论文 LaTeX 模板独立副本 |
| 检索工具 | TavilySearchResults (langchain-community) | — | 可选联网查资料工具（模型自主决定是否调用） |
| 可观测 | loguru + 自研用量记账 | — | LLM 调用日志、token/缓存命中分桶、AI 使用事件留痕 |


### 质量闭环与合规留痕

- **机器质检**：求解完成后结构化裁决"方案与结果是否可信自洽"，不通过携带原因与修正建议打回；
- **人工审核**：`interrupt` 收集对最终总结的意见，回流重写（清空对话上下文，避免惯性）；
- **数值真实性铁律**：prompt 级约束所有关键数值必须出自脚本真实 stdout（运行报告为唯一可信数值源）；
- **合规留痕**：AI 调用事件（时间/模型/提示摘录）独立记账，自动生成《AI 工具使用详情》PDF。



## 此项目有DeepSeek/GPT (OpenAI)/GLM (智谱 Zhipu)/通义千问 (Qwen)/Kimi (Moonshot)/MiniMax (Mimo)，在项目运行过程中，可以做到多模型的热切换

## 需要本机下载texlive才可正常生成论文

## 目前本次更新支持在前端自由删减或增添题目

