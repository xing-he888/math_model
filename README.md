# 数学建模 Agent

一个基于 LangGraph 的数学建模自动化 Agent：读取题面与数据集 → 建模分析 → 四种方法并行求解 → 执行代码出图 → 汇总对比 → 人工终审。`src/` 里的 agent 逻辑不动，`backend/`、`frontend/` 是后来加的展示层。

```
question/   题目（txt/md/pdf）
dataset/    数据（csv/xlsx/json…）
code/       生成的求解脚本（自动落盘）
paper/      每问的思路 md + 最终总结
photo/      生成图片（自动移入）
src/        agent 本体（agent.py + tool.py）
backend/    FastAPI + SSE 流式接口
frontend/   Electron 桌面端，实时展示
```

```powershell
conda activate langgraph
python backend\server.py
```

保持这个窗口开着。再新开一个窗口起前端：

```powershell
cd frontend
npm install   # 第一次
npm start
```

桌面弹出控制台窗口，右上角「后端在线」变绿后点「开始运行」。

## 流程交互

流程中途会停在三个地方等人输入，底部输入框直接回车 = 跳过：

1. **建模思路**（可跳过，模型自拟）
2. **建模审核**：回车通过；输入"打回"重新分析；输入别的当建议
3. **最终审查**：回车认可；不满意写意见打回重写总结

左侧实时刷运行日志，右侧看思路、代码、图片（运行中每 3 秒自动刷新）。「重新开始」清空本轮状态。

## 常见问题

- 后端起不来：确认用的是 `langgraph` 环境的 python，且在根目录执行
- 前端显示后端离线：后端没起，或 8000 端口被占
- 出图脚本执行：`run_solutions` 只跑含 `savefig`/`matplotlib` 的脚本；单个脚本失败会重试 2 次，仍失败则提交 LLM 诊断并改写代码（最多 2 轮），全部失败才中断交由人工处理。



## 此项目有DeepSeek/GPT (OpenAI)/GLM (智谱 Zhipu)/通义千问 (Qwen)/Kimi (Moonshot)/MiniMax (Mimo)