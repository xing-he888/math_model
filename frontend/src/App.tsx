import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  startRun,
  fetchState,
  resetThread,
  checkHealth,
  getBackendUrl,
  listFiles,
  readFile,
  imageUrl,
  type SseEvent,
  type WorkspaceFile,
} from "./api";
import "./App.css";

const NODE_LABELS: Record<string, string> = {
  load_problem: "读取题目",
  question_structed: "解析问题结构",
  read_dataset: "读取数据集",
  modeling: "建模分析",
  tool_node: "工具执行",
  review_modeling_analysis: "建模审核",
  send_problem_index: "分发问题",
  collect_branches: "等待分支汇合",
  solve_with_method: "方法求解",
  run_solutions: "执行求解代码",
  compare_summarize: "汇总对比",
  compare_tool_node: "汇总工具",
  final_analysis: "最终审查",
};

type LogItem =
  | { id: number; t: string; kind: "node"; label: string; method?: string }
  | { id: number; t: string; kind: "tool"; label: string; content: string }
  | { id: number; t: string; kind: "ai"; label: string; content: string }
  | { id: number; t: string; kind: "interrupt"; label: string; question: string }
  | { id: number; t: string; kind: "user"; label: string; content: string }
  | { id: number; t: string; kind: "info"; label: string; content: string }
  | { id: number; t: string; kind: "error"; label: string; content: string };

type Status = "idle" | "running" | "awaiting_input" | "done" | "error";

type DistributiveOmit<T, K extends keyof any> = T extends any ? Omit<T, K> : never;

let seq = 0;
const now = () => new Date().toLocaleTimeString("zh-CN", { hour12: false });

/** 把 markdown 中的代码块折叠成 <details>，正文干净展示 */
function FoldingMarkdown({ source }: { source: string }) {
  const parts = source.split(/```[^\n]*\n?/g);
  const blocks = source.match(/```[^\n]*\n?[\s\S]*?```/g) || [];
  const out: React.ReactNode[] = [];
  parts.forEach((p, i) => {
    if (p.trim()) out.push(<ReactMarkdown key={`t${i}`}>{p}</ReactMarkdown>);
    if (i < blocks.length) {
      const lang = (blocks[i].match(/```([^\n]*)/)?.[1] || "").trim() || "code";
      out.push(
        <details key={`c${i}`} className="folded-code">
          <summary>[代码块] {lang}</summary>
          <pre>{blocks[i].replace(/^```[^\n]*\n?/, "").replace(/```$/, "")}</pre>
        </details>,
      );
    }
  });
  return <>{out}</>;
}

export default function App() {
  const threadId = "1111";
  const [status, setStatus] = useState<Status>("idle");
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [question, setQuestion] = useState<string | null>(null);
  const [answer, setAnswer] = useState("");
  const [finalSummary, setFinalSummary] = useState("");
  const [runReports, setRunReports] = useState<any[]>([]);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  const [papers, setPapers] = useState<WorkspaceFile[]>([]);
  const [codes, setCodes] = useState<WorkspaceFile[]>([]);
  const [images, setImages] = useState<WorkspaceFile[]>([]);
  // 点击图片放大时记录文件名(show in 灯箱)，null 表示未放大
  const [zoomImage, setZoomImage] = useState<string | null>(null);
  // 灯箱内大图的缩放倍率(滚轮控制)
  const [zoomScale, setZoomScale] = useState(1);
  // 灯箱根节点 ref，用于绑定原生非被动 wheel 监听和全屏
  const lightboxRef = useRef<HTMLDivElement | null>(null);
  const [selected, setSelected] = useState<{ dir: string; name: string; content: string } | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef<Status>(status);
  statusRef.current = status;

  const push = useCallback((item: DistributiveOmit<LogItem, "id" | "t">) => {
    setLogs((ls) => [...ls, { ...item, id: ++seq, t: now() }]);
  }, []);

  const refreshWorkspace = useCallback(async () => {
    const [p, c, im] = await Promise.all([
      listFiles("paper"),
      listFiles("code"),
      listFiles("photo"),
    ]);
    setPapers(p.filter((f) => f.name.endsWith(".md")));
    setCodes(c.filter((f) => f.name.endsWith(".py")));
    setImages(im.filter((f) => /\.(png|jpe?g|gif|bmp)$/i.test(f.name)));
  }, []);

  useEffect(() => {
    checkHealth().then(setBackendOk);
    const timer = setInterval(() => checkHealth().then(setBackendOk), 5000);
    refreshWorkspace();
    return () => clearInterval(timer);
  }, [refreshWorkspace]);

  // 运行期间每 3 秒刷新一次工作区文件清单, 实时感知新生成的思路/代码/图片
  useEffect(() => {
    if (status !== "running") return;
    const timer = setInterval(refreshWorkspace, 3000);
    return () => clearInterval(timer);
  }, [status, refreshWorkspace]);

  useEffect(() => {
    logBoxRef.current?.scrollTo({
      top: logBoxRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [logs]);

  // 灯箱打开时: ESC 关闭, ← → 切换上一张/下一张(切换后重置缩放)
  useEffect(() => {
    if (!zoomImage) return;
    const idx = images.findIndex((f) => f.name === zoomImage);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setZoomImage(null);
      } else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && idx >= 0 && images.length > 1) {
        const dir = e.key === "ArrowRight" ? 1 : -1;
        const next = (idx + dir + images.length) % images.length;
        setZoomImage(images[next].name);
        setZoomScale(1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoomImage, images]);

  // 滚轮缩放大图: 原生非被动监听, 否则 preventDefault 会被浏览器忽略
  useEffect(() => {
    if (!zoomImage) return;
    const el = lightboxRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      setZoomScale((s) => Math.min(6, Math.max(0.4, s * factor)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomImage]);

  const handleEvent = useCallback(
    (ev: SseEvent) => {
      if (ev.type === "update" && ev.node) {
        const label = NODE_LABELS[ev.node] ?? ev.node;
        const isSolve = ev.node === "solve_with_method";
        push({ kind: "node", label, method: isSolve ? ev.data?.method : undefined });

        const msgs: any[] = ev.data?.messages ?? [];
        for (const m of msgs.slice(0, 2).reverse()) {
          if (m.role === "tool") {
            push({ kind: "tool", label: `工具:${m.name ?? ""}`, content: m.content || "" });
          } else if (m.role === "ai" && m.content?.trim()) {
            push({
              kind: "ai",
              label: isSolve ? `${ev.data?.method ?? "未知方法"} · 模型输出` : "模型输出",
              content: m.content,
            });
          }
        }
        if (ev.data?.code_files?.length) {
          refreshWorkspace();
        }
        if (ev.data?.run_report) {
          setRunReports(ev.data.run_report);
          refreshWorkspace();
        }
      } else if (ev.type === "interrupt") {
        const label = NODE_LABELS[ev.node ?? ""] ?? ev.node ?? "等待输入";
        setQuestion(ev.value ?? "请输入：");
        push({ kind: "interrupt", label, question: ev.value ?? "" });
        setStatus("awaiting_input");
      } else if (ev.type === "suspended") {
        setStatus("awaiting_input");
      } else if (ev.type === "done") {
        setStatus("done");
        refreshWorkspace();
        fetchState(threadId)
          .then((s) => {
            const sum = s.values?.final_summary || "";
            if (sum) setFinalSummary(sum);
            push({ kind: "info", label: "运行完成", content: "全部节点执行完毕" });
          })
          .catch(() => push({ kind: "error", label: "获取状态失败", content: "请检查后端" }));
      } else if (ev.type === "error") {
        setStatus("error");
        push({ kind: "error", label: "运行出错", content: ev.error ?? "" });
      }
    },
    [push, threadId, refreshWorkspace],
  );

  const run = useCallback(
    async (resume: string | null) => {
      abortRef.current = new AbortController();
      setStatus("running");
      try {
        await startRun(threadId, resume, handleEvent, abortRef.current.signal);
      } catch (e: any) {
        if (e?.name !== "AbortError") {
          setStatus("error");
          push({ kind: "error", label: "连接失败", content: String(e?.message ?? e) });
        }
      } finally {
        abortRef.current = null;
      }
    },
    [threadId, handleEvent, push],
  );

  const onStart = useCallback(() => {
    if (status === "running") return;
    setLogs([]);
    setFinalSummary("");
    setRunReports([]);
    setSelected(null);
    run(null);
  }, [status, run]);

  const onSubmitAnswer = useCallback(() => {
    if (status !== "awaiting_input") return;
    const text = answer.trim();
    push({ kind: "user", label: "人工输入", content: text || "（回车跳过）" });
    setAnswer("");
    setQuestion(null);
    run(text);
  }, [status, answer, run, push]);

  const onReset = useCallback(async () => {
    if (status === "running") {
      abortRef.current?.abort();
    }
    await resetThread(threadId);
    setLogs([]);
    setFinalSummary("");
    setRunReports([]);
    setQuestion(null);
    setAnswer("");
    setSelected(null);
    setStatus("idle");
  }, [status, threadId]);

  const openFile = useCallback(async (dir: string, name: string) => {
    setLoadingFile(true);
    const content = await readFile(dir, name);
    setSelected({ dir, name, content });
    setLoadingFile(false);
  }, []);

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>数学建模 Agent 控制台</h1>
          <span className="thread">会话 {threadId}</span>
        </div>
        <div className="header-actions">
          <div className={`backend ${backendOk === false ? "down" : ""}`}>
            <span className="dot" />
            后端 {backendOk === null ? "检测中" : backendOk ? `在线 (${getBackendUrl()})` : "离线"}
          </div>
          <div className="status-badge" data-status={status}>
            {status === "idle" && "待命"}
            {status === "running" && "运行中…"}
            {status === "awaiting_input" && "等待人工输入"}
            {status === "done" && "已完成"}
            {status === "error" && "出错"}
          </div>
          <button className="btn ghost" onClick={onReset} disabled={status === "idle" && !logs.length}>
            重新开始
          </button>
          <button className="btn primary" onClick={onStart} disabled={status === "running"} title="启动一次完整运行">
            开始运行
          </button>
        </div>
      </header>

      <main className="main">
        <section className="panel log-panel">
          <h2>运行日志</h2>
          <div className="log-box" ref={logBoxRef}>
            {logs.length === 0 && <div className="empty">点击「开始运行」启动 agent 流程</div>}
            {logs.map((item) => (
              <div key={item.id} className={`log-item ${item.kind}`}>
                <span className="log-time">{item.t}</span>
                <span className="log-label">{item.label}</span>
                <pre className="log-content">
                  {"content" in item ? item.content : ""}
                </pre>
                {"question" in item && typeof item.question === "string" ? (
                  <span className="log-question">{item.question}</span>
                ) : null}
              </div>
            ))}
          </div>
        </section>

        <section className="panel result-panel">
          <h2>最终总结</h2>
          <div className="summary-box">
            {finalSummary ? (
              <ReactMarkdown>{finalSummary}</ReactMarkdown>
            ) : (
              <div className="empty">运行结束后在此展示《最终总结.md》内容</div>
            )}
          </div>

          <h2>思路与代码（实时）</h2>
          <div className="ws-box">
            <div className="ws-files">
              <div className="ws-group">
                <span className="ws-group-title">paper/ 思路</span>
                {papers.length === 0 && <span className="ws-none">（暂无）</span>}
                {papers.map((f) => (
                  <button
                    key={f.name}
                    className={`ws-item ${selected?.name === f.name && selected.dir === "paper" ? "active" : ""}`}
                    onClick={() => openFile("paper", f.name)}
                  >
                    {f.name}
                  </button>
                ))}
              </div>
              <div className="ws-group">
                <span className="ws-group-title">code/ 代码</span>
                {codes.length === 0 && <span className="ws-none">（暂无）</span>}
                {codes.map((f) => (
                  <button
                    key={f.name}
                    className={`ws-item ${selected?.name === f.name && selected.dir === "code" ? "active" : ""}`}
                    onClick={() => openFile("code", f.name)}
                  >
                    {f.name}
                  </button>
                ))}
              </div>
            </div>
            <div className="ws-preview">
              {loadingFile && <div className="empty">加载中…</div>}
              {!loadingFile && !selected && <div className="empty">点击左侧文件查看内容</div>}
              {!loadingFile && selected && (
                selected.dir === "code" ? (
                  <pre className="code-preview">{selected.content}</pre>
                ) : (
                  <div className="md-preview">
                    <FoldingMarkdown source={selected.content} />
                  </div>
                )
              )}
            </div>
          </div>

          <h2>生成的图片（实时）</h2>
          <div className="img-box">
            {images.length === 0 && <div className="empty">（暂无）运行到「执行求解代码」阶段后自动生成</div>}
            <div className="img-grid">
              {images.map((f) => (
                <figure key={f.name} className="img-item">
                  <img
                    src={imageUrl("photo", f.name)}
                    alt={f.name}
                    loading="lazy"
                    className="zoomable"
                    title="点击放大"
                    onClick={() => setZoomImage(f.name)}
                  />
                  <figcaption>{f.name}</figcaption>
                </figure>
              ))}
            </div>
          </div>

          <h2>运行报告</h2>
          <div className="report-box">
            {runReports.length === 0 && <div className="empty">暂无代码运行报告</div>}
            {runReports.map((r, i) => (
              <div key={i} className="report-item">
                <b>{r.file}</b>
                <span className={r.status?.includes("失败") || r.status?.includes("超时") ? "bad" : "good"}>
                  {r.status}
                </span>
                {r.output && <pre>stdout: {r.output}</pre>}
                {r.error && <pre className="err">stderr: {r.error}</pre>}
              </div>
            ))}
          </div>
        </section>
      </main>

      {zoomImage && (
        <div
          className="img-lightbox"
          ref={lightboxRef}
          onClick={() => setZoomImage(null)}
        >
          <button
            className="img-lightbox-close"
            title="关闭 (Esc)"
            onClick={(e) => {
              e.stopPropagation();
              setZoomImage(null);
            }}
          >
            ×
          </button>
          <img
            src={imageUrl("photo", zoomImage)}
            alt={zoomImage}
            style={{ transform: `scale(${zoomScale})` }}
            onClick={(e) => e.stopPropagation()}
            onDoubleClick={(e) => {
              e.stopPropagation();
              const el = lightboxRef.current;
              if (!document.fullscreenElement) el?.requestFullscreen?.();
              else document.exitFullscreen?.();
            }}
          />
          <div className="img-lightbox-cap">
            {zoomImage} · 滚轮缩放 / 双击全屏 / ← → 切换 · {Math.round(zoomScale * 100)}%
          </div>
        </div>
      )}

      <footer className="footer">
        {status === "awaiting_input" && question !== null ? (
          <div className="interrupt-bar">
            <div className="interrupt-q">✋ {question}</div>
            <input
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onSubmitAnswer()}
              placeholder="直接回车=跳过，或输入思路/审核意见后回车…"
            />
            <button className="btn primary" onClick={onSubmitAnswer}>
              提交
            </button>
          </div>
        ) : (
          <div className="hint">
            {status === "running" ? "正在运行，日志与工作区文件实时刷新中…" : "按回车提交后继续流程 · 输入「打回」可要求重新分析"}
          </div>
        )}
      </footer>
    </div>
  );
}