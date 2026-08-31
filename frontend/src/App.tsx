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
  fetchModels,
  fetchModelConfig,
  fetchUsage,
  listPdfs,
  pdfUrl,
  saveKeys,
  fetchUiConfig,
  saveUiConfig,
  mediaUrl,
  listMedia,
  uploadMedia,
  thumbUrl,
  transcodedUrl,
  startTranscode,
  transcodeStatus,
  fetchWeWallpapers,
  wePreviewUrl,
  weFileUrl,
  fetchWorkspaces,
  createWorkspace,
  deleteWorkspace,
  activateWorkspace,
  uploadWorkspaceFiles,
  deleteArtifact,
  deleteWsFile,
  DEFAULT_UI,
  FALLBACK_MODELS,
  type SseEvent,
  type WeWallpaper,
  type WorkspaceInfo,
  type WorkspaceFile,
  type ModelOption,
  type UiConfig,
  type UsageStats,
  type PdfInfo,
} from "./api";
import "./App.css";
import bgImage from "./assets/bg-miku.jpg";

/* dsh 同款配色预设（dsh-wallpaper-engine ACCENT_PRESETS / GLASS_COLOR_PRESETS） */
const ACCENT_PRESETS = ["#4f8cff", "#67DCE7", "#DD8FAC", "#F3B75F", "#F1717F", "#CBE77D"];
const GLASS_COLOR_PRESETS = ["#ffffff", "#0d1524", "#67DCE7", "#DD8FAC", "#F3B75F", "#F1717F"];
/* 文字主色预设(""=默认由单独按钮承载);深浅都给了, 配深色玻璃或浅色玻璃都能搭 */
const TEXT_COLOR_PRESETS = ["#16233c", "#0f172a", "#334155", "#f8fafc", "#e2e8f0", "#cbd5e1"];
const IMAGE_EXT = /\.(png|jpe?g|gif|bmp|webp)$/i;
const VIDEO_EXT = /\.(mp4|webm)$/i;
const RATING_LABEL: Record<string, string> = {
  everyone: "全年龄",
  pg13: "PG13",
  mature: "成人",
  unrated: "未分级",
};

const NODE_LABELS: Record<string, string> = {
  load_problem: "读取题目",
  question_structed: "解析问题结构",
  read_dataset: "读取数据集",
  modeling: "建模分析",
  debate_plan: "方案辩论",
  tool_node: "工具执行",
  review_modeling_analysis: "建模审核",
  send_problem_index: "分发问题",
  collect_branches: "等待分支汇合",
  solve_with_method: "方法求解",
  run_solutions: "执行求解代码",
  compare_summarize: "汇总对比",
  compare_tool_node: "汇总工具",
  final_analysis: "最终审查",
  feedback_check: "建模质检",
  write_article: "论文撰写",
  fill_document_meta: "封面元信息",
};

/* 设计稿 6 步工作流：由后端真实节点事件归并驱动（纯前端展示映射） */
const STEPS = [
  { label: "问题理解", hint: "analyzing…" },
  { label: "假设与符号", hint: "建模分析进行中" },
  { label: "模型建立", hint: "构建与审核方程" },
  { label: "求解计算", hint: "算法实现与数值求解" },
  { label: "验证与灵敏度", hint: "汇总对比与检验" },
  { label: "论文撰写", hint: "最终审查与成文" },
];

/* 后端节点 → 设计稿步骤（1-6）；未列出的节点保持当前步骤不变。
   按真实图结构归并：质检打回时 modeling 事件会把步骤拉回 2（允许回退），
   tool_node 只服务 modeling 的工具循环故归步骤 2，compare 有独立节点归 5 */
const NODE_TO_STEP: Record<string, number> = {
  load_problem: 1,
  question_structed: 1,
  read_dataset: 1,
  modeling: 2,
  debate_plan: 3,
  tool_node: 2,
  review_modeling_analysis: 3,
  send_problem_index: 4,
  solve_with_method: 4,
  feedback_check: 4,
  collect_branches: 4,
  run_solutions: 4,
  compare_summarize: 5,
  compare_tool_node: 5,
  final_analysis: 6,
  write_article: 6,
  fill_document_meta: 6,
};

const fmtElapsed = (sec: number) => {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return h > 0
    ? `${h}h ${m}m ${s}s`
    : m > 0
      ? `${m}m ${s}s`
      : `${s}s`;
};

/** token 数缩写：48231 → 48.2k */
const fmtTokens = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n));

/** 文件大小缩写：1234567 → 1.2MB */
const fmtSize = (n: number) =>
  n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)}MB` : `${Math.max(1, Math.round(n / 1024))}KB`;

type LogItem =
  | { id: number; t: string; kind: "node"; label: string; method?: string }
  | { id: number; t: string; kind: "tool"; label: string; content: string }
  | { id: number; t: string; kind: "ai"; label: string; content: string }
  | { id: number; t: string; kind: "thinking"; label: string; content: string }
  | { id: number; t: string; kind: "ai_stream"; label: string; content: string }
  | { id: number; t: string; kind: "interrupt"; label: string; question: string }
  | { id: number; t: string; kind: "user"; label: string; content: string }
  | { id: number; t: string; kind: "info"; label: string; content: string }
  | { id: number; t: string; kind: "log"; label: string; content: string }
  | { id: number; t: string; kind: "error"; label: string; content: string };

type Status = "idle" | "running" | "awaiting_input" | "done" | "error";

/** 每题一个状态桶:切换题目只是换视图,各题的运行与日志互不干扰 */
interface WsState {
  logs: LogItem[];
  status: Status;
  question: string | null;   // 当前待回答的提问(卡片+输入占位);暂停提示也走这里
  step: number;              // 工作流步骤(0=未开始/1-6)
  finalSummary: string;
  runReports: any[];
  runStart: number | null;   // 本轮运行起点(算 elapsed 用;续跑保留旧值=计时延续)
}

const EMPTY_WS: WsState = {
  logs: [], status: "idle", question: null, step: 0,
  finalSummary: "", runReports: [], runStart: null,
};

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
  // 题目工作区：题目 id 即 thread_id（多题隔离，切换题目 = 切换线程状态与产物目录）
  const [currentWs, setCurrentWs] = useState("default");
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [showWsModal, setShowWsModal] = useState(false);
  const [newWsTitle, setNewWsTitle] = useState("");
  const [wsUploading, setWsUploading] = useState(false);
  const wsQRef = useRef<HTMLInputElement | null>(null);  // 题目文件
  const wsDRef = useRef<HTMLInputElement | null>(null);  // 数据文件
  const [wsDdOpen, setWsDdOpen] = useState(false);       // 顶栏题目下拉(自绘, 行内可删题)
  const wsDdRef = useRef<HTMLDivElement | null>(null);
  // —— 多题并行:每题一个状态桶,切换题目只是换视图,绝不掐断运行 ——
  // 已知限制(留档): a) 并行两题选不同模型时,后启动运行的 set_model 会影响先启动题的后续调用
  // (后端全局模型实例为既有机制,改动需动 src/agent.py); b) 日志在前端内存,刷新页面会丢。
  const [wsMap, setWsMap] = useState<Record<string, WsState>>({});
  const wsMapRef = useRef<Record<string, WsState>>({});
  wsMapRef.current = wsMap;
  const [answer, setAnswer] = useState("");
  const [backendOk, setBackendOk] = useState<boolean | null>(null);
  // Token 消耗/缓存命中统计（/api/usage 轮询；后端旧版本无该接口时保持 null，界面自动隐藏该行）
  const [usage, setUsage] = useState<UsageStats | null>(null);
  // PDF 产物列表与当前预览
  const [pdfs, setPdfs] = useState<PdfInfo[]>([]);
  const [viewPdf, setViewPdf] = useState<PdfInfo | null>(null);
  const [papers, setPapers] = useState<WorkspaceFile[]>([]);
  const [codes, setCodes] = useState<WorkspaceFile[]>([]);
  const [images, setImages] = useState<WorkspaceFile[]>([]);
  // 模型选择：下拉框选项 / 当前选择 / 已保存的 key / 弹窗与保存状态
  const [models, setModels] = useState<ModelOption[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>("deepseek");
  const [savedKeys, setSavedKeys] = useState<Record<string, string>>({});
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [modalKeys, setModalKeys] = useState<Record<string, string>>({});
  const [savingKeys, setSavingKeys] = useState(false);
  // 点击图片放大时记录文件名(show in 灯箱)，null 表示未放大
  const [zoomImage, setZoomImage] = useState<string | null>(null);
  // 灯箱内大图的缩放倍率(滚轮控制)
  const [zoomScale, setZoomScale] = useState(1);
  // 灯箱根节点 ref，用于绑定原生非被动 wheel 监听和全屏
  const lightboxRef = useRef<HTMLDivElement | null>(null);
  const [selected, setSelected] = useState<{ dir: string; name: string; content: string } | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);
  // 液态玻璃外观设置（dsh 同款）：配置来自 ui_config.json，实时预览，保存时 PUT
  const [uiTheme, setUiTheme] = useState<UiConfig>(DEFAULT_UI);
  const [showAppearance, setShowAppearance] = useState(false);
  const [mediaFiles, setMediaFiles] = useState<WorkspaceFile[]>([]);
  const [carouselIdx, setCarouselIdx] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // 素材库弹窗（dsh 壁纸选择器同款）
  const [showMediaLib, setShowMediaLib] = useState(false);
  const [mediaSearch, setMediaSearch] = useState("");
  const [showHidden, setShowHidden] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [transcodedReady, setTranscodedReady] = useState(false);
  const fileRef = useRef<HTMLInputElement | null>(null);
  // WE 壁纸库（dsh 同款）
  const [weWallpapers, setWeWallpapers] = useState<WeWallpaper[]>([]);
  const [weLibTab, setWeLibTab] = useState<"local" | "we">("local");
  const [weSearch, setWeSearch] = useState("");
  // 每题独立的运行控制器/运行代数/暂停标记(切题互不影响, 后台题继续跑)
  const abortMapRef = useRef<Record<string, AbortController | null>>({});
  const epochMapRef = useRef<Record<string, number>>({});
  const pausedMapRef = useRef<Record<string, boolean>>({});
  const logBoxRef = useRef<HTMLDivElement>(null);
  const currentWsRef = useRef(currentWs);
  currentWsRef.current = currentWs;
  const [elapsed, setElapsed] = useState(0);

  // 当前题的状态桶——界面全部读这里的派生值
  const cur: WsState = wsMap[currentWs] ?? EMPTY_WS;
  const { logs, status, question, step, finalSummary, runReports } = cur;
  // 可拖拽分栏：左右栏列宽由分隔条拖动实时调整，中栏自适应剩余空间
  const [leftWidth, setLeftWidth] = useState(232);
  const [rightWidth, setRightWidth] = useState(312);
  const splitDragRef = useRef<{
    side: "left" | "right";
    startX: number;
    startWidth: number;
  } | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const papersSecRef = useRef<HTMLDivElement | null>(null);
  const imagesSecRef = useRef<HTMLDivElement | null>(null);

  // 文档弹窗尺寸: null=走 CSS 默认;拖过之后 inline 生效并记忆(localStorage), 文件/PDF 两个弹窗共用
  const [docSize, setDocSize] = useState<{ w: number; h: number } | null>(() => {
    try {
      const v = JSON.parse(localStorage.getItem("doc-modal-size") || "null");
      // 恢复值同样钳制合法区间, 防止手改存储后弹出异常尺寸
      if (v && typeof v.w === "number" && typeof v.h === "number") {
        return {
          w: Math.min(window.innerWidth * 0.94, Math.max(420, v.w)),
          h: Math.min(window.innerHeight * 0.96, Math.max(320, v.h)),
        };
      }
    } catch { /* 存储损坏则忽略, 回落 CSS 默认 */ }
    return null;
  });

  // 弹窗右下角拖拽调大小: 宽高独立生效, 范围钳在 420x320 ~ 94vw/96vh, 松手写入 localStorage
  const startDocResize = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const el = (e.currentTarget as HTMLElement).closest(".modal") as HTMLElement | null;
    const drag = { x: e.clientX, y: e.clientY, w: el?.offsetWidth ?? 900, h: el?.offsetHeight ?? 700 };
    const clamp = (w: number, h: number) => ({
      w: Math.min(window.innerWidth * 0.94, Math.max(420, w)),
      h: Math.min(window.innerHeight * 0.96, Math.max(320, h)),
    });
    let latest = clamp(drag.w, drag.h);
    const onMove = (ev: MouseEvent) => {
      latest = clamp(drag.w + (ev.clientX - drag.x), drag.h + (ev.clientY - drag.y));
      setDocSize(latest);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.classList.remove("doc-resizing");
      try { localStorage.setItem("doc-modal-size", JSON.stringify(latest)); } catch { /* 忽略 */ }
    };
    document.body.classList.add("doc-resizing");
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  // 会话计时:当前题运行/等待输入期间每秒刷新(runStart 存桶内,切回运行中的题能对上真实时长)
  useEffect(() => {
    if (status !== "running" && status !== "awaiting_input") return;
    const tick = () => {
      const rs = wsMapRef.current[currentWs]?.runStart;
      setElapsed(rs != null ? Math.floor((Date.now() - rs) / 1000) : 0);
    };
    tick();
    const timer = setInterval(tick, 1000);
    return () => clearInterval(timer);
  }, [status, currentWs]);

  // interrupt 挂起时自动聚焦底部输入框(切到正在等输入的题也会触发)
  useEffect(() => {
    if (status === "awaiting_input") chatInputRef.current?.focus();
  }, [status, question, currentWs]);

  // 自绘题目下拉: 点击面板外自动收起
  useEffect(() => {
    if (!wsDdOpen) return;
    const onDoc = (e: MouseEvent) => {
      if (wsDdRef.current && !wsDdRef.current.contains(e.target as Node)) setWsDdOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [wsDdOpen]);

  const patchWs = useCallback((id: string, patch: Partial<WsState>) => {
    setWsMap((prev) => ({ ...prev, [id]: { ...(prev[id] ?? EMPTY_WS), ...patch } }));
  }, []);
  // 向指定题的聊天桶追加一条(后台运行题的事件写入它自己的桶,不打扰当前视图)
  const push = useCallback((id: string, item: DistributiveOmit<LogItem, "id" | "t">) => {
    setWsMap((prev) => {
      const s = prev[id] ?? EMPTY_WS;
      return { ...prev, [id]: { ...s, logs: [...s.logs, { ...item, id: ++seq, t: now() }] } };
    });
  }, []);

  // 流式打字: 按 (LLM消息id mid + 节点) 分组追加"思考过程/模型输出"增量;
  // mid 每次新 LLM 调用必然变化, 分组天然自终, 无需在节点 update 时清空
  // (清空反而会拆散并行 Send 分支下仍在流式的其他分支的同一条消息)
  const streamMapRef = useRef<Record<string, { mid: string; node: string; thinkId: number | null; textId: number | null } | null>>({});
  // 最近一次正文流式所属节点: 节点 update 到达时若匹配则跳过 400 字截断摘要(避免重复)
  const streamedNodeMapRef = useRef<Record<string, string | null>>({});
  // 最近一个已实时宣告(推过节点行)的节点: 流式首个增量即宣告节点行+步骤, 不等节点结束;
  // update 到达时对比去重(并行 Send 分支的多条同名 update 也靠它只推一行) —— 均按题键控
  const announcedMapRef = useRef<Record<string, string | null>>({});

  const appendStream = useCallback((id: string, ev: SseEvent, kind: "thinking" | "ai_stream") => {
    const text = ev.text ?? "";
    if (!text) return;
    const mid = ev.mid ?? "";
    const node = ev.node ?? "";
    let sr = streamMapRef.current[id];
    if (!sr || sr.mid !== mid || sr.node !== node) {
      sr = streamMapRef.current[id] = { mid, node, thinkId: null, textId: null };
    }
    const idKey = kind === "thinking" ? "thinkId" : "textId";
    const existing = sr[idKey];
    if (existing == null) {
      // 本节点首个增量: 先宣告节点行(气泡出现在节点名之下, 时序正确)
      if (node && announcedMapRef.current[id] !== node) {
        announcedMapRef.current[id] = node;
        push(id, { kind: "node", label: NODE_LABELS[node] ?? node });
        const st = NODE_TO_STEP[node];
        if (st) patchWs(id, { step: st });
      }
      const nid = ++seq;
      sr[idKey] = nid;
      if (kind === "ai_stream") streamedNodeMapRef.current[id] = node;
      const label = kind === "thinking" ? "思考过程" : (node && NODE_LABELS[node]) || "模型输出";
      setWsMap((prev) => {
        const s = prev[id] ?? EMPTY_WS;
        return { ...prev, [id]: { ...s, logs: [...s.logs, { id: nid, t: now(), kind, label, content: text }] } };
      });
    } else {
      setWsMap((prev) => {
        const s = prev[id] ?? EMPTY_WS;
        return {
          ...prev,
          [id]: {
            ...s,
            logs: s.logs.map((it) =>
              it.id === existing && (it.kind === "thinking" || it.kind === "ai_stream")
                ? { ...it, content: (it.content + text).slice(-30000) }
                : it,
            ),
          },
        };
      });
    }
  }, [push, patchWs]);

  // 产物清单:按题查询(thread_id 参数)——多题并行下全局兜底指针会被后启动的运行占用,不传会串台
  const refreshWorkspace = useCallback(async (wsId?: string) => {
    const id = wsId ?? currentWsRef.current;
    const [p, c, im] = await Promise.all([
      listFiles("paper", id),
      listFiles("code", id),
      listFiles("photo", id),
    ]);
    setPapers(p.filter((f) => f.name.endsWith(".md")));
    setCodes(c.filter((f) => f.name.endsWith(".py")));
    setImages(im.filter((f) => /\.(png|jpe?g|gif|bmp)$/i.test(f.name)));
    // 运行状态为空时回退读 paper/最终总结.md（模拟工作区 / 中断恢复场景）
    // 运行中不回退: 避免旧文件内容中途回显, 新总结落定前保持占位
    const s = wsMapRef.current[id] ?? EMPTY_WS;
    if (!s.finalSummary && s.status !== "running") {
      readFile("paper", "最终总结.md", id)
        .then((c) => {
          if (c && c.trim()) patchWs(id, { finalSummary: c.trim() });
        })
        .catch(() => {});
    }
  }, [patchWs]);

  // 仅当该题正被查看时才刷新产物面板(后台题的产物留给切回时的 refreshWorkspace)
  const refreshArtifactsIfVisible = useCallback((id: string) => {
    if (id === currentWsRef.current) refreshWorkspace(id);
  }, [refreshWorkspace]);

  useEffect(() => {
    checkHealth().then(setBackendOk);
    const timer = setInterval(() => checkHealth().then(setBackendOk), 5000);
    refreshWorkspace();
    return () => clearInterval(timer);
  }, [refreshWorkspace]);

  // Token/缓存命中: 3 秒轮询（与产物刷新同节奏），重置/新运行由后端清零；PDF 清单按题刷新
  useEffect(() => {
    fetchUsage().then(setUsage);
    listPdfs(currentWsRef.current).then(setPdfs);
    const timer = setInterval(() => {
      fetchUsage().then(setUsage);
      listPdfs(currentWsRef.current).then(setPdfs);
    }, 3000);
    return () => clearInterval(timer);
  }, [currentWs]);

  // 启动时拉取可选模型与已保存的默认模型（配置一次一直可用）
  // 若后端不可用，回退到内置模型清单，保证下拉框与 API Key 输入永不卡死
  useEffect(() => {
    let cancelled = false;
    setModelsLoading(true);
    fetchModels()
      .then((ms) => { if (!cancelled) setModels(ms.length ? ms : FALLBACK_MODELS); })
      .catch(() => { if (!cancelled) setModels(FALLBACK_MODELS); })
      .finally(() => { if (!cancelled) setModelsLoading(false); });
    fetchModelConfig().then((cfg) => {
      setSelectedModel(cfg.model);
      setSavedKeys(cfg.keys ?? {});
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // 启动时加载外观配置（液态玻璃主题 + 背景模式）+ 背景素材清单；失败静默用默认值
  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchUiConfig(), listMedia(), fetchWeWallpapers()])
      .then(([cfg, files, we]) => {
        if (cancelled) return;
        setUiTheme(cfg);
        setMediaFiles(files);
        setWeWallpapers(we);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // uiTheme 变化 → 写入 CSS 变量（dsh 同款联动：saturate = 1.15 + blur*0.028，模糊越大颜色融化越强）
  useEffect(() => {
    const s = document.documentElement.style;
    const blur = uiTheme.glassWindow ? uiTheme.blur : 0; // 玻璃总开关：off → 取消毛玻璃
    s.setProperty("--we-accent", uiTheme.accent);
    // 文字主色: 用户选了色则全局覆盖 --text, 清空(默认)则回落 :root 主题值
    if (uiTheme.textColor) s.setProperty("--text", uiTheme.textColor);
    else s.removeProperty("--text");
    s.setProperty("--we-glass-alpha", String(uiTheme.glassAlpha / 100));
    s.setProperty("--we-glass-color", uiTheme.glassColor);
    s.setProperty("--we-blur", `${blur}px`);
    s.setProperty("--we-saturate", String(1.15 + blur * 0.028));
    s.setProperty("--we-scrim-color", `rgba(0,0,0,${uiTheme.scrim})`);
    s.setProperty("--we-border-alpha", String(uiTheme.border));
    s.setProperty("--we-wallpaper-blur", `${uiTheme.wallpaperBlur}px`);
    s.setProperty(
      "--we-media-filter",
      uiTheme.wallpaperBlur > 0 ? `blur(${uiTheme.wallpaperBlur}px)` : "none",
    );
    s.setProperty("--we-wallpaper-scale", (1 + uiTheme.wallpaperBlur * 0.006).toFixed(4));
    s.setProperty("--we-wallpaper-flip", uiTheme.flip ? "-1" : "1");
    s.setProperty("--we-object-fit", uiTheme.objectFit);
  }, [uiTheme]);

  // 遮挡暂停（dsh 同款）：页面隐藏 / 窗口失焦 / 电池供电时暂停视频壁纸，恢复自动继续
  useEffect(() => {
    const v = () => videoRef.current;
    const onVis = () => {
      if (document.hidden && uiTheme.pauseOnHidden) v()?.pause();
      else if (!document.hidden) v()?.play().catch(() => {});
    };
    const onBlur = () => { if (uiTheme.pauseOnBlur) v()?.pause(); };
    const onFocus = () => { if (uiTheme.pauseOnBlur) v()?.play().catch(() => {}); };
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("blur", onBlur);
    window.addEventListener("focus", onFocus);
    let batt: any = null;
    let onCharge = () => {};
    if (uiTheme.pauseOnBattery && (navigator as any).getBattery) {
      (navigator as any).getBattery().then((b: any) => {
        batt = b;
        onCharge = () => { if (batt && !batt.charging) v()?.pause(); else v()?.play().catch(() => {}); };
        b.addEventListener("chargingchange", onCharge);
        if (!b.charging) v()?.pause();
      }).catch(() => {});
    }
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("blur", onBlur);
      window.removeEventListener("focus", onFocus);
      batt?.removeEventListener?.("chargingchange", onCharge);
    };
  }, [uiTheme.pauseOnHidden, uiTheme.pauseOnBlur, uiTheme.pauseOnBattery]);

  // 视频壁纸低帧率转码（dsh 同款 fpsCap 思路）：触发 + 轮询，ready 后切转码版降 GPU
  useEffect(() => {
    if (uiTheme.bgSource === "we" || uiTheme.bgMode !== "video" || !VIDEO_EXT.test(uiTheme.bgFile)) return;
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;
    startTranscode(uiTheme.bgFile, 24)
      .then((st) => {
        if (cancelled || st.phase !== "working") return;
        timer = setInterval(async () => {
          if (cancelled) return;
          const s = await transcodeStatus(uiTheme.bgFile);
          if (s.phase === "ready" || s.phase === "error" || s.phase === "skipped") {
            if (timer) clearInterval(timer);
            if (s.phase === "ready") setTranscodedReady(true);
          }
        }, 2000);
      })
      .catch(() => {});
    return () => { cancelled = true; if (timer) clearInterval(timer); setTranscodedReady(false); };
  }, [uiTheme.bgMode, uiTheme.bgFile]);

  // 轮播：bgMode=carousel 且可见图片素材 ≥2 时按间隔切换（shuffle 时随机跳转）
  useEffect(() => {
    const imgs = mediaFiles.filter((f) => IMAGE_EXT.test(f.name) && !uiTheme.hiddenIds.includes(f.name));
    if (uiTheme.bgMode !== "carousel" || imgs.length < 2) return;
    const timer = setInterval(() => {
      if (uiTheme.shuffle) {
        setCarouselIdx(Math.floor(Math.random() * imgs.length));
      } else {
        setCarouselIdx((i) => (i + 1) % imgs.length);
      }
    }, Math.max(3, uiTheme.carouselSecs) * 1000);
    return () => clearInterval(timer);
  }, [uiTheme.bgMode, uiTheme.carouselSecs, uiTheme.shuffle, uiTheme.hiddenIds, mediaFiles]);

  // 打开素材库时刷新 WE 壁纸清单（后端晚于页面就绪也能加载，不必刷新页面）
  useEffect(() => {
    if (!showMediaLib) return;
    fetchWeWallpapers().then(setWeWallpapers).catch(() => {});
  }, [showMediaLib]);

  // 启动时拉取题目列表并激活当前题（题目 id = thread_id）
  useEffect(() => {
    let cancelled = false;
    fetchWorkspaces().then(({ workspaces: wsList, current }) => {
      if (cancelled) return;
      setWorkspaces(wsList);
      const target = current || wsList[0]?.id || "default";
      setCurrentWs(target);
      activateWorkspace(target).catch(() => {});
    }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // —— 题目管理：切换/新建/删除/上传（每题的题目、数据、产物、线程状态互相隔离）——
  const switchWs = useCallback(async (id: string) => {
    if (id === currentWs) return;
    // 多题并行:切换只换视图,绝不掐断旧题的 SSE——旧运行继续跑,事件继续写它自己的桶;
    // 切回时历史与实时进度都在(产物面板由 refreshWorkspace 按题刷新)
    setCurrentWs(id);
    setAnswer("");
    // 预览类浮层里嵌着按当前题取的资源(thread_id),切题不关会静默串成新题的同名文件
    setSelected(null);
    setZoomImage(null);
    setViewPdf(null);
    setElapsed(0); // 若目标题在跑,计时 effect 会立即按它的 runStart 校正
    try { await activateWorkspace(id); } catch { /* 后端离线时本地先切 */ }
    refreshWorkspace(id);
  }, [currentWs, refreshWorkspace]);

  const onCreateWs = useCallback(async () => {
    const title = newWsTitle.trim();
    if (!title) return;
    try {
      const ws = await createWorkspace(title);
      setNewWsTitle("");
      setWorkspaces((prev) => [ws, ...prev.filter((w) => w.id !== ws.id)]);
      await switchWs(ws.id);
      setShowWsModal(false);
      push(ws.id, { kind: "info", label: "题目已创建", content: `${ws.title} (${ws.id})` });
    } catch (e: any) {
      push(currentWsRef.current, { kind: "error", label: "新建失败", content: String(e?.message ?? e) });
    }
  }, [newWsTitle, switchWs, push]);

  const onDeleteWs = useCallback(async (id: string) => {
    const title = workspaces.find((w) => w.id === id)?.title ?? id;
    if (!window.confirm(`确定删除题目「${title}」？\n将连带删除该题全部产物与运行状态，不可恢复！`)) return;
    try {
      await deleteWorkspace(id);
      // 清掉该题的前端状态桶与运行控制器(后端 409 保护:运行中的题删不掉,不会走到这里)
      abortMapRef.current[id]?.abort();
      delete abortMapRef.current[id];
      delete epochMapRef.current[id];
      delete pausedMapRef.current[id];
      delete streamMapRef.current[id];
      delete streamedNodeMapRef.current[id];
      delete announcedMapRef.current[id];
      setWsMap((prev) => {
        if (!(id in prev)) return prev;
        const { [id]: _gone, ...rest } = prev;
        return rest;
      });
      setWorkspaces((prev) => prev.filter((w) => w.id !== id));
      let bubbleTarget = currentWs;
      if (id === currentWs) {
        const next = workspaces.find((w) => w.id !== id);
        bubbleTarget = next?.id ?? "default";
        await switchWs(bubbleTarget);
      }
      push(bubbleTarget, { kind: "info", label: "题目已删除", content: title });
    } catch (e: any) {
      push(currentWsRef.current, { kind: "error", label: "删除失败", content: String(e?.message ?? e) });
    }
  }, [workspaces, currentWs, switchWs, push]);

  const onWsUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>, target: "question" | "dataset") => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setWsUploading(true);
    try {
      const info = await uploadWorkspaceFiles(currentWs, target, files);
      setWorkspaces((prev) => prev.map((w) => (w.id === currentWs ? info : w)));
      refreshWorkspace(currentWs);
      push(currentWs, { kind: "info", label: "上传成功", content: `${files.length} 个${target === "question" ? "题目" : "数据"}文件` });
    } catch (err: any) {
      push(currentWs, { kind: "error", label: "上传失败", content: String(err?.message ?? err) });
    } finally {
      setWsUploading(false);
      e.target.value = "";
    }
  }, [currentWs, push, refreshWorkspace]);

  const curWsInfo = workspaces.find((w) => w.id === currentWs);

  const setTheme = useCallback((patch: Partial<UiConfig>) => {
    setUiTheme((t) => ({ ...t, ...patch }));
  }, []);

  const onSaveAppearance = useCallback(async () => {
    try {
      const merged = await saveUiConfig(uiTheme);
      setUiTheme(merged);
      setShowAppearance(false);
      push(currentWsRef.current, { kind: "info", label: "外观已保存", content: "液态玻璃设置已持久化到 ui_config.json" });
    } catch (e: any) {
      push(currentWsRef.current, { kind: "error", label: "保存外观失败", content: String(e?.message ?? e) });
    }
  }, [uiTheme, push]);

  const onResetAppearance = useCallback(() => {
    setUiTheme(DEFAULT_UI);
    saveUiConfig(DEFAULT_UI).catch(() => {});
  }, []);

  // —— 素材库（dsh 壁纸选择器同款：类型过滤 + 隐藏/恢复 + 搜索）——
  const visibleMedia = mediaFiles.filter((f) => {
    if (!showHidden && uiTheme.hiddenIds.includes(f.name)) return false;
    if (uiTheme.typeFilter === "image" && !IMAGE_EXT.test(f.name)) return false;
    if (uiTheme.typeFilter === "video" && !VIDEO_EXT.test(f.name)) return false;
    if (mediaSearch && !f.name.toLowerCase().includes(mediaSearch.toLowerCase())) return false;
    return true;
  });

  const toggleHidden = useCallback((name: string) => {
    setUiTheme((t) => {
      const cur = new Set(t.hiddenIds);
      if (cur.has(name)) cur.delete(name);
      else cur.add(name);
      return { ...t, hiddenIds: [...cur] };
    });
  }, []);

  const onUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const r = await uploadMedia(file);
      setMediaFiles(await listMedia());
      push(currentWsRef.current, { kind: "info", label: "上传成功", content: r.name });
    } catch (err: any) {
      push(currentWsRef.current, { kind: "error", label: "上传失败", content: String(err?.message ?? err) });
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }, [push]);

  // WE 壁纸库过滤（分级 + 搜索）
  const weVisible = weWallpapers.filter((w) => {
    const f = uiTheme.contentRatingFilter;
    if (f === "everyone" && w.rating !== "everyone") return false;
    if (f === "pg13" && w.rating !== "pg13" && w.rating !== "mature") return false;
    if (f === "mature" && w.rating !== "mature") return false;
    if (f === "unrated" && w.rating !== "unrated") return false;
    if (weSearch && !w.title.toLowerCase().includes(weSearch.toLowerCase())) return false;
    return true;
  });
  const setWeBg = useCallback((w: WeWallpaper) => {
    setTheme({ bgSource: "we", weId: w.id, bgMode: w.type === "video" ? "video" : "image" });
  }, [setTheme]);

  // 运行期间每 3 秒刷新一次工作区文件清单, 实时感知新生成的思路/代码/图片(按当前查看的题)
  useEffect(() => {
    if (status !== "running") return;
    const timer = setInterval(() => refreshWorkspace(currentWsRef.current), 3000);
    return () => clearInterval(timer);
  }, [status, currentWs, refreshWorkspace]);

  // 自动滚动: 流式打字每秒触发多次, rAF 合帧 + 仅在已接近底部时跟随, 不打断用户回看
  const scrollPendRef = useRef(false);
  useEffect(() => {
    if (scrollPendRef.current) return;
    scrollPendRef.current = true;
    requestAnimationFrame(() => {
      scrollPendRef.current = false;
      const el = logBoxRef.current;
      if (!el) return;
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 160) {
        el.scrollTo({ top: el.scrollHeight });
      }
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

  // 文件预览弹窗: ESC 关闭
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

  // PDF 预览弹窗: ESC 关闭
  useEffect(() => {
    if (!viewPdf) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setViewPdf(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [viewPdf]);

  // 当前选中模型是否已具备可用的 key：统一以 .env（进程环境）是否就绪为准
  const currentModel = models.find((m) => m.key === selectedModel);
  const currentEnv = currentModel?.api_key_env;
  const keyReady = Boolean(currentEnv && currentModel?.key_set);
  const missingCount = models.filter(
    (m) => m.api_key_env && !(m.key_set || savedKeys[m.api_key_env]),
  ).length;
  // DeepSeek Key 为全局必需:问题提取/质检等结构化步骤固定走 DeepSeek 非思考模式,
  // 不随上方模型选择变化,缺失会导致这些步骤降级
  const deepseekMissing = !(
    models.find((m) => m.key === "deepseek")?.key_set || savedKeys["DEEPSEEK_API_KEY"]
  );

  const handleEvent = useCallback(
    (id: string, ev: SseEvent) => {
      if (ev.type === "update" && ev.node) {
        const label = NODE_LABELS[ev.node] ?? ev.node;
        const isSolve = ev.node === "solve_with_method";
        // 流式期间已实时宣告过该节点则不重复推节点行; solve_with_method 每方法一个
        // 并行 Send 分支(多条同名 update), 也靠这个对比只保留一行
        if (announcedMapRef.current[id] !== ev.node) {
          announcedMapRef.current[id] = ev.node;
          push(id, { kind: "node", label, method: isSolve ? ev.data?.method : undefined });
        }
        const mapped = NODE_TO_STEP[ev.node];
        if (mapped) patchWs(id, { step: mapped }); // 直接赋值：SSE 事件有序，质检/审核打回时步骤可回退

        // 该节点正文已流式上屏 → 完整文本已在屏上, 跳过 400 字截断摘要避免重复。
        // 并行分支下可能被先完成的同名分支消费掉, 最坏情况多一条截断摘要, 可接受
        const hasStreamedText = streamedNodeMapRef.current[id] != null && streamedNodeMapRef.current[id] === ev.node;
        if (hasStreamedText) streamedNodeMapRef.current[id] = null;
        const msgs: any[] = ev.data?.messages ?? [];
        for (const m of msgs.slice(0, 2).reverse()) {
          if (m.role === "tool") {
            push(id, { kind: "tool", label: `工具:${m.name ?? ""}`, content: m.content || "" });
          } else if (m.role === "ai" && m.content?.trim() && !hasStreamedText) {
            push(id, {
              kind: "ai",
              label: isSolve ? `${ev.data?.method ?? "未知方法"} · 模型输出` : "模型输出",
              content: m.content,
            });
          }
        }
        if (ev.data?.code_files?.length) {
          refreshArtifactsIfVisible(id);
        }
        if (ev.data?.run_report) {
          patchWs(id, { runReports: ev.data.run_report });
          refreshArtifactsIfVisible(id);
        }
      } else if (ev.type === "reasoning") {
        appendStream(id, ev, "thinking");
      } else if (ev.type === "token") {
        appendStream(id, ev, "ai_stream");
      } else if (ev.type === "interrupt") {
        streamMapRef.current[id] = null; streamedNodeMapRef.current[id] = null;
        const label = NODE_LABELS[ev.node ?? ""] ?? ev.node ?? "等待输入";
        patchWs(id, { question: ev.value ?? "请输入：", status: "awaiting_input" });
        push(id, { kind: "interrupt", label, question: ev.value ?? "" });
      } else if (ev.type === "suspended") {
        streamMapRef.current[id] = null; streamedNodeMapRef.current[id] = null;
        patchWs(id, { status: "awaiting_input" });
      } else if (ev.type === "done") {
        streamMapRef.current[id] = null; streamedNodeMapRef.current[id] = null;
        patchWs(id, { status: "done" });
        refreshArtifactsIfVisible(id);
        fetchState(id)
          .then((s) => {
            const sum = s.values?.final_summary || "";
            if (sum) patchWs(id, { finalSummary: sum });
            push(id, { kind: "info", label: "运行完成", content: "全部节点执行完毕" });
          })
          .catch(() => push(id, { kind: "error", label: "获取状态失败", content: "请检查后端" }));
      } else if (ev.type === "log") {
        // 后端桥接的"LLM调用 → 模型[角色] · 用途"实时日志
        push(id, { kind: "log", label: "模型调用", content: ev.text ?? "" });
      } else if (ev.type === "error") {
        streamMapRef.current[id] = null; streamedNodeMapRef.current[id] = null;
        patchWs(id, { status: "error" });
        push(id, { kind: "error", label: "运行出错", content: ev.error ?? "" });
      }
    },
    [push, appendStream, patchWs, refreshArtifactsIfVisible],
  );

  const run = useCallback(
    async (id: string, resume: string | null, model?: string, apiKey?: string, continueRun?: boolean) => {
      const controller = new AbortController();
      abortMapRef.current[id] = controller;
      const myEpoch = (epochMapRef.current[id] ?? 0) + 1;
      epochMapRef.current[id] = myEpoch;
      streamMapRef.current[id] = null; streamedNodeMapRef.current[id] = null; // 每次运行(含续跑)都从干净的流式分组开始
      announcedMapRef.current[id] = null;
      patchWs(id, { status: "running" });
      try {
        await startRun(
          id,
          resume,
          (ev) => { if (epochMapRef.current[id] === myEpoch) handleEvent(id, ev); },
          controller.signal,
          model,
          apiKey,
          continueRun,
        );
      } catch (e: any) {
        if (e?.name !== "AbortError" && epochMapRef.current[id] === myEpoch) {
          patchWs(id, { status: "error" });
          push(id, { kind: "error", label: "连接失败", content: String(e?.message ?? e) });
        }
      } finally {
        // 仅当仍是自己的 controller 时才清空，防止旧运行的 finally 覆盖新运行的控制器
        if (abortMapRef.current[id] === controller) abortMapRef.current[id] = null;
      }
    },
    [patchWs, handleEvent, push],
  );

  const onStart = useCallback(() => {
    if (wsMapRef.current[currentWs]?.status === "running") return;
    pausedMapRef.current[currentWs] = false;
    setSelected(null);
    setElapsed(0);
    // 乐观置第 1 步，首个节点事件到达后按真实节点校正;runStart 从本轮重计
    patchWs(currentWs, { logs: [], finalSummary: "", runReports: [], question: null, step: 1, runStart: Date.now() });
    // 配置统一来源于 .env，环境已就绪则无需前端下发 key，后端自读；否则兜底用已保存值
    const runKey = currentModel?.key_set ? undefined : (savedKeys[currentEnv ?? ""] || undefined);
    run(currentWs, null, selectedModel, runKey);
  }, [currentWs, run, selectedModel, currentEnv, currentModel, savedKeys, patchWs]);

  // 设计稿「暂停」：中断 SSE 流（后端取消图，状态停在最近 checkpoint）。
  // 「下一步」走 continue_run 从检查点续跑，不再强制重新开始
  const onPause = useCallback(() => {
    if (wsMapRef.current[currentWs]?.status !== "running") return;
    abortMapRef.current[currentWs]?.abort();
    pausedMapRef.current[currentWs] = true;
    streamMapRef.current[currentWs] = null; streamedNodeMapRef.current[currentWs] = null; // 本轮流式收束, 续跑的节点重跑从新气泡开始
    patchWs(currentWs, { status: "awaiting_input", question: "已暂停 · 点「下一步」从最近检查点继续" });
    push(currentWs, {
      kind: "interrupt",
      label: "已暂停",
      question: "运行已中止，点「下一步」将从最近检查点继续（中断处的节点会重跑一遍）",
    });
  }, [currentWs, patchWs, push]);

  const onSubmitAnswer = useCallback(() => {
    if (status !== "awaiting_input") return;
    const text = answer.trim();
    push(currentWs, { kind: "user", label: "人工输入", content: text || "（回车跳过）" });
    setAnswer("");
    patchWs(currentWs, { question: null });
    const runKey = currentModel?.key_set ? undefined : (savedKeys[currentEnv ?? ""] || undefined);
    run(currentWs, text, selectedModel, runKey);
  }, [status, answer, currentWs, run, push, patchWs, selectedModel, currentEnv, currentModel, savedKeys]);

  // 从最近 checkpoint 续跑（出错/暂停后）：有挂起中断则跳过当前提问，否则原地继续
  const onResume = useCallback(() => {
    if (status === "running") return;
    pausedMapRef.current[currentWs] = false;
    const runKey = currentModel?.key_set ? undefined : (savedKeys[currentEnv ?? ""] || undefined);
    push(currentWs, { kind: "user", label: "人工输入", content: "（从最近检查点继续运行）" });
    run(currentWs, null, selectedModel, runKey, true);
  }, [status, currentWs, run, selectedModel, currentEnv, currentModel, savedKeys, push]);

  /* 设计稿「下一步」：待命=开始运行；暂停/出错=检查点续跑；真实 interrupt=提交输入继续 */
  const onNext = useCallback(() => {
    if (status === "idle") {
      onStart();
    } else if (status === "awaiting_input") {
      if (pausedMapRef.current[currentWs]) {
        onResume();
      } else {
        onSubmitAnswer();
      }
    } else if (status === "error") {
      onResume();
    }
  }, [status, currentWs, onStart, onSubmitAnswer, onResume]);

  const onSaveKeys = useCallback(async () => {
    setSavingKeys(true);
    try {
      const keys: Record<string, string> = {};
      for (const [env, val] of Object.entries(modalKeys)) {
        if (val && val.trim()) keys[env] = val.trim();
      }
      const res = await saveKeys(keys, selectedModel);
      setSavedKeys(res.keys ?? {});
      setModels((prev) =>
        prev.map((m) => ({ ...m, key_set: res.present?.[m.api_key_env] ?? m.key_set })),
      );
      setShowKeyModal(false);
      setModalKeys({});
      push(currentWsRef.current, {
        kind: "info",
        label: "API Key 已保存",
        content: `已持久化 ${Object.keys(keys).length} 项；下次启动默认模型：${res.model}`,
      });
    } catch (e: any) {
      push(currentWsRef.current, { kind: "error", label: "保存失败", content: String(e?.message ?? e) });
    } finally {
      setSavingKeys(false);
    }
  }, [modalKeys, selectedModel, push]);

  const onReset = useCallback(async () => {
    if (wsMapRef.current[currentWs]?.status === "running") {
      abortMapRef.current[currentWs]?.abort();
    }
    epochMapRef.current[currentWs] = (epochMapRef.current[currentWs] ?? 0) + 1; // 作废该题旧运行残余事件
    pausedMapRef.current[currentWs] = false;
    abortMapRef.current[currentWs] = null;
    await resetThread(currentWs);
    streamMapRef.current[currentWs] = null; streamedNodeMapRef.current[currentWs] = null;
    announcedMapRef.current[currentWs] = null;
    patchWs(currentWs, { logs: [], finalSummary: "", runReports: [], question: null, step: 0, status: "idle", runStart: null });
    setAnswer("");
    setSelected(null);
    setElapsed(0);
  }, [currentWs, patchWs]);

  // 拖拽分隔条调整左右栏宽度（左栏 160-560px / 右栏 200-620px，中栏自适应）
  const startDrag = useCallback((side: "left" | "right", e: React.MouseEvent) => {
    e.preventDefault();
    splitDragRef.current = {
      side,
      startX: e.clientX,
      startWidth: side === "left" ? leftWidth : rightWidth,
    };
    document.body.classList.add("resizing");
    const onMove = (ev: MouseEvent) => {
      const d = splitDragRef.current;
      if (!d) return;
      const dx = ev.clientX - d.startX;
      const next =
        d.side === "left"
          ? Math.max(160, Math.min(560, d.startWidth + dx))
          : Math.max(200, Math.min(620, d.startWidth - dx)); // 右栏：分隔条右移 → 右栏变窄
      if (d.side === "left") setLeftWidth(next);
      else setRightWidth(next);
    };
    const onUp = () => {
      splitDragRef.current = null;
      document.body.classList.remove("resizing");
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [leftWidth, rightWidth]);

  const openFile = useCallback(async (dir: string, name: string) => {
    setLoadingFile(true);
    const content = await readFile(dir, name, currentWs);
    setSelected({ dir, name, content });
    setLoadingFile(false);
  }, [currentWs]);

  // 手动删除产物(思路/代码/图片/PDF): 带确认;运行中的题后端会以 409 拒绝
  const onDeleteArtifact = useCallback(async (relPath: string, label: string) => {
    if (!window.confirm(`确定删除产物「${label}」？\n文件将被移除，不可恢复！`)) return;
    try {
      await deleteArtifact(relPath, currentWs);
      // 删的是最终总结 → 同步清掉内存里的总结卡片, 不等刷新
      if (relPath === "paper/最终总结.md") {
        patchWs(currentWs, { finalSummary: "" });
      }
      refreshWorkspace(currentWs);
      push(currentWs, { kind: "info", label: "产物已删除", content: label });
    } catch (e: any) {
      push(currentWs, { kind: "error", label: "删除失败", content: String(e?.message ?? e) });
    }
  }, [currentWs, refreshWorkspace, push, patchWs]);

  // 删除题目弹窗里上传的题目/数据文件(带确认, 本地清单同步移除)
  const onDeleteWsFile = useCallback(async (target: "question" | "dataset", name: string) => {
    if (!window.confirm(`确定删除${target === "question" ? "题目" : "数据"}文件「${name}」？\n文件将被移除，不可恢复！`)) return;
    try {
      await deleteWsFile(currentWs, target, name);
      setWorkspaces((prev) => prev.map((w) => (w.id === currentWs ? {
        ...w,
        questionFiles: target === "question" ? w.questionFiles.filter((x) => x !== name) : w.questionFiles,
        datasetFiles: target === "dataset" ? w.datasetFiles.filter((x) => x !== name) : w.datasetFiles,
      } : w)));
      push(currentWs, { kind: "info", label: "文件已删除", content: name });
    } catch (e: any) {
      push(currentWs, { kind: "error", label: "删除失败", content: String(e?.message ?? e) });
    }
  }, [currentWs, push]);

  // 背景素材解析：WE 壁纸库优先（bgSource=we），否则 media 目录，默认 bg-miku.jpg 走本地资源
  const weWallpaper = uiTheme.bgSource === "we" && uiTheme.weId
    ? weWallpapers.find((w) => w.id === uiTheme.weId) ?? null
    : null;
  const mediaNames = new Set(mediaFiles.map((f) => f.name));
  const carouselImgs = mediaFiles.filter(
    (f) => IMAGE_EXT.test(f.name) && !uiTheme.hiddenIds.includes(f.name)
      && (uiTheme.typeFilter === "all" || uiTheme.typeFilter === "image"),
  );
  const curMedia = carouselImgs[carouselIdx % Math.max(1, carouselImgs.length)];
  const videoSrc = transcodedReady
    ? transcodedUrl(`${uiTheme.bgFile}.24fps.mp4`)
    : mediaUrl(uiTheme.bgFile);
  let bgNode: React.ReactNode;
  if (weWallpaper) {
    // WE 壁纸：视频直接播原片；scene/web/image 用 preview 静态帧（dsh 同款思路）
    if (weWallpaper.type === "video") {
      bgNode = (
        <video
          key={weWallpaper.id}
          ref={videoRef}
          className="app-bg-media"
          src={weFileUrl(weWallpaper.id)}
          autoPlay
          loop
          muted
          playsInline
          onLoadedMetadata={(e) => {
            e.currentTarget.playbackRate = uiTheme.playbackRate;
          }}
        />
      );
    } else {
      bgNode = (
        <img
          key={weWallpaper.id}
          className="app-bg-media"
          src={weFileUrl(weWallpaper.id)}
          alt=""
          draggable={false}
        />
      );
    }
  } else if (uiTheme.bgMode === "video" && mediaNames.has(uiTheme.bgFile) && VIDEO_EXT.test(uiTheme.bgFile)) {
    bgNode = (
      <video
        key={uiTheme.bgFile}
        ref={videoRef}
        className="app-bg-media"
        src={videoSrc}
        autoPlay
        loop
        muted
        playsInline
        onLoadedMetadata={(e) => {
          e.currentTarget.playbackRate = uiTheme.playbackRate;
        }}
      />
    );
  } else if (uiTheme.bgMode === "carousel" && carouselImgs.length > 0) {
    bgNode = (
      <img
        key={curMedia?.name}
        className="app-bg-media"
        src={mediaUrl(curMedia!.name)}
        alt=""
        draggable={false}
      />
    );
  } else {
    bgNode = (
      <img
        className="app-bg-media"
        src={mediaNames.has(uiTheme.bgFile) ? mediaUrl(uiTheme.bgFile) : bgImage}
        alt=""
        draggable={false}
      />
    );
  }

  return (
    <div className="app">
      <div className="app-bg">{bgNode}</div>
      <div
      className="layout"
      style={{ gridTemplateColumns: `${leftWidth}px 10px minmax(0, 1fr) 10px ${rightWidth}px` }}
    >
      {/* 左栏 · 工作流侧边栏（设计稿 220px） */}
        <aside className="panel sidebar">
          <div className="side-brand">
            <h1>MathModel Agent</h1>
            <div className="side-session">
              <div className="ws-dd" ref={wsDdRef}>
                <button
                  type="button"
                  className="ws-select side-ws-select ws-dd-btn"
                  onClick={() => setWsDdOpen((o) => !o)}
                  title="切换题目（行尾 ✕ 可删题）：每题的题目/数据/产物/运行状态互相隔离"
                >
                  <span className="ws-dd-cur">{curWsInfo?.title ?? currentWs}</span>
                  <span className="ws-dd-arrow">▾</span>
                </button>
                {wsDdOpen && (
                  <div className="ws-dd-list">
                    {(workspaces.length
                      ? workspaces
                      : [{ id: "default", title: "默认题目", createdAt: "", questionFiles: [], datasetFiles: [], hasState: false } as WorkspaceInfo]
                    ).map((w) => (
                      <div
                        key={w.id}
                        className={`ws-dd-item ${w.id === currentWs ? "active" : ""}`}
                        onClick={() => { setWsDdOpen(false); if (w.id !== currentWs) switchWs(w.id); }}
                        title="点击切换到该题"
                      >
                        <span className="ws-dd-name">
                          {w.title}
                          {wsMap[w.id]?.status === "running" ? "（运行中）" : ""}
                        </span>
                        <button
                          className="file-del"
                          title="删除该题目（连带全部产物与运行状态，需确认）"
                          onClick={(e) => { e.stopPropagation(); onDeleteWs(w.id); }}
                        >
                          ✕
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <span className="side-session-id">session #{currentWs}</span>
            </div>
          </div>

          <div className="side-status">
            <div
              className={`backend ${backendOk === false ? "down" : ""}`}
              title={backendOk ? `后端在线 ${getBackendUrl()}` : "后端离线"}
            >
              <span className="dot" />
              {backendOk === null ? "检测中" : backendOk ? "在线" : "离线"}
            </div>
            <div className="status-badge" data-status={status}>
              {status === "idle" && "待命"}
              {status === "running" && "运行中…"}
              {status === "awaiting_input" && "等待输入"}
              {status === "done" && "已完成"}
              {status === "error" && "出错"}
            </div>
          </div>

          <div className="side-section-title">WORKFLOW</div>
          <div className="workflow-list">
            {STEPS.map((s, i) => {
              const n = i + 1;
              const isDone = status === "done" ? true : step > n;
              const isActive = status !== "done" && step === n;
              return (
                <div
                  key={s.label}
                  className={`wf-step ${isActive ? "active" : isDone && step > 0 ? "done" : ""}`}
                  title={isActive ? s.hint : s.label}
                >
                  <span className="wf-num">{isDone && step > n ? "✓" : n}</span>
                  <span className="wf-label">{s.label}</span>
                  {isActive && <span className="wf-dot" />}
                </div>
              );
            })}
          </div>

          <div className="side-foot">
            <div
              className="model-picker"
              title="选择本次运行使用的模型；问题提取/质检固定走 DeepSeek 非思考模式，DeepSeek API Key 必须配置"
            >
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={status === "running"}
              >
                {modelsLoading && models.length === 0 && <option value="deepseek">deepseek（加载中…）</option>}
                {models.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
            <div className="side-btns">
              <button className="btn ghost sm" onClick={() => setShowWsModal(true)} title="新建题目 / 上传题目与数据 / 删除">
                题目
              </button>
              <button className="btn ghost sm" onClick={() => setShowKeyModal(true)} disabled={status === "running"}
                title="配置各模型 API Key（已存在的环境变量无需重复填写）">
                Key{missingCount > 0 ? ` (${missingCount})` : ""}
              </button>
              <button className="btn ghost sm" onClick={() => setShowAppearance(true)} title="液态玻璃外观：配色 / 透明度 / 模糊 / 背景模式">
                外观
              </button>
              <button className="btn ghost sm" onClick={onReset} disabled={status === "idle" && !logs.length} title="清空线程并重新开始">
                重置
              </button>
            </div>
            {usage && usage.calls > 0 && (
              <div className="usage" title="本次 LLM 调用累计用量（重置/新运行时清零）">
                <div className="usage-row">
                  Token {fmtTokens(usage.prompt_tokens + usage.completion_tokens)}
                  <span className="usage-sub">
                    （入 {fmtTokens(usage.prompt_tokens)} / 出 {fmtTokens(usage.completion_tokens)}）
                  </span>
                </div>
              </div>
            )}
            <div className="elapsed">elapsed {fmtElapsed(elapsed)}</div>
          </div>
        </aside>

        <div className="splitter" onMouseDown={(e) => startDrag("left", e)} title="拖拽调整栏宽" />

        {/* 中栏 · 对话流（设计稿自适应宽度） */}
        <section className="panel chat-panel">
          <div className="chat-head">
            <div className="chat-title">
              <h2>{step > 0 ? STEPS[Math.min(step, STEPS.length) - 1].label : "MathModel Agent"}</h2>
              <span className="step-badge">{step > 0 ? `step ${Math.min(step, STEPS.length)} / ${STEPS.length}` : "standby"}</span>
            </div>
            <div className="chat-actions">
              <button className="btn secondary" onClick={onPause} disabled={status !== "running"} title="终止本轮运行（后端图被取消，无法断点续跑，「下一步」将重新开始）">
                暂停
              </button>
              <button
                className="btn primary"
                onClick={onNext}
                disabled={status === "running" || ((status === "idle" || status === "error") && !keyReady)}
                title={status === "idle" ? "启动一次完整运行" : status === "error" ? "从最近检查点恢复运行" : status === "awaiting_input" ? "提交输入继续流程" : ""}
              >
                {status === "error" ? "继续运行" : "下一步"}
              </button>
            </div>
          </div>

          <div className="chat-scroll" ref={logBoxRef}>
            {logs.length === 0 && (
              <div className="empty chat-empty">
                点击「下一步」启动 agent 流程
                <br />
                <span>运行过程将以对话气泡形式实时展示</span>
              </div>
            )}
            {logs.map((item, idx) => {
              if (item.kind === "user") {
                return (
                  <div key={item.id} className="msg user">
                    <div className="bubble">{item.content}</div>
                  </div>
                );
              }
              if (item.kind === "thinking") {
                // 思考过程: 流式打字中(details 常开+内滚跟随), 结束后默认折叠可回看
                const live = status === "running" && idx === logs.length - 1;
                return (
                  <div key={item.id} className="msg agent">
                    <div className="avatar">❖</div>
                    <div className="msg-body">
                      <div className="msg-meta">{item.label}{live ? " · 思考中" : ""}</div>
                      <details className="bubble think" open={live || undefined}>
                        <summary>💭 思考过程</summary>
                        <div
                          className="think-text"
                          ref={(el) => { if (el && live) el.scrollTop = el.scrollHeight; }}
                        >
                          {item.content || "…"}
                        </div>
                      </details>
                    </div>
                  </div>
                );
              }
              if (item.kind === "ai_stream" || item.kind === "ai") {
                const live = item.kind === "ai_stream" && status === "running" && idx === logs.length - 1;
                return (
                  <div key={item.id} className="msg agent">
                    <div className="avatar">❖</div>
                    <div className="msg-body">
                      <div className="msg-meta">MathModel Agent · now</div>
                      <div className="bubble md">
                        <FoldingMarkdown source={item.content} />
                        {live && <span className="stream-caret" />}
                      </div>
                    </div>
                  </div>
                );
              }
              if (item.kind === "interrupt") {
                return (
                  <div key={item.id} className="msg agent">
                    <div className="avatar">❖</div>
                    <div className="msg-body">
                      <div className="msg-meta">{item.label}</div>
                      <div className="bubble ask">✋ {item.question || "请输入："}</div>
                    </div>
                  </div>
                );
              }
              if (item.kind === "error") {
                return (
                  <div key={item.id} className="msg agent">
                    <div className="avatar err">!</div>
                    <div className="msg-body">
                      <div className="msg-meta err">{item.label}</div>
                      <div className="bubble error">{item.content}</div>
                    </div>
                  </div>
                );
              }
              if (item.kind === "node") {
                return (
                  <div key={item.id} className="sys-line">
                    <span className="sys-dot" />
                    {item.label}
                    {item.method ? ` · ${item.method}` : ""}
                  </div>
                );
              }
              return null;
            })}
          </div>

          <div className="chip-row">
            <button className="q-chip" onClick={() => setShowWsModal(true)}>上传数据</button>
            <button className="q-chip" onClick={() => papersSecRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}>
              参考论文
            </button>
            <button className="q-chip" onClick={() => imagesSecRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}>
              生成图片
            </button>
          </div>

          <div className="chat-input-bar">
            <textarea
              ref={chatInputRef}
              value={answer}
              rows={1}
              onChange={(e) => {
                setAnswer(e.target.value);
                // 自增高：内容多行时撑高（CSS min/max-height 兜底），清空后回落
                const el = e.target;
                el.style.height = "auto";
                el.style.height = `${Math.min(120, el.scrollHeight)}px`;
              }}
              onKeyDown={(e) => {
                // Enter 发送；Shift+Enter 换行；中文输入法组词中的回车不发送
                if (e.key !== "Enter" || e.shiftKey || e.nativeEvent.isComposing) return;
                if (status === "running") return;
                e.preventDefault();
                onNext();
              }}
              placeholder={
                status === "awaiting_input"
                  ? (question ?? "输入回复后回车继续…（Shift+Enter 换行）")
                  : status === "running"
                    ? "运行中，可点「暂停」…"
                    : status === "error"
                      ? "已出错 · 回车从最近检查点继续运行"
                      : "输入指令或问题…（回车开始运行）"
              }
              disabled={status === "running"}
            />
            <button
              className="send-btn"
              onClick={onNext}
              disabled={status === "running" || ((status === "idle" || status === "error") && !keyReady)}
              title={status === "awaiting_input" ? "提交（直接回车=跳过）" : status === "error" ? "从最近检查点继续" : "开始运行"}
            >
              ▶
            </button>
          </div>
        </section>


        <div className="splitter" onMouseDown={(e) => startDrag("right", e)} title="拖拽调整栏宽" />

        {/* 右栏 · 中间产物（设计稿 280px） */}
        <aside className="panel artifacts">
          <div className="art-head">
            <h2>中间产物</h2>
            <span className="art-count">{papers.length + codes.length + images.length} items</span>
          </div>
          <div className="art-scroll">
            <div
              className={`art-card ${finalSummary ? "clickable" : ""}`}
              onClick={() => {
                if (finalSummary) setSelected({ dir: "paper", name: "最终总结.md", content: finalSummary });
              }}
              title={finalSummary ? "点击放大查看" : undefined}
            >
              <div className="art-card-title">最终总结</div>
              <div className="art-summary">
                {finalSummary ? (
                  <FoldingMarkdown source={finalSummary} />
                ) : (
                  <div className="art-none">运行结束后在此展示《最终总结.md》</div>
                )}
              </div>
            </div>

            <div className="art-card" ref={papersSecRef}>
              <div className="art-card-title">paper/ 思路</div>
              <div className="art-files">
                {papers.length === 0 && <span className="art-none">（暂无）</span>}
                {papers.map((f) => (
                  <div key={f.name} className="ws-item-row">
                    <button
                      className={`ws-item ${selected?.name === f.name && selected.dir === "paper" ? "active" : ""}`}
                      onClick={() => openFile("paper", f.name)}
                      title={`${f.name} · 点击预览`}
                    >
                      {f.name}
                    </button>
                    <button
                      className="file-del"
                      title="删除该文件"
                      onClick={(e) => { e.stopPropagation(); onDeleteArtifact(`paper/${f.name}`, f.name); }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="art-card">
              <div className="art-card-title">code/ 代码</div>
              <div className="art-files">
                {codes.length === 0 && <span className="art-none">（暂无）</span>}
                {codes.map((f) => (
                  <div key={f.name} className="ws-item-row">
                    <button
                      className={`ws-item ${selected?.name === f.name && selected.dir === "code" ? "active" : ""}`}
                      onClick={() => openFile("code", f.name)}
                      title={`${f.name} · 点击预览`}
                    >
                      {f.name}
                    </button>
                    <button
                      className="file-del"
                      title="删除该文件"
                      onClick={(e) => { e.stopPropagation(); onDeleteArtifact(`code/${f.name}`, f.name); }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>

            <div className="art-card" ref={imagesSecRef}>
              <div className="art-card-title">生成的图片</div>
              {images.length === 0 && <div className="art-none">（暂无）运行到求解阶段后自动生成</div>}
              <div className="thumb-grid">
                {images.map((f) => (
                  <div key={f.name} className="thumb-wrap">
                    <button
                      className="thumb"
                      title={`${f.name} · 点击放大`}
                      onClick={() => setZoomImage(f.name)}
                    >
                      <img src={imageUrl("photo", f.name, currentWs)} alt={f.name} loading="lazy" />
                    </button>
                    <button
                      className="file-del"
                      title="删除该图片"
                      onClick={(e) => { e.stopPropagation(); onDeleteArtifact(`photo/${f.name}`, f.name); }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            </div>

            {pdfs.length > 0 && (
              <div className="art-card">
                <div className="art-card-title">PDF 文档</div>
                <div className="art-files">
                  {pdfs.map((f) => (
                    <div key={f.path} className="ws-item-row">
                      <button
                        className="ws-item"
                        onClick={() => setViewPdf(f)}
                        title={`${f.name} · 点击预览`}
                      >
                        {f.name}
                        <span className="pdf-size"> {fmtSize(f.size)}</span>
                      </button>
                      <button
                        className="file-del"
                        title="删除该 PDF"
                        onClick={(e) => { e.stopPropagation(); onDeleteArtifact(f.path, f.name); }}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="art-card">
              <div className="art-card-title">运行报告</div>
              {runReports.length === 0 && <div className="art-none">暂无代码运行报告</div>}
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

            <div className="art-card log-card">
              <div className="art-card-title">运行日志</div>
              <div className="mini-log-list">
                {logs.length === 0 && <div className="art-none">（空）</div>}
                {logs.map((item) => (
                  <div key={item.id} className={`mini-log ${item.kind}`}>
                    <span className="log-time">{item.t}</span>
                    <span className="mini-label">{item.label}</span>
                    {"content" in item && item.content ? (
                      <span className="mini-content" title={item.content}>{item.content}</span>
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>


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
            src={imageUrl("photo", zoomImage, currentWs)}
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

      {showKeyModal && (
        <div className="modal-mask" onClick={() => setShowKeyModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>配置 API Key</h3>
              <button className="modal-close" onClick={() => setShowKeyModal(false)} title="关闭">×</button>
            </div>
            <p className="modal-tip">
              已存在于系统环境 / 项目根 <code>.env</code> 的 Key 会标记为「已配置」，无需重复填写；
              只填缺失的项即可。配置持久化到 <code>model_config.json</code>，下次启动默认模型为当前下拉所选。
              <br />
              <strong>DeepSeek API Key 为必需项</strong>
              ：问题提取、建模质检等结构化步骤固定使用 DeepSeek 非思考模式（不随上方模型下拉切换），
              无论选用哪个模型都必须配置。
            </p>
            {deepseekMissing && (
              <p className="modal-warn">
                注意：尚未检测到 DeepSeek API Key。缺少它时问题提取/质检将走降级链路，流程质量下降，请务必填写。
              </p>
            )}
            <div className="key-rows">
              {models.map((m) => {
                const env = m.api_key_env;
                const set = Boolean(m.key_set || savedKeys[env]);
                return (
                  <div className="key-row" key={m.key}>
                    <div className="key-meta">
                      <span className="key-name">
                        {m.label}
                        {m.key === "deepseek" && <em className="key-required">必需</em>}
                      </span>
                      <span className={`key-badge ${set ? "ok" : "no"}`}>
                        {set ? "✓ 已配置" : "未配置"}
                      </span>
                    </div>
                    <input
                      type="password"
                      placeholder={m.key_label}
                      value={modalKeys[env] ?? ""}
                      onChange={(e) => setModalKeys((prev) => ({ ...prev, [env]: e.target.value }))}
                    />
                  </div>
                );
              })}
            </div>
            <div className="modal-foot">
              <button className="btn ghost" onClick={() => setShowKeyModal(false)}>取消</button>
              <button className="btn primary" onClick={onSaveKeys} disabled={savingKeys}>
                {savingKeys ? "保存中…" : "保存"}
              </button>
            </div>
          </div>
        </div>
      )}

      {showAppearance && (
        <div className="modal-mask" onClick={() => setShowAppearance(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>外观设置 · 液态玻璃</h3>
              <button className="modal-close" onClick={() => setShowAppearance(false)} title="关闭">×</button>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">配色</span>
              <div className="appearance-swatches">
                {ACCENT_PRESETS.map((hex) => (
                  <button
                    key={hex}
                    className={`appearance-swatch ${uiTheme.accent === hex ? "appearance-swatch--active" : ""}`}
                    style={{ background: hex }}
                    onClick={() => setTheme({ accent: hex })}
                    title={hex}
                  />
                ))}
                <label className="appearance-swatch appearance-swatch--custom" title="自定义配色">
                  <input type="color" value={uiTheme.accent} onChange={(e) => setTheme({ accent: e.target.value })} />
                </label>
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">玻璃颜色</span>
              <div className="appearance-swatches">
                {GLASS_COLOR_PRESETS.map((hex) => (
                  <button
                    key={hex}
                    className={`appearance-swatch ${uiTheme.glassColor === hex ? "appearance-swatch--active" : ""}`}
                    style={{ background: hex }}
                    onClick={() => setTheme({ glassColor: hex })}
                    title={hex}
                  />
                ))}
                <label className="appearance-swatch appearance-swatch--custom" title="自定义玻璃色">
                  <input type="color" value={uiTheme.glassColor} onChange={(e) => setTheme({ glassColor: e.target.value })} />
                </label>
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">文字颜色</span>
              <div className="appearance-swatches">
                <button
                  className={`appearance-swatch appearance-swatch--textdefault ${uiTheme.textColor === "" ? "appearance-swatch--active" : ""}`}
                  onClick={() => setTheme({ textColor: "" })}
                  title="默认（跟随主题）"
                >
                  默认
                </button>
                {TEXT_COLOR_PRESETS.map((hex) => (
                  <button
                    key={hex}
                    className={`appearance-swatch ${uiTheme.textColor === hex ? "appearance-swatch--active" : ""}`}
                    style={{ background: hex }}
                    onClick={() => setTheme({ textColor: hex })}
                    title={hex}
                  />
                ))}
                <label className="appearance-swatch appearance-swatch--custom" title="自定义文字颜色">
                  <input type="color" value={uiTheme.textColor || "#16233c"} onChange={(e) => setTheme({ textColor: e.target.value })} />
                </label>
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">玻璃模糊 <span className="val">{uiTheme.blur}px</span></span>
              <input className="appearance-range" type="range" min={0} max={60} value={uiTheme.blur}
                onChange={(e) => setTheme({ blur: Number(e.target.value) })} />
            </div>

            <div className="appearance-row">
              <span className="appearance-label">玻璃透明度（越大越透） <span className="val">{uiTheme.glassAlpha}%</span></span>
              <input className="appearance-range" type="range" min={0} max={60} step={5} value={uiTheme.glassAlpha}
                onChange={(e) => setTheme({ glassAlpha: Number(e.target.value) })} />
            </div>

            <div className="appearance-row">
              <span className="appearance-label">背景压暗 <span className="val">{Math.round(uiTheme.scrim * 100)}%</span></span>
              <input className="appearance-range" type="range" min={0} max={100} value={Math.round(uiTheme.scrim * 100)}
                onChange={(e) => setTheme({ scrim: Number(e.target.value) / 100 })} />
            </div>

            <div className="appearance-row">
              <span className="appearance-label">边框强调 <span className="val">{Math.round(uiTheme.border * 100)}%</span></span>
              <input className="appearance-range" type="range" min={0} max={100} value={Math.round(uiTheme.border * 100)}
                onChange={(e) => setTheme({ border: Number(e.target.value) / 100 })} />
            </div>

            <div className="appearance-row">
              <span className="appearance-label">背景模糊 <span className="val">{uiTheme.wallpaperBlur}px</span></span>
              <input className="appearance-range" type="range" min={0} max={60} value={uiTheme.wallpaperBlur}
                onChange={(e) => setTheme({ wallpaperBlur: Number(e.target.value) })} />
            </div>

            <div className="appearance-row">
              <span className="appearance-label">背景模式</span>
              <div className="appearance-seg">
                {(["image", "video", "carousel"] as const).map((m) => (
                  <button key={m} className={uiTheme.bgMode === m ? "active" : ""} onClick={() => setTheme({ bgMode: m })}>
                    {m === "image" ? "静态图" : m === "video" ? "视频" : "轮播"}
                  </button>
                ))}
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">背景素材（media/ 目录）</span>
              <div className="appearance-media-list">
                {mediaFiles.length === 0 && <span style={{ color: "var(--muted)", fontSize: 12 }}>（空）把图片/视频放到项目 media/ 目录，或选下面的默认背景</span>}
                {mediaFiles.map((f) => (
                  <button key={f.name} className={`appearance-media-item ${uiTheme.bgFile === f.name && uiTheme.bgMode !== "carousel" ? "active" : ""}`}
                    onClick={() => setTheme({ bgFile: f.name })}>
                    {f.name} {VIDEO_EXT.test(f.name) ? "▶" : ""}
                  </button>
                ))}
                <button className={`appearance-media-item ${uiTheme.bgFile === "bg-miku.jpg" && uiTheme.bgMode !== "carousel" ? "active" : ""}`}
                  onClick={() => setTheme({ bgFile: "bg-miku.jpg" })}>
                  bg-miku.jpg（内置）
                </button>
              </div>
            </div>

            <div className="appearance-row">
              <button className="btn ghost" onClick={() => { setWeLibTab("we"); setShowMediaLib(true); }} style={{ width: "100%" }}>
                WE 壁纸库 · 上传 / 隐藏 / 搜索 / 转码
              </button>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">图片适配</span>
              <div className="appearance-seg">
                {(["cover", "contain", "center", "fill"] as const).map((m) => (
                  <button key={m} className={uiTheme.objectFit === m ? "active" : ""} onClick={() => setTheme({ objectFit: m })}>
                    {m === "cover" ? "覆盖" : m === "contain" ? "填充" : m === "center" ? "居中" : "拉伸"}
                  </button>
                ))}
              </div>
            </div>

            <div className="appearance-row">
              <label className="appearance-label" style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={uiTheme.flip} onChange={(e) => setTheme({ flip: e.target.checked })} />
                水平翻转
              </label>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">玻璃总开关</span>
              <div className="appearance-seg">
                <button className={uiTheme.glassWindow ? "active" : ""} onClick={() => setTheme({ glassWindow: true })}>开启</button>
                <button className={!uiTheme.glassWindow ? "active" : ""} onClick={() => setTheme({ glassWindow: false })}>关闭</button>
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">遮挡暂停（视频壁纸）</span>
              <label className="appearance-label" style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={uiTheme.pauseOnHidden} onChange={(e) => setTheme({ pauseOnHidden: e.target.checked })} />
                最小化/切页时暂停
              </label>
              <label className="appearance-label" style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={uiTheme.pauseOnBlur} onChange={(e) => setTheme({ pauseOnBlur: e.target.checked })} />
                窗口失焦时暂停
              </label>
              <label className="appearance-label" style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                <input type="checkbox" checked={uiTheme.pauseOnBattery} onChange={(e) => setTheme({ pauseOnBattery: e.target.checked })} />
                电池供电时暂停
              </label>
            </div>

            {uiTheme.bgMode === "carousel" && (
              <div className="appearance-row">
                <label className="appearance-label" style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
                  <input type="checkbox" checked={uiTheme.shuffle} onChange={(e) => setTheme({ shuffle: e.target.checked })} />
                  随机顺序
                </label>
              </div>
            )}

            {uiTheme.bgMode === "video" && (
              <div className="appearance-row">
                <span className="appearance-label">视频倍速</span>
                <div className="appearance-seg">
                  {[0.5, 1, 1.5, 2].map((r) => (
                    <button key={r} className={uiTheme.playbackRate === r ? "active" : ""} onClick={() => setTheme({ playbackRate: r })}>
                      {r}×
                    </button>
                  ))}
                </div>
              </div>
            )}

            {uiTheme.bgMode === "carousel" && (
              <div className="appearance-row">
                <span className="appearance-label">轮播间隔 <span className="val">{uiTheme.carouselSecs}s</span></span>
                <input className="appearance-range" type="range" min={3} max={120} value={uiTheme.carouselSecs}
                  onChange={(e) => setTheme({ carouselSecs: Number(e.target.value) })} />
              </div>
            )}

            <div className="modal-foot">
              <button className="btn ghost" onClick={onResetAppearance}>恢复默认</button>
              <button className="btn ghost" onClick={() => setShowAppearance(false)}>取消</button>
              <button className="btn primary" onClick={onSaveAppearance}>保存</button>
            </div>
          </div>
        </div>
      )}

      {showMediaLib && (
        <div className="modal-mask" onClick={() => setShowMediaLib(false)}>
          <div className="modal media-lib" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>背景素材库</h3>
              <button className="modal-close" onClick={() => setShowMediaLib(false)} title="关闭">×</button>
            </div>

            <div className="appearance-row" style={{ marginBottom: 10 }}>
              <div className="appearance-seg">
                <button className={weLibTab === "local" ? "active" : ""} onClick={() => setWeLibTab("local")}>本地素材</button>
                <button className={weLibTab === "we" ? "active" : ""} onClick={() => setWeLibTab("we")}>
                  WE 壁纸库{weWallpapers.length > 0 ? `（${weWallpapers.length}）` : ""}
                </button>
              </div>
            </div>

            {weLibTab === "we" ? (
              <>
                <div className="media-lib-bar">
                  <input
                    className="media-lib-search"
                    placeholder="搜索壁纸标题…"
                    value={weSearch}
                    onChange={(e) => setWeSearch(e.target.value)}
                  />
                </div>
                <div className="media-lib-grid">
                  {weVisible.length === 0 && (
                    <div className="empty">没有匹配的壁纸（分级过滤或搜索条件过严）</div>
                  )}
                  {weVisible.map((w) => {
                    const isActive = uiTheme.bgSource === "we" && uiTheme.weId === w.id;
                    return (
                      <div key={w.id} className={`media-lib-card ${isActive ? "active" : ""}`}>
                        <div className="media-lib-video-thumb">
                          <img src={wePreviewUrl(w.id)} alt={w.title} loading="lazy" />
                          {w.type !== "image" && (
                            <span className="media-lib-play">{w.type === "video" ? "▶" : "❖"}</span>
                          )}
                          <span className="we-rating">{RATING_LABEL[w.rating] ?? w.rating}</span>
                        </div>
                        <span className="media-lib-name" title={w.title}>{w.title}</span>
                        <div className="media-lib-actions">
                          <button className="btn ghost" onClick={() => setWeBg(w)}>
                            {isActive ? "使用中" : "设为背景"}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            ) : (
              <>
                <div className="media-lib-bar">
                  <input
                    className="media-lib-search"
                    placeholder="搜索素材…"
                    value={mediaSearch}
                    onChange={(e) => setMediaSearch(e.target.value)}
                  />
                  <div className="appearance-seg">
                    {(["all", "image", "video"] as const).map((m) => (
                      <button key={m} className={uiTheme.typeFilter === m ? "active" : ""}
                        onClick={() => setTheme({ typeFilter: m })}>
                        {m === "all" ? "全部" : m === "image" ? "图片" : "视频"}
                      </button>
                    ))}
                  </div>
                  <label className="appearance-label" style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", margin: 0, whiteSpace: "nowrap" }}>
                    <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} />
                    显示已隐藏
                  </label>
                  <button className="btn ghost" onClick={() => fileRef.current?.click()} disabled={uploading}>
                    {uploading ? "上传中…" : "上传素材"}
                  </button>
                  <input ref={fileRef} type="file" hidden accept=".jpg,.jpeg,.png,.webp,.gif,.mp4,.webm" onChange={onUpload} />
                </div>

                <div className="media-lib-grid">
                  {visibleMedia.length === 0 && (
                    <div className="empty">没有匹配的素材，点击「上传素材」添加图片或视频</div>
                  )}
                  {visibleMedia.map((f) => {
                    const hidden = uiTheme.hiddenIds.includes(f.name);
                    const isImg = IMAGE_EXT.test(f.name);
                    const isActive = uiTheme.bgSource === "local" && uiTheme.bgFile === f.name && uiTheme.bgMode !== "carousel";
                    return (
                      <div key={f.name} className={`media-lib-card ${isActive ? "active" : ""} ${hidden ? "hidden" : ""}`}>
                        {isImg ? (
                          <img src={mediaUrl(f.name)} alt={f.name} loading="lazy" />
                        ) : (
                          <div className="media-lib-video-thumb">
                            <img src={thumbUrl(f.name)} alt="" loading="lazy"
                              onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }} />
                            <span className="media-lib-play">▶</span>
                          </div>
                        )}
                        <span className="media-lib-name">{f.name}</span>
                        <div className="media-lib-actions">
                          <button className="btn ghost" onClick={() => setTheme({ bgSource: "local", bgFile: f.name, bgMode: isImg ? "image" : "video" })}>
                            设为背景
                          </button>
                          <button className="btn ghost" onClick={() => toggleHidden(f.name)}>
                            {hidden ? "恢复" : "隐藏"}
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {showWsModal && (
        <div className="modal-mask" onClick={() => setShowWsModal(false)}>
          <div className="modal ws-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>题目管理</h3>
              <button className="modal-close" onClick={() => setShowWsModal(false)} title="关闭">×</button>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">新建题目</span>
              <div style={{ display: "flex", gap: 8 }}>
                <input
                  className="media-lib-search"
                  placeholder="题目标题，如：2024B题"
                  value={newWsTitle}
                  onChange={(e) => setNewWsTitle(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && onCreateWs()}
                />
                <button className="btn primary" onClick={onCreateWs}>新建</button>
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">当前题：{curWsInfo?.title ?? currentWs}</span>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn ghost" onClick={() => wsQRef.current?.click()} disabled={wsUploading} style={{ flex: 1 }}>
                  {wsUploading ? "上传中…" : "上传题目（txt / md / pdf）"}
                </button>
                <input ref={wsQRef} type="file" hidden multiple accept=".txt,.md,.pdf" onChange={(e) => onWsUpload(e, "question")} />
                <button className="btn ghost" onClick={() => wsDRef.current?.click()} disabled={wsUploading} style={{ flex: 1 }}>
                  {wsUploading ? "上传中…" : "上传数据（csv / xlsx / json）"}
                </button>
                <input ref={wsDRef} type="file" hidden multiple accept=".csv,.tsv,.xlsx,.xls,.json,.jsonl" onChange={(e) => onWsUpload(e, "dataset")} />
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">本题题目文件</span>
              <div className="appearance-media-list" style={{ maxHeight: 90 }}>
                {(curWsInfo?.questionFiles.length ?? 0) === 0 && <span style={{ color: "var(--muted)", fontSize: 12 }}>（无，请上传）</span>}
                {curWsInfo?.questionFiles.map((f) => (
                  <span key={f} className="appearance-media-item" style={{ cursor: "default" }}>
                    {f}
                    <button className="file-del" title="删除该文件" onClick={() => onDeleteWsFile("question", f)}>✕</button>
                  </span>
                ))}
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">本题数据集文件</span>
              <div className="appearance-media-list" style={{ maxHeight: 90 }}>
                {(curWsInfo?.datasetFiles.length ?? 0) === 0 && <span style={{ color: "var(--muted)", fontSize: 12 }}>（无，请上传）</span>}
                {curWsInfo?.datasetFiles.map((f) => (
                  <span key={f} className="appearance-media-item" style={{ cursor: "default" }}>
                    {f}
                    <button className="file-del" title="删除该文件" onClick={() => onDeleteWsFile("dataset", f)}>✕</button>
                  </span>
                ))}
              </div>
            </div>

            <div className="modal-foot">
              <button className="btn ghost" onClick={() => onDeleteWs(currentWs)} style={{ color: "var(--bad)" }}>
                删除当前题
              </button>
              <button className="btn primary" onClick={() => setShowWsModal(false)}>关闭</button>
            </div>
          </div>
        </div>
      )}

      {/* PDF 预览弹窗（内嵌渲染论文/合规文档，Esc 关闭） */}
      {viewPdf && (
        <div className="modal-mask" onClick={() => setViewPdf(null)}>
          <div
            className="modal pdf-modal"
            style={docSize ? { width: docSize.w, height: docSize.h } : undefined}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-head">
              <h3>{viewPdf.name}</h3>
              <div className="modal-head-actions">
                <button
                  className="btn ghost sm"
                  onClick={() => window.open(pdfUrl(viewPdf.path, currentWs), "_blank")}
                  title="在系统浏览器中打开（内嵌预览异常时的备选）"
                >
                  浏览器打开
                </button>
                <button className="modal-close" onClick={() => setViewPdf(null)} title="关闭 (Esc)">×</button>
              </div>
            </div>
            <iframe src={pdfUrl(viewPdf.path, currentWs)} title={viewPdf.name} className="pdf-frame" />
            <div className="modal-resize" onMouseDown={startDocResize} title="拖拽调整大小" />
          </div>
        </div>
      )}

      {/* 文件预览弹窗（替代原内联预览区） */}
      {selected && (
        <div className="modal-mask" onClick={() => setSelected(null)}>
          <div
            className="modal file-modal"
            style={docSize ? { width: docSize.w, height: docSize.h } : undefined}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-head">
              <h3>{selected.dir}/{selected.name}</h3>
              <button className="modal-close" onClick={() => setSelected(null)} title="关闭 (Esc)">×</button>
            </div>
            <div className="file-modal-body">
              {loadingFile ? (
                <div className="empty">加载中…</div>
              ) : selected.dir === "code" ? (
                <pre className="code-preview">{selected.content}</pre>
              ) : (
                <div className="md-preview">
                  <FoldingMarkdown source={selected.content} />
                </div>
              )}
            </div>
            <div className="modal-resize" onMouseDown={startDocResize} title="拖拽调整大小" />
          </div>
        </div>
      )}
    </div>
  );
}