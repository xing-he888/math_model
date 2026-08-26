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
  DEFAULT_UI,
  FALLBACK_MODELS,
  type SseEvent,
  type WeWallpaper,
  type WorkspaceInfo,
  type WorkspaceFile,
  type ModelOption,
  type UiConfig,
} from "./api";
import "./App.css";
import bgImage from "./assets/bg-miku.jpg";

/* dsh 同款配色预设（dsh-wallpaper-engine ACCENT_PRESETS / GLASS_COLOR_PRESETS） */
const ACCENT_PRESETS = ["#4f8cff", "#67DCE7", "#DD8FAC", "#F3B75F", "#F1717F", "#CBE77D"];
const GLASS_COLOR_PRESETS = ["#ffffff", "#0d1524", "#67DCE7", "#DD8FAC", "#F3B75F", "#F1717F"];
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

type LogItem =
  | { id: number; t: string; kind: "node"; label: string; method?: string }
  | { id: number; t: string; kind: "tool"; label: string; content: string }
  | { id: number; t: string; kind: "ai"; label: string; content: string }
  | { id: number; t: string; kind: "interrupt"; label: string; question: string }
  | { id: number; t: string; kind: "user"; label: string; content: string }
  | { id: number; t: string; kind: "info"; label: string; content: string }
  | { id: number; t: string; kind: "log"; label: string; content: string }
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
  // 题目工作区：题目 id 即 thread_id（多题隔离，切换题目 = 切换线程状态与产物目录）
  const [currentWs, setCurrentWs] = useState("default");
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [showWsModal, setShowWsModal] = useState(false);
  const [newWsTitle, setNewWsTitle] = useState("");
  const [wsUploading, setWsUploading] = useState(false);
  const wsQRef = useRef<HTMLInputElement | null>(null);  // 题目文件
  const wsDRef = useRef<HTMLInputElement | null>(null);  // 数据文件
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
  const abortRef = useRef<AbortController | null>(null);
  const logBoxRef = useRef<HTMLDivElement>(null);
  const statusRef = useRef<Status>(status);
  statusRef.current = status;
  // 运行代数：切题/重置/新运行时自增，旧运行残余事件按代数丢弃，防串台
  const epochRef = useRef(0);
  // 暂停标记：暂停 = abort 杀图，线程无挂起 interrupt，无法断点续跑，「下一步」须重新开始
  const pausedRef = useRef(false);
  // 设计稿三栏布局：当前工作流步骤(0=未开始/1-6) + 会话计时
  const [step, setStep] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const runStartRef = useRef<number | null>(null);
  const chatInputRef = useRef<HTMLTextAreaElement | null>(null);
  const papersSecRef = useRef<HTMLDivElement | null>(null);
  const imagesSecRef = useRef<HTMLDivElement | null>(null);

  // 会话计时：运行/等待输入期间每秒刷新
  useEffect(() => {
    if (status !== "running" && status !== "awaiting_input") return;
    if (runStartRef.current === null) runStartRef.current = Date.now() - elapsed * 1000;
    const timer = setInterval(() => {
      setElapsed(Math.floor((Date.now() - (runStartRef.current ?? Date.now())) / 1000));
    }, 1000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  // interrupt 挂起时自动聚焦底部输入框
  useEffect(() => {
    if (status === "awaiting_input") chatInputRef.current?.focus();
  }, [status, question]);

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
    // 先掐断旧 SSE 流（后端会取消旧题的图），再清状态：
    // 否则旧运行的事件继续推送、后端全局工作区也被占用
    abortRef.current?.abort();
    epochRef.current++;          // 旧运行残余事件按代数丢弃，不串台到新题目
    pausedRef.current = false;
    setStatus("idle");
    setLogs([]);
    setFinalSummary("");
    setRunReports([]);
    setSelected(null);
    setQuestion(null);
    setAnswer("");
    setStep(0);
    setElapsed(0);
    runStartRef.current = null;
    setCurrentWs(id);
    try { await activateWorkspace(id); } catch { /* 后端离线时本地先切 */ }
    refreshWorkspace();
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
      push({ kind: "info", label: "题目已创建", content: `${ws.title} (${ws.id})` });
    } catch (e: any) {
      push({ kind: "error", label: "新建失败", content: String(e?.message ?? e) });
    }
  }, [newWsTitle, switchWs, push]);

  const onDeleteWs = useCallback(async (id: string) => {
    const title = workspaces.find((w) => w.id === id)?.title ?? id;
    if (!window.confirm(`确定删除题目「${title}」？\n将连带删除该题全部产物与运行状态，不可恢复！`)) return;
    try {
      await deleteWorkspace(id);
      setWorkspaces((prev) => prev.filter((w) => w.id !== id));
      if (id === currentWs) {
        const next = workspaces.find((w) => w.id !== id);
        await switchWs(next?.id ?? "default");
      }
      push({ kind: "info", label: "题目已删除", content: title });
    } catch (e: any) {
      push({ kind: "error", label: "删除失败", content: String(e?.message ?? e) });
    }
  }, [workspaces, currentWs, switchWs, push]);

  const onWsUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>, target: "question" | "dataset") => {
    const files = Array.from(e.target.files ?? []);
    if (!files.length) return;
    setWsUploading(true);
    try {
      const info = await uploadWorkspaceFiles(currentWs, target, files);
      setWorkspaces((prev) => prev.map((w) => (w.id === currentWs ? info : w)));
      refreshWorkspace();
      push({ kind: "info", label: "上传成功", content: `${files.length} 个${target === "question" ? "题目" : "数据"}文件` });
    } catch (err: any) {
      push({ kind: "error", label: "上传失败", content: String(err?.message ?? err) });
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
      push({ kind: "info", label: "外观已保存", content: "液态玻璃设置已持久化到 ui_config.json" });
    } catch (e: any) {
      push({ kind: "error", label: "保存外观失败", content: String(e?.message ?? e) });
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
      push({ kind: "info", label: "上传成功", content: r.name });
    } catch (err: any) {
      push({ kind: "error", label: "上传失败", content: String(err?.message ?? err) });
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

  // 文件预览弹窗: ESC 关闭
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected]);

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
    (ev: SseEvent) => {
      if (ev.type === "update" && ev.node) {
        const label = NODE_LABELS[ev.node] ?? ev.node;
        const isSolve = ev.node === "solve_with_method";
        push({ kind: "node", label, method: isSolve ? ev.data?.method : undefined });
        const mapped = NODE_TO_STEP[ev.node];
        if (mapped) setStep(mapped); // 直接赋值：SSE 事件有序，质检/审核打回时步骤可回退

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
        fetchState(currentWs)
          .then((s) => {
            const sum = s.values?.final_summary || "";
            if (sum) setFinalSummary(sum);
            push({ kind: "info", label: "运行完成", content: "全部节点执行完毕" });
          })
          .catch(() => push({ kind: "error", label: "获取状态失败", content: "请检查后端" }));
      } else if (ev.type === "log") {
        // 后端桥接的"LLM调用 → 模型[角色] · 用途"实时日志
        push({ kind: "log", label: "模型调用", content: ev.text ?? "" });
      } else if (ev.type === "error") {
        setStatus("error");
        push({ kind: "error", label: "运行出错", content: ev.error ?? "" });
      }
    },
    [push, currentWs, refreshWorkspace],
  );

  const run = useCallback(
    async (resume: string | null, model?: string, apiKey?: string, continueRun?: boolean) => {
      const controller = new AbortController();
      abortRef.current = controller;
      const myEpoch = ++epochRef.current;
      setStatus("running");
      try {
        await startRun(
          currentWs,
          resume,
          (ev) => { if (epochRef.current === myEpoch) handleEvent(ev); },
          controller.signal,
          model,
          apiKey,
          continueRun,
        );
      } catch (e: any) {
        if (e?.name !== "AbortError" && epochRef.current === myEpoch) {
          setStatus("error");
          push({ kind: "error", label: "连接失败", content: String(e?.message ?? e) });
        }
      } finally {
        // 仅当仍是自己的 controller 时才清空，防止旧运行的 finally 覆盖新运行的控制器
        if (abortRef.current === controller) abortRef.current = null;
      }
    },
    [currentWs, handleEvent, push],
  );

  const onStart = useCallback(() => {
    if (status === "running") return;
    pausedRef.current = false;
    setLogs([]);
    setFinalSummary("");
    setRunReports([]);
    setSelected(null);
    setStep(1); // 乐观置第 1 步，首个节点事件到达后按真实节点校正
    runStartRef.current = null;
    // 配置统一来源于 .env，环境已就绪则无需前端下发 key，后端自读；否则兜底用已保存值
    const runKey = currentModel?.key_set ? undefined : (savedKeys[currentEnv ?? ""] || undefined);
    run(null, selectedModel, runKey);
  }, [status, run, selectedModel, currentEnv, currentModel, savedKeys]);

  // 设计稿「暂停」：中断 SSE 流（后端取消图，状态停在最近 checkpoint）。
  // 「下一步」走 continue_run 从检查点续跑，不再强制重新开始
  const onPause = useCallback(() => {
    if (status !== "running") return;
    abortRef.current?.abort();
    pausedRef.current = true;
    setStatus("awaiting_input");
    setQuestion("已暂停 · 点「下一步」从最近检查点继续");
    push({
      kind: "interrupt",
      label: "已暂停",
      question: "运行已中止，点「下一步」将从最近检查点继续（中断处的节点会重跑一遍）",
    });
  }, [status, push]);

  const onSubmitAnswer = useCallback(() => {
    if (status !== "awaiting_input") return;
    const text = answer.trim();
    push({ kind: "user", label: "人工输入", content: text || "（回车跳过）" });
    setAnswer("");
    setQuestion(null);
    const runKey = currentModel?.key_set ? undefined : (savedKeys[currentEnv ?? ""] || undefined);
    run(text, selectedModel, runKey);
  }, [status, answer, run, push, selectedModel, currentEnv, currentModel, savedKeys]);

  // 从最近 checkpoint 续跑（出错/暂停后）：有挂起中断则跳过当前提问，否则原地继续
  const onResume = useCallback(() => {
    if (status === "running") return;
    pausedRef.current = false;
    const runKey = currentModel?.key_set ? undefined : (savedKeys[currentEnv ?? ""] || undefined);
    push({ kind: "user", label: "人工输入", content: "（从最近检查点继续运行）" });
    run(null, selectedModel, runKey, true);
  }, [status, run, selectedModel, currentEnv, currentModel, savedKeys, push]);

  /* 设计稿「下一步」：待命=开始运行；暂停/出错=检查点续跑；真实 interrupt=提交输入继续 */
  const onNext = useCallback(() => {
    if (status === "idle") {
      onStart();
    } else if (status === "awaiting_input") {
      if (pausedRef.current) {
        onResume();
      } else {
        onSubmitAnswer();
      }
    } else if (status === "error") {
      onResume();
    }
  }, [status, onStart, onSubmitAnswer, onResume]);

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
      push({
        kind: "info",
        label: "API Key 已保存",
        content: `已持久化 ${Object.keys(keys).length} 项；下次启动默认模型：${res.model}`,
      });
    } catch (e: any) {
      push({ kind: "error", label: "保存失败", content: String(e?.message ?? e) });
    } finally {
      setSavingKeys(false);
    }
  }, [modalKeys, selectedModel, push]);

  const onReset = useCallback(async () => {
    if (status === "running") {
      abortRef.current?.abort();
    }
    epochRef.current++;        // 作废旧运行残余事件
    pausedRef.current = false;
    await resetThread(currentWs);
    setLogs([]);
    setFinalSummary("");
    setRunReports([]);
    setQuestion(null);
    setAnswer("");
    setSelected(null);
    setStep(0);
    setElapsed(0);
    runStartRef.current = null;
    setStatus("idle");
  }, [status, currentWs]);

  const openFile = useCallback(async (dir: string, name: string) => {
    setLoadingFile(true);
    const content = await readFile(dir, name);
    setSelected({ dir, name, content });
    setLoadingFile(false);
  }, []);

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
      <div className="layout">
        {/* 左栏 · 工作流侧边栏（设计稿 220px） */}
        <aside className="panel sidebar">
          <div className="side-brand">
            <h1>MathModel Agent</h1>
            <div className="side-session">
              <select
                className="ws-select side-ws-select"
                value={currentWs}
                onChange={(e) => switchWs(e.target.value)}
                title="切换题目：每题的题目/数据/产物/运行状态互相隔离"
              >
                {workspaces.map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.title}
                  </option>
                ))}
                {workspaces.length === 0 && <option value="default">默认题目</option>}
              </select>
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
            <div className="elapsed">elapsed {fmtElapsed(elapsed)}</div>
          </div>
        </aside>

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
            {logs.map((item) => {
              if (item.kind === "user") {
                return (
                  <div key={item.id} className="msg user">
                    <div className="bubble">{item.content}</div>
                  </div>
                );
              }
              if (item.kind === "ai") {
                return (
                  <div key={item.id} className="msg agent">
                    <div className="avatar">❖</div>
                    <div className="msg-body">
                      <div className="msg-meta">MathModel Agent · now</div>
                      <div className="bubble md">
                        <FoldingMarkdown source={item.content} />
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


        {/* 右栏 · 中间产物（设计稿 280px） */}
        <aside className="panel artifacts">
          <div className="art-head">
            <h2>中间产物</h2>
            <span className="art-count">{papers.length + codes.length + images.length} items</span>
          </div>
          <div className="art-scroll">
            <div className="art-card">
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
                  <button
                    key={f.name}
                    className={`ws-item ${selected?.name === f.name && selected.dir === "paper" ? "active" : ""}`}
                    onClick={() => openFile("paper", f.name)}
                    title={`${f.name} · 点击预览`}
                  >
                    {f.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="art-card">
              <div className="art-card-title">code/ 代码</div>
              <div className="art-files">
                {codes.length === 0 && <span className="art-none">（暂无）</span>}
                {codes.map((f) => (
                  <button
                    key={f.name}
                    className={`ws-item ${selected?.name === f.name && selected.dir === "code" ? "active" : ""}`}
                    onClick={() => openFile("code", f.name)}
                    title={`${f.name} · 点击预览`}
                  >
                    {f.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="art-card" ref={imagesSecRef}>
              <div className="art-card-title">生成的图片</div>
              {images.length === 0 && <div className="art-none">（暂无）运行到求解阶段后自动生成</div>}
              <div className="thumb-grid">
                {images.map((f) => (
                  <button
                    key={f.name}
                    className="thumb"
                    title={`${f.name} · 点击放大`}
                    onClick={() => setZoomImage(f.name)}
                  >
                    <img src={imageUrl("photo", f.name)} alt={f.name} loading="lazy" />
                  </button>
                ))}
              </div>
            </div>

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
                  <span key={f} className="appearance-media-item" style={{ cursor: "default" }}>{f}</span>
                ))}
              </div>
            </div>

            <div className="appearance-row">
              <span className="appearance-label">本题数据集文件</span>
              <div className="appearance-media-list" style={{ maxHeight: 90 }}>
                {(curWsInfo?.datasetFiles.length ?? 0) === 0 && <span style={{ color: "var(--muted)", fontSize: 12 }}>（无，请上传）</span>}
                {curWsInfo?.datasetFiles.map((f) => (
                  <span key={f} className="appearance-media-item" style={{ cursor: "default" }}>{f}</span>
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

      {/* 文件预览弹窗（替代原内联预览区） */}
      {selected && (
        <div className="modal-mask" onClick={() => setSelected(null)}>
          <div className="modal file-modal" onClick={(e) => e.stopPropagation()}>
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
          </div>
        </div>
      )}
    </div>
  );
}