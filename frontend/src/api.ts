export interface SseEvent {
  type: "update" | "interrupt" | "suspended" | "done" | "error" | "log";
  node?: string;
  data?: any;
  value?: string;
  error?: string;
  text?: string;
}

declare global {
  interface Window {
    api?: { backendUrl: () => string; platform: string };
  }
}

const backendUrl = () =>
  window.api?.backendUrl?.() ??
  (import.meta.env.VITE_API_URL as string | undefined) ??
  "http://127.0.0.1:8000";

export const getBackendUrl = backendUrl;

async function readSseStream(
  resp: Response,
  onEvent: (ev: SseEvent) => void,
) {
  if (!resp.body) throw new Error("响应没有可读流");
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const dataLine = chunk
        .split("\n")
        .find((l) => l.startsWith("data:"))
        ?.slice(5)
        .trim();
      if (!dataLine) continue;
      try {
        onEvent(JSON.parse(dataLine) as SseEvent);
      } catch {
        /* 忽略无法解析的碎片 */
      }
    }
  }
}

export interface ModelOption {
  key: string;
  label: string;
  model: string;
  api_key_env: string;
  key_label: string;
  /** 该模型的 key 环境变量当前是否已在进程环境中配置（系统环境 / .env） */
  key_set?: boolean;
}

export interface ModelConfig {
  model: string;
  keys: Record<string, string>;
}

/** 后端不可用时兜底，保证下拉框与 API Key 输入永不卡死 */
export const FALLBACK_MODELS: ModelOption[] = [
  { key: "deepseek", label: "DeepSeek", model: "deepseek-v4-flash", api_key_env: "DEEPSEEK_API_KEY", key_label: "DeepSeek API Key" },
  { key: "gpt", label: "GPT (OpenAI)", model: "gpt-4o", api_key_env: "OPENAI_API_KEY", key_label: "OpenAI API Key" },
  { key: "glm", label: "GLM (智谱 Zhipu)", model: "glm-4-flash", api_key_env: "ZHIPU_API_KEY", key_label: "智谱 API Key" },
  { key: "qwen", label: "通义千问 (Qwen)", model: "qwen-plus", api_key_env: "DASHSCOPE_API_KEY", key_label: "通义千问 API Key (DashScope)" },
  { key: "kimi", label: "Kimi (Moonshot)", model: "moonshot-v1-8k", api_key_env: "MOONSHOT_API_KEY", key_label: "Kimi API Key (Moonshot)" },
  { key: "mimo", label: "MiniMax (Mimo)", model: "MiniMax-Text-01", api_key_env: "MINIMAX_API_KEY", key_label: "MiniMax API Key" },
  { key: "openrouter", label: "OpenRouter", model: "stealth/ox-alpha", api_key_env: "OPENROUTER_API_KEY", key_label: "OpenRouter API Key" },
];

export async function fetchModels(): Promise<ModelOption[]> {
  try {
    const resp = await fetch(`${backendUrl()}/api/models`);
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.models ?? [];
  } catch {
    return [];
  }
}

export async function fetchModelConfig(): Promise<ModelConfig> {
  try {
    const resp = await fetch(`${backendUrl()}/api/model`);
    if (!resp.ok) return { model: "deepseek", keys: {} };
    const data = await resp.json();
    return { model: data.model ?? "deepseek", keys: data.keys ?? {} };
  } catch {
    return { model: "deepseek", keys: {} };
  }
}

export interface SaveKeysResult {
  model: string;
  keys: Record<string, string>;
  present: Record<string, boolean>;
}

/** 批量保存 API Key（可选同时设定默认模型），后端持久化到 model_config.json */
export async function saveKeys(keys: Record<string, string>, model?: string): Promise<SaveKeysResult> {
  const resp = await fetch(`${backendUrl()}/api/keys`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model ?? "", keys }),
  });
  if (!resp.ok) throw new Error(`保存失败: ${resp.status}`);
  return resp.json();
}

export async function startRun(
  threadId: string,
  resume: string | null,
  onEvent: (ev: SseEvent) => void,
  signal?: AbortSignal,
  model?: string,
  apiKey?: string,
  continueRun?: boolean,
): Promise<void> {
  const body: Record<string, unknown> = continueRun
    ? { thread_id: threadId, continue_run: true }
    : resume === null
      ? { thread_id: threadId }
      : { thread_id: threadId, resume };
  if (model) body.model = model;
  if (apiKey) body.api_key = apiKey;
  const resp = await fetch(`${backendUrl()}/api/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`后端错误 ${resp.status}: ${text.slice(0, 200)}`);
  }
  await readSseStream(resp, onEvent);
}

export async function fetchState(threadId: string): Promise<any> {
  const resp = await fetch(
    `${backendUrl()}/api/state?thread_id=${encodeURIComponent(threadId)}`,
  );
  if (!resp.ok) throw new Error(`获取状态失败: ${resp.status}`);
  return resp.json();
}

export async function resetThread(threadId: string): Promise<void> {
  await fetch(`${backendUrl()}/api/reset?thread_id=${encodeURIComponent(threadId)}`, {
    method: "POST",
  });
}

export async function checkHealth(): Promise<boolean> {
  try {
    const resp = await fetch(`${backendUrl()}/api/health`, { signal: AbortSignal.timeout(3000) });
    return resp.ok;
  } catch {
    return false;
  }
}

export interface WorkspaceFile {
  name: string;
  size: number;
}

export async function listFiles(workDir: string): Promise<WorkspaceFile[]> {
  const resp = await fetch(
    `${backendUrl()}/api/files?work_dir=${encodeURIComponent(workDir)}`,
    { signal: AbortSignal.timeout(5000) },
  );
  if (!resp.ok) return [];
  const data = await resp.json();
  return data.files ?? [];
}

export async function readFile(workDir: string, relPath: string): Promise<string> {
  const resp = await fetch(
    `${backendUrl()}/api/file?work_dir=${encodeURIComponent(workDir)}&rel_path=${encodeURIComponent(relPath)}`,
    { signal: AbortSignal.timeout(10000) },
  );
  if (!resp.ok) return "";
  const data = await resp.json();
  return data.content ?? "";
}

export const imageUrl = (workDir: string, relPath: string) =>
  `${backendUrl()}/api/image?work_dir=${encodeURIComponent(workDir)}&rel_path=${encodeURIComponent(relPath)}`;

export const mediaUrl = (name: string) =>
  `${backendUrl()}/api/media/file?name=${encodeURIComponent(name)}`;

export interface UiConfig {
  accent: string;
  glassAlpha: number;     // 0-60，越大越透
  glassColor: string;
  blur: number;           // 玻璃模糊 px
  scrim: number;          // 背景压暗 0-1
  border: number;         // 边框强调 0-1
  wallpaperBlur: number;  // 背景自身模糊 px
  objectFit: "cover" | "contain" | "center" | "fill";
  flip: boolean;
  bgMode: "image" | "video" | "carousel";
  bgFile: string;
  carouselSecs: number;
  playbackRate: number;
  glassWindow: boolean;
  // —— dsh 同款（第二批）——
  pauseOnHidden: boolean;   // 页面隐藏时暂停视频
  pauseOnBlur: boolean;     // 窗口失焦时暂停视频
  pauseOnBattery: boolean;  // 电池供电时暂停视频
  typeFilter: "all" | "image" | "video";
  hiddenIds: string[];      // 隐藏（软删除）的素材文件名
  shuffle: boolean;         // 轮播随机顺序
  mediaDir: string;         // 背景素材目录
  // —— WE 壁纸库（dsh 同款）——
  bgSource: "local" | "we";        // local（media 目录）/ we（Wallpaper Engine）
  weId: string;                    // 当前选中的 WE 壁纸 id
  contentRatingFilter: "all" | "everyone" | "pg13" | "mature" | "unrated";
}

export const DEFAULT_UI: UiConfig = {
  accent: "#4f8cff",
  glassAlpha: 12,
  glassColor: "#ffffff",
  blur: 16,
  scrim: 0.25,
  border: 0.35,
  wallpaperBlur: 0,
  objectFit: "cover",
  flip: false,
  bgMode: "image",
  bgFile: "bg-miku.jpg",
  carouselSecs: 30,
  playbackRate: 1,
  glassWindow: true,
  pauseOnHidden: true,
  pauseOnBlur: false,
  pauseOnBattery: false,
  typeFilter: "all",
  hiddenIds: [],
  shuffle: false,
  mediaDir: "media",
  bgSource: "local",
  weId: "",
  contentRatingFilter: "all",
};

/** 拉取前端外观配置；后端不可用/未配置时回退默认值，绝不阻塞界面 */
export async function fetchUiConfig(): Promise<UiConfig> {
  try {
    const resp = await fetch(`${backendUrl()}/api/ui-config`, {
      signal: AbortSignal.timeout(4000),
    });
    if (!resp.ok) return DEFAULT_UI;
    const d = await resp.json();
    return { ...DEFAULT_UI, ...d };
  } catch {
    return DEFAULT_UI;
  }
}

/** 保存前端外观配置（部分字段即可，后端合并 + 裁剪区间） */
export async function saveUiConfig(patch: Partial<UiConfig>): Promise<UiConfig> {
  const resp = await fetch(`${backendUrl()}/api/ui-config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
    signal: AbortSignal.timeout(4000),
  });
  if (!resp.ok) throw new Error(`保存外观配置失败: ${resp.status}`);
  const d = await resp.json();
  return { ...DEFAULT_UI, ...d };
}

export async function listMedia(): Promise<WorkspaceFile[]> {
  try {
    const resp = await fetch(`${backendUrl()}/api/media`, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.files ?? [];
  } catch {
    return [];
  }
}

export const thumbUrl = (name: string) => `${backendUrl()}/api/media/thumb?name=${encodeURIComponent(name)}`;
export const transcodedUrl = (name: string) => `${backendUrl()}/api/media/transcoded?name=${encodeURIComponent(name)}`;

/** 上传背景素材（jpg/png/webp/mp4/webm） */
export async function uploadMedia(file: File): Promise<{ name: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch(`${backendUrl()}/api/media/upload`, { method: "POST", body: fd });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`上传失败: ${resp.status} ${text.slice(0, 120)}`);
  }
  return resp.json();
}

export interface TranscodeStatus {
  name?: string;
  phase: "idle" | "working" | "ready" | "error" | "skipped";
  percent: number;
  error?: string | null;
  file?: string | null;
}

/** 触发视频低帧率转码（无 ffmpeg 时返回 skipped） */
export async function startTranscode(name: string, fps = 24): Promise<TranscodeStatus> {
  const resp = await fetch(`${backendUrl()}/api/media/transcode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, fps }),
  });
  if (!resp.ok) throw new Error(`转码启动失败: ${resp.status}`);
  return resp.json();
}

/** 查询转码进度 */
export async function transcodeStatus(name: string): Promise<TranscodeStatus> {
  try {
    const resp = await fetch(`${backendUrl()}/api/media/transcode-status?name=${encodeURIComponent(name)}`, {
      signal: AbortSignal.timeout(4000),
    });
    if (!resp.ok) return { phase: "idle", percent: 0 };
    return resp.json();
  } catch {
    return { phase: "idle", percent: 0 };
  }
}

// ---------- WE 壁纸库（dsh 同款：Steam workshop） ----------
export interface WeWallpaper {
  id: string;
  title: string;
  type: "video" | "scene" | "web" | "image";
  rating: "everyone" | "pg13" | "mature" | "unrated";
  preview: string;
  file: string;
}

export async function fetchWeWallpapers(): Promise<WeWallpaper[]> {
  try {
    const resp = await fetch(`${backendUrl()}/api/we/inventory`, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) return [];
    const data = await resp.json();
    return data.wallpapers ?? [];
  } catch {
    return [];
  }
}

export const wePreviewUrl = (id: string) => `${backendUrl()}/api/we/preview?id=${encodeURIComponent(id)}`;
export const weFileUrl = (id: string) => `${backendUrl()}/api/we/file?id=${encodeURIComponent(id)}`;

// ---------- 题目工作区（多题隔离：thread_id = 题目 id） ----------
export interface WorkspaceInfo {
  id: string;
  title: string;
  createdAt: string;
  questionFiles: string[];
  datasetFiles: string[];
  hasState: boolean; // 有未结束状态 → 显示「继续运行」
}

export interface WorkspacesResp {
  workspaces: WorkspaceInfo[];
  current: string;
}

export async function fetchWorkspaces(): Promise<WorkspacesResp> {
  try {
    const resp = await fetch(`${backendUrl()}/api/workspaces`, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) return { workspaces: [], current: "default" };
    return resp.json();
  } catch {
    return { workspaces: [], current: "default" };
  }
}

export async function createWorkspace(title: string): Promise<WorkspaceInfo> {
  const resp = await fetch(`${backendUrl()}/api/workspaces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!resp.ok) throw new Error(`新建失败: ${resp.status}`);
  return resp.json();
}

export async function deleteWorkspace(id: string): Promise<void> {
  const resp = await fetch(`${backendUrl()}/api/workspaces/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!resp.ok) throw new Error(`删除失败: ${resp.status}`);
}

export async function activateWorkspace(id: string): Promise<string> {
  const resp = await fetch(`${backendUrl()}/api/workspaces/${encodeURIComponent(id)}/activate`, { method: "POST" });
  if (!resp.ok) throw new Error(`切换失败: ${resp.status}`);
  const d = await resp.json();
  return d.current ?? id;
}

export async function uploadWorkspaceFiles(
  id: string,
  target: "question" | "dataset",
  files: File[],
): Promise<WorkspaceInfo> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const resp = await fetch(
    `${backendUrl()}/api/workspaces/${encodeURIComponent(id)}/upload?target=${target}`,
    { method: "POST", body: fd, signal: AbortSignal.timeout(30000) },
  );
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`上传失败: ${resp.status} ${text.slice(0, 120)}`);
  }
  return resp.json();
}