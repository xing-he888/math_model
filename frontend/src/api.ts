export interface SseEvent {
  type: "update" | "interrupt" | "suspended" | "done" | "error";
  node?: string;
  data?: any;
  value?: string;
  error?: string;
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
): Promise<void> {
  const body: Record<string, unknown> =
    resume === null ? { thread_id: threadId } : { thread_id: threadId, resume };
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