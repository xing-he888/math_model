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

export async function startRun(
  threadId: string,
  resume: string | null,
  onEvent: (ev: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(`${backendUrl()}/api/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      resume === null ? { thread_id: threadId } : { thread_id: threadId, resume },
    ),
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