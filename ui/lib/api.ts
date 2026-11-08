// Browser-side client for the backend (through the /api proxy).

export type Mode = "agent" | "pipeline";

export type Citation = {
  source: string;
  tags?: string[];
  document: string;
  page: number;
  section: string | null;
  chunk_id: number;
};

export type Step = {
  step: number;
  tool: string;
  args: Record<string, unknown>;
  summary: string;
};

export type Usage = { prompt_tokens: number; completion_tokens: number; usd: number };

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  steps?: Step[];
  usage?: Usage;
  error?: string;
  streaming?: boolean;
  traceUrl?: string; // Langfuse trace of this turn, when tracing is on
};

export type Conversation = { id: number; title: string; mode: Mode; created_at: string };

export type DocumentInfo = {
  id: number;
  filename: string;
  company: string;
  status: "pending" | "processing" | "ready" | "failed";
  error: string | null;
  pages: number | null;
  chunks: number;
};

export type Passage = { id: number; document: string; page: number; section: string | null; content: string };

export const REFUSAL_PREFIX = "N/A";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

// Each browser gets its own user id; conversations and the daily budget are per user.
export function getUserId(): string {
  try {
    let id = localStorage.getItem("rag-user-id");
    if (!id) {
      id = `user-${crypto.randomUUID().slice(0, 8)}`;
      localStorage.setItem("rag-user-id", id);
    }
    return id;
  } catch {
    return "anonymous";
  }
}

export function setUserId(id: string) {
  try {
    localStorage.setItem("rag-user-id", id);
  } catch {
    /* storage blocked: the id lasts for this page only */
  }
}

// What the user should read for each failure, instead of a bare status code.
function explain(status: number, detail: string): string {
  if (status === 429) return `Daily budget reached: ${detail}. Try again tomorrow.`;
  if (status === 503) return detail || "The API is not reachable.";
  if (status === 413) return "The file is larger than 50 MB.";
  if (status === 415) return "Only PDF files can be uploaded.";
  return detail || `Request failed (${status}).`;
}

async function errorFrom(res: Response): Promise<ApiError> {
  let detail = "";
  try {
    const body = await res.json();
    detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* not JSON */
  }
  return new ApiError(res.status, explain(res.status, detail));
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { "X-User-Id": getUserId(), ...(init.body ? { "Content-Type": "application/json" } : {}), ...init.headers },
    });
  } catch {
    throw new ApiError(0, "The UI server is not reachable.");
  }
  if (!res.ok) throw await errorFrom(res);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

// Upload with progress: fetch() cannot report upload progress, XMLHttpRequest can.
export function uploadPdf(file: File, onProgress: (fraction: number) => void): Promise<DocumentInfo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/documents");
    xhr.setRequestHeader("X-User-Id", getUserId());
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let detail = "";
      try {
        detail = JSON.parse(xhr.responseText).detail ?? "";
      } catch {
        /* not JSON */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
      else reject(new ApiError(xhr.status, explain(xhr.status, detail)));
    };
    xhr.onerror = () => reject(new ApiError(0, "The UI server is not reachable."));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}

export type StreamEvent =
  | { type: "step"; step: number; tool: string; args: Record<string, unknown>; summary: string }
  | { type: "token"; text: string }
  | { type: "done"; answer: string; citations: Citation[]; steps: Step[]; usage: Usage; conversation_id: number; trace_url?: string | null }
  | { type: "error"; message: string };

// POST /chat/stream and call onEvent for each server-sent event.
// Returns the conversation id (known from the response header before any event).
export async function streamChat(
  body: { question: string; conversation_id: number | null; mode: Mode },
  onStart: (conversationId: number) => void,
  onEvent: (event: StreamEvent) => void,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-User-Id": getUserId() },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "The UI server is not reachable.");
  }
  if (!res.ok || !res.body) throw await errorFrom(res);
  onStart(Number(res.headers.get("x-conversation-id")));

  // SSE frames are separated by a blank line; the JSON is on the "data:" line.
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = frame.split("\n").find((line) => line.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)) as StreamEvent);
    }
  }
}
