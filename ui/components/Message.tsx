"use client";

import { ChatMessage, Citation, REFUSAL_PREFIX, Step } from "@/lib/api";

function stepLabel(s: Step): string {
  const a = s.args as Record<string, any>;
  switch (s.tool) {
    case "search_reports":
      return `Search “${a.query ?? ""}”${a.companies?.length ? ` in ${a.companies.join(", ")}` : ""}`;
    case "read_page":
      return `Read page ${a.page} of ${a.document}`;
    case "list_documents":
      return "List the reports";
    case "calculator":
      return `Calculate ${a.expression}`;
    default:
      return s.tool;
  }
}

function Steps({ steps, live }: { steps: Step[]; live: boolean }) {
  if (!steps.length) return null;
  return (
    <details className="steps" open={live}>
      <summary>
        {live ? "Working" : "Steps"} · {steps.length} tool call{steps.length === 1 ? "" : "s"}
      </summary>
      <ol>
        {steps.map((s, i) => (
          <li key={i}>
            <span className="step-tool">{stepLabel(s)}</span>
            <span className={`step-result${s.summary === "error" || s.summary === "0 results" ? " warn" : ""}`}>
              → {s.summary}
            </span>
          </li>
        ))}
      </ol>
    </details>
  );
}

// The answer with each [S#] turned into a numbered button for its citation.
function Answer({ text, citations, onOpen }: { text: string; citations: Citation[]; onOpen: (c: Citation) => void }) {
  const index = new Map<string, number>();
  citations.forEach((c, i) => (c.tags ?? [c.source]).forEach((t) => index.set(t, i)));

  const parts = text.split(/\[S(\d+)\]/g); // odd positions hold the numbers
  const out: React.ReactNode[] = [];
  let previous = -1;
  parts.forEach((part, i) => {
    if (i % 2 === 0) {
      out.push(part);
      if (part.trim()) previous = -1;
      return;
    }
    const n = index.get(`S${part}`);
    if (n === undefined || n === previous) return; // unknown tag, or the same source twice in a row
    previous = n;
    out.push(
      <button key={i} className="cite" onClick={() => onOpen(citations[n])} title={`${citations[n].document}, p. ${citations[n].page}`}>
        {n + 1}
      </button>,
    );
  });
  return <div className="answer">{out}</div>;
}

export function Message({ message, onOpen }: { message: ChatMessage; onOpen: (c: Citation) => void }) {
  if (message.role === "user") return <div className="msg user">{message.content}</div>;

  const citations = message.citations ?? [];
  const refusal = !message.streaming && message.content.trim().startsWith(REFUSAL_PREFIX);
  const waiting = message.streaming && !message.content;

  return (
    <div className={`msg assistant${refusal ? " refusal" : ""}`}>
      <Steps steps={message.steps ?? []} live={!!message.streaming} />
      {refusal && <div className="refusal-label">Not found in the reports</div>}
      {waiting ? (
        <div className="thinking">Searching the reports…</div>
      ) : (
        <Answer text={message.content} citations={citations} onOpen={onOpen} />
      )}
      {message.error && <div className="error">{message.error}</div>}
      {citations.length > 0 && (
        <ol className="sources">
          {citations.map((c, i) => (
            <li key={i}>
              <button onClick={() => onOpen(c)}>
                <span className="source-num">{i + 1}</span>
                <span className="source-doc">{c.document}</span>
                <span className="source-page">p. {c.page}</span>
                {c.section && <span className="source-section">{c.section}</span>}
              </button>
            </li>
          ))}
        </ol>
      )}
      {message.usage && !message.streaming && (
        <div className="usage">
          ${message.usage.usd.toFixed(4)} · {(message.usage.prompt_tokens + message.usage.completion_tokens).toLocaleString()} tokens
          {message.traceUrl && (
            <>
              {" · "}
              <a href={message.traceUrl} target="_blank" rel="noreferrer">
                View trace
              </a>
            </>
          )}
        </div>
      )}
    </div>
  );
}
