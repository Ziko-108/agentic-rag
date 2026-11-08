"use client";

import { useEffect, useState } from "react";
import { api, Citation, Passage } from "@/lib/api";

// The passage behind a citation, so the reader can check the answer.
export function SourcePanel({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  const [passage, setPassage] = useState<Passage | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setPassage(null);
    setError("");
    api<Passage>(`/chunks/${citation.chunk_id}`).then(setPassage, (e) => setError(e.message));
  }, [citation.chunk_id]);

  // the stored chunk starts with a "[Company | p.N | Section]" header line; the panel shows that above
  const body = passage?.content.replace(/^\[[^\]\n]*\]\n/, "");

  return (
    <aside className="source-panel" aria-label="Source passage">
      <header>
        <div>
          <div className="panel-doc">{citation.document}</div>
          <div className="panel-meta">
            Page {citation.page}
            {citation.section ? ` · ${citation.section}` : ""}
          </div>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close">
          ×
        </button>
      </header>
      {error && <div className="error">{error}</div>}
      {!passage && !error && <div className="muted">Loading…</div>}
      {body && <pre className="passage">{body}</pre>}
    </aside>
  );
}
