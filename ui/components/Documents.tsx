"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, DocumentInfo, uploadPdf } from "@/lib/api";

type Upload = { name: string; progress: number; error?: string };

// The reports in the database: upload, follow ingestion, delete.
export function Documents() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [error, setError] = useState("");
  const input = useRef<HTMLInputElement>(null);

  const refresh = useCallback(() => {
    api<DocumentInfo[]>("/documents").then(
      (d) => {
        setDocs(d);
        setError("");
      },
      (e) => setError(e.message),
    );
  }, []);

  // poll quickly while something is being ingested, slowly otherwise
  const busy = docs.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, busy ? 2000 : 15000);
    return () => clearInterval(timer);
  }, [refresh, busy]);

  async function upload(files: FileList | null) {
    for (const file of Array.from(files ?? [])) {
      setUploads((u) => [...u, { name: file.name, progress: 0 }]);
      const update = (patch: Partial<Upload>) =>
        setUploads((u) => u.map((x) => (x.name === file.name ? { ...x, ...patch } : x)));
      try {
        await uploadPdf(file, (progress) => update({ progress }));
        setUploads((u) => u.filter((x) => x.name !== file.name)); // now listed with its ingestion status
        refresh();
      } catch (e) {
        update({ error: (e as Error).message });
      }
    }
    if (input.current) input.current.value = "";
  }

  async function remove(doc: DocumentInfo) {
    if (!confirm(`Delete ${doc.filename} and its ${doc.chunks} passages?`)) return;
    try {
      await api(`/documents/${doc.id}`, { method: "DELETE" });
      refresh();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const ready = docs.filter((d) => d.status === "ready").length;

  return (
    <section className="documents">
      <div className="section-head">
        <h2>
          Reports <span className="muted">{ready}/{docs.length} ready</span>
        </h2>
        <button className="small-btn" onClick={() => input.current?.click()}>
          Upload PDF
        </button>
        <input ref={input} type="file" accept="application/pdf,.pdf" multiple hidden onChange={(e) => upload(e.target.files)} />
      </div>
      {error && <div className="error">{error}</div>}
      <ul>
        {uploads.map((u) => (
          <li key={`up-${u.name}`} className="doc">
            <span className="doc-name">{u.name}</span>
            {u.error ? (
              <span className="error small">
                {u.error}{" "}
                <button className="link" onClick={() => setUploads((x) => x.filter((y) => y.name !== u.name))}>
                  dismiss
                </button>
              </span>
            ) : (
              <progress value={u.progress} max={1} aria-label={`Uploading ${u.name}`} />
            )}
          </li>
        ))}
        {docs.map((d) => (
          <li key={d.id} className="doc">
            <span className="doc-name" title={d.filename}>
              {d.company}
            </span>
            <span className={`badge ${d.status}`} title={d.error ?? undefined}>
              {d.status === "ready" ? `${d.pages} p` : d.status}
            </span>
            <button className="icon-btn" onClick={() => remove(d)} aria-label={`Delete ${d.filename}`}>
              ×
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
