"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Documents } from "@/components/Documents";
import { Message } from "@/components/Message";
import { SourcePanel } from "@/components/SourcePanel";
import {
  api,
  ChatMessage,
  Citation,
  Conversation,
  getUserId,
  Mode,
  setUserId,
  Step,
  streamChat,
  Usage,
} from "@/lib/api";

type StoredMessage = { role: "user" | "assistant"; content: string; citations: Citation[] | null; steps: Step[] | null; usage: Usage };

const EXAMPLES = [
  "Which reports can I ask about?",
  "Who are the members of Datalogic's board of directors?",
  "Compare the total revenue of HCA Healthcare and Albany International.",
];

export default function Home() {
  const [user, setUser] = useState("");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [current, setCurrent] = useState<number | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<Mode>("agent");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState("");
  const [source, setSource] = useState<Citation | null>(null);
  const [health, setHealth] = useState<"ok" | "down" | "unknown">("unknown");
  const [menuOpen, setMenuOpen] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => setUser(getUserId()), []);

  const loadConversations = useCallback(() => {
    api<Conversation[]>("/conversations").then(setConversations, () => {});
  }, []);
  useEffect(() => {
    if (user) loadConversations();
  }, [user, loadConversations]);

  // API status dot in the header
  useEffect(() => {
    const check = () => api("/health").then(() => setHealth("ok"), () => setHealth("down"));
    check();
    const timer = setInterval(check, 15000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" }), [messages]);

  function newChat() {
    if (busy) return;
    setCurrent(null);
    setMessages([]);
    setSource(null);
    setBanner("");
    setMenuOpen(false);
  }

  async function openConversation(id: number) {
    if (busy) return;
    try {
      const conv = await api<{ mode: Mode; messages: StoredMessage[] }>(`/conversations/${id}`);
      setCurrent(id);
      setMode(conv.mode);
      setMessages(conv.messages.map((m) => ({ ...m, citations: m.citations ?? [], steps: m.steps ?? [] })));
      setSource(null);
      setBanner("");
      setMenuOpen(false);
    } catch (e) {
      setBanner((e as Error).message);
      loadConversations();
    }
  }

  async function deleteConversation(id: number) {
    if (!confirm("Delete this conversation?")) return;
    try {
      await api(`/conversations/${id}`, { method: "DELETE" });
      if (id === current) newChat();
      loadConversations();
    } catch (e) {
      setBanner((e as Error).message);
    }
  }

  function changeUser(id: string) {
    const clean = id.trim();
    if (!clean || clean === user) return;
    setUserId(clean);
    setUser(clean);
    newChat();
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    setBanner("");
    setMessages((m) => [...m, { role: "user", content: question }, { role: "assistant", content: "", steps: [], streaming: true }]);
    const patchLast = (f: (m: ChatMessage) => ChatMessage) =>
      setMessages((ms) => [...ms.slice(0, -1), f(ms[ms.length - 1])]);

    let started = false;
    try {
      await streamChat(
        { question, conversation_id: current, mode },
        (id) => {
          started = true;
          setCurrent(id);
        },
        (ev) => {
          if (ev.type === "step") patchLast((m) => ({ ...m, steps: [...(m.steps ?? []), ev] }));
          else if (ev.type === "token") patchLast((m) => ({ ...m, content: m.content + ev.text }));
          else if (ev.type === "done")
            patchLast((m) => ({ ...m, content: ev.answer, citations: ev.citations, steps: ev.steps, usage: ev.usage, streaming: false, traceUrl: ev.trace_url ?? undefined }));
          else if (ev.type === "error") patchLast((m) => ({ ...m, streaming: false, error: ev.message }));
        },
      );
      patchLast((m) => (m.streaming ? { ...m, streaming: false, error: "The answer was interrupted." } : m));
    } catch (e) {
      if (started) {
        patchLast((m) => ({ ...m, streaming: false, error: (e as Error).message }));
      } else {
        // rejected before any answer (budget, API down): take the question back
        setMessages((ms) => ms.slice(0, -2));
        setInput(question);
        setBanner((e as Error).message);
      }
    } finally {
      setBusy(false);
      loadConversations();
    }
  }

  return (
    <div className={`app${source ? " with-panel" : ""}`}>
      <nav className={`sidebar${menuOpen ? " open" : ""}`}>
        <button className="primary-btn" onClick={newChat} disabled={busy}>
          + New chat
        </button>
        <section className="conversations">
          <h2>Conversations</h2>
          {conversations.length === 0 && <p className="muted small">No conversations yet.</p>}
          <ul>
            {conversations.map((c) => (
              <li key={c.id} className={c.id === current ? "active" : ""}>
                <button className="conv-title" onClick={() => openConversation(c.id)} title={c.title}>
                  {c.title}
                </button>
                <button className="icon-btn" onClick={() => deleteConversation(c.id)} aria-label="Delete conversation">
                  ×
                </button>
              </li>
            ))}
          </ul>
        </section>
        <Documents />
        <label className="user">
          <span className="muted small">User</span>
          <input key={user} defaultValue={user} onBlur={(e) => changeUser(e.target.value)} onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()} />
        </label>
      </nav>

      <main className="chat">
        <header className="topbar">
          <button className="icon-btn menu" onClick={() => setMenuOpen((o) => !o)} aria-label="Menu">
            ☰
          </button>
          <h1>Annual Reports Assistant</h1>
          <div className="mode" role="radiogroup" aria-label="Answering mode">
            {(["agent", "pipeline"] as Mode[]).map((m) => (
              <button key={m} role="radio" aria-checked={mode === m} className={mode === m ? "on" : ""} onClick={() => setMode(m)} disabled={busy}>
                {m === "agent" ? "Agent" : "Single pass"}
              </button>
            ))}
          </div>
          <span className={`health ${health}`} title={health === "down" ? "API is not reachable" : "API is up"}>
            {health === "down" ? "API down" : health === "ok" ? "API ok" : "…"}
          </span>
        </header>

        {banner && (
          <div className="banner" role="alert">
            {banner}
            <button className="icon-btn" onClick={() => setBanner("")} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}

        <div className="messages">
          {messages.length === 0 ? (
            <div className="empty">
              <h2>Ask about the annual reports</h2>
              <p className="muted">
                Answers cite the report and page they come from. If the reports don&apos;t contain the answer, you get N/A
                instead of a guess.
              </p>
              <div className="examples">
                {EXAMPLES.map((q) => (
                  <button key={q} onClick={() => send(q)} disabled={busy}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => <Message key={i} message={m} onOpen={setSource} />)
          )}
          <div ref={bottom} />
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder="Ask a question about the reports…"
            rows={2}
            maxLength={2000}
            aria-label="Question"
          />
          <button className="primary-btn" type="submit" disabled={busy || !input.trim()}>
            {busy ? "Answering…" : "Send"}
          </button>
        </form>
      </main>

      {source && <SourcePanel citation={source} onClose={() => setSource(null)} />}
    </div>
  );
}
