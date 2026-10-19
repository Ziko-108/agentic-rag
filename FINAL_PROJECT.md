# Final Project: Agentic RAG over PDFs

**Format:** take-home, individual
**Deadline:** 10.11.2026

## The task

Build a chatbot that answers questions about a set of annual reports in PDF, end to end:
from uploading a document to a cited answer in a browser, with the whole stack
started by one command.

The difference from the reference app is that the answer comes from an **agent**.
The model decides what to search for, reads what comes back, searches again if
something is missing, and only then answers.

Everyone starts from the same documents and the same questions. The points below
say what has to exist. How you build each part is up to you.

## What you get

The `starter/` folder (details in its [README](starter/README.md)):

- `data/pdfs/`: 14 annual reports, the corpus. Expect long filings, tables and
  multi-column pages.
- `data/golden_set.json`: 16 questions about them, with expected answers and the
  pages the answers are on. Three of them can't be answered from the reports, and
  one compares five reports.
- `eval.py`: asks your app every question and prints a scorecard. You change
  only the function that calls your API. The scoring is fixed and deterministic,
  the same for everyone.

## Ground rules

- **Use the reference app (`m9-l8`) as a guide, not a source to copy.** Look at
  how it splits the layers, which problems it solves and how, then build your
  own version. Copying its code teaches you nothing, and you won't be able to
  explain or change what you didn't write. Most of the grade is for the agent,
  retrieval that works on these reports, and decisions you back with numbers.
- **You may swap any component.** For example: pgvector instead of Qdrant, Gradio
  or React instead of Streamlit, LangGraph instead of a hand-written loop, or
  another LLM provider. Say why in the README.
- **It doesn't have to be production-ready,** but the README has to be honest
  about what doesn't work.

## AI-assisted development

Use Cursor, Claude Code, Codex, Copilot or anything else you like. That's fine,
and it's how you'll work in a real job too.

But AI-assisted development is not the same as vibe coding:

- **Vibe coding:** you describe the app in a few prompts, accept whatever comes
  back, and ship it without knowing how it works.
- **AI-assisted development:** the AI writes much of the code, but you stay in
  charge. You make the main decisions, know what each piece does and how the
  pieces connect, and can say why it's built this way and not another.

We expect the second. For anything in your project, you should be able to
explain what it does, why you chose it, and what you'd change if it broke. If
you can't explain a part, it doesn't count as your work. **Simpler system that you understand is far more valuable than more complex, "cutting-edge", but vibe-coded one.**

## What we expect (100 points)

### 1. The agent (18)

- A tool-calling loop. The model calls a search tool with its own query, reads
  the results, may search again, then answers.
- Tools written for the model: clear names and descriptions, and error messages
  that say what to try next. Add tools beyond search if your domain needs them
  (list documents, read a page, calculator, and so on).
- Limits that make it always stop: an iteration cap, a token or cost budget, and
  a guard against repeating the same call.
- A tool error goes back to the model as a result. It does not crash the request.

### 2. PDF ingestion (10)

- PDFs are uploaded through the API and processed in the background. You can
  check each one's status (pending / ready / failed).
- Parsing that copes with these reports. Look at what comes out of the tables,
  headers and multi-column pages.
- A chunking strategy chosen for these documents, which you can explain.
- Every chunk keeps the metadata a citation needs: document and page, and section
  if you have it.
- Deleting a document removes it from everywhere.

### 3. Storage (4)

- A vector database for search.
- A database that is the source of truth for documents, chunks, conversations
  and messages.
- The vector index can be rebuilt from that database without parsing the PDFs
  again.

### 4. Retrieval (10)

- Hybrid search: dense plus lexical (BM25 or similar), fused.
- A reranker over the shortlist.
- A relevance threshold, so a search can return nothing.
- Settings (top-k, chunk size, threshold) chosen by measuring with `eval.py`,
  not guessing.

### 5. Grounded answers (7)

- Claims cite their document and page (the PDF file name and a 1-based page, so
  `eval.py` can check them).
- When the documents don't contain the answer, the bot says so.
- The user can open the passage behind a citation.

### 6. Backend API (7)

- An HTTP API (FastAPI or similar) for documents, chat and conversations.
- Answers stream as they are generated (SSE or similar).
- Sensible errors and status codes, and a health endpoint.

### 7. Conversations, cost and safety (6)

- Conversations are saved and survive a restart. Follow-up questions work.
- Long histories stay within a token budget.
- Tokens and cost are recorded per turn, with a per-user limit (a user-id header
  is enough).
- A basic defence against instructions planted in uploaded PDFs.
- A fallback when the main LLM fails, such as a local model through Ollama.

### 8. User interface (6)

- Chat with streamed answers and sources.
- Upload with progress, and a list of past conversations.
- The user can see what the agent did: which searches it ran and how many steps
  it took.
- Clear messages for refusals, a spent budget, and an API that is down.

### 9. Containers (8)

- `docker compose up` from a fresh clone starts everything.
- Secrets come from `.env` (with a committed `.env.example`) and are never in git
  or in an image.
- Data lives on volumes. The clean-clone test passes (`scripts/clean_clone.py`
  or your own).

### 10. Tests and tracing (6)

- Unit tests for the parts whose output depends only on their input.
- At least one integration test through the running API, with a fake LLM.
- Traces (Langfuse or similar) where each agent step is visible: tool calls,
  retrieval, and each LLM call with its tokens and cost.

### 11. Evaluation (12)

This section is about working **eval-driven**: change something, measure it,
keep it or throw it away based on the numbers. We grade that process, not the
final score.

So don't game the eval. Sixteen questions are easy to overfit, and a high score
earned that way is worth nothing to us. Examples of what we mean:

- prompts or few-shot examples written around these exact questions;
- patches that only work here, like special cases for these companies or
  regexes for these phrasings;
- answers shaped to fit the scorer rather than the user.

A fair test: would the change still help with a question about a report that
isn't in the set? If you're not sure, say so in the write-up.

What we want to see:

- Run `eval.py` on the golden set and commit the scorecards to `runs/`.
- **Compare the agent with a single-pass pipeline** (one retrieval, one LLM call)
  on the same questions: accuracy, refusals, evidence cited, time and cost. End
  with your answer to "is the agent worth it?". If the agent loses and you say so
  with numbers, that counts as a good result.
- At least one improvement backed by the eval: change one thing (chunk size,
  reranker, a prompt, a tool description), run `eval.py` before and after, and
  show both scorecards.
- A short eval log: a section of the README or a separate `EVAL.md`. For each
  change you tried, give what you changed and why, the scorecard before and
  after, and whether you kept it. Changes that didn't help belong there too.

### 12. README (6)

- What the app does.
- An architecture diagram.
- How to run it from a fresh clone.
- The scorecard and the agent-vs-pipeline comparison.
- The main decisions and their trade-offs.
- **What doesn't work:** known failures and limitations.

## Advanced (bonus, outside the 100)

These are things the course did not build. A bonus counts only if it works and
the README shows the evidence: a number, a screenshot or a trace. Together they
are worth up to **+30**.

| extra | points |
|---|---|
| Deployed to the cloud with a public URL | +4 |
| OCR or a vision model for scanned pages, charts and figures | +4 |
| Real authentication (login, tokens) instead of a user-id header | +3 |
| A real web frontend (React, Vue, Svelte, ...) instead of Streamlit | +3 |
| Tables extracted as structured data the agent can query | +3 |
| An LLM judge for groundedness, checked against your own hand labels | +3 |
| Metadata filters taken from the question (company, year, document type) | +2 |
| A task queue with a separate worker for ingestion (Celery, RQ, Arq, ...) | +2 |
| A prompt-injection test set, with results | +2 |
| User feedback (thumbs up/down) saved and linked to traces | +2 |
| Your own idea, agreed with us first | up to +2 |

## What to submit

- A link to your git repository, public or shared with us.
- The README described above, with the scorecards committed.
- A 3-5 minute demo video showing an upload, the comparison question, a refusal
  and a trace.

## How we grade

- We clone your repo and run `docker compose up`. Anything we can't run, we can
  only partly grade.
- Each section can earn partial points.
- We run `eval.py` ourselves. No accuracy threshold is required: we grade how
  you got your numbers and how honestly you report them, not how high they are.
- Numbers count for more than adjectives. "Reranking raised accuracy from 9/13 to
  11/13" beats "reranking improved results".
