"""Clean-clone test: does `docker compose up` work from a fresh clone?

1. git clone the repository (what is committed, nothing else) into a temp folder;
2. copy only .env next to it (the one file that is not in git);
3. `docker compose up --build` under another project name and other ports, so a
   stack that is already running is not touched;
4. wait until the API is healthy and every starter PDF is ingested from scratch;
5. ask one question through the UI's /api proxy and check its citation;
6. `docker compose down -v` and delete the clone (unless --keep).

    python scripts/clean_clone.py              # clone this repository's HEAD
    python scripts/clean_clone.py --repo https://github.com/Ziko-108/agentic-rag.git

Ingesting the 14 reports takes a while (one low-priority parser process) and
embeds them again (about $0.03).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "ragcleanclone"
PORTS = {"DB_PORT": "15432", "API_PORT": "18000", "UI_PORT": "13000"}


def get(url: str, body: dict | None = None, timeout: float = 10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "X-User-Id": "clean-clone"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or "null")


def wait(what: str, check, timeout: float, every: float = 5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if check():
                print(f"  ok: {what} ({time.time() - start:.0f}s)")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(every)
    raise SystemExit(f"FAILED: {what} not reached within {timeout:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=str(ROOT), help="what to clone (default: this repository)")
    parser.add_argument("--keep", action="store_true", help="leave the stack and the clone running")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds to wait for ingestion")
    args = parser.parse_args()

    if not (ROOT / ".env").exists():
        raise SystemExit("needs .env in the repository root (copy .env.example and fill it in)")
    work = Path(tempfile.mkdtemp(prefix="clean-clone-"))
    clone = work / "repo"
    env = {**os.environ, **PORTS, "COMPOSE_PROJECT_NAME": PROJECT}
    compose = ["docker", "compose"]
    api, ui = f"http://127.0.0.1:{PORTS['API_PORT']}", f"http://127.0.0.1:{PORTS['UI_PORT']}"

    try:
        print(f"1. cloning {args.repo}")
        subprocess.run(["git", "clone", "--quiet", args.repo, str(clone)], check=True)
        shutil.copy(ROOT / ".env", clone / ".env")
        expected = len(list((clone / "starter/data/pdfs").glob("*.pdf")))

        print("2. docker compose up --build")
        subprocess.run([*compose, "up", "-d", "--build", "--quiet-pull"], cwd=clone, env=env, check=True)

        print("3. waiting")
        wait("API healthy", lambda: get(f"{api}/health")["status"] == "ok", timeout=300)
        wait("UI serves the page", lambda: urllib.request.urlopen(ui, timeout=5).status == 200, timeout=120)

        def ingested():
            docs = get(f"{api}/documents")
            failed = [d["filename"] for d in docs if d["status"] == "failed"]
            if failed:
                raise SystemExit(f"FAILED: ingestion failed for {failed}")
            ready = sum(d["status"] == "ready" for d in docs)
            print(f"     {ready}/{expected} reports ready", flush=True)
            return ready == expected

        wait(f"all {expected} reports ingested", ingested, timeout=args.timeout, every=30)

        print("4. one question through the UI proxy")
        answer = get(f"{ui}/api/chat", {"question": "What was RWE's total installed capacity?"}, timeout=180)
        cited = [(c["document"], c["page"]) for c in answer["citations"]]
        print(f"     {answer['answer'][:120]!r}\n     citations: {cited}")
        if not cited:
            raise SystemExit("FAILED: the answer has no citation")
        print("PASSED: a fresh clone starts, ingests and answers with citations")
    finally:
        if args.keep:
            print(f"kept: {clone} (stop with: docker compose -p {PROJECT} down -v)")
        else:
            print("5. cleaning up")
            subprocess.run([*compose, "down", "-v", "--remove-orphans"], cwd=clone, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
