// Every /api/* request from the browser is forwarded to the FastAPI backend.
// The browser then talks to one origin (no CORS), and the backend address
// stays a server-side setting (API_URL=http://api:8000 inside docker compose).

const API_URL = process.env.API_URL ?? "http://localhost:8000";

export const dynamic = "force-dynamic";

const FORWARD_REQUEST = ["content-type", "x-user-id", "accept"];
const FORWARD_RESPONSE = ["content-type", "cache-control", "x-conversation-id"];

async function proxy(req: Request, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = `${API_URL}/${path.map(encodeURIComponent).join("/")}${new URL(req.url).search}`;

  const headers = new Headers();
  for (const name of FORWARD_REQUEST) {
    const value = req.headers.get(name);
    if (value) headers.set(name, value);
  }
  const hasBody = req.method !== "GET" && req.method !== "HEAD" && req.method !== "DELETE";

  let res: Response;
  try {
    res = await fetch(target, {
      method: req.method,
      headers,
      body: hasBody ? await req.arrayBuffer() : undefined, // uploads are at most 50 MB
      cache: "no-store",
    });
  } catch {
    return Response.json({ detail: "The API is not reachable. Is the backend running?" }, { status: 503 });
  }

  const out = new Headers();
  for (const name of FORWARD_RESPONSE) {
    const value = res.headers.get(name);
    if (value) out.set(name, value);
  }
  return new Response(res.body, { status: res.status, headers: out }); // streams SSE through
}

export { proxy as GET, proxy as POST, proxy as DELETE };
