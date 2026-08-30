"""PriceRanger MCP — agent-native value check (LangChain + LangGraph example).

An open example of how an agentic process qualifies the PriceRanger MCP server:
what it is, what it is not, and how to grade it yourself before wiring it in.

STACK
  LangChain supplies the pieces (the chat model, the @tool decorator, and the
  MCP adapter that turns the server's tools into LangChain tools). LangGraph
  supplies the runtime: create_react_agent returns a CompiledStateGraph, so the
  reason-act loop is a state graph rather than a hand-rolled while-loop. The
  graph is a prebuilt -- nothing here authors nodes or edges -- which is the
  point: wiring an MCP server into an agent should not require graph plumbing.

  The grader is swappable. config.yaml picks openai or anthropic; the tools,
  the questions and the loop are identical either way. A service worth wiring
  in should survive being graded by more than one model.

WHAT THIS SERVICE IS
  Risk telemetry with receipts. Per asset, per hour: a calibrated range band
  that carries its own measured coverage against the 90% it promises, and a
  price ladder whose rungs carry graded touch odds and dwell time. The honest
  contract, on every card: this is not a forecast that beats the market, and it
  does not describe itself as one.

WHAT YOU CANNOT GET ELSEWHERE
  Not the numbers — the *evidence the numbers keep*. A broker API hands you
  price. This hands you a band plus its realized coverage, a verdict, the list
  of calibration tests it failed, a multiple-testing correction, and per-rung
  fill evidence — read by an agent in one call, at machine speed, over an open
  standard (MCP). The premium lane (models that beat the persistence baseline)
  is graded against that same baseline and only ships where it wins; when it
  does, it lands behind the same tools.

HOW TO RUN
  pip install -r requirements.txt
  export PRICERANGER_TOKEN=<your minted token>     # signup at priceranger.ai
  export OPENAI_API_KEY=<your key>                 # or ANTHROPIC_API_KEY
  python priceranger_mcp_eval.py

  Pick the grader in config.yaml (provider: openai | anthropic), or point
  PRICERANGER_EVAL_CONFIG at another file. Only the selected provider is
  imported, so you need just the one installed. The resolved provider, model
  and config source are printed at the top of every run.

  The routing check reads the published edge-universe JSON by default. Override
  it with PRICERANGER_EDGE_UNIVERSE_URL, or point PRICERANGER_EDGE_UNIVERSE_PATH
  at a local copy. If it is unreachable the eval skips that step rather than
  failing.

THE POINT
  The agent does not take the marketing at its word. It reads the routing
  surface, then opens the graded receipts on both the "use us" and the "use the
  free baseline" assets, and checks whether the claim survives its own data.
  That is how this service is meant to be graded: not "would you use it," but
  "does the receipt back the claim." The answer below is the model's, not ours.
"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.request

import httpx
import yaml

try:
    from langchain_core.tools import tool
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.prebuilt import create_react_agent
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "Missing deps: pip install -r requirements.txt"
    ) from exc

MCP_URL = "https://priceranger.ai/mcp/"

# The public endpoint rate-limits by IP (nginx: ~10 req/s, short burst), and the
# MCP adapter opens a fresh session per tool call (initialize + call + teardown
# is several requests). A fast agent fanning out across the fleet trips that
# budget and gets 429s as transport errors instead of tool payloads. The polite,
# production-correct behavior to model: serialize requests with a minimum gap so
# the answer comes back as data, not a transport error.
MIN_CALL_GAP_SECONDS = 1.0

# The routing artifact is published beside the widget cards. If the public JSON
# route is not live yet, point this at a local copy or leave it None; the eval
# then skips the routing step instead of failing.
UNIVERSE_URL = os.environ.get(
    "PRICERANGER_EDGE_UNIVERSE_URL",
    "https://priceranger.ai/widgets/edge_universe.json",
)
UNIVERSE_PATH = os.environ.get("PRICERANGER_EDGE_UNIVERSE_PATH")  # local override

TOKEN = os.environ.get("PRICERANGER_TOKEN", "").strip()
if not TOKEN:
    raise SystemExit(
        "PRICERANGER_TOKEN is not set. Mint a token from your priceranger.ai "
        "account; the MCP has no anonymous tier, so an agent needs one to read."
    )

# Which LLM grades the service. The agent, the tools and the questions are
# identical either way -- only the reasoner changes, which is the point: a
# service worth wiring in should survive being graded by more than one model.
CONFIG_PATH = os.environ.get(
    "PRICERANGER_EVAL_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
)

_PROVIDER_PACKAGES = {"openai": "langchain-openai",
                      "anthropic": "langchain-anthropic"}

DEFAULT_CONFIG = {
    "provider": "openai",
    "openai": {"model": "gpt-4o", "temperature": 0,
               "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"model": "claude-sonnet-4-5", "temperature": 0,
                  "api_key_env": "ANTHROPIC_API_KEY"},
}


def load_config() -> tuple[dict, str]:
    """Merge config.yaml over the built-in defaults; report where it came from."""
    try:
        with open(CONFIG_PATH) as fh:
            loaded = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return DEFAULT_CONFIG, "built-in defaults (no config.yaml)"
    except yaml.YAMLError as exc:
        raise SystemExit(f"{CONFIG_PATH} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"{CONFIG_PATH} must contain a YAML mapping.")
    cfg = {**DEFAULT_CONFIG, **loaded}
    for name in _PROVIDER_PACKAGES:
        cfg[name] = {**DEFAULT_CONFIG[name], **(loaded.get(name) or {})}
    return cfg, CONFIG_PATH


def build_llm(cfg: dict):
    """Instantiate the configured chat model, importing only that provider."""
    provider = str(cfg.get("provider", "")).strip().lower()
    if provider not in _PROVIDER_PACKAGES:
        raise SystemExit(
            f"config: 'provider' must be one of "
            f"{sorted(_PROVIDER_PACKAGES)}; got {cfg.get('provider')!r}"
        )
    settings = cfg[provider]
    key_env = settings.get("api_key_env") or ""
    if not os.environ.get(key_env, "").strip():
        raise SystemExit(
            f"provider is '{provider}', so {key_env} must be set. "
            f"Export it, or switch 'provider' in {CONFIG_PATH}."
        )
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI as Chat
        else:
            from langchain_anthropic import ChatAnthropic as Chat
    except ImportError as exc:
        raise SystemExit(
            f"provider '{provider}' needs: "
            f"pip install {_PROVIDER_PACKAGES[provider]}"
        ) from exc
    return Chat(model=settings["model"],
                temperature=settings.get("temperature", 0))


CONFIG, CONFIG_SOURCE = load_config()


class Bearer(httpx.Auth):
    """Attach the token as an Auth flow.

    The MCP adapter does not forward a bare Authorization header on tool calls,
    so a plain `headers={'Authorization': ...}` is silently dropped. An
    httpx.Auth subclass survives the call path.
    """

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {TOKEN}"
        yield request


class _PacedTransport(httpx.AsyncHTTPTransport):
    """Serialize and pace requests to the public MCP inside its rate budget.

    A class-level lock + clock so every client the adapter spins up shares the
    same pacing: each request waits its turn and leaves a minimum gap behind the
    last one. On a 429 the request waits out the window and is retried by the
    caller, so the tool's payload -- not a transport error -- carries the answer.
    """

    _lock = asyncio.Lock()
    _last = 0.0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self._lock:
            loop = asyncio.get_event_loop()
            gap = MIN_CALL_GAP_SECONDS - (loop.time() - self._last)
            if gap > 0:
                await asyncio.sleep(gap)
            response = await super().handle_async_request(request)
            self._last = loop.time()
            return response


def _paced_client_factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=headers, timeout=timeout, auth=auth,
                             follow_redirects=True,
                             transport=_PacedTransport())


def fetch_universe() -> dict:
    """Read the routing surface: which assets beat the free EWMA baseline."""
    try:
        if UNIVERSE_PATH:
            return json.loads(open(UNIVERSE_PATH).read())
        req = urllib.request.Request(UNIVERSE_URL,
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode()
        return json.loads(body)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the eval
        return {"error": f"{type(exc).__name__}: {exc}"}


QUESTION = """You are qualifying a market-data service to wire into a trading
agent. Grade it on evidence, not on its own descriptions. Cost is not the
question -- whether it earns a place in your stack is.

The service's contract, in its own words: risk telemetry with receipts, not
alpha. A calibrated hour-ahead band per asset with its measured coverage, and a
price ladder whose rungs carry graded touch odds and dwell time.

STEP 1 (if the read_edge_universe tool returns data): call it first. It lists
which assets the service's own forward-graded receipts say beat a free EWMA
baseline (use_us) and which do not (use_ewma). The whole claim is routing:
"use us where we beat the free baseline; elsewhere the free one is sharper."

STEP 2: verify the receipts. Pick two assets from use_us and two from use_ewma
(or any four assets if there is no universe), and for each call get_agent_brief
and get_price_ladder. Check: does the band publish its realized coverage against
its target, a verdict, and its failed tests? Do the ladder rungs carry measured
touch rates and dwell, or bare geometry? Do the use_us assets actually look
sharper than the use_ewma ones when you read their receipts?

Then answer, briefly and without flattery:
1. WIRE IT IN? YES or NO, one sentence why.
2. RECEIPTS BACK THE CLAIM? Does the coverage/verdict/failed-tests evidence
   actually travel with the numbers, and does the routing hold up under them?
3. UNIQUE? Name a specific fact you could NOT get from a broker API or a free
   feed. If none, write exactly: NOTHING.
4. THE PREMIUM HOOK: the service grades a model lane against the same
   persistence baseline and only ships it where the receipts say it wins, so a
   subscriber gets a sharper band the moment it earns its place -- without
   changing how they integrate. Is that a reason to stay subscribed as it
   lands? One sentence.
"""


async def main() -> None:
    client = MultiServerMCPClient(
        {"priceranger": {"url": MCP_URL, "transport": "streamable_http",
                         "auth": Bearer(),
                         "httpx_client_factory": _paced_client_factory}}
    )
    mcp_tools = await client.get_tools()

    @tool
    def read_edge_universe() -> dict:
        """The routing surface: which assets the service's forward-graded
        receipts say beat the free EWMA baseline (use_us) and which do not
        (use_ewma), with margin and receipt count for each. Read this before
        judging any asset. May be unavailable if the public JSON route is not
        live; in that case the eval proceeds without the routing step."""
        u = fetch_universe()
        if "error" in u:
            return {"available": False, "error": u["error"],
                    "note": "routing artifact not reachable; grade on receipts"}
        return {
            "generated_at": u.get("generated_at"),
            "baseline": u.get("baseline"),
            "horizon": u.get("horizon"),
            "means": u.get("means"),
            "use_us": u.get("use_us"),
            "use_ewma": u.get("use_ewma"),
            "counts": {"use_us": len(u.get("use_us", [])),
                       "use_ewma": len(u.get("use_ewma", []))},
        }

    tools = list(mcp_tools) + [read_edge_universe]
    print(f"[wired {len(mcp_tools)} MCP tools + edge universe; token from env]")

    llm = build_llm(CONFIG)
    active = CONFIG[CONFIG["provider"]]
    print(f"[grader: {CONFIG['provider']}/{active['model']} "
          f"temp={active.get('temperature', 0)} · config: {CONFIG_SOURCE}]")
    agent = create_react_agent(llm, tools)
    out = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
    msgs = out["messages"]
    calls = [tc.get("name") for m in msgs
             for tc in (getattr(m, "tool_calls", None) or [])]
    print(f"\ntools called: {calls}\n")
    print("=" * 72)
    print(msgs[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
