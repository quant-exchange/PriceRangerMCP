"""PriceRanger MCP — full tool sweep. Deterministic, no LLM, exits non-zero on failure.

WHAT THIS IS
  The sibling of priceranger_mcp_eval.py. That one asks a language model to
    *judge* the service. This one asks nothing: it calls known public tools the server
    advertises, checks each payload actually contains the evidence the tool
  claims to carry, and prints what each tool is worth. No API key, no grader,
  no opinions — just the surface, exercised.

    Run it as a test. Exit 0 means contract checks passed; review unavailable,
    collecting, stale and skipped rows. Exit 1 means a contract or catalog failed.

WHY IT EXISTS
  Three failure modes it catches that a docs page cannot:

  1. A tool that 200s but has quietly stopped carrying its evidence. The band
     without its coverage receipt is just a number, and a number is what
     everyone else already sells you.
  2. A tool that appears on the public surface and should not. This endpoint is
     read-only research; if a name that writes, trades or cancels ever shows up
     in tools/list, that is a product bug and this run fails on it.
  3. Drift between what is documented and what is served. Anything advertised
     but unplanned here is reported as UNDOCUMENTED rather than skipped.

HOW TO RUN
  pip install -r requirements.txt
    export PRICERANGER_MCP_TOKEN=<your minted token> # signup at priceranger.ai
  python priceranger_mcp_tool_eval.py

  Options:
    --json            machine-readable result instead of the report
    --asset SYMBOL    pin the sweep to one asset (default: first you can read)
    --composition     print the cross-MCP composition notes and exit
    --catalog         print the tool catalog as markdown and exit

  No LLM provider is needed, so config.yaml is not read. Set
  PRICERANGER_OPERATOR only if you hold the owner's admin token; a minted user
  token already identifies its operator.

USING THIS WITH YOUR OWN TOOLS, AND WITH OTHER MCP SERVERS
  PriceRanger answers one question — "how wide is the honest range, and how
  well has that band actually held?" — and deliberately answers nothing else.
  It publishes no bid/ask, no account, no fills and no orders. That is not a
  gap to apologise for; it is what makes it safe to compose. The value shows up
  when you put it next to a server that does have those things.

  The obvious partner is Alpaca's MCP server (alpacahq/alpaca-mcp-server),
  which brings quotes, positions, orders and the market clock. Both are MCP, so
  an agent holds both at once and no glue code is written:

      from langchain_mcp_adapters.client import MultiServerMCPClient

      client = MultiServerMCPClient({
          "priceranger": {                      # research: read-only, remote
              "url": "https://priceranger.ai/mcp",
              "transport": "streamable_http",
              "auth": Bearer(),                 # see the class below
          },
          "alpaca": {                           # execution: local, stdio
              "command": "uvx",
              "args": ["alpaca-mcp-server"],
              "transport": "stdio",
              "env": {"ALPACA_API_KEY": "...",
                      "ALPACA_SECRET_KEY": "...",
                      "ALPACA_PAPER_TRADE": "true"},
          },
      })
      tools = await client.get_tools()          # both servers, one tool list

  Three handoffs that are worth more than either server alone:

  * PRICE THE SPREAD YOU ACTUALLY PAY. plan_placement ranks resting rungs by
    P(fill) x price improvement, but PriceRanger publishes no bid/ask, so
    cost_bps defaults to 0 and every figure is GROSS. Alpaca's
    get_stock_latest_quote / get_crypto_latest_quote gives you the real spread.
    Convert it to basis points and pass it in — a rung a fraction of a bp from
    center is inside the spread and is not a real placement. This single wiring
    turns a gross ranking into a net one.

  * SIZE OFF THE BAND, NOT THE FORECAST. get_agent_brief carries band_coverage
    against its target plus band_failed_tests. Pair that width with Alpaca's
    get_account_info buying power and get_all_positions to size a position
    against a *measured* hour-ahead width instead of a guessed one. When
    band_failed_tests contains "independence", breaches arrive in clusters —
    the failure mode that empties an account — and that is a reason to size
    down, which no price feed will ever tell you.

  * DO NOT READ A STALE CARD. Equity cards only refresh during market sessions,
    so they read stale off-hours by design. Alpaca's get_clock tells you
    whether the venue is open before you trust a stock card's freshness block.

  Where the boundary sits, explicitly: PriceRanger is read-only research and is
  not execution-proven. Alpaca places real orders. The handoff between them is
  where money starts moving, and it should be your code and your risk limits —
  not an inference chain. Keep ALPACA_PAPER_TRADE=true until your own receipts
  say otherwise, and use ALPACA_TOOLSETS to hand the agent only the toolsets it
  needs (e.g. "stock-data,crypto-data" for a research loop that cannot trade at
  all). The same shape works for any other MCP server: let PriceRanger answer
  the range question and let the specialist answer its own.

NOTE ON DUPLICATION
  The Bearer auth class and the paced transport below also appear in
    priceranger_mcp_eval.py. Keep those two files together: the grader imports
    the sweep's public catalog and payload/scope helpers to prevent drift.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time

import httpx

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit("Missing deps: pip install -r requirements.txt") from exc

MCP_URL = os.environ.get("PRICERANGER_MCP_URL", "https://priceranger.ai/mcp")

# nginx rate-limits the public endpoint (~10 req/s per IP). This sweep holds one
# session open for the whole run, so it is far lighter than an agent that opens
# a session per call — but it still paces itself rather than sprinting.
MIN_CALL_GAP_SECONDS = 0.35

TOKEN = (os.environ.get("PRICERANGER_MCP_TOKEN", "").strip()
         or os.environ.get("PRICERANGER_TOKEN", "").strip())
# A minted user token already identifies its operator. The owner's admin token
# does not, so it names one explicitly. Normal users leave this unset.
OPERATOR = os.environ.get("PRICERANGER_OPERATOR", "").strip()

# A read-only research endpoint has no business advertising these. If any tool
# name starts with one of these verbs, the run fails: it means a desk-tier
# surface reached the public catalog. This is not hypothetical -- a dependency
# upgrade silently disabled the catalog filter once and published 41 tools
# instead of 15, including trade and cancel_order, for about a day.
#
# Prefixes, not substrings: "get_multi_asset_brief" contains "set_" and
# "get_trade_guide" contains "trade", and both are reads. Anything else that
# leaks is caught as UNDOCUMENTED below.
FORBIDDEN_PREFIXES = (
    "trade", "place_", "cancel_", "replace_", "claim_", "release_",
    "set_", "run_", "stop_", "prepare_", "manage_", "watch_", "plan_order",
)


class Bearer(httpx.Auth):
    """Attach the token as an Auth flow.

    The MCP adapter does not forward a bare Authorization header on tool calls,
    so a plain `headers={'Authorization': ...}` is silently dropped. An
    httpx.Auth subclass survives the call path.
    """

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {TOKEN}"
        if OPERATOR:
            request.headers["X-Operator-Id"] = OPERATOR
        yield request


class _PacedTransport(httpx.AsyncHTTPTransport):
    """Serialize and pace requests so the answer comes back as data, not a 429."""

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


# ---------------------------------------------------------------------------
# The plan: every public tool, what it is for, and the fields that prove it.
#
# `evidence` is the point of this file. Any tool can return 200 and a shape.
# These are the keys that carry the receipt -- the coverage number, the verdict,
# the failed tests, the cost model, the honest caveat. A tool that answers
# without them has stopped being worth calling, and the sweep says so.
# ---------------------------------------------------------------------------
PLAN: dict[str, dict] = {
    "whoami": {
        "value": "Identity and entitlement before you spend a call: which "
                 "assets this token may read, and which endpoint profile "
                 "answered.",
        "args": lambda ctx: {},
        "evidence": ("operator", "allowed_assets", "endpoint_profile", "token_role"),
    },
    "list_assets": {
        "value": "The published pool and its sector grouping, plus "
                 "allowed_to_you -- iterate that, not the pool, or expect "
                 "per-asset permission errors.",
        "args": lambda ctx: {},
        "evidence": ("pool", "allowed_to_you", "access_note"),
    },
    "get_mcp_methodology": {
        "value": "The full research contract in one call: what the grades "
                 "score, what the fill model does not model, and the "
                 "independent baseline every band is measured against.",
        "args": lambda ctx: {"asset": ctx["asset"], "env": "paper"},
        "evidence": ("about", "methodology", "safety_boundaries"),
    },
    "get_agent_brief": {
        "value": "The intended entry point and the cheapest honest read on the "
                 "server: one asset's coverage against target, band verdict, "
                 "failed calibration tests, and skill vs the free EWMA "
                 "baseline -- around 1KB.",
        "args": lambda ctx: {"asset": ctx["asset"], "compact": True},
        "evidence": ("band_coverage", "band_verdict", "band_failed_tests",
                     "band_baseline_skill_pct", "center_beats_baseline",
                     "recommended_next_tool"),
    },
    "get_multi_asset_brief": {
        "value": "The same triage row across your whole allow-list in one "
                 "call, so an agent scans first and opens the expensive tools "
                 "only where the receipts justify it.",
        "args": lambda ctx: {},
        "evidence": ("briefs", "count", "note"),
    },
    "get_status": {
        "value": "The graded current-hour report card: accuracy grade with its "
                 "scope stated, freshness, and which lane produced the numbers.",
        "args": lambda ctx: {"asset": ctx["asset"]},
        "evidence": ("forecast_accuracy", "freshness", "construction", "about"),
    },
    "get_price_ladder": {
        "value": "The primary artifact. Numeric buy/sell rungs from the "
                 "model-free band, with the calibration audit and range-skill "
                 "card that say how much trust they have earned. The neural "
                 "lane's own ladder is fenced under research, graded against "
                 "the band, never merged into it.",
        "args": lambda ctx: {"asset": ctx["asset"]},
        "evidence": ("buy_ladder", "sell_ladder", "range_skill_card",
                     "calibration_audit", "walk_forward", "tail_zone",
                     "construction"),
    },
    "get_range_forecast": {
        "value": "The h+1..h+4 fan, each horizon written before its bar existed "
                 "and graded after it closed, carrying its own coverage state "
                 "and verdict. Direct heads, so h+4 error does not compound.",
        "args": lambda ctx: {"asset": ctx["asset"]},
        "evidence": ("horizons", "coverage_state", "band_readiness", "lane"),
    },
    "get_range_edge": {
        "value": "Rung net-capture, stop-first corrected: a breached stop is a "
                 "realised loss even if price later recovered. Declares its own "
                 "staleness and whether the edge is execution-validated or "
                 "still a counterfactual estimate.",
        "args": lambda ctx: {"asset": ctx["asset"]},
        "evidence": ("range_edge", "schema_version", "freshness"),
    },
    "get_touch_probability": {
        "value": "How reachable an arbitrary price is, interpolated from the "
                 "asset's own graded touch curve rather than assumed. Says "
                 "plainly that it is a touch estimate, not a fill and not a "
                 "direction call.",
        "args": lambda ctx: {"asset": ctx["asset"], "price": ctx["probe_price"],
                             "horizon_hours": 1},
        "evidence": ("touch_probability_pct", "touch_calibration",
                     "tail_zone_evidence", "fill_model", "means"),
    },
    "plan_placement": {
        "value": "Where to rest a limit order and what waiting is worth: every "
                 "rung ranked by P(fill by deadline) x price improvement. "
                 "Ships its own cost_model and a not_modelled field admitting "
                 "it does not price the cost of NOT filling.",
        "args": lambda ctx: {"asset": ctx["asset"], "side": "buy",
                             "deadline_minutes": 60.0, "cost_bps": 0.0},
        "evidence": ("recommended", "alternatives", "cost_model",
                     "not_modelled", "ranking_objective"),
    },
    "get_shadow_frequency": {
        "value": "Compare current 1H and 4H research receipts, sample counts, "
                 "coverage and limitations without assuming either horizon "
                 "wins. Shadow research is not routed or tradable.",
        "args": lambda ctx: {},
        "evidence": ("fleet", "theory", "not_routing", "signal_lanes",
                     "lanes_omitted"),
        # The per-asset scope is the same tool with an argument. Exercise both,
        # or the merge that replaced get_shadow_frequency_asset goes untested.
        "also": {
            "label": "get_shadow_frequency(asset=)",
            "args": lambda ctx: {"asset": ctx["asset"]},
            "evidence": ("h1_band", "h4_band", "h1_direction", "h4_direction"),
            "value": "The same card scoped to one symbol: does its 4h band "
                     "cover where its 1h band misses?",
        },
    },
    "get_trade_guide": {
        "value": "The numbers in plain English: how to read a ladder, which "
                 "levels are reference points, and where the read-only "
                 "boundary sits.",
        "args": lambda ctx: {"asset": ctx["asset"], "env": "paper"},
        "evidence": ("how_to_read", "next_calls", "plain_english"),
    },
}


def _payload(result) -> dict:
    """Unwrap an MCP CallToolResult into the tool's dict payload."""
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        # FastMCP wraps a non-dict return under 'result'; ours return dicts.
        value = data.get("result") if set(data) == {"result"} else data
        return value if isinstance(value, dict) else {"_invalid_payload_type": type(value).__name__}
    for block in (getattr(result, "content", None) or []):
        text = getattr(block, "text", None)
        if text:
            try:
                value = json.loads(text)
                return value if isinstance(value, dict) else {"_invalid_payload_type": type(value).__name__}
            except json.JSONDecodeError:
                return {"_text": text}
    return {}


def _forbidden(name: str) -> str | None:
    for prefix in FORBIDDEN_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return None


def _choose_asset(catalog: dict, override: str | None) -> str:
    allowed = catalog.get("allowed_to_you")
    if not isinstance(allowed, list) or not allowed or not all(isinstance(asset, str) and asset for asset in allowed):
        raise SystemExit("No valid allowed_to_you asset scope. Check the token's access; no asset was guessed.")
    selected = override.upper() if override else allowed[0]
    if selected not in allowed:
        raise SystemExit(f"{selected} is outside this token's allowed_to_you scope.")
    return selected


def _probe_price(brief: dict) -> float | None:
    center = brief.get("forecast_center")
    if isinstance(center, bool) or not isinstance(center, (int, float)) or not math.isfinite(center) or center <= 0:
        return None
    return float(center) * 0.99


_DICT_EVIDENCE = {
    "about", "band_coverage", "forecast_accuracy", "freshness", "construction",
    "range_skill_card", "calibration_audit", "walk_forward", "tail_zone",
    "band_readiness", "range_edge", "touch_calibration", "tail_zone_evidence",
    "fill_model", "recommended", "cost_model", "how_to_read", "fleet",
    "h1_band", "h4_band", "h1_direction", "h4_direction",
}
_LIST_EVIDENCE = {
    "allowed_assets", "pool", "allowed_to_you", "methodology", "safety_boundaries",
    "band_failed_tests", "briefs", "horizons", "alternatives", "signal_lanes", "next_calls",
}
_TEXT_EVIDENCE = {
    "operator", "endpoint_profile", "token_role", "access_note", "band_verdict",
    "recommended_next_tool", "note", "coverage_state", "lane", "schema_version",
    "means", "ranking_objective", "plain_english",
}
_PENDING_STATES = {"calibrating", "collecting", "insufficient", "insufficient_evidence", "missing", "unavailable"}


def _evidence_status(payload: dict, required: tuple[str, ...]) -> tuple[str, list[str]]:
    if not isinstance(payload, dict) or not payload:
        return "THIN", ["payload: expected a nonempty object"]
    if "available" in payload and not isinstance(payload["available"], bool):
        return "THIN", ["available: expected a boolean"]
    if payload.get("available") is False:
        return "UNAVAILABLE", []
    coverage = payload.get("band_coverage")
    state = coverage.get("state") if isinstance(coverage, dict) else payload.get("coverage_state")
    collecting = state in _PENDING_STATES
    issues = []
    for key in required:
        value = payload.get(key)
        if key not in payload:
            issues.append(key)
        elif key in _DICT_EVIDENCE and (not isinstance(value, dict) or not value):
            issues.append(f"{key}: expected a nonempty object")
        elif key in _LIST_EVIDENCE and not isinstance(value, list):
            issues.append(f"{key}: expected an array")
        elif key in _TEXT_EVIDENCE and (not isinstance(value, str) or not value.strip()):
            issues.append(f"{key}: expected nonempty text")
        elif key in {"band_baseline_skill_pct", "touch_probability_pct"}:
            if value is None and collecting:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                issues.append(f"{key}: expected a finite number or explicitly immature evidence")
            elif key == "touch_probability_pct" and not 0 <= value <= 100:
                issues.append(f"{key}: outside 0..100")
        elif key == "center_beats_baseline":
            if not (isinstance(value, bool) or value is None and collecting):
                issues.append(f"{key}: expected a boolean or explicitly immature evidence")
        elif key == "not_routing" and value is not True:
            issues.append("not_routing: expected true")
        elif key == "count" and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            issues.append("count: expected a nonnegative integer")
        elif value is None:
            issues.append(f"{key}: null evidence without an explicit unavailable state")
    if "endpoint_profile" in required and payload.get("endpoint_profile") != "analytics":
        issues.append("endpoint_profile: expected analytics")
    if isinstance(coverage, dict) and coverage.get("state") == "measured":
        for key in ("band_coverage_pct", "coverage_target_pct"):
            value = coverage.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                issues.append(f"band_coverage.{key}: invalid measured percentage")
    if "schema_version" in required and payload.get("schema_version") != "range_edge.v2":
        issues.append("schema_version: unsupported range-edge contract")
    edge = payload.get("range_edge")
    if "range_edge" in required and isinstance(edge, dict):
        if edge.get("published") is not True or edge.get("accounting") != "live_policy_stop_first":
            issues.append("range_edge: missing corrected public accounting contract")
        execution = edge.get("execution")
        if not isinstance(execution, dict) or not isinstance(execution.get("validated"), bool):
            issues.append("range_edge.execution.validated: expected a boolean")
        if not isinstance(edge.get("stale"), bool):
            issues.append("range_edge.stale: expected a boolean")
        for key in ("netcap_pct", "sample_windows"):
            value = edge.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                issues.append(f"range_edge.{key}: expected finite evidence or an explicit null")
    if issues:
        return "THIN", issues
    if collecting:
        return "COLLECTING", []
    if payload.get("stale") is True or (isinstance(edge, dict) and edge.get("stale") is True) or (isinstance(payload.get("freshness"), dict) and
                                       (payload["freshness"].get("stale") is True or payload["freshness"].get("state") == "stale")):
        return "STALE", []
    return "OK", []


async def sweep(asset_override: str | None) -> dict:
    client = MultiServerMCPClient({
        "priceranger": {"url": MCP_URL, "transport": "streamable_http",
                        "auth": Bearer(),
                        "httpx_client_factory": _paced_client_factory}
    })

    async with client.session("priceranger") as session:
        advertised = [t.name for t in (await session.list_tools()).tools]

        async def call(name: str, args: dict) -> tuple[dict, float, str | None]:
            started = time.perf_counter()
            try:
                result = await session.call_tool(name, args)
            except Exception as exc:  # noqa: BLE001 - a failed tool is a result
                return {}, (time.perf_counter() - started) * 1000, \
                    f"{type(exc).__name__}: {exc}"
            elapsed = (time.perf_counter() - started) * 1000
            if getattr(result, "isError", False):
                detail = " ".join(
                    (getattr(b, "text", "") or "")
                    for b in (getattr(result, "content", None) or [])
                ).strip()
                return {}, elapsed, detail or "tool returned isError"
            return _payload(result), elapsed, None

        # Context for the arg builders: read the allow-list, then anchor the
        # touch probe on this asset's own forecast center rather than a guess.
        who, _, err = await call("whoami", {})
        if err:
            raise SystemExit(f"whoami failed, cannot sweep: {err}")
        if who.get("endpoint_profile") != "analytics":
            raise SystemExit("This example requires the public analytics profile, not a private desk endpoint.")
        catalog, _, err = await call("list_assets", {})
        if err:
            raise SystemExit(f"list_assets failed, cannot resolve scope: {err}")
        asset = _choose_asset(catalog, asset_override)
        brief, _, _ = await call("get_agent_brief", {"asset": asset, "compact": True})
        ctx = {
            "asset": asset,
            "operator": who.get("operator"),
            # A shade below center: a plausible buy-side resting level, not a
            # tail probe, so the touch curve is exercised in its measured range.
            "probe_price": _probe_price(brief),
        }

        async def grade(label: str, name: str, case: dict) -> dict:
            if name == "get_touch_probability" and ctx["probe_price"] is None:
                return {"tool": label, "status": "SKIPPED", "ms": 0, "bytes": 0,
                        "missing": [], "error": None, "value": case["value"],
                        "reason": "No valid forecast_center was published; no price was invented."}
            payload, elapsed, error = await call(name, case["args"](ctx))
            if error:
                status, missing = "FAIL", []
            else:
                status, missing = _evidence_status(payload, case["evidence"])
            return {"tool": label, "status": status, "ms": round(elapsed),
                    "bytes": len(json.dumps(payload)), "missing": missing,
                    "error": error, "value": case["value"]}

        rows = []
        for name in advertised:
            case = PLAN.get(name)
            if case is None:
                rows.append({"tool": name, "status": "UNDOCUMENTED", "ms": 0,
                             "bytes": 0, "missing": [], "value": None})
                continue
            rows.append(await grade(name, name, case))
            also = case.get("also")
            if also:
                rows.append(await grade(also["label"], name, also))

        planned_missing = sorted(set(PLAN) - set(advertised))
        leaked = {n: f for n in advertised if (f := _forbidden(n))}

    return {"url": MCP_URL, "asset": ctx["asset"], "operator": ctx["operator"],
            "advertised": len(advertised), "rows": rows,
            "not_advertised": planned_missing, "leaked_write_tools": leaked}


COMPOSITION = """\
COMPOSING PRICERANGER WITH OTHER MCP SERVERS
  PriceRanger answers one question -- how wide is the honest range, and how well
  has that band actually held -- and publishes no bid/ask, account, fill or
  order. That is what makes it safe to compose. Point it at a server that has
  those things (Alpaca's MCP is the obvious one) and hold both in one agent:

    client = MultiServerMCPClient({
        "priceranger": {"url": "https://priceranger.ai/mcp",
                        "transport": "streamable_http", "auth": Bearer()},
        "alpaca":      {"command": "uvx", "args": ["alpaca-mcp-server"],
                        "transport": "stdio",
                        "env": {"ALPACA_API_KEY": "...",
                                "ALPACA_SECRET_KEY": "...",
                                "ALPACA_PAPER_TRADE": "true"}},
    })

  Handoffs worth more than either server alone:

    plan_placement(cost_bps=0)        -> gross ranking, no spread published
      + alpaca get_stock_latest_quote -> real spread in bps
      = a NET ranking. A rung a fraction of a bp from center is inside the
        spread and was never a real placement.

    get_agent_brief.band_coverage     -> a measured hour-ahead width
      + alpaca get_account_info / get_all_positions
      = size against evidence. band_failed_tests containing "independence"
        means breaches cluster -- the failure mode that empties an account.

    get_status.freshness (stocks read stale off-hours, by design)
      + alpaca get_clock
      = never act on a card the venue was closed for.

  The boundary, stated plainly: PriceRanger is read-only research and is not
  execution-proven. Alpaca places real orders. The step between them is where
  money starts moving and it should be your code and your risk limits, not an
  inference chain. Keep ALPACA_PAPER_TRADE=true until your own receipts say
  otherwise, and use ALPACA_TOOLSETS to hand the agent only what it needs.
"""


def report(result: dict) -> int:
    icon = {"OK": "PASS", "THIN": "THIN", "FAIL": "FAIL",
            "UNAVAILABLE": "N/A ", "UNDOCUMENTED": "NEW ",
            "SKIPPED": "SKIP", "COLLECTING": "WAIT", "STALE": "OLD "}
    print(f"PriceRanger MCP tool sweep — {result['url']}")
    print(f"operator: {result['operator']}   asset: {result['asset']}   "
          f"tools advertised: {result['advertised']}\n")

    total = 0
    for row in result["rows"]:
        print(f"[{icon[row['status']]}] {row['tool']:<24} "
              f"{row['bytes']:>7,}B {row['ms']:>5}ms")
        if row["value"]:
            for line in _wrap(row["value"], 74):
                print(f"         {line}")
        if row.get("missing"):
            print(f"         !! missing evidence: {', '.join(row['missing'])}")
        if row.get("error"):
            print(f"         !! {row['error']}")
        if row.get("reason"):
            print(f"         {row['reason']}")
        total += row["bytes"]
        print()

    print("=" * 78)
    counts: dict[str, int] = {}
    for row in result["rows"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"whole surface, one pass: {total:,}B (~{total // 4:,} tokens)")

    failures = [r["tool"] for r in result["rows"]
                if r["status"] in ("FAIL", "THIN", "UNDOCUMENTED")]
    if result["not_advertised"]:
        print(f"\nplanned but NOT advertised: {result['not_advertised']}")
    if result["leaked_write_tools"]:
        print("\nWRITE-CAPABLE TOOL ON A READ-ONLY ENDPOINT: "
              f"{result['leaked_write_tools']}")
    print()
    print(COMPOSITION)

    if failures or result["not_advertised"] or result["leaked_write_tools"]:
        print(f"RESULT: FAILED — {failures or ''} "
              f"{result['not_advertised'] or ''} "
              f"{result['leaked_write_tools'] or ''}".strip())
        return 1
    print("RESULT: PASSED — contract checks passed; review any SKIPPED, UNAVAILABLE, COLLECTING or STALE rows.")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


def catalog_markdown() -> str:
    """The catalog as a markdown table, so the README never hand-types it.

    Prose in a docs file drifts from the code silently, and always in the
    flattering direction. Regenerate with --catalog instead of editing by hand.
    """
    rows = ["| Tool | What a dev gets | Evidence it must carry |",
            "|---|---|---|"]
    for name, case in PLAN.items():
        value = " ".join(case["value"].split()).replace("|", "\\|")
        keys = ", ".join(f"`{k}`" for k in case["evidence"])
        rows.append(f"| `{name}` | {value} | {keys} |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="emit the raw result instead of the report")
    parser.add_argument("--asset", help="pin the sweep to one symbol")
    parser.add_argument("--composition", action="store_true",
                        help="print the cross-MCP composition notes and exit")
    parser.add_argument("--catalog", action="store_true",
                        help="print the tool catalog as markdown and exit")
    args = parser.parse_args()

    if args.composition:
        print(COMPOSITION)
        return
    if args.catalog:
        print(catalog_markdown())
        return
    if not TOKEN:
        raise SystemExit(
            "PRICERANGER_MCP_TOKEN is not set (PRICERANGER_TOKEN is a legacy alias). Mint a token from your "
            "priceranger.ai account; the MCP has no anonymous tier."
        )

    result = asyncio.run(sweep(args.asset))
    if args.json:
        print(json.dumps(result, indent=2))
        bad = (result["not_advertised"] or result["leaked_write_tools"]
               or [r for r in result["rows"]
                   if r["status"] in ("FAIL", "THIN", "UNDOCUMENTED")])
        sys.exit(1 if bad else 0)
    sys.exit(report(result))


if __name__ == "__main__":
    main()
