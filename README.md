<img width="1944" height="432" alt="priceranger_logo" src="https://github.com/user-attachments/assets/f5123b8e-4315-43e6-96ee-4a7cabd08b54" />

# PriceRanger MCP Examples

Python clients for the public, read-only [PriceRanger MCP](https://priceranger.ai/mcp).
Access is free during Open Research & Development. These examples read published
ranges, forecast results, coverage, uncertainty, and warnings. They do not place
trades, manage accounts, or expose our private trading desk.

## Start Here: One Research Brief

No LLM, provider API key, or broker account is needed.

```bash
python -m pip install "fastmcp>=3,<4"
```

Generate a personal token in [Settings](https://priceranger.ai/settings#mcp-token).
Keep it in your environment or secret manager, not in the script or source control.

### Linux, macOS, or WSL

```bash
export PRICERANGER_MCP_TOKEN='your_personal_token'
python priceranger_mcp_quickstart.py
```

### PowerShell

```powershell
$env:PRICERANGER_MCP_TOKEN = 'your_personal_token'
python priceranger_mcp_quickstart.py
```

### Windows Command Prompt

```bat
set PRICERANGER_MCP_TOKEN=your_personal_token
python priceranger_mcp_quickstart.py
```

The script confirms the public analytics profile, calls `list_assets`, selects
from `allowed_to_you`, and prints a compact `get_agent_brief` response. It uses
the same workflow as the MCP and Settings Python reference. It does not assume
that BTC, or any other particular symbol, belongs to your token's scope.

Use `https://priceranger.ai/mcp` without a trailing slash. A personal token needs
only an Authorization bearer header. Never send broker keys to this endpoint.
The examples accept `PRICERANGER_TOKEN` as a legacy alias; when both are set,
`PRICERANGER_MCP_TOKEN` wins. Rotate expired or lost tokens in Settings.

## Optional Advanced Examples

| File | Purpose | LLM required? |
|---|---|---|
| `priceranger_mcp_quickstart.py` | Read one permitted research brief | No |
| `priceranger_mcp_tool_eval.py` | Check the public catalog and evidence contracts | No |
| `priceranger_mcp_eval.py` | Ask an LLM to assess current evidence and integration limits | Yes |
| `test_examples.py` | Synthetic regression tests; no network or provider calls | No |

For the advanced examples and tests:

```bash
python -m pip install -r requirements.txt
```

The full install includes both optional LLM providers. Only the selected
provider is imported when building the grader. The LangGraph prebuilt agent is
pinned to its supported 1.2 line; its deprecation warning is not a service result.

## Deterministic Tool Sweep

<img width="1604" height="3164" alt="tool-sweep-dark" src="https://github.com/user-attachments/assets/783e86d3-9207-4df7-91c4-1717d12643c0" />

```bash
python priceranger_mcp_tool_eval.py
python priceranger_mcp_tool_eval.py --asset ETH
python priceranger_mcp_tool_eval.py --json
```

The sweep expects the 13 public analytics tools. It reports undocumented tools,
missing tools, and private/write-capable names as failures and never invokes
unknown tools. It requires the public analytics profile and resolves asset scope
through `list_assets.allowed_to_you`. An empty scope or an out-of-scope override
stops the run instead of guessing BTC.

Evidence checks validate presence, types, finite values, and explicit states.
Present-but-null evidence is not a pass. Legitimate immature or unavailable
results are labeled rather than converted into numerical claims. Touch probes
use a valid published forecast center; if no center exists, that probe is skipped
with a reason instead of inventing a price.

| Status | Meaning |
|---|---|
| `OK` | The checked evidence contract is usable, not proof of profitable trading |
| `COLLECTING` | Explicitly immature evidence; not a measured win or loss |
| `STALE` | Structurally valid historical evidence, not a current forecast |
| `UNAVAILABLE` | The endpoint explicitly reports no available data |
| `SKIPPED` | A required input is unavailable; no guessed probe was sent |
| `THIN` | Missing, null, malformed, or unsupported evidence |
| `FAIL` | The tool call failed |
| `UNDOCUMENTED` | An unexpected tool appeared in the catalog and was not called |

`FAIL`, `THIN`, catalog drift, or leaked private/write tools produce a nonzero
exit. A zero exit means contract checks passed; review all other states before
using the data. This is a client-side diagnostic, not independent certification.

```bash
python priceranger_mcp_tool_eval.py --catalog
python priceranger_mcp_tool_eval.py --composition
```

Those two documentation commands need no token. The catalog text is generated
from the sweep's plan, so it stays aligned with the checks it actually performs.

## Optional LLM Grader

<img width="2264" height="5317" alt="how-it-works-dark" src="https://github.com/user-attachments/assets/76985a6a-8a06-4754-bf18-6c461a576655" />

```bash
export OPENAI_API_KEY='your_provider_key'
python priceranger_mcp_eval.py
```

Or select Anthropic in your local `config.yaml` and set `ANTHROPIC_API_KEY`:

```yaml
provider: anthropic
anthropic:
  model: claude-sonnet-4-5
  temperature: 0
  api_key_env: ANTHROPIC_API_KEY
```

With no config file, the grader uses its built-in OpenAI defaults. The
`api_key_env` field is an environment-variable name, never a key. Do not commit
your actual tokens, keys, or local config secrets. Provider calls can incur
charges; the deterministic sweep and quick start do not need a provider key.

The grader verifies the catalog before binding tools, checks identity and scope,
and filters the public routing artifact to permitted assets. It asks for current
metrics, timestamps, baselines, and limitations. It does not assume that one
horizon wins, that a result is unique, or that a future paid/model service exists.
The LLM's answer is a judgment, not a contract test or a trading signal.

## Configuration

| Variable | Purpose |
|---|---|
| `PRICERANGER_MCP_TOKEN` | Canonical personal bearer token; required for reads |
| `PRICERANGER_TOKEN` | Legacy fallback when the canonical token is empty |
| `PRICERANGER_MCP_URL` | Endpoint override for all examples; defaults to the public HTTPS URL |
| `PRICERANGER_OPERATOR` | Advanced shared-admin-token identity only; personal tokens do not need it |
| `PRICERANGER_EVAL_CONFIG` | Optional grader configuration file |
| `PRICERANGER_EDGE_UNIVERSE_URL` | Optional public routing artifact URL |
| `PRICERANGER_EDGE_UNIVERSE_PATH` | Optional local routing artifact for the grader |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Optional grader provider credential |

`env.example` is a secret-free template. No dotenv loader runs automatically.
On POSIX shells, you can copy it to a gitignored `.env` and load it explicitly:

```bash
cp env.example .env
set -a
source .env
set +a
```

On PowerShell or Command Prompt, set process environment variables directly or
use your secret manager. Values containing shell metacharacters need the quoting
rules of that shell. Do not print tokens while troubleshooting.

## Read the Results Honestly

- Check freshness, sample counts, target coverage, baseline comparisons, and warnings together.
- Price-center grades are not buy/sell signals. A positive estimate is not proof of a tradable edge.
- Keep forward evidence, development holdouts, and replay separate.
- Touch/fill estimates are not actual order-book execution.
- Corrected range-edge accounting is stop-first; check execution reconciliation before treating a counterfactual estimate as execution-proven.
- Missing, collecting, stale, and losing results remain part of the record.
- Your software and broker connection are yours. These examples do not expose or operate our private desk.
- Bring-your-own-model service remains future R&D; model uploads are not available yet.

Bearer authentication and paced requests remain in the advanced clients. Polling
faster does not create fresher research. Respect rate-limit and retry responses.

## Tests

```bash
python -m unittest discover -s . -p test_examples.py -v
```

Tests use synthetic responses only. They cover null evidence, invalid types,
asset scope, missing probe prices, stale/collecting/unavailable results, neutral
grader wording, and refusal of unexpected/private tool catalogs.
