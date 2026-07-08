# Hermes Token Usage Dashboard

A **local-only** Streamlit dashboard for your [Hermes Agent](https://hermes-agent.nousresearch.com) token usage.

It reads Hermes's local `state.db`, enriches each session with OpenRouter
per-token rates (cached 24h), and shows token usage + estimated cost broken
down by provider, model, and time — plus a set of cost-control "Insights"
views (cache hit-rate, cached-input savings, model adoption, activity
volume, reasoning-token share, and more).

**No data leaves your machine.** Nothing is uploaded; prices are fetched
read-only from OpenRouter's public API. Your `state.db` is read locally and
never transmitted.

## Screenshot

Run it and open `http://localhost:8501` — KPI row, per-provider / model /
period tables with cost %, a Top-N cost trend, and an Insights tab with 11
cost-control views.

## Install & run

> **Use Python 3.11.** The venv must match the numpy/pandas wheel ABI — building
> against 3.14 (linuxbrew default) loads 3.11-compiled C-extensions and crashes
> on import. `python3.11` is the safe target here.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Then open the printed `http://localhost:8501` URL.

## Files

- `app.py` — the Streamlit dashboard (KPIs, tabs, Insights views).
- `usage_loader.py` — reads Hermes `state.db` (provider / model / token buckets).
- `pricing.py` — OpenRouter price fetch + manual per-model override support.
- `README.md` — this file.

## Cost model

Hermes does **not** persist real costs (`actual_cost_usd` is `NULL` in
`state.db`), so cost is *estimated*. `pricing.py` fetches live per-token
rates from OpenRouter's public API (`/api/v1/models`) and caches them for 24h
in `~/.cache/hermes-usage-dashboard/prices.json`. If a model isn't found
there (local/unknown models), you can drop a `pricing_overrides.json` in this
folder:

```json
{
  "custom/llama-3.1-8b": {"prompt": 0.0, "completion": 0.0, "cache_read": 0.0}
}
```

Keys match the `model` column shown in the dashboard. Unknown models show as
`$0.00` (marked unpriced) until overridden.

## Features

- **KPI row**: non-cached input, cached input, output, total tokens, estimated cost.
- **Cost breakdown**: token volume by segment (cached vs non-cached input vs output) + cost by provider.
- **Tabs**:
  - *By Provider* — totals per `billing_provider`, with **% of cost**.
  - *By Model* — per `(provider, model)` with all token buckets + cost + % of cost.
  - *By Period* — grouped **daily / weekly / monthly** (sidebar toggle), with a trend line.
  - *Top-N Trend* — slider (3–15) + line chart of daily estimated cost per top model.
  - *Insights* — eleven cost-control views:
    - **Cache hit-rate** by model (`cached / (cached + non-cached)`).
    - **Cached-input savings ($)** — what caching saved vs full input price.
    - **Token composition** 100% stacked area over time (input / cached / output / reasoning).
    - **Spend treemap** provider → model (box size = cost).
    - **Cost vs cache hit-rate** scatter (find cheap *and* well-cached models).
    - **Model adoption over time** (sessions per model per day — migration tracker).
    - **Activity volume** (sessions/day + total tokens/day).
    - **Avg cost per session** over time (efficiency trend).
    - **Reasoning-token share** by model.
    - **Cache write vs read balance** per model (spot wasted writes).
    - **Session cost distribution** histogram (spot outlier burns).
- **Provider / model filters** + **date-range picker** in the sidebar.
- **Auto-refresh** toggle (re-reads state.db every 60s).

## License

MIT — see [LICENSE](LICENSE).
