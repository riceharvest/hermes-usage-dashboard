"""Hermes Token Usage Dashboard (local-only, Streamlit).

Reads ~/.hermes/state.db, prices every (provider, model) with the cheapest
OpenRouter per-token rates (cached 24h), and shows token usage + cost broken
out by cached input / non-cached input / output, per provider, per model, and
across daily / weekly / monthly / all periods.

Run:
    pip install streamlit
    python3 app.py
Then open the printed localhost URL.
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import pricing
import usage_loader

# ── Material light + teal theme ────────────────────────────────────────────
TEAL = "#009688"
TEAL_DARK = "#00796b"
TEAL_LIGHT = "#e0f2f1"

MATERIAL_CSS = f"""
<style>
  :root {{
    --md-teal: {TEAL};
    --md-teal-dark: {TEAL_DARK};
    --md-teal-light: {TEAL_LIGHT};
  }}
  html, body, [class*="stApp"] {{
    background-color: #fafafa;
    color: #212121;
    font-family: "Roboto", "Segoe UI", system-ui, sans-serif;
  }}
  /* Top bar */
  header[data-testid="stHeader"] {{ background: {TEAL}; }}
  .stApp > header {{ background: {TEAL}; }}
  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background-color: #ffffff;
    border-right: 1px solid #e0e0e0;
  }}
  /* Headings */
  h1, h2, h3 {{ color: #212121; font-weight: 500; }}
  h1 {{ border-bottom: 3px solid {TEAL}; padding-bottom: 6px; }}
  /* Primary buttons */
  .stButton > button {{
    background-color: {TEAL}; color: white; border: none;
    border-radius: 4px; text-transform: uppercase; font-weight: 500;
    box-shadow: 0 2px 4px rgba(0,0,0,.15);
  }}
  .stButton > button:hover {{ background-color: {TEAL_DARK}; }}
  /* Metric cards -> Material elevation */
  [data-testid="stMetric"] {{
    background: #ffffff; border-radius: 8px; padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 1px 2px rgba(0,0,0,.08);
    border-top: 3px solid {TEAL};
  }}
  [data-testid="stMetric"] label {{ color: {TEAL_DARK}; font-weight: 500; }}
  /* Tables */
  .stDataFrame {{ border-radius: 8px; }}
  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ border-bottom: 2px solid #e0e0e0; }}
  .stTabs [data-baseweb="tab"] {{ color: #616161; }}
  .stTabs [aria-selected="true"] {{ color: {TEAL_DARK} !important; border-bottom: 2px solid {TEAL} !important; }}
  /* Selectbox / multiselect */
  .stSelectbox label, .stMultiSelect label {{ color: {TEAL_DARK}; font-weight: 500; }}
  /* Captions */
  .stCaption, .stInfo {{ color: #616161; }}
  .stAlert {{ border-radius: 6px; }}
  hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 1rem 0; }}
</style>
"""


@st.cache_data(ttl=3600, show_spinner="Reading local Hermes state.db…")
def load_data(db_path: str | None):
    rows = usage_loader.load_usage(db_path)
    return rows


@st.cache_data(ttl=3600, show_spinner="Fetching OpenRouter pricing…")
def load_prices(force: bool):
    return pricing.get_prices(force=force)


def build_dataframe(rows, prices, overrides=None):
    """Build a per-session dataframe with computed cost + period keys."""
    recs = []
    for r in rows:
        price = pricing.price_for(r.model, prices, overrides)
        cost = pricing.compute_cost(
            price,
            r.input_tokens,
            r.output_tokens,
            r.cache_read_tokens,
            r.cache_write_tokens,
        )
        non_cached = max(0, r.input_tokens - r.cache_read_tokens)
        recs.append(
            {
                "provider": r.provider,
                "model": r.model,
                "started_at": datetime.fromtimestamp(r.started_at, tz=timezone.utc),
                "non_cached_input": non_cached,
                "cached_input": r.cache_read_tokens,
                "cache_write": r.cache_write_tokens,
                "output": r.output_tokens,
                "reasoning": r.reasoning_tokens,
                "total_tokens": r.input_tokens + r.output_tokens + r.cache_read_tokens + r.cache_write_tokens,
                "cost_usd": cost,
                "priced": price is not None,
                "cost_status": r.cost_status,
            }
        )
    df = pd.DataFrame(recs)
    df["date"] = df["started_at"].dt.strftime("%Y-%m-%d")
    df["week"] = df["started_at"].apply(lambda d: f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}")
    df["month"] = df["started_at"].dt.strftime("%Y-%m")
    return df


def agg(df, group_cols):
    g = df.groupby(group_cols, dropna=False).agg(
        sessions=("model", "size"),
        non_cached_input=("non_cached_input", "sum"),
        cached_input=("cached_input", "sum"),
        cache_write=("cache_write", "sum"),
        output=("output", "sum"),
        reasoning=("reasoning", "sum"),
        total_tokens=("total_tokens", "sum"),
        cost_usd=("cost_usd", "sum"),
        priced_sessions=("priced", "sum"),
    ).reset_index()
    return g


def fmt_tokens(n):
    if n is None:
        return "—"
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def fmt_usd(n):
    if n is None:
        return "—"
    return f"${n:,.2f}"


def main():
    st.set_page_config(page_title="Hermes Usage", layout="wide", page_icon="📊")
    st.markdown(MATERIAL_CSS, unsafe_allow_html=True)

    st.title("📊 Hermes Token Usage Dashboard")
    st.caption("Local-only · reads ~/.hermes/state.db · priced with cheapest OpenRouter rates")

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        db_path = st.text_input(
            "state.db path (blank = auto)",
            value="",
            help="Leave blank to auto-detect ~/.hermes/state.db",
        )
        db_path = db_path.strip() or None

        if st.button("🔄 Refresh OpenRouter pricing"):
            st.cache_data.clear()
            st.rerun()

        period = st.radio("Period", ["all", "daily", "weekly", "monthly"], index=1)
        st.divider()
        st.caption("Pricing cached 24h at ~/.hermes/cache/openrouter_pricing.json")

    # ── Load ─────────────────────────────────────────────────────────────
    with st.spinner("Loading…"):
        rows = load_data(db_path)
        prices = load_prices(force=False)
        overrides = pricing.load_overrides()

        # ── Manual price mapping (for unknowns / any model) ──────────────
        with st.sidebar:
            st.divider()
            st.subheader("💲 Model price mapping")
            st.caption("Pick an OpenRouter model to price any unmapped model (e.g. unknowns, local GGUF).")
            distinct_models = sorted({r.model for r in rows if r.model})
            # unknowns (currently unpriced) first
            priced_now = {r.model for r in rows if pricing.price_for(r.model, prices, overrides)}
            unpriced_models = [m for m in distinct_models if m not in priced_now]
            ordered = unpriced_models + [m for m in distinct_models if m not in unpriced_models]
            target = st.selectbox(
                "Model to map",
                ordered,
                format_func=lambda m: f"{m}  (unpriced)" if m in unpriced_models else m,
            )
            or_ids = sorted(prices.keys())
            current = overrides.get(target, "")
            idx = 0
            if current in or_ids:
                idx = or_ids.index(current) + 1
            choice = st.selectbox(
                f"Price '{target}' as OpenRouter model",
                ["" ] + or_ids,
                index=idx,
                format_func=lambda x: "— default/unmapped —" if x == "" else x,
            )
            c1, c2 = st.columns(2)
            if c1.button("Save mapping"):
                new_overrides = dict(overrides)
                if choice:
                    new_overrides[target] = choice
                elif target in new_overrides:
                    del new_overrides[target]
                pricing.save_overrides(new_overrides)
                st.rerun()
            if c2.button("Clear mapping") and target in overrides:
                new_overrides = dict(overrides)
                del new_overrides[target]
                pricing.save_overrides(new_overrides)
                st.rerun()

            # Date-range filter
            st.divider()
            st.subheader("📅 Date range")
            dmin = datetime.fromtimestamp(min(r.started_at for r in rows), tz=timezone.utc).date()
            dmax = datetime.fromtimestamp(max(r.started_at for r in rows), tz=timezone.utc).date()
            rng = st.date_input("From / to", value=(dmin, dmax), min_value=dmin, max_value=dmax)
            auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)

        df = build_dataframe(rows, prices, overrides)

        # Apply date-range filter
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            lo, hi = rng
            if lo is not None:
                df = df[df["started_at"].dt.date >= lo]
            if hi is not None:
                df = df[df["started_at"].dt.date <= hi]

    if df.empty:
        st.warning("No session data found in state.db.")
        return

    # ── Provider / model filter ──────────────────────────────────────────
    providers = sorted(df["provider"].unique())
    sel_providers = st.multiselect("Providers", providers, default=providers)
    fdf = df[df["provider"].isin(sel_providers)] if sel_providers else df

    models = sorted(fdf["model"].unique())
    sel_models = st.multiselect("Models", models, default=models)
    if sel_models:
        fdf = fdf[fdf["model"].isin(sel_models)]

    # ── Totals (KPI row) ─────────────────────────────────────────────────
    tot_noncached = fdf["non_cached_input"].sum()
    tot_cached = fdf["cached_input"].sum()
    tot_out = fdf["output"].sum()
    tot_all = fdf["total_tokens"].sum()
    tot_cost = fdf["cost_usd"].sum()
    priced_cost = fdf[fdf["priced"]]["cost_usd"].sum()
    unpriced = int((~fdf["priced"]).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Non-cached input", fmt_tokens(tot_noncached))
    c2.metric("Cached input", fmt_tokens(tot_cached))
    c3.metric("Output", fmt_tokens(tot_out))
    c4.metric("Total tokens", fmt_tokens(tot_all))
    c5.metric("Est. cost (OR)", fmt_usd(tot_cost))

    if unpriced:
        st.info(
            f"{unpriced:,} sessions use models with no OpenRouter route "
            f"(e.g. custom GGUF, native Codex) — shown as n/a cost. "
            f"Priced cost above: {fmt_usd(priced_cost)}."
        )

    st.divider()

    # ── Cost split donut (cached vs non-cached input vs output) ──────────
    st.subheader("Cost breakdown")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        split = pd.DataFrame(
            {
                "segment": ["Non-cached input", "Cached input", "Cache write", "Output"],
                "tokens": [tot_noncached, tot_cached, fdf["cache_write"].sum(), tot_out],
            }
        )
        st.bar_chart(split.set_index("segment")["tokens"], width="stretch")
        st.caption("Token volume by segment (cached vs non-cached input vs output)")
    with col_b:
        # Cost by provider
        by_prov = agg(fdf, ["provider"]).sort_values("cost_usd", ascending=False)
        chart_df = by_prov[["provider", "cost_usd"]].fillna(0).set_index("provider")
        st.bar_chart(chart_df["cost_usd"], width="stretch")
        st.caption("Estimated cost (USD) by provider — cheapest OpenRouter rates")

    st.divider()

    # ── Tabs: by provider / by model / by period / top-N trend / insights ──
    tab_prov, tab_model, tab_period, tab_topn, tab_insights = st.tabs(
        ["By Provider", "By Model", "By Period", "Top-N Trend", "Insights"]
    )

    with tab_prov:
        g = agg(fdf, ["provider"]).sort_values("cost_usd", ascending=False)
        total_cost = float(g["cost_usd"].sum())
        g["cost_pct"] = (g["cost_usd"] / total_cost * 100).round(1) if total_cost else 0.0
        show = g[
            ["provider", "sessions", "non_cached_input", "cached_input",
             "cache_write", "output", "total_tokens", "cost_usd", "cost_pct", "priced_sessions"]
        ].copy()
        show.columns = ["Provider", "Sessions", "Non-cached in", "Cached in",
                        "Cache write", "Output", "Total tok", "Cost (OR)", "% cost", "Priced"]
        for col in ["Non-cached in", "Cached in", "Cache write", "Output", "Total tok"]:
            show[col] = show[col].map(fmt_tokens)
        show["Cost (OR)"] = show["Cost (OR)"].map(fmt_usd)
        st.dataframe(show, width="stretch", hide_index=True)

    with tab_model:
        g = agg(fdf, ["provider", "model"]).sort_values("cost_usd", ascending=False)
        total_cost = float(g["cost_usd"].sum())
        g["cost_pct"] = (g["cost_usd"] / total_cost * 100).round(1) if total_cost else 0.0
        show = g[
            ["provider", "model", "sessions", "non_cached_input", "cached_input",
             "cache_write", "output", "total_tokens", "cost_usd", "cost_pct", "priced_sessions"]
        ].copy()
        show.columns = ["Provider", "Model", "Sessions", "Non-cached in", "Cached in",
                        "Cache write", "Output", "Total tok", "Cost (OR)", "% cost", "Priced"]
        for col in ["Non-cached in", "Cached in", "Cache write", "Output", "Total tok"]:
            show[col] = show[col].map(fmt_tokens)
        show["Cost (OR)"] = show["Cost (OR)"].map(fmt_usd)
        st.dataframe(show, width="stretch", hide_index=True)

    with tab_period:
        pk = {"daily": "date", "weekly": "week", "monthly": "month", "all": None}[period]
        if pk is None:
            st.caption("Select a period (daily / weekly / monthly) in the sidebar to group by time.")
            g = agg(fdf, ["provider"])
        else:
            g = agg(fdf, [pk, "provider"]).sort_values([pk, "total_tokens"], ascending=[True, False])
        if pk:
            show = g[[pk, "provider", "sessions", "non_cached_input", "cached_input",
                      "output", "total_tokens", "cost_usd"]].copy()
            show.columns = [period.capitalize(), "Provider", "Sessions", "Non-cached in",
                            "Cached in", "Output", "Total tok", "Cost (OR)"]
            for col in ["Non-cached in", "Cached in", "Output", "Total tok"]:
                show[col] = show[col].map(fmt_tokens)
            show["Cost (OR)"] = show["Cost (OR)"].map(fmt_usd)
            st.dataframe(show, width="stretch", hide_index=True)

            # Trend line: total tokens + cost over period
            st.subheader(f"{period.capitalize()} trend")
            trend = agg(fdf, [pk]).sort_values(pk)
            tchart = trend[[pk, "total_tokens", "cost_usd"]].set_index(pk)
            st.line_chart(tchart, width="stretch")
        else:
            show = g[["provider", "sessions", "total_tokens", "cost_usd"]].copy()
            show.columns = ["Provider", "Sessions", "Total tok", "Cost (OR)"]
            show["Total tok"] = show["Total tok"].map(fmt_tokens)
            show["Cost (OR)"] = show["Cost (OR)"].map(fmt_usd)
            st.dataframe(show, width="stretch", hide_index=True)

    # ── Top-N model cost trend (sparkline over daily buckets) ───────────
    with tab_topn:
        topn = st.slider("Top N models", 3, 15, 5)
        # pick the most expensive models (by current filtered cost)
        model_cost = fdf.groupby("model")["cost_usd"].sum().sort_values(ascending=False)
        top_models = list(model_cost.head(topn).index)
        if top_models:
            pk = {"daily": "date", "weekly": "week", "monthly": "month", "all": "date"}[period]
            sub = fdf[fdf["model"].isin(top_models)]
            piv = sub.pivot_table(index=pk, columns="model", values="cost_usd", aggfunc="sum").fillna(0)
            piv = piv.sort_index()
            st.line_chart(piv, width="stretch")
            st.caption("Estimated cost (USD) per top model over time — narrow with the date-range filter.")
        else:
            st.info("No priced models in the current filter.")

    # ── Insights: cache hit-rate, composition, treemap, scatter, reasoning, histogram ──
    with tab_insights:
        import plotly.express as px

        st.subheader("🎯 Cache hit-rate by model")
        st.caption("cached_input / (cached_input + non_cached_input). Higher = cheaper prompts.")
        cr = (
            fdf.groupby(["provider", "model"])
            .agg(cached=("cached_input", "sum"), noncached=("non_cached_input", "sum"))
            .reset_index()
        )
        cr["hit_rate"] = (
            cr["cached"] / (cr["cached"] + cr["noncached"]).replace(0, 1) * 100
        ).round(1)
        cr = cr[cr["cached"] + cr["noncached"] > 0].sort_values("hit_rate", ascending=False)
        cr_disp = cr.copy()
        cr_disp["Model"] = cr_disp["provider"] + " / " + cr_disp["model"]
        st.bar_chart(cr_disp.set_index("Model")["hit_rate"], width="stretch")

        st.subheader("🧱 Token composition over time")
        st.caption("100% stacked area: non-cached input / cached input / output / reasoning.")
        pk = {"daily": "date", "weekly": "week", "monthly": "month", "all": "date"}[period]
        comp = (
            fdf.groupby(pk)
            .agg(
                noncached=("non_cached_input", "sum"),
                cached=("cached_input", "sum"),
                output=("output", "sum"),
                reasoning=("reasoning", "sum"),
            )
            .sort_index()
        )
        comp_pct = comp.div(comp.sum(axis=1).replace(0, 1), axis=0) * 100
        st.area_chart(comp_pct, width="stretch")

        st.subheader("🌳 Spend treemap (provider → model)")
        st.caption("Box size = estimated cost. Click a provider to drill into models.")
        treemap = fdf.groupby(["provider", "model"]).agg(cost=("cost_usd", "sum")).reset_index()
        treemap = treemap[treemap["cost"].notna()]
        fig = px.treemap(
            treemap,
            path=[px.Constant("all"), "provider", "model"],
            values="cost",
            color="cost",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0))
        st.plotly_chart(fig, width="stretch")

        st.subheader("💸 Cost vs cache hit-rate (per model)")
        st.caption("Each point = a model. Top-right = cheap AND well-cached.")
        scatter = cr.copy()
        cost_by_model = fdf.groupby("model")["cost_usd"].sum()
        scatter["cost"] = scatter["model"].map(cost_by_model)
        scatter = scatter.dropna(subset=["cost"])
        if not scatter.empty:
            fig2 = px.scatter(
                scatter,
                x="hit_rate",
                y="cost",
                text="model",
                size=cost_by_model[scatter["model"]].values,
                color="provider",
                labels={"hit_rate": "cache hit-rate %", "cost": "est. cost (USD)"},
                size_max=40,
            )
            fig2.update_traces(textposition="top center", marker=dict(opacity=0.7))
            fig2.update_layout(margin=dict(t=10, l=0, r=0, b=0))
            st.plotly_chart(fig2, width="stretch")

        st.subheader("🧠 Reasoning-token share by model")
        st.caption("reasoning / total_tokens %. Flags reasoning-heavy (expensive) models.")
        rs = (
            fdf.groupby(["provider", "model"])
            .agg(reasoning=("reasoning", "sum"), total=("total_tokens", "sum"))
            .reset_index()
        )
        rs["reason_pct"] = (rs["reasoning"] / rs["total"].replace(0, 1) * 100).round(1)
        rs = rs[rs["total"] > 0].sort_values("reason_pct", ascending=False)
        rs_disp = rs.copy()
        rs_disp["Model"] = rs_disp["provider"] + " / " + rs_disp["model"]
        st.bar_chart(rs_disp.set_index("Model")["reason_pct"], width="stretch")

        st.subheader("📊 Session cost distribution")
        st.caption("Histogram of per-session estimated cost — reveals outlier burns.")
        sess_cost = fdf[fdf["cost_usd"].notna()]["cost_usd"]
        if len(sess_cost) > 1:
            fig3 = px.histogram(sess_cost, nbins=40, color_discrete_sequence=[TEAL])
            fig3.update_layout(
                margin=dict(t=10, l=0, r=0, b=0),
                xaxis_title="est. cost per session (USD)",
                yaxis_title="sessions",
            )
            st.plotly_chart(fig3, width="stretch")
        else:
            st.info("Not enough priced sessions in the current filter for a histogram.")

        st.subheader("💰 Cached-input savings ($)")
        st.caption("What you'd have paid at full input price minus what caching cost. Bigger = caching pays off.")
        # compute per-session savings using price.prompt vs price.cache_read
        def _savings(row):
            p = pricing.price_for(row["model"], prices, overrides)
            if p is None or p.cache_read >= p.prompt:
                return 0.0
            return row["cached_input"] * (p.prompt - p.cache_read)
        fdf2 = fdf.copy()
        fdf2["savings"] = fdf2.apply(_savings, axis=1)
        sav_total = fdf2["savings"].sum()
        st.metric("Total cached-input savings", fmt_usd(sav_total))
        sav_by_provider = (
            fdf2.groupby("provider")["savings"].sum().sort_values(ascending=False)
        )
        st.bar_chart(sav_by_provider, width="stretch")
        st.caption("Cached-input savings by provider (USD)")

        st.subheader("📈 Model adoption over time")
        st.caption("Stacked area of sessions per model per day — tracks model migration.")
        adopt = (
            fdf.groupby([pk, "model"]).size().unstack(fill_value=0).sort_index()
        )
        # keep top 12 models by total sessions, group rest as 'other'
        top_models = fdf["model"].value_counts().head(12).index.tolist()
        adopt_top = adopt[top_models] if all(m in adopt.columns for m in top_models) else adopt
        st.area_chart(adopt_top, width="stretch")

        st.subheader("🔥 Activity volume (sessions & tokens per day)")
        st.caption("Dual view: how hard you're running vs how expensive.")
        act = fdf.groupby(pk).agg(sessions=("model", "size"), tokens=("total_tokens", "sum")).sort_index()
        st.line_chart(act, width="stretch")
        st.caption("Blue = sessions/day, orange = total tokens/day")

        st.subheader("⚡ Avg cost per session over time")
        st.caption("Efficiency: estimated cost per session. Rising = each run getting pricier.")
        cps = fdf[fdf["cost_usd"].notna()].groupby(pk)["cost_usd"].mean().sort_index()
        st.line_chart(cps, width="stretch")

        st.subheader("🔄 Cache write vs read balance (per model)")
        st.caption("Models where you write cache you never read = wasted writes.")
        wr = (
            fdf.groupby(["provider", "model"])
            .agg(write=("cache_write", "sum"), read=("cached_input", "sum"))
            .reset_index()
        )
        wr = wr[wr["write"] + wr["read"] > 0].sort_values("write", ascending=False)
        wr_disp = wr.copy()
        wr_disp["Model"] = wr_disp["provider"] + " / " + wr_disp["model"]
        wr_disp = wr_disp.set_index("Model")[["write", "read"]]
        st.bar_chart(wr_disp, width="stretch")
        st.caption("write = cache_write tokens, read = cached_input tokens")

    st.caption(
        f"Data: {len(df):,} sessions · "
        f"OpenRouter prices: {len(prices)} models · "
        f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    # Auto-refresh: re-run the script every 60s to re-read state.db
    if auto_refresh:
        import time as _time
        _time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
