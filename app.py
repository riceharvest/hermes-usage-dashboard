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

import os, time, json
from collections import defaultdict
from datetime import datetime, timezone, date, timedelta

import pandas as pd
import streamlit as st
import plotly.express as px

import pricing
import usage_loader

# ── Material light + teal theme ────────────────────────────────────────────
TEAL = "#009688"
TEAL_DARK = "#00796b"
TEAL_LIGHT = "#e0f2f1"

MATERIAL_CSS = f"""
<style>
  html, body, [class*="stApp"] {{
    font-family: "Inter", "Roboto", "Segoe UI", system-ui, sans-serif;
  }}
  
  /* Modern elevated Metric cards with theme-matching background and border */
  [data-testid="stMetric"] {{
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-left: 5px solid var(--primary-color);
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }}
  
  [data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
  }}
  
  /* Make metric label bold and theme-colored */
  [data-testid="stMetric"] label {{
    color: var(--primary-color);
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  
  /* Style Tabs cleanly */
  .stTabs [data-baseweb="tab-list"] {{
    border-bottom: 2px solid rgba(128, 128, 128, 0.15);
    gap: 8px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 4px 4px 0 0;
    padding: 8px 16px;
    font-weight: 500;
  }}
  
  /* Sidebar styling */
  section[data-testid="stSidebar"] {{
    border-right: 1px solid rgba(128, 128, 128, 0.15);
  }}
  
  /* Info alerts style */
  .stAlert {{
    border-radius: 8px;
    border: 1px solid rgba(128, 128, 128, 0.15);
  }}
  
  /* Premium h1 gradient title */
  h1 {{
    background: linear-gradient(135deg, var(--primary-color), #00bfa5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
  }}
</style>
"""


@st.cache_data(ttl=10, show_spinner="Reading local Hermes state.db…")
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
        non_cached = max(0, r.input_tokens - r.cache_read_tokens)
        if price is not None:
            cost = pricing.compute_cost(
                price,
                r.input_tokens,
                r.output_tokens,
                r.cache_read_tokens,
                r.cache_write_tokens,
            )
            cost_non_cached = non_cached * price.prompt
            cost_cached = r.cache_read_tokens * price.cache_read
            cost_write = r.cache_write_tokens * price.cache_write
            cost_out = r.output_tokens * price.completion
            if price.cache_read < price.prompt:
                savings = r.cache_read_tokens * (price.prompt - price.cache_read)
            else:
                savings = 0.0
        else:
            cost = 0.0
            cost_non_cached = 0.0
            cost_cached = 0.0
            cost_write = 0.0
            cost_out = 0.0
            savings = 0.0

        recs.append(
            {
                "provider": r.provider or "unknown",
                "model": r.model or "unknown",
                "started_at": datetime.fromtimestamp(r.started_at, tz=timezone.utc),
                "non_cached_input": non_cached,
                "cached_input": r.cache_read_tokens,
                "cache_write": r.cache_write_tokens,
                "output": r.output_tokens,
                "reasoning": r.reasoning_tokens,
                "total_tokens": r.input_tokens + r.output_tokens + r.cache_read_tokens + r.cache_write_tokens,
                "cost_usd": cost,
                "cost_non_cached_input": cost_non_cached,
                "cost_cached_input": cost_cached,
                "cost_cache_write": cost_write,
                "cost_output": cost_out,
                "savings_usd": savings,
                "priced": price is not None,
                "cost_status": r.cost_status,
            }
        )
    if not recs:
        return pd.DataFrame(columns=[
            "provider", "model", "started_at", "non_cached_input", "cached_input",
            "cache_write", "output", "reasoning", "total_tokens", "cost_usd",
            "cost_non_cached_input", "cost_cached_input", "cost_cache_write", "cost_output",
            "savings_usd", "priced", "cost_status", "date", "week", "month"
        ])
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
        cost_non_cached_input=("cost_non_cached_input", "sum"),
        cost_cached_input=("cost_cached_input", "sum"),
        cost_cache_write=("cost_cache_write", "sum"),
        cost_output=("cost_output", "sum"),
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


def suggest_model_match(target_name: str, options: list[str]) -> str:
    t = target_name.lower().replace("-", " ").replace("/", " ").replace("_", " ")
    best_match = ""
    best_score = 0
    for opt in options:
        o = opt.lower().replace("-", " ").replace("/", " ").replace("_", " ")
        # Token overlap
        t_words = set(t.split())
        o_words = set(o.split())
        common = t_words.intersection(o_words)
        score = len(common)
        
        # Check specific prefix/suffix matches
        t_last = target_name.split("/")[-1].lower()
        o_last = opt.split("/")[-1].lower()
        if o_last.startswith(t_last) or t_last.startswith(o_last):
            score += 3
        if score > best_score:
            best_score = score
            best_match = opt
    return best_match if best_score > 0 else ""

CURRENCIES = {
    "USD ($)": ("$", 1.0),
    "EUR (€)": ("€", 0.92),
    "GBP (£)": ("£", 0.78),
    "JPY (¥)": ("¥", 160.0),
    "CAD (C$)": ("C$", 1.36),
    "AUD (A$)": ("A$", 1.50),
}

def fmt_usd(n):
    if n is None:
        return "—"
    n = float(n)
    symbol = st.session_state.get("currency_symbol", "$")
    rate = st.session_state.get("currency_rate", 1.0)
    n_conv = n * rate
    if n_conv == 0.0:
        return f"{symbol}0.00"
    if n_conv < 0.01:
        return f"{symbol}{n_conv:.4f}"
    return f"{symbol}{n_conv:,.2f}"



@st.fragment(run_every=60)
def autorefresh_handler(enabled: bool):
    if enabled:
        st.rerun()


def main():
    st.set_page_config(page_title="Hermes Usage", layout="wide", page_icon="📊")
    st.markdown(MATERIAL_CSS, unsafe_allow_html=True)

    if "toast_msg" in st.session_state:
        st.toast(st.session_state["toast_msg"], icon="✅")
        del st.session_state["toast_msg"]

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

        period = st.radio("Time Bucket", ["all", "daily", "weekly", "monthly"], index=1, key="filter_time_bucket")
        st.divider()
        st.caption("Pricing cached 24h at ~/.hermes/cache/openrouter_pricing.json")

    # ── Load ─────────────────────────────────────────────────────────────
    with st.spinner("Loading…"):
        try:
            rows = load_data(db_path)
        except FileNotFoundError as e:
            st.error(f"⚠️ **Database File Not Found**: {str(e)}")
            st.info("Please verify the database path in the sidebar, or ensure your Hermes instance is running and has generated `state.db`.")
            return
        except Exception as e:
            st.error(f"⚠️ **Failed to load database**: {str(e)}")
            return

        if not rows:
            st.warning("No session data found in state.db.")
            return

        prices = load_prices(force=False)
        overrides = pricing.load_overrides()

        # Date-range and auto refresh filter in sidebar
        with st.sidebar:
            st.divider()
            st.subheader("📅 Date range")
            dmin = datetime.fromtimestamp(min(r.started_at for r in rows), tz=timezone.utc).date()
            dmax = datetime.fromtimestamp(max(r.started_at for r in rows), tz=timezone.utc).date()
            
            # Only reset the date range to a default when the user has not explicitly set it
            # AND the time bucket selection changed.  This preserves user-selected ranges.
            current_bucket = st.session_state.get("filter_time_bucket", "daily")
            prev_bucket = st.session_state.get("prev_time_bucket", "")
            if current_bucket != prev_bucket:
                st.session_state["prev_time_bucket"] = current_bucket
                # Only set a preset if the user hasn't touched the range yet
                if "filter_date_range_user_set" not in st.session_state:
                    if current_bucket == "daily":
                        st.session_state["filter_date_range"] = (max(dmin, dmax - timedelta(days=6)), dmax)
                    elif current_bucket == "weekly":
                        st.session_state["filter_date_range"] = (max(dmin, dmax - timedelta(days=29)), dmax)
                    elif current_bucket == "monthly":
                        st.session_state["filter_date_range"] = (max(dmin, dmax - timedelta(days=89)), dmax)
                    else: # "all"
                        st.session_state["filter_date_range"] = (dmin, dmax)
            
            if "filter_date_range" not in st.session_state:
                st.session_state["filter_date_range"] = (dmin, dmax)
            else:
                val = st.session_state["filter_date_range"]
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    v_lo, v_hi = val
                    if v_lo < dmin or v_hi > dmax:
                        st.session_state["filter_date_range"] = (dmin, dmax)
            
            rng = st.date_input(
                "From / to",
                min_value=dmin,
                max_value=dmax,
                key="filter_date_range",
                on_change=lambda: st.session_state.update({"filter_date_range_user_set": True}),
            )
            auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
            
            st.divider()
            st.subheader("📁 Database Diagnostics")
            db_resolved_path = usage_loader._resolve_db_path(db_path)
            db_sz_mb = 0.0
            if os.path.exists(db_resolved_path):
                db_sz_mb = os.path.getsize(db_resolved_path) / (1024 * 1024)
            st.metric("Total Sessions", f"{len(rows):,}")
            st.metric("Database File Size", f"{db_sz_mb:.1f} MB")
            
            # Currency Settings
            st.divider()
            with st.expander("💱 Currency Converter"):
                curr_choice = st.selectbox(
                    "Display Currency",
                    options=list(CURRENCIES.keys()),
                    index=0,
                    key="curr_selection"
                )
                def_sym, def_rate = CURRENCIES[curr_choice]
                custom_rate = st.number_input(
                    f"Exchange rate (1 USD = ? {def_sym})",
                    min_value=0.0001,
                    max_value=10000.0,
                    value=def_rate,
                    format="%.4f"
                )
                st.session_state["currency_symbol"] = def_sym
                st.session_state["currency_rate"] = custom_rate
            
            # Reset Filters Button
            st.divider()
            if st.button("🧹 Reset All Filters", key="btn_reset_filters", width=160):
                st.session_state.pop("filter_providers", None)
                st.session_state.pop("filter_models", None)
                st.session_state.pop("filter_date_range", None)
                st.session_state.pop("filter_time_bucket", None)
                st.session_state["toast_msg"] = "All filters reset to defaults!"
                st.rerun()

        autorefresh_handler(auto_refresh)

        df = build_dataframe(rows, prices, overrides)

        # Apply date-range filter
        if isinstance(rng, (tuple, list)):
            if len(rng) == 2:
                lo, hi = rng
                if lo is not None:
                    df = df[df["started_at"].dt.date >= lo]
                if hi is not None:
                    df = df[df["started_at"].dt.date <= hi]
            elif len(rng) == 1:
                lo = rng[0]
                if lo is not None:
                    df = df[df["started_at"].dt.date >= lo]
        elif isinstance(rng, date):
            df = df[df["started_at"].dt.date == rng]

    if df.empty:
        st.warning("No session data found in state.db matching date filter.")
        return

    # ── Provider & Model Selection Flow (Cascading) ──────────────────────
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        providers = sorted(df["provider"].unique())
        sel_providers = st.multiselect(
            "Filter by Provider",
            options=providers,
            default=[],
            placeholder="All Providers",
            key="filter_providers",
            help="Select one or more providers to filter. Leave empty to show all."
        )
        fdf = df[df["provider"].isin(sel_providers)] if sel_providers else df

    with col_f2:
        # Cascading options: only show models available for the selected providers
        models = sorted(fdf["model"].unique())
        sel_models = st.multiselect(
            "Filter by Model",
            options=models,
            default=[],
            placeholder="All Models",
            key="filter_models",
            help="Select one or more models to filter. Leave empty to show all."
        )
        if sel_models:
            fdf = fdf[fdf["model"].isin(sel_models)]

    if fdf.empty:
        st.info("No data matches the selected filters. Please adjust your Provider / Model / Date Range filters.")
        return

    # Apply currency rate scaling to fdf
    symbol = st.session_state.get("currency_symbol", "$")
    rate = st.session_state.get("currency_rate", 1.0)
    if rate != 1.0:
        cost_cols = [c for c in fdf.columns if "cost" in c or "savings" in c]
        for c in cost_cols:
            fdf[c] = fdf[c] * rate
            
    # Define SHARED_COLUMN_CONFIG dynamically using current symbol
    SHARED_COLUMN_CONFIG = {
        "Provider": st.column_config.TextColumn("Provider"),
        "Model": st.column_config.TextColumn("Model"),
        "Sessions": st.column_config.NumberColumn("Sessions", format="%,d"),
        "Non-cached in": st.column_config.NumberColumn("Non-cached in", format="%,d"),
        "Cached in": st.column_config.NumberColumn("Cached in", format="%,d"),
        "Cache write": st.column_config.NumberColumn("Cache write", format="%,d"),
        "Output": st.column_config.NumberColumn("Output", format="%,d"),
        "Total tok": st.column_config.NumberColumn("Total tok", format="%,d"),
        "Cost (OR)": st.column_config.NumberColumn("Cost (OR)", format=f"{symbol}%.4f"),
        "% cost": st.column_config.NumberColumn("% cost", format="%.1f%%"),
        "Priced": st.column_config.NumberColumn("Priced", format="%,d"),
        "Daily": st.column_config.TextColumn("Daily"),
        "Weekly": st.column_config.TextColumn("Weekly"),
        "Monthly": st.column_config.TextColumn("Monthly"),
        "Started At": st.column_config.DatetimeColumn("Started At", format="YYYY-MM-DD HH:mm:ss"),
    }

    # Active filters summary row
    filter_badges = []
    if sel_providers:
        filter_badges.append(f"☁️ Providers: `{', '.join(sel_providers)}`")
    if sel_models:
        filter_badges.append(f"🤖 Models: `{', '.join(sel_models)}`")
    if isinstance(rng, (tuple, list)) and len(rng) == 2:
        filter_badges.append(f"📅 Range: `{rng[0]}` to `{rng[1]}`")
    
    if filter_badges:
        st.caption("Active Filters: " + " • ".join(filter_badges))

    # ── Totals (KPI row) ─────────────────────────────────────────────────
    tot_noncached = fdf["non_cached_input"].sum()
    tot_cached = fdf["cached_input"].sum()
    tot_out = fdf["output"].sum()
    tot_all = fdf["total_tokens"].sum()
    tot_cost = fdf["cost_usd"].sum()
    tot_savings = fdf["savings_usd"].sum()
    priced_cost = fdf[fdf["priced"]]["cost_usd"].sum()
    unpriced = int((~fdf["priced"]).sum())

    hit_rate = (tot_cached / (tot_cached + tot_noncached) * 100) if (tot_cached + tot_noncached) > 0 else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Non-cached input", fmt_tokens(tot_noncached))
    c2.metric("Cached input", fmt_tokens(tot_cached))
    c3.metric("Output", fmt_tokens(tot_out))
    c4.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
    c5.metric("Est. cost (OR)", fmt_usd(tot_cost))
    c6.metric("Realized Savings", fmt_usd(tot_savings))

    if unpriced:
        st.info(
            f"{unpriced:,} sessions use models with no OpenRouter route "
            f"(e.g. custom GGUF, native Codex) — shown as n/a cost. "
            f"Priced cost above: {fmt_usd(priced_cost)}."
        )

    st.divider()

    # ── Volume & Cost Breakdown ──────────────────────────────────────────
    st.subheader("📊 Volume & Cost Breakdown")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("##### Token Volume by Segment")
        split_tokens = pd.DataFrame(
            {
                "segment": ["Non-cached input", "Cached input", "Cache write", "Output"],
                "tokens": [tot_noncached, tot_cached, fdf["cache_write"].sum(), tot_out],
            }
        )
        fig_tok = px.pie(
            split_tokens,
            names="segment",
            values="tokens",
            hole=0.4,
            color_discrete_sequence=["#26a69a", "#4db6ac", "#80cbc4", "#00796b"],
        )
        fig_tok.update_traces(
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Tokens: %{value:,.0f}<br>Percentage: %{percent:.1%}<extra></extra>"
        )
        fig_tok.update_layout(
            margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            height=280,
            title=dict(text="Token volume by segment", font=dict(size=12)),
        )
        st.plotly_chart(fig_tok, width="stretch")
        
    with col_b:
        st.markdown("##### Estimated Cost by Segment")
        split_cost = pd.DataFrame(
            {
                "segment": ["Non-cached input", "Cached input", "Cache write", "Output"],
                "cost": [
                    fdf["cost_non_cached_input"].sum(),
                    fdf["cost_cached_input"].sum(),
                    fdf["cost_cache_write"].sum(),
                    fdf["cost_output"].sum(),
                ],
            }
        )
        fig_cost = px.pie(
            split_cost,
            names="segment",
            values="cost",
            hole=0.4,
            color_discrete_sequence=["#26a69a", "#4db6ac", "#80cbc4", "#00796b"],
        )
        fig_cost.update_traces(
            textinfo='percent+label',
            hovertemplate=f"<b>%{{label}}</b><br>Cost: {symbol}%{{value:,.4f}}<br>Percentage: %{{percent:.1%}}<extra></extra>"
        )
        fig_cost.update_layout(
            margin=dict(t=20, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            height=280,
            title=dict(text=f"Cost by segment ({symbol})", font=dict(size=12)),
        )
        st.plotly_chart(fig_cost, width="stretch")
        
    with col_c:
        st.markdown("##### Estimated Cost by Provider")
        by_prov = agg(fdf, ["provider"]).sort_values("cost_usd", ascending=False)
        fig_prov = px.bar(
            by_prov,
            x="provider",
            y="cost_usd",
            labels={"provider": "Provider", "cost_usd": f"Cost ({symbol})"},
            color="cost_usd",
            color_continuous_scale="Teal",
        )
        fig_prov.update_traces(
            hovertemplate=f"<b>%{{x}}</b><br>Cost: {symbol}%{{y:,.4f}}<extra></extra>"
        )
        fig_prov.update_layout(
            margin=dict(t=20, b=10, l=10, r=10),
            coloraxis_showscale=False,
            height=280,
            xaxis={'categoryorder': 'total descending'},
        )
        st.plotly_chart(fig_prov, width="stretch")

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_prov, tab_model, tab_period, tab_topn, tab_logs, tab_catalog, tab_insights = st.tabs(
        ["By Provider", "By Model", "By Period", "Top-N Trend", "Session Logs", "Model Catalog", "Insights"]
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
        st.dataframe(show, width="stretch", hide_index=True, column_config=SHARED_COLUMN_CONFIG)

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
        st.dataframe(show, width="stretch", hide_index=True, column_config=SHARED_COLUMN_CONFIG)

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
            st.dataframe(show, width="stretch", hide_index=True, column_config=SHARED_COLUMN_CONFIG)

            # Trend lines: total tokens & cost over period (split scales!)
            trend_pk = pk
            trend_label = period.capitalize()
            test_trend = agg(fdf, [pk])
            if len(test_trend) < 3 and pk != "date":
                trend_pk = "date"
                trend_label = "Daily (automatic fallback for trend detail)"
                trend = agg(fdf, ["date"]).sort_values("date")
            else:
                trend = test_trend.sort_values(pk)
                
            st.subheader(f"{trend_label} Trend")
            tchart = trend[[trend_pk, "total_tokens", "cost_usd"]].set_index(trend_pk)
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("##### Total Tokens")
                st.line_chart(tchart["total_tokens"], width="stretch")
            with col_t2:
                st.markdown(f"##### Estimated Cost ({symbol})")
                st.line_chart(tchart["cost_usd"], width="stretch")
        else:
            show = g[["provider", "sessions", "total_tokens", "cost_usd"]].copy()
            show.columns = ["Provider", "Sessions", "Total tok", "Cost (OR)"]
            st.dataframe(show, width="stretch", hide_index=True, column_config=SHARED_COLUMN_CONFIG)

    with tab_topn:
        col_n1, col_n2 = st.columns([1, 2])
        with col_n1:
            topn = st.slider("Top N models", 3, 15, 5)
        with col_n2:
            metric_choice = st.radio("Metric to plot", [f"Cost ({symbol})", "Token Volume"], horizontal=True, key="topn_metric")
            
        val_col = "cost_usd" if metric_choice == f"Cost ({symbol})" else "total_tokens"
        model_cost = fdf.groupby("model")[val_col].sum().sort_values(ascending=False)
        top_models = list(model_cost.head(topn).index)
        if top_models:
            pk_trend = {"daily": "date", "weekly": "week", "monthly": "month", "all": "date"}[period]
            sub = fdf[fdf["model"].isin(top_models)]
            test_piv = sub.pivot_table(index=pk_trend, columns="model", values=val_col, aggfunc="sum").fillna(0)
            if len(test_piv) < 3 and pk_trend != "date":
                piv = sub.pivot_table(index="date", columns="model", values=val_col, aggfunc="sum").fillna(0)
                piv.columns = [str(c).replace(":", " - ") for c in piv.columns]
                st.line_chart(piv, width="stretch")
                st.caption(f"{metric_choice} per top model over time (grouped by day for detail) — narrow with the date-range filter.")
            else:
                piv = test_piv.sort_index()
                piv.columns = [str(c).replace(":", " - ") for c in piv.columns]
                st.line_chart(piv, width="stretch")
                st.caption(f"{metric_choice} per top model over time ({period}) — narrow with the date-range filter.")
        else:
            st.info("No priced models in the current filter.")

    with tab_logs:
        st.subheader("📋 Session Logs")
        st.caption("Detailed view of individual execution runs. Sort by cost, date, or token throughput. Click a row to inspect its token breakdown.")
        session_show = fdf[[
            "started_at", "provider", "model", "non_cached_input", "cached_input", "output", "total_tokens", "cost_usd"
        ]].copy()
        session_show.columns = ["Started At", "Provider", "Model", "Non-cached in", "Cached in", "Output", "Total tok", "Cost (OR)"]
        session_show = session_show.sort_values("Started At", ascending=False)
        
        # Color highlight expensive runs (cost > $0.10) using pandas styling
        def highlight_expensive(val):
            try:
                val_f = float(val)
                if val_f >= 0.10:
                    return 'background-color: rgba(255, 152, 0, 0.25); color: #ff9800; font-weight: bold;'
                elif val_f >= 0.02:
                    return 'background-color: rgba(255, 193, 7, 0.15); font-weight: 500;'
            except Exception:
                pass
            return ''
            
        styled_show = session_show.style.map(highlight_expensive, subset=["Cost (OR)"])
        
        selected = st.dataframe(
            styled_show,
            width="stretch",
            hide_index=True,
            column_config=SHARED_COLUMN_CONFIG,
            selection_mode="single-row",
            on_select="rerun"
        )
        
        selected_rows = selected.get("selection", {}).get("rows", [])
        if selected_rows:
            idx = selected_rows[0]
            row_data = session_show.iloc[idx]
            st.markdown("---")
            st.markdown("### 🔍 Session Detail Inspector")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.markdown(f"🗓️ **Started At**: `{row_data['Started At']}`")
                st.markdown(f"☁️ **Provider**: `{row_data['Provider']}`")
                st.markdown(f"🤖 **Model**: `{row_data['Model']}`")
            with col_d2:
                st.markdown(f"📥 **Non-cached Input**: `{fmt_tokens(row_data['Non-cached in'])}` tokens")
                st.markdown(f"💾 **Cached Input**: `{fmt_tokens(row_data['Cached in'])}` tokens")
                st.markdown(f"📤 **Output**: `{fmt_tokens(row_data['Output'])}` tokens")
            with col_d3:
                st.markdown(f"📊 **Total Tokens**: `{fmt_tokens(row_data['Total tok'])}` tokens")
                st.markdown(f"💵 **Estimated Cost**: `{fmt_usd(row_data['Cost (OR)'])}`")
            
            # Quick Map Option
            is_priced = pricing.price_for(row_data['Model'], prices, overrides) is not None
            if not is_priced:
                st.markdown("---")
                st.info(f"⚠️ **Model Price Mapping Missing**: `{row_data['Model']}` has no price mapping and defaults to $0.00. Map it now:")
                col_qm1, col_qm2 = st.columns([3, 1])
                with col_qm1:
                    or_ids = sorted(prices.keys())
                    qm_choice = st.selectbox(
                        "Price as OpenRouter model ID",
                        [""] + or_ids,
                        key=f"quick_map_{row_data['Model']}",
                        format_func=lambda x: "— select OpenRouter model —" if x == "" else x
                    )
                with col_qm2:
                    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("Save Quick Map", key=f"btn_qm_{row_data['Model']}"):
                        if qm_choice:
                            new_overrides = dict(overrides)
                            new_overrides[row_data['Model']] = qm_choice
                            pricing.save_overrides(new_overrides)
                            st.session_state["toast_msg"] = f"Mapped '{row_data['Model']}' to '{qm_choice}'"
                            st.rerun()
        
        # Download button
        csv_data = session_show.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Session Logs (CSV)",
            data=csv_data,
            file_name="hermes_session_logs.csv",
            mime="text/csv",
            key="download_session_logs"
        )

    with tab_catalog:
        st.subheader("📖 Model Price Catalog")
        st.caption("Reference list of cached OpenRouter model prices per 1 million tokens. Prices are cached locally.")
        catalog_data = []
        for model_id, p in prices.items():
            catalog_data.append({
                "Model ID": model_id,
                f"Prompt ({symbol}/1M)": p.prompt * 1e6 * rate,
                f"Completion ({symbol}/1M)": p.completion * 1e6 * rate,
                f"Cache Read ({symbol}/1M)": p.cache_read * 1e6 * rate,
                f"Cache Write ({symbol}/1M)": p.cache_write * 1e6 * rate,
            })
        catalog_df = pd.DataFrame(catalog_data)
        catalog_search = st.text_input("🔍 Search Models by name", value="", placeholder="Search model IDs...", key="catalog_search_input")
        hide_free = st.checkbox("Hide free models", value=False, key="catalog_hide_free")
        if not catalog_df.empty:
            if catalog_search:
                catalog_df = catalog_df[catalog_df["Model ID"].str.contains(catalog_search, case=False, na=False)]
            if hide_free:
                catalog_df = catalog_df[
                    (catalog_df[f"Prompt ({symbol}/1M)"] > 0) | (catalog_df[f"Completion ({symbol}/1M)"] > 0)
                ]
            catalog_df = catalog_df.sort_values("Model ID")
            CATALOG_COLUMN_CONFIG = {
                "Model ID": st.column_config.TextColumn("Model ID"),
                f"Prompt ({symbol}/1M)": st.column_config.NumberColumn(f"Prompt ({symbol}/1M)", format=f"{symbol}%.2f"),
                f"Completion ({symbol}/1M)": st.column_config.NumberColumn(f"Completion ({symbol}/1M)", format=f"{symbol}%.2f"),
                f"Cache Read ({symbol}/1M)": st.column_config.NumberColumn(f"Cache Read ({symbol}/1M)", format=f"{symbol}%.2f"),
                f"Cache Write ({symbol}/1M)": st.column_config.NumberColumn(f"Cache Write ({symbol}/1M)", format=f"{symbol}%.2f"),
            }
            st.dataframe(catalog_df, width="stretch", hide_index=True, column_config=CATALOG_COLUMN_CONFIG)
            
            # Download button for catalog
            csv_catalog = catalog_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Price Catalog (CSV)",
                data=csv_catalog,
                file_name="hermes_model_price_catalog.csv",
                mime="text/csv",
                key="download_price_catalog"
            )
        else:
            st.info("No pricing models catalog loaded.")

        # Cache Diagnostics
        st.markdown("---")
        st.subheader("⚙️ Pricing Cache Diagnostics")
        cache_exists = os.path.exists(pricing.CACHE_PATH)
        if cache_exists:
            cache_sz_kb = os.path.getsize(pricing.CACHE_PATH) / 1024
            mtime = os.path.getmtime(pricing.CACHE_PATH)
            age_hours = (time.time() - mtime) / 3600
            
            c_d1, c_d2, c_d3 = st.columns(3)
            c_d1.metric("Cached Models Count", f"{len(prices):,}")
            c_d2.metric("Cache File Size", f"{cache_sz_kb:.1f} KB")
            c_d3.metric("Cache File Age", f"{age_hours:.1f} hours ago")
            
            st.caption(f"Cache location: `{pricing.CACHE_PATH}`")
        else:
            st.warning("No local pricing cache file found. Using live fetched prices.")

        # Request Cost Calculator
        st.markdown("---")
        st.subheader("🧮 Request Cost Estimator")
        st.caption("Enter hypothetical tokens to estimate and compare costs across all cached models.")
        
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        with calc_col1:
            calc_in = st.number_input("Hypothetical Input tokens", min_value=0, max_value=None, value=50000, step=5000, key="calc_in_tokens")
        with calc_col2:
            calc_out = st.number_input("Hypothetical Output tokens", min_value=0, max_value=None, value=2000, step=500, key="calc_out_tokens")
        with calc_col3:
            calc_cached = st.checkbox("Assume prompt cache hit (read)", value=True, key="calc_caching_used")
            
        calc_rows = []
        for model_id, p in prices.items():
            prompt_rate = p.cache_read if (calc_cached and p.cache_read < p.prompt) else p.prompt
            est_cost = (calc_in * prompt_rate) + (calc_out * p.completion)
            calc_rows.append({
                "Model ID": model_id,
                "Prompt Port": calc_in * prompt_rate * rate,
                "Completion Port": calc_out * p.completion * rate,
                "Total Est. Cost": est_cost * rate,
            })
            
        model_options = sorted(prices.keys())
        selected_compare_models = st.multiselect(
            "Select specific models to compare",
            options=model_options,
            default=[],
            placeholder="Search/Select specific models... (leave empty to show top 10 cheapest)",
            key="calc_models_select"
        )
            
        calc_df = pd.DataFrame(calc_rows)
        if not calc_df.empty:
            if selected_compare_models:
                calc_df = calc_df[calc_df["Model ID"].isin(selected_compare_models)]
                calc_df = calc_df.sort_values("Total Est. Cost")
            else:
                calc_df = calc_df.sort_values("Total Est. Cost").head(10)
            CALC_COLUMN_CONFIG = {
                "Model ID": st.column_config.TextColumn("Model ID"),
                "Prompt Port": st.column_config.NumberColumn("Prompt Portion", format=f"{symbol}%.4f"),
                "Completion Port": st.column_config.NumberColumn("Completion Portion", format=f"{symbol}%.4f"),
                "Total Est. Cost": st.column_config.NumberColumn("Total Cost", format=f"{symbol}%.4f"),
            }
            st.dataframe(calc_df, width="stretch", hide_index=True, column_config=CALC_COLUMN_CONFIG)

    with tab_insights:
        sub_tab_cost, sub_tab_cache, sub_tab_activity = st.tabs([
            "💰 Cost & Savings", 
            "⚡ Caching & Efficiency", 
            "📈 Activity & Adoption"
        ])
        
        # ── SUB-TAB 1: Cost & Savings ──
        with sub_tab_cost:
            st.subheader("🌳 Spend treemap (provider → model)")
            st.caption("Box size = estimated cost. Click a provider to drill into models.")
            treemap = fdf.groupby(["provider", "model"]).agg(cost=("cost_usd", "sum")).reset_index()
            treemap = treemap[treemap["cost"].notna() & (treemap["cost"] > 0)]
            if not treemap.empty:
                fig = px.treemap(
                    treemap,
                    path=[px.Constant("all"), "provider", "model"],
                    values="cost",
                    color="cost",
                    color_continuous_scale="Tealgrn",
                )
                fig.update_traces(
                    texttemplate=f"<b>%{{label}}</b><br>{symbol}%{{value:,.2f}}",
                    hovertemplate=f"<b>%{{label}}</b><br>Cost: {symbol}%{{value:,.4f}}<extra></extra>"
                )
                fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), coloraxis_colorbar=dict(title=f"Cost ({symbol})"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No cost data to display treemap.")

            st.subheader("💰 Cached-input savings ($)")
            st.caption("What you'd have paid at full input price minus what caching cost. Bigger = caching pays off.")
            sav_total = fdf["savings_usd"].sum()
            
            # Show savings metrics side-by-side with bar chart
            col_sav1, col_sav2 = st.columns([1, 2])
            with col_sav1:
                st.metric("Total Cached-Input Savings", fmt_usd(sav_total))
            with col_sav2:
                sav_by_provider = (
                    fdf.groupby("provider")["savings_usd"].sum().sort_values(ascending=False)
                )
                st.bar_chart(sav_by_provider, width="stretch")
                st.caption("Cached-input savings by provider (USD)")

            st.subheader("🥞 Input vs Output Cost split (per provider)")
            st.caption("Stacked bar of prompt input costs vs completion output costs by provider.")
            ratio_agg = fdf.groupby("provider").agg(
                cost_non_cached_input=("cost_non_cached_input", "sum"),
                cost_cached_input=("cost_cached_input", "sum"),
                cost_cache_write=("cost_cache_write", "sum"),
                cost_output=("cost_output", "sum"),
            ).reset_index()
            
            ratio_agg["Prompt (Input)"] = ratio_agg["cost_non_cached_input"] + ratio_agg["cost_cached_input"] + ratio_agg["cost_cache_write"]
            ratio_agg["Completion (Output)"] = ratio_agg["cost_output"]
            
            ratio_long = ratio_agg.melt(
                id_vars=["provider"],
                value_vars=["Prompt (Input)", "Completion (Output)"],
                var_name="Cost Type",
                value_name="Cost (USD)"
            )
            ratio_long = ratio_long[ratio_long["Cost (USD)"] > 0]
            
            if not ratio_long.empty:
                fig_ratio = px.bar(
                    ratio_long,
                    x="provider",
                    y="Cost (USD)",
                    color="Cost Type",
                    barmode="stack",
                    labels={"provider": "Provider", "Cost (USD)": "Cost ($)"},
                    color_discrete_map={"Prompt (Input)": "#009688", "Completion (Output)": "#ff9800"}
                )
                fig_ratio.update_traces(
                    hovertemplate=f"<b>%{{x}}</b><br>Cost: {symbol}%{{y:,.4f}}<extra></extra>"
                )
                fig_ratio.update_layout(
                    margin=dict(t=10, l=0, r=0, b=0),
                    height=280,
                    xaxis={'categoryorder': 'total descending'}
                )
                st.plotly_chart(fig_ratio, width="stretch")
            else:
                st.info("No priced cost to show Prompt/Completion ratio.")

            st.subheader("📊 Session cost distribution")
            st.caption("Histogram of per-session estimated cost — reveals outlier burns.")
            sess_cost = fdf[fdf["cost_usd"].notna() & (fdf["cost_usd"] > 0)]["cost_usd"]
            if len(sess_cost) > 1:
                log_y = st.checkbox("Logarithmic scale (Y-axis)", value=False, key="hist_log_y")
                fig3 = px.histogram(sess_cost, nbins=40, color_discrete_sequence=["#009688"], log_y=log_y)
                fig3.update_layout(
                    margin=dict(t=10, l=0, r=0, b=0),
                    xaxis_title="est. cost per session (USD)",
                    yaxis_title="sessions",
                )
                st.plotly_chart(fig3, width="stretch")
            else:
                st.info("Not enough priced sessions in the current filter for a histogram.")

            st.subheader("⚡ Avg cost per session over time")
            st.caption("Efficiency: estimated cost per session. Rising = each run getting pricier.")
            pk_time = {"daily": "date", "weekly": "week", "monthly": "month", "all": "date"}[period]
            cps = fdf[fdf["priced"]].groupby(pk_time)["cost_usd"].mean().sort_index()
            if not cps.empty:
                st.line_chart(cps, width="stretch")
            else:
                st.info("No pricing data to show average cost trend.")

            st.subheader("📉 Blended Cost per 1K Tokens over time")
            st.caption("Blended rate: (total_cost / total_tokens * 1000) for priced sessions. Lower is more token-efficient.")
            priced_df = fdf[fdf["priced"]]
            if not priced_df.empty:
                # Group by time and aggregate sum of cost & sum of tokens
                blended_agg = priced_df.groupby(pk_time).agg(tot_c=("cost_usd", "sum"), tot_t=("total_tokens", "sum"))
                blended = blended_agg.apply(lambda r: (r["tot_c"] / r["tot_t"] * 1000) if r["tot_t"] > 0 else 0.0, axis=1).sort_index()
                if not blended.empty:
                    st.line_chart(blended, width="stretch")
                else:
                    st.info("Not enough token data to show blended cost trend.")
            else:
                st.info("No pricing data to show blended cost trend.")

        # ── SUB-TAB 2: Caching & Efficiency ──
        with sub_tab_cache:
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
            if not cr.empty:
                cr_disp = cr.copy()
                cr_disp["Model"] = cr_disp["provider"] + " / " + cr_disp["model"]
                st.bar_chart(cr_disp.set_index("Model")["hit_rate"], width="stretch")
            else:
                st.info("No cache usage data found.")

            st.subheader("🔄 Cache write vs read balance (per model)")
            st.caption("Models where you write cache you never read = wasted writes.")
            wr = (
                fdf.groupby(["provider", "model"])
                .agg(write=("cache_write", "sum"), read=("cached_input", "sum"))
                .reset_index()
            )
            wr = wr[wr["write"] + wr["read"] > 0].sort_values("write", ascending=False)
            if not wr.empty:
                wr_disp = wr.copy()
                wr_disp["Model"] = wr_disp["provider"] + " / " + wr_disp["model"]
                wr_disp = wr_disp.set_index("Model")[["write", "read"]]
                st.bar_chart(wr_disp, width="stretch")
                st.caption("write = cache_write tokens, read = cached_input tokens")
            else:
                st.info("No cache read/write data found.")

            st.subheader("🧠 Reasoning-token share by model")
            st.caption("reasoning / total_tokens %. Flags reasoning-heavy (expensive) models.")
            rs = (
                fdf.groupby(["provider", "model"])
                .agg(reasoning=("reasoning", "sum"), total=("total_tokens", "sum"))
                .reset_index()
            )
            rs["reason_pct"] = (rs["reasoning"] / rs["total"].replace(0, 1) * 100).round(1)
            rs = rs[rs["total"] > 0].sort_values("reason_pct", ascending=False)
            if not rs.empty:
                rs_disp = rs.copy()
                rs_disp["Model"] = rs_disp["provider"] + " / " + rs_disp["model"]
                st.bar_chart(rs_disp.set_index("Model")["reason_pct"], width="stretch")
            else:
                st.info("No tokens found to compute reasoning share.")

            st.subheader("💸 Cost vs cache hit-rate (per model)")
            st.caption("Each point = a model. Top-right = cheap AND well-cached.")
            if not cr.empty:
                cost_by_model = fdf.groupby(["provider", "model"])["cost_usd"].sum().reset_index()
                scatter = pd.merge(cr, cost_by_model, on=["provider", "model"])
                scatter = scatter.dropna(subset=["cost_usd"])
                if not scatter.empty:
                    show_labels = len(scatter) <= 10
                    fig2 = px.scatter(
                        scatter,
                        x="hit_rate",
                        y="cost_usd",
                        text="model" if show_labels else None,
                        size="cost_usd",
                        color="provider",
                        labels={"hit_rate": "Cache Hit-Rate %", "cost_usd": "Est. Cost (USD)"},
                        size_max=40,
                    )
                    fig2.update_traces(textposition="top center", marker=dict(opacity=0.7))
                    fig2.update_layout(margin=dict(t=10, l=0, r=0, b=0))
                    st.plotly_chart(fig2, width="stretch")
                else:
                    st.info("No cost data to plot scatter.")
            else:
                st.info("No cache data to plot scatter.")

        # ── SUB-TAB 3: Activity & Adoption ──
        with sub_tab_activity:
            st.subheader("🔥 Activity volume")
            st.caption("Daily usage volume showing count of sessions and total token throughput.")
            pk_time = {"daily": "date", "weekly": "week", "monthly": "month", "all": "date"}[period]
            test_act = fdf.groupby(pk_time).agg(sessions=("model", "size"), tokens=("total_tokens", "sum")).sort_index()
            if len(test_act) < 3 and pk_time != "date":
                pk_time = "date"
                act = fdf.groupby("date").agg(sessions=("model", "size"), tokens=("total_tokens", "sum")).sort_index()
                st.caption("ℹ️ Trend lines automatically grouped by day to provide enough details.")
            else:
                act = test_act
            
            col_act1, col_act2 = st.columns(2)
            with col_act1:
                st.markdown("##### Sessions per day")
                st.line_chart(act["sessions"], width="stretch")
            with col_act2:
                st.markdown("##### Total tokens per day")
                st.line_chart(act["tokens"], width="stretch")

            st.subheader("🧱 Token composition over time")
            st.caption("100% stacked area: non-cached input / cached input / output / reasoning.")
            comp = (
                fdf.groupby(pk_time)
                .agg(
                    noncached=("non_cached_input", "sum"),
                    cached=("cached_input", "sum"),
                    output=("output", "sum"),
                    reasoning=("reasoning", "sum"),
                )
                .sort_index()
            )
            if not comp.empty and comp.sum().sum() > 0:
                comp_pct = comp.div(comp.sum(axis=1).replace(0, 1), axis=0) * 100
                st.area_chart(comp_pct, width="stretch")
            else:
                st.info("No token usage data to show composition.")

            st.subheader("📈 Model adoption over time")
            st.caption("Stacked area of sessions per model per day — tracks model migration.")
            adopt = (
                fdf.groupby([pk_time, "model"]).size().unstack(fill_value=0).sort_index()
            )
            if not adopt.empty:
                # keep top 12 models by total sessions, group rest as 'other'
                top_models = fdf["model"].value_counts().head(12).index.tolist()
                adopt_top = adopt[top_models] if all(m in adopt.columns for m in top_models) else adopt
                # Escape colons in column names to prevent Altair parsing crashes
                adopt_top.columns = [str(c).replace(":", " - ") for c in adopt_top.columns]
                st.area_chart(adopt_top, width="stretch")
            else:
                st.info("No adoption data to show.")

    # ── Model Price Mapping relocated to main area ──────────────────────
    st.divider()
    with st.expander("💲 Manage Model Price Mappings", expanded=False):
        st.caption("Pick an OpenRouter model to price any unmapped model (e.g. unknowns, local GGUF).")
        distinct_models = sorted({r.model for r in rows if r.model})
        # unknowns (currently unpriced) first
        priced_now = {r.model for r in rows if pricing.price_for(r.model, prices, overrides)}
        unpriced_models = [m for m in distinct_models if m not in priced_now]
        ordered = unpriced_models + [m for m in distinct_models if m not in unpriced_models]
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            target = st.selectbox(
                "Model to map",
                ordered,
                format_func=lambda m: f"{m}  (unpriced)" if m in unpriced_models else m,
                key="map_target_select",
                on_change=lambda: st.session_state.pop("map_choice_select", None)
            )
            
            # Auto suggest button
            or_ids = sorted(prices.keys())
            if st.button("🧙 Auto-Suggest Match", key="btn_auto_suggest"):
                best = suggest_model_match(target, or_ids)
                if best:
                    st.session_state["map_choice_select"] = best
                    st.session_state["toast_msg"] = f"Suggested matching model: '{best}'"
                    st.rerun()
                else:
                    st.warning("Could not find a close matching model name in the catalog.")
                    
        with col_m2:
            current = overrides.get(target, "")
            if "map_choice_select" not in st.session_state:
                st.session_state["map_choice_select"] = current if current in or_ids else ""
                
            choice = st.selectbox(
                f"Price '{target}' as OpenRouter model",
                [""] + or_ids,
                format_func=lambda x: "— default/unmapped —" if x == "" else x,
                key="map_choice_select"
            )
        
        c1, c2 = st.columns(2)
        if c1.button("Save mapping"):
            new_overrides = dict(overrides)
            if choice:
                new_overrides[target] = choice
                st.session_state["toast_msg"] = f"Mapped '{target}' to '{choice}'"
            elif target in new_overrides:
                del new_overrides[target]
                st.session_state["toast_msg"] = f"Removed mapping for '{target}'"
            pricing.save_overrides(new_overrides)
            st.rerun()
        if c2.button("Clear mapping") and target in overrides:
            new_overrides = dict(overrides)
            del new_overrides[target]
            st.session_state["toast_msg"] = f"Cleared mapping for '{target}'"
            pricing.save_overrides(new_overrides)
            st.rerun()
            
        # Download overrides mapping
        if overrides:
            st.markdown("---")
            overrides_json = json.dumps({"overrides": overrides}, indent=2).encode('utf-8')
            st.download_button(
                label="📥 Export Mappings as JSON",
                data=overrides_json,
                file_name="usage_dashboard_overrides.json",
                mime="application/json",
                key="btn_download_overrides"
            )

    st.caption(
        f"Data: {len(df):,} sessions · "
        f"OpenRouter prices: {len(prices)} models · "
        f"generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )


if __name__ == "__main__":
    main()
