"""
ICEBREAKER — ARB Dashboard
Spread Monitor for KC/RC and CC/LCC pairs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ── Config ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ARB Monitor", layout="wide")

DB = Path(__file__).parent.parent / "Database"

KC_FACTOR = 22.0462   # ¢/lbs  →  $/MT

# ── Theme ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* KPI cards */
.kpi-row { display:flex; gap:12px; margin-bottom:8px; }
.kpi {
    flex:1; background:#ffffff;
    border:1px solid #e5e7eb;
    border-top: 3px solid #2563eb;
    border-radius:6px; padding:16px 18px 12px;
}
.kpi.green  { border-top-color:#16a34a; }
.kpi.red    { border-top-color:#dc2626; }
.kpi.amber  { border-top-color:#d97706; }
.kpi.teal   { border-top-color:#2563eb; }
.kpi.purple { border-top-color:#7c3aed; }
.kpi-label {
    font-size:10px; font-weight:700; letter-spacing:.1em;
    text-transform:uppercase; color:#6b7280; margin-bottom:6px;
}
.kpi-value { font-size:24px; font-weight:700; color:#111827; line-height:1; }
.kpi-sub   { font-size:11px; color:#9ca3af; margin-top:5px; }

/* Section tag */
.sec-tag {
    display:inline-block; font-size:10px; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase;
    color:#6b7280; border-bottom:1px solid #e5e7eb;
    padding-bottom:4px; margin-bottom:12px; width:100%;
}
/* Stat table */
.stat-row { display:flex; justify-content:space-between;
    border-bottom:1px solid #f3f4f6;
    padding:5px 0; font-size:12px; }
.stat-key { color:#6b7280; }
.stat-val { color:#111827; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────

PAPER  = "rgba(0,0,0,0)"
PLOT   = "rgba(0,0,0,0)"
GRID   = "rgba(0,0,0,0.06)"
FONT   = "#374151"
MUTED  = "#9ca3af"
TEAL   = "#2563eb"
GREEN  = "#16a34a"
RED    = "#dc2626"
AMBER  = "#d97706"
PURPLE = "#7c3aed"

def base_layout(fig, **kw):
    ax = dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
               zerolinecolor=GRID, zerolinewidth=1)
    xo = kw.pop("xaxis", {})
    yo = kw.pop("yaxis", {})
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PLOT,
        font=dict(color=FONT, size=12),
        title_font=dict(color="#111827", size=13),
        margin=dict(t=36, b=20, l=8, r=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT)),
        xaxis={**ax, **xo}, yaxis={**ax, **yo},
        **kw,
    )
    return fig

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_all():
    kc  = pd.read_parquet(DB / "arb_KC.parquet")["rollex_px"].rename("KC")
    rc  = pd.read_parquet(DB / "arb_RC.parquet")["rollex_px"].rename("RC")
    cc  = pd.read_parquet(DB / "arb_CC.parquet")["rollex_px"].rename("CC")
    lcc = pd.read_parquet(DB / "arb_LCC.parquet")["rollex_px"].rename("LCC")
    gbp = pd.read_parquet(DB / "fx_gbp.parquet")["GBP_USD"]
    return kc, rc, cc, lcc, gbp

kc_raw, rc_raw, cc_raw, lcc_raw, gbp_raw = load_all()

# ── Analytics helpers ─────────────────────────────────────────────────────────

def half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    lag   = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag, delta = lag.align(delta, join="inner")
    X    = add_constant(lag)
    beta = OLS(delta, X).fit().params.iloc[1]
    if beta >= 0:
        return np.nan
    return -np.log(2) / np.log(1 + beta)

def rolling_half_life(spread: pd.Series, window: int) -> pd.Series:
    return spread.rolling(window).apply(lambda s: half_life(pd.Series(s)), raw=False)

def zscore(spread: pd.Series, window: int) -> pd.Series:
    mu  = spread.rolling(window).mean()
    sig = spread.rolling(window).std()
    return (spread - mu) / sig

def rolling_coint_pval(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([s1, s2], axis=1).dropna()
    vals = []
    for i in range(window, len(aligned) + 1):
        chunk = aligned.iloc[i - window: i]
        try:
            _, pv, _ = coint(chunk.iloc[:, 0], chunk.iloc[:, 1])
        except Exception:
            pv = np.nan
        vals.append((aligned.index[i - 1], pv))
    idx, data = zip(*vals)
    return pd.Series(data, index=idx)

def adf_pvalue(spread: pd.Series) -> float:
    try:
        return adfuller(spread.dropna())[1]
    except Exception:
        return np.nan

def percentile_rank(spread: pd.Series, window: int) -> pd.Series:
    def pct(arr):
        return (arr[:-1] < arr[-1]).sum() / (len(arr) - 1) * 100
    return spread.rolling(window).apply(pct, raw=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Configuration")
    pair = st.radio("Pair", ["KC / RC  (Arabica vs Robusta)", "CC / LCC  (NY vs London Cocoa)"],
                    index=0, label_visibility="collapsed")
    pair_key = "KCRC" if pair.startswith("KC") else "CCLCC"

    st.divider()
    st.markdown("**Windows**")
    zscore_win  = st.slider("Z-score lookback (days)",  60, 504, 252, step=21)
    corr_win    = st.slider("Rolling correlation (days)", 30, 252, 60,  step=10)
    hl_win      = st.slider("Rolling half-life (days)",  60, 504, 126, step=21)

    st.divider()
    st.markdown("**Signal thresholds**")
    z_entry = st.number_input("Entry z-score",  value=1.5, step=0.1, format="%.1f")
    z_exit  = st.number_input("Exit z-score",   value=0.5, step=0.1, format="%.1f")

# ── Build spread ──────────────────────────────────────────────────────────────

if pair_key == "KCRC":
    kc_mt   = kc_raw * KC_FACTOR
    spread  = (kc_mt - rc_raw).dropna()
    leg1_label, leg2_label = "KC ($/MT)", "RC ($/MT)"
    spread_label = "Arabica Premium over Robusta ($/MT)"
    pair_title   = "KC / RC  —  Arabica vs Robusta"
    has_fx       = False
else:
    lcc_usd = (lcc_raw * gbp_raw).dropna()
    spread  = (cc_raw - lcc_usd).dropna()
    leg1_label, leg2_label = "CC ($/MT)", "LCC in USD ($/MT)"
    spread_label = "NY Premium over London Cocoa ($/MT)"
    pair_title   = "CC / LCC  —  NY vs London Cocoa"
    has_fx       = True

# ── Date range ────────────────────────────────────────────────────────────────

date_min = spread.index.min().date()
date_max = spread.index.max().date()

st.title("ARB Monitor")
st.caption(pair_title)
st.markdown("**Date range**")

d_start, d_end = st.slider(
    "range", min_value=date_min, max_value=date_max,
    value=(date_min, date_max), format="DD MMM YYYY",
    label_visibility="collapsed",
)
cal_l, cal_r, _ = st.columns([1, 1, 4])
with cal_l:
    d_start = st.date_input("Start", value=d_start, min_value=date_min,
                             max_value=date_max, key="ds")
with cal_r:
    d_end   = st.date_input("End",   value=d_end,   min_value=date_min,
                             max_value=date_max, key="de")

spread  = spread.loc[str(d_start): str(d_end)]
gbp_raw = gbp_raw.loc[str(d_start): str(d_end)]

st.divider()

# ── Compute ───────────────────────────────────────────────────────────────────

z = zscore(spread, zscore_win)
mu  = spread.rolling(zscore_win).mean()
sig = spread.rolling(zscore_win).std()

cur_spread = spread.iloc[-1]
cur_z      = z.iloc[-1]
cur_pct    = percentile_rank(spread, zscore_win).iloc[-1]
hl_val     = half_life(spread)

# Days since last mean cross
crosses   = (z * z.shift(1)) < 0
last_cross = (spread.index[-1] - spread.index[crosses][-1]).days if crosses.any() else None

# ── KPIs ──────────────────────────────────────────────────────────────────────

z_color   = "red"   if abs(cur_z) > z_entry else "green" if abs(cur_z) < z_exit else "amber"
pct_color = "red"   if cur_pct > 85        else "green" if cur_pct < 15       else "teal"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi teal">
    <div class="kpi-label">Current Spread</div>
    <div class="kpi-value">${cur_spread:,.1f}</div>
    <div class="kpi-sub">{spread_label.split('(')[0].strip()}</div>
  </div>
  <div class="kpi {z_color}">
    <div class="kpi-label">Z-Score  ({zscore_win}d)</div>
    <div class="kpi-value">{cur_z:+.2f}</div>
    <div class="kpi-sub">Entry threshold: &plusmn;{z_entry:.1f}</div>
  </div>
  <div class="kpi {pct_color}">
    <div class="kpi-label">Percentile Rank</div>
    <div class="kpi-value">{cur_pct:.0f}<span style="font-size:14px">th</span></div>
    <div class="kpi-sub">vs {zscore_win}d history</div>
  </div>
  <div class="kpi purple">
    <div class="kpi-label">Half-Life</div>
    <div class="kpi-value">{"—" if np.isnan(hl_val) else f"{hl_val:.0f}d"}</div>
    <div class="kpi-sub">Expected mean reversion</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-label">Days Since Cross</div>
    <div class="kpi-value">{last_cross if last_cross else "—"}</div>
    <div class="kpi-sub">Last zero-line cross</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Spread Monitor
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("Section 1 — Spread Monitor", expanded=True):
    st.caption("Spread level with rolling mean and 1/2 standard deviation bands. "
               "The z-score panel shows where the spread sits relative to its own history.")

    # — Spread + bands —
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu + 2*sig, name="+2σ",
        line=dict(color=RED, width=1, dash="dot"), showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu + sig, name="+1σ",
        line=dict(color=AMBER, width=1, dash="dash"), showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu, name="Mean",
        line=dict(color=MUTED, width=1.5), showlegend=True))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu - sig, name="-1σ",
        line=dict(color=AMBER, width=1, dash="dash"), showlegend=False))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=mu - 2*sig, name="-2σ",
        line=dict(color=RED, width=1, dash="dot"), showlegend=False))
    fig_sp.add_trace(go.Scatter(
        x=spread.index, y=spread, name="Spread",
        line=dict(color=TEAL, width=2), showlegend=True))
    base_layout(fig_sp, title=spread_label)
    st.plotly_chart(fig_sp, use_container_width=True)

    # — Z-score —
    fig_z = go.Figure()
    fig_z.add_hrect(y0=z_entry, y1=4,   fillcolor=RED,   opacity=0.06, line_width=0)
    fig_z.add_hrect(y0=-4, y1=-z_entry, fillcolor=GREEN, opacity=0.06, line_width=0)
    fig_z.add_trace(go.Scatter(
        x=z.index, y=z, name="Z-score",
        line=dict(color=TEAL, width=1.5)))
    for level, color in [(z_entry, RED), (-z_entry, GREEN), (z_exit, AMBER), (-z_exit, AMBER)]:
        fig_z.add_hline(y=level, line_dash="dot", line_color=color, line_width=1)
    fig_z.add_hline(y=0, line_color=MUTED, line_width=1)
    base_layout(fig_z, title=f"Z-Score  ({zscore_win}d rolling)",
                yaxis=dict(gridcolor=GRID, linecolor=GRID,
                           tickfont=dict(color=MUTED), range=[-4, 4]))
    st.plotly_chart(fig_z, use_container_width=True)

    # — Individual legs —
    if pair_key == "KCRC":
        l1 = (kc_raw * KC_FACTOR).loc[str(d_start):str(d_end)]
        l2 = rc_raw.loc[str(d_start):str(d_end)]
    else:
        l1 = cc_raw.loc[str(d_start):str(d_end)]
        l2 = (lcc_raw * gbp_raw).dropna().loc[str(d_start):str(d_end)]

    fig_legs = go.Figure()
    fig_legs.add_trace(go.Scatter(x=l1.index, y=l1, name=leg1_label,
                                  line=dict(color=TEAL, width=1.5)))
    fig_legs.add_trace(go.Scatter(x=l2.index, y=l2, name=leg2_label,
                                  line=dict(color=AMBER, width=1.5)))
    base_layout(fig_legs, title="Individual Legs ($/MT)",
                yaxis=dict(gridcolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED)))
    st.plotly_chart(fig_legs, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FX Decomposition (CC/LCC) or Ratio (KC/RC)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("Section 2 — FX Decomposition" if has_fx else "Section 2 — Ratio & Basis", expanded=True):

    if has_fx:
        st.caption("Decomposes the daily spread change into a price component (actual exchange "
                   "divergence) and an FX component (GBP/USD move). Trade the price component; "
                   "be aware of the FX component.")

        cc_al  = cc_raw.loc[str(d_start):str(d_end)]
        lcc_al = lcc_raw.loc[str(d_start):str(d_end)]
        gbp_al = gbp_raw.loc[str(d_start):str(d_end)]

        aligned = pd.concat([cc_al, lcc_al, gbp_al], axis=1).dropna()
        aligned.columns = ["CC", "LCC_GBP", "GBP_USD"]

        d_cc     = aligned["CC"].diff()
        d_lcc    = aligned["LCC_GBP"].diff()
        d_gbp    = aligned["GBP_USD"].diff()

        # ΔSpread = ΔCC - Δ(LCC_GBP × GBP_USD)
        # Price component : ΔCC - ΔLCC_GBP × GBP_USD
        # FX component    : -LCC_GBP × ΔGBP_USD
        px_comp  = d_cc - d_lcc * aligned["GBP_USD"]
        fx_comp  = -aligned["LCC_GBP"] * d_gbp
        d_spread = px_comp + fx_comp

        fig_fx = go.Figure()
        fig_fx.add_trace(go.Bar(x=aligned.index, y=px_comp, name="Price component",
                                marker_color=TEAL, opacity=0.8))
        fig_fx.add_trace(go.Bar(x=aligned.index, y=fx_comp, name="FX component",
                                marker_color=AMBER, opacity=0.8))
        base_layout(fig_fx, title="Daily Spread Change — Price vs FX Contribution ($/MT)",
                    barmode="relative")
        st.plotly_chart(fig_fx, use_container_width=True)

        # GBP/USD overlay
        c1, c2 = st.columns(2)
        with c1:
            fig_gbp = go.Figure()
            fig_gbp.add_trace(go.Scatter(x=gbp_al.index, y=gbp_al,
                                         line=dict(color=PURPLE, width=1.5), name="GBP/USD"))
            base_layout(fig_gbp, title="GBP/USD Spot")
            st.plotly_chart(fig_gbp, use_container_width=True)
        with c2:
            roll_corr_fx = px_comp.rolling(corr_win).corr(fx_comp)
            fig_corr_fx = go.Figure()
            fig_corr_fx.add_trace(go.Scatter(x=roll_corr_fx.index, y=roll_corr_fx,
                                              line=dict(color=AMBER, width=1.5),
                                              name="Price/FX correlation"))
            fig_corr_fx.add_hline(y=0, line_color=MUTED, line_width=1)
            base_layout(fig_corr_fx, title=f"Rolling Corr: Price vs FX Component ({corr_win}d)")
            st.plotly_chart(fig_corr_fx, use_container_width=True)

    else:
        st.caption("Ratio of Arabica to Robusta price (both in $/MT). "
                   "Roasters blend the two; extreme ratios historically mean-revert "
                   "as substitution economics kick in.")

        ratio = l1 / l2
        mu_r  = ratio.rolling(zscore_win).mean()
        sig_r = ratio.rolling(zscore_win).std()

        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r + sig_r,
                                       line=dict(color=AMBER, width=1, dash="dash"), name="+1σ"))
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r - sig_r,
                                       line=dict(color=AMBER, width=1, dash="dash"),
                                       name="-1σ", showlegend=False))
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=mu_r,
                                       line=dict(color=MUTED, width=1), name="Mean"))
        fig_ratio.add_trace(go.Scatter(x=ratio.index, y=ratio,
                                       line=dict(color=TEAL, width=2), name="KC/RC Ratio"))
        base_layout(fig_ratio, title="KC/RC Price Ratio (Arabica/Robusta, $/MT)")
        st.plotly_chart(fig_ratio, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            roll_corr = l1.rolling(corr_win).corr(l2)
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=roll_corr.index, y=roll_corr,
                                         line=dict(color=GREEN, width=1.5), name="Correlation"))
            fig_rc.add_hline(y=0, line_color=MUTED, line_width=1)
            base_layout(fig_rc, title=f"Rolling Correlation KC vs RC ({corr_win}d)")
            st.plotly_chart(fig_rc, use_container_width=True)
        with c2:
            pct_rank = percentile_rank(spread, zscore_win)
            fig_pct = go.Figure()
            fig_pct.add_trace(go.Scatter(x=pct_rank.index, y=pct_rank,
                                          line=dict(color=PURPLE, width=1.5),
                                          name="Percentile rank"))
            fig_pct.add_hline(y=85, line_color=RED,   line_dash="dot", line_width=1)
            fig_pct.add_hline(y=15, line_color=GREEN, line_dash="dot", line_width=1)
            fig_pct.add_hline(y=50, line_color=MUTED, line_width=1)
            base_layout(fig_pct, title=f"Spread Percentile Rank ({zscore_win}d)",
                        yaxis=dict(gridcolor=GRID, linecolor=GRID,
                                   tickfont=dict(color=MUTED), range=[0, 100]))
            st.plotly_chart(fig_pct, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Advanced Analytics (collapsed)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("Section 3 — Advanced Analytics", expanded=False):
    st.caption("Statistical tests for mean reversion and cointegration. "
               "These confirm whether the spread relationship is structurally intact "
               "and how quickly it tends to correct.")

    # ── Static stats panel ────────────────────────────────────────────────────
    adf_p   = adf_pvalue(spread)
    try:
        _, coint_p, _ = coint(l1.reindex(spread.index).dropna(),
                               l2.reindex(spread.index).dropna())
    except Exception:
        coint_p = np.nan

    spread_mean = spread.mean()
    spread_std  = spread.std()
    spread_skew = spread.skew()
    pct5  = spread.quantile(0.05)
    pct95 = spread.quantile(0.95)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-tag">Spread Statistics</div>', unsafe_allow_html=True)
        rows = [
            ("Mean",            f"${spread_mean:,.1f}"),
            ("Std Dev",         f"${spread_std:,.1f}"),
            ("Skewness",        f"{spread_skew:.2f}"),
            ("5th / 95th pct",  f"${pct5:,.1f}  /  ${pct95:,.1f}"),
            ("Current spread",  f"${cur_spread:,.1f}"),
            ("Current z-score", f"{cur_z:+.2f}"),
        ]
        for k, v in rows:
            st.markdown(f'<div class="stat-row"><span class="stat-key">{k}</span>'
                        f'<span class="stat-val">{v}</span></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="sec-tag">Mean Reversion Tests</div>', unsafe_allow_html=True)
        adf_sig  = "Yes" if adf_p < 0.05 else "Borderline" if adf_p < 0.10 else "No"
        co_sig   = "Yes" if coint_p < 0.05 else "Borderline" if coint_p < 0.10 else "No"
        rows2 = [
            ("ADF p-value",         f"{adf_p:.4f}"),
            ("Spread stationary",   adf_sig),
            ("Cointegration p-val", f"{coint_p:.4f}"),
            ("Cointegrated",        co_sig),
            ("Half-life (full)",    f"{hl_val:.0f}d" if not np.isnan(hl_val) else "—"),
            ("Days since cross",    str(last_cross) if last_cross else "—"),
        ]
        for k, v in rows2:
            color = ""
            if v == "Yes": color = f"color:{GREEN};"
            if v == "No":  color = f"color:{RED};"
            st.markdown(f'<div class="stat-row"><span class="stat-key">{k}</span>'
                        f'<span class="stat-val" style="{color}">{v}</span></div>',
                        unsafe_allow_html=True)

    st.divider()

    # ── Rolling half-life ─────────────────────────────────────────────────────
    with st.spinner("Computing rolling half-life..."):
        rhl = rolling_half_life(spread, hl_win)

    fig_hl = go.Figure()
    fig_hl.add_trace(go.Scatter(x=rhl.index, y=rhl.clip(0, 365),
                                 line=dict(color=PURPLE, width=1.5), name="Half-life (days)"))
    fig_hl.add_hline(y=hl_val, line_color=MUTED, line_dash="dot", line_width=1,
                     annotation_text=f"Full-period: {hl_val:.0f}d" if not np.isnan(hl_val) else "",
                     annotation_font=dict(color=MUTED, size=11))
    base_layout(fig_hl, title=f"Rolling Half-Life ({hl_win}d window)  — days to expected mean reversion")
    st.plotly_chart(fig_hl, use_container_width=True)

    # ── Rolling cointegration p-value ─────────────────────────────────────────
    st.markdown("**Rolling cointegration** — p-value below 0.05 means the pair is statistically "
                "cointegrated in that window. Elevated p-values warn that the ARB relationship "
                "may be breaking down.")
    coint_win = st.slider("Cointegration window (days)", 60, 252, 120, step=20,
                           key="coint_win")
    with st.spinner("Computing rolling cointegration..."):
        rcoint = rolling_coint_pval(
            l1.reindex(spread.index).dropna(),
            l2.reindex(spread.index).dropna(),
            coint_win,
        )

    fig_co = go.Figure()
    fig_co.add_hrect(y0=0, y1=0.05, fillcolor=GREEN, opacity=0.08, line_width=0)
    fig_co.add_hrect(y0=0.05, y1=0.10, fillcolor=AMBER, opacity=0.06, line_width=0)
    fig_co.add_trace(go.Scatter(x=rcoint.index, y=rcoint,
                                 line=dict(color=TEAL, width=1.5), name="p-value"))
    fig_co.add_hline(y=0.05, line_color=GREEN, line_dash="dot", line_width=1)
    fig_co.add_hline(y=0.10, line_color=AMBER, line_dash="dot", line_width=1)
    base_layout(fig_co, title=f"Rolling Cointegration p-value ({coint_win}d)",
                yaxis=dict(gridcolor=GRID, linecolor=GRID,
                           tickfont=dict(color=MUTED), range=[0, 1]))
    st.plotly_chart(fig_co, use_container_width=True)

    # ── Rolling correlation ───────────────────────────────────────────────────
    roll_corr_main = l1.rolling(corr_win).corr(l2)
    fig_corr = go.Figure()
    fig_corr.add_trace(go.Scatter(x=roll_corr_main.index, y=roll_corr_main,
                                   line=dict(color=GREEN, width=1.5), name="Correlation"))
    fig_corr.add_hline(y=0.9, line_color=MUTED, line_dash="dot", line_width=1)
    fig_corr.add_hline(y=0,   line_color=MUTED, line_width=1)
    base_layout(fig_corr, title=f"Rolling Correlation ({corr_win}d)")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Spread distribution ───────────────────────────────────────────────────
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=spread, nbinsx=80, name="Spread distribution",
                                     marker_color=TEAL, opacity=0.75))
    fig_dist.add_vline(x=cur_spread, line_color=RED,   line_width=2,
                       annotation_text="Current", annotation_font=dict(color=RED, size=11))
    fig_dist.add_vline(x=spread_mean, line_color=MUTED, line_dash="dot", line_width=1,
                       annotation_text="Mean", annotation_font=dict(color=MUTED, size=11))
    base_layout(fig_dist, title="Spread Distribution — where is today vs history")
    st.plotly_chart(fig_dist, use_container_width=True)

st.caption("ICEBREAKER ARB  —  Data: Rollex parquets + ICE GBP/USD")
