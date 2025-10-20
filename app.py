# Streamlit app — SkillCorner Analyzer (Runs + Under Pressure)
# Author: GPT-5 Thinking
# Purpose: Upload CSVs exported from SkillCorner, map columns, compute KPIs for
# (A) passes to teammate runs and (B) actions under pressure; rank players,
# visualize trade-offs, and export results. Includes an "About & Methods" tab
# with metric definitions and references.
#
# How to run locally:
#   1) pip install -r requirements.txt
#   2) streamlit run streamlit_app_skillcorner_runs.py
#
# Notes
# - Auto-detects semicolon delimiters and decimal commas; override in the sidebar.
# - If your column names differ, use the column mapping controls (sidebar → “Column mapping”).
# - Expected content is aggregated per player (per-match rates). Team-level rows also work.

from __future__ import annotations
import io
import re
import unicodedata
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="SkillCorner Analyzer", page_icon="⚽", layout="wide")
st.title("⚽ SkillCorner Analyzer")
st.caption("Upload CSVs, map columns if needed, explore KPIs, visualize, and download leaderboards.")

# --------------------------
# Utilities
# --------------------------

def normalize_colname(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO, semicolon: bool, comma_decimal: bool) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    sep = ";" if semicolon else ","
    if comma_decimal:
        raw = file.read()
        text = raw.decode("utf-8", errors="ignore")
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)  # replace decimal comma
        return pd.read_csv(io.StringIO(text), sep=sep)
    else:
        file.seek(0)
        return pd.read_csv(file, sep=sep)

# Canonical columns for the RUNS module
RUNS_EXPECTED: Dict[str, list[str]] = {
    "player": ["player", "name"],
    "third": ["third", "third of the pitch", "zone third"],
    "channel": ["channel", "lane", "corridor"],
    "minutes_pm": ["minutes played per match", "mins per match", "minutes/match"],
    "opp_runs_pm": ["count opportunities to pass to runs per match"],
    "att_runs_pm": ["count pass attempts for runs per match"],
    "comp_ratio_runs": ["pass completion ratio to runs"],
    "threat_completed_pm": ["threat of runs to which a pass was completed per match"],
    "completed_runs_pm": ["count completed passes for runs per match"],
    "completed_to_shot_pm": ["count completed passes leading to shot for runs per match"],
    "completed_to_goal_pm": ["count completed passes leading to goal for runs per match"],
    "opp_danger_runs_pm": ["count opportunities to pass to dangerous runs per match"],
    "att_danger_runs_pm": ["count pass attempts for dangerous runs per match"],
    "completed_danger_runs_pm": ["count completed passes for dangerous runs per match"],
    "threat_all_opp_pm": ["threat of all opportunities to pass to runs per match"],
    "opp_runs_in_sample": ["count opportunities to pass to runs in sample"],
}

# Canonical columns for the UNDER PRESSURE module
PRESS_EXPECTED: Dict[str, list[str]] = {
    "player": ["player", "name"],
    "third": ["third"],
    "channel": ["channel"],
    "minutes_pm": ["minutes played per match"],
    "pressures_pm": ["count pressures received per match"],
    "pass_att_press_pm": ["count pass attempts under pressure per match"],
    "comp_ratio_press": ["pass completion ratio under pressure"],
    "comp_ratio_press_danger": ["dangerous pass completion ratio under pressure"],
    "comp_ratio_press_difficult": ["difficult pass completion ratio under pressure"],
    "retentions_press_pm": ["count ball retentions under pressure per match"],
    "forced_losses_press_pm": ["count forced losses under pressure per match"],
}

DEFAULT_WEIGHTS_RUNS = {
    # Composite score to highlight creators who try & complete high-value passes to runs
    "threat_completed_pm": 0.30,
    "completed_runs_pm": 0.20,
    "comp_ratio_runs": 0.20,
    "attempt_rate_runs": 0.20,
    "conversion_to_shot": 0.10,
}

# --------------------------
# Sidebar — Data input & parsing
# --------------------------

st.sidebar.header("1) Upload CSVs")
runs_file = st.sidebar.file_uploader("Upload RUNS CSV (player-level)", type=["csv"], key="runs")
press_file = st.sidebar.file_uploader("Upload UNDER PRESSURE CSV (player-level)", type=["csv"], key="press")

st.sidebar.subheader("Delimiter & decimal")
use_semicolon = st.sidebar.checkbox("CSV uses semicolon (;) as delimiter", value=True)
use_comma_decimal = st.sidebar.checkbox("Numbers use comma as decimal (e.g., 12,3)", value=False)

runs_df_raw = load_csv(runs_file, use_semicolon, use_comma_decimal)
press_df_raw = load_csv(press_file, use_semicolon, use_comma_decimal)

# --------------------------
# Tabs
# --------------------------

TAB_RUNS, TAB_PRESS, TAB_NOTES = st.tabs(["Runs", "Under Pressure", "About & Methods"])

# ==========================
# RUNS TAB
# ==========================
with TAB_RUNS:
    st.subheader("Passes to Runs — KPIs, Leaderboards & Visualizations")

    if runs_df_raw.empty:
        st.info("Upload a RUNS CSV to use this tab.")
    else:
        st.success(f"Loaded RUNS data: {runs_df_raw.shape[0]} rows × {runs_df_raw.shape[1]} columns")

        st.markdown("### Column mapping")
        norm_cols = {normalize_colname(c): c for c in runs_df_raw.columns}
        auto_map: Dict[str, Optional[str]] = {}
        for canon, aliases in RUNS_EXPECTED.items():
            found = None
            for alias in aliases + [canon]:
                alias_norm = normalize_colname(alias)
                if alias_norm in norm_cols:
                    found = norm_cols[alias_norm]
                    break
                for k_norm, orig in norm_cols.items():
                    if alias_norm in k_norm:
                        found = orig
                        break
                if found:
                    break
            auto_map[canon] = found

        mapping: Dict[str, Optional[str]] = {}
        cols = st.columns(3)
        items = list(RUNS_EXPECTED.keys())
        for i, canon in enumerate(items):
            with cols[i % 3]:
                options = [None] + list(runs_df_raw.columns)
                mapping[canon] = st.selectbox(
                    f"{canon}", options=options, index=(options.index(auto_map[canon]) if auto_map[canon] in options else 0), key=f"map_runs_{canon}"
                )

        required = [
            "player", "opp_runs_pm", "att_runs_pm", "comp_ratio_runs",
            "threat_completed_pm", "completed_runs_pm",
            "completed_to_shot_pm", "completed_to_goal_pm",
            "opp_danger_runs_pm", "att_danger_runs_pm", "completed_danger_runs_pm",
        ]
        missing = [c for c in required if mapping.get(c) is None]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            cols_to_take = [c for c in mapping.values() if c is not None]
            df = runs_df_raw[cols_to_take].copy()
            df.columns = [k for k, v in mapping.items() if v is not None]

            for opt in ["third", "channel", "minutes_pm", "threat_all_opp_pm", "opp_runs_in_sample"]:
                if opt not in df.columns:
                    df[opt] = np.nan

            # KPIs
            df["attempt_rate_runs"] = df["att_runs_pm"] / df["opp_runs_pm"].replace({0: np.nan})
            df["attempt_rate_danger"] = df["att_danger_runs_pm"] / df["opp_danger_runs_pm"].replace({0: np.nan})
            df["productivity_runs"] = df["completed_runs_pm"]
            df["conversion_to_shot"] = df["completed_to_shot_pm"] / df["completed_runs_pm"].replace({0: np.nan})
            df["conversion_to_goal"] = df["completed_to_goal_pm"] / df["completed_runs_pm"].replace({0: np.nan})

            pct = lambda s: s.rank(pct=True, na_option="keep")
            for col in [
                "attempt_rate_runs", "comp_ratio_runs", "threat_completed_pm",
                "productivity_runs", "conversion_to_shot", "conversion_to_goal",
            ]:
                df[f"p_{col}"] = pct(df[col])

            st.markdown("### Composite weights")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                w_threat = st.number_input("Threat", 0.0, 1.0, value=float(DEFAULT_WEIGHTS_RUNS["threat_completed_pm"]))
            with c2:
                w_prod = st.number_input("Completed runs", 0.0, 1.0, value=float(DEFAULT_WEIGHTS_RUNS["completed_runs_pm"]))
            with c3:
                w_comp = st.number_input("Completion", 0.0, 1.0, value=float(DEFAULT_WEIGHTS_RUNS["comp_ratio_runs"]))
            with c4:
                w_attempt = st.number_input("Attempt rate", 0.0, 1.0, value=float(DEFAULT_WEIGHTS_RUNS["attempt_rate_runs"]))
            with c5:
                w_convshot = st.number_input("Conversion→shot", 0.0, 1.0, value=float(DEFAULT_WEIGHTS_RUNS["conversion_to_shot"]))

            sum_w = w_threat + w_prod + w_comp + w_attempt + w_convshot
            if sum_w == 0:
                st.warning("All weights are zero. Using equal weights (0.2).")
                w_threat = w_prod = w_comp = w_attempt = w_convshot = 0.2
                sum_w = 1.0
            w_threat, w_prod, w_comp, w_attempt, w_convshot = [w / sum_w for w in [w_threat, w_prod, w_comp, w_attempt, w_convshot]]

            df["composite_runs_creator"] = (
                w_threat * df["p_threat_completed_pm"]
                + w_prod * df["p_productivity_runs"]
                + w_comp * df["p_comp_ratio_runs"]
                + w_attempt * df["p_attempt_rate_runs"]
                + w_convshot * df["p_conversion_to_shot"]
            )

            st.markdown("### Filters")
            min_minutes = st.number_input("Min minutes per match", 0.0, value=0.0, step=1.0)
            sel_third = st.multiselect("Third(s)", options=sorted([x for x in df["third"].dropna().unique()]))
            sel_channel = st.multiselect("Channel(s)", options=sorted([x for x in df["channel"].dropna().unique()]))

            mask = pd.Series(True, index=df.index)
            if not pd.isna(df["minutes_pm"]).all():
                mask &= (df["minutes_pm"].fillna(0) >= min_minutes)
            if sel_third:
                mask &= df["third"].isin(sel_third)
            if sel_channel:
                mask &= df["channel"].isin(sel_channel)
            df_view = df.loc[mask].copy()

            st.markdown("### Leaderboards")
            n_top = st.slider("Top N", 5, 100, 20, step=5, key="runs_topn")

            cols_creator = [
                "player", "third", "channel", "opp_runs_pm", "att_runs_pm", "attempt_rate_runs",
                "comp_ratio_runs", "threat_completed_pm", "completed_runs_pm",
                "completed_to_shot_pm", "completed_to_goal_pm", "composite_runs_creator",
            ]
            leader_creator = df_view.sort_values("composite_runs_creator", ascending=False)[cols_creator].head(n_top)

            cols_danger = [
                "player", "third", "channel", "opp_danger_runs_pm", "att_danger_runs_pm",
                "attempt_rate_danger", "completed_danger_runs_pm",
            ]
            leader_danger = df_view.sort_values(["completed_danger_runs_pm", "att_danger_runs_pm"], ascending=[False, False])[cols_danger].head(n_top)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top creators to runs (composite score)**")
                st.dataframe(leader_creator.round(3), use_container_width=True)
                st.download_button(
                    "Download CSV — Top creators",
                    data=leader_creator.to_csv(index=False).encode("utf-8"),
                    file_name="skillcorner_top_runs_creators.csv",
                    mime="text/csv",
                )
            with c2:
                st.markdown("**Top for dangerous runs**")
                st.dataframe(leader_danger.round(3), use_container_width=True)
                st.download_button(
                    "Download CSV — Dangerous runs",
                    data=leader_danger.to_csv(index=False).encode("utf-8"),
                    file_name="skillcorner_top_danger_runs.csv",
                    mime="text/csv",
                )

            st.markdown("### Visualizations")
            base = alt.Chart(df_view.dropna(subset=["attempt_rate_runs", "comp_ratio_runs"])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X("attempt_rate_runs:Q", title="Attempt rate to runs (attempts / opportunities)"),
                    y=alt.Y("comp_ratio_runs:Q", title="Pass completion to runs"),
                    tooltip=["player", "attempt_rate_runs", "comp_ratio_runs", "threat_completed_pm", "completed_runs_pm"],
                )
            color_choice = st.selectbox("Color points by", ["None", "third", "channel"], index=0, key="runs_color")
            chart = base.encode(color=alt.Color(f"{color_choice}:N")) if color_choice != "None" else base
            st.altair_chart(chart.properties(title="Attempt vs. completion"), use_container_width=True)

            chart2 = alt.Chart(df_view.dropna(subset=["threat_completed_pm", "completed_runs_pm"])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X("threat_completed_pm:Q", title="Threat of completed runs per match"),
                    y=alt.Y("completed_runs_pm:Q", title="Completed passes to runs per match"),
                    tooltip=["player", "threat_completed_pm", "completed_runs_pm"],
                )
            st.altair_chart(chart2.properties(title="Value (threat) vs. volume (completed runs)"), use_container_width=True)

            # --- Exports: XLSX & HTML charts (Runs)
            st.markdown("### Exports")
            import io as _io
            _buf = _io.BytesIO()
            try:
                with pd.ExcelWriter(_buf, engine="xlsxwriter") as xw:
                    leader_creator.to_excel(xw, sheet_name="Top creators", index=False)
                    leader_danger.to_excel(xw, sheet_name="Dangerous runs", index=False)
                    df_view.to_excel(xw, sheet_name="Mapped data", index=False)
                st.download_button("Download XLSX — Runs module", data=_buf.getvalue(), file_name="runs_module_leaderboards.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"XLSX export unavailable: {e}")

            # Rebuild charts to export as standalone HTML with tooltips
            _chart_a = alt.Chart(df_view.dropna(subset=["attempt_rate_runs", "comp_ratio_runs"])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X("attempt_rate_runs:Q", title="Attempt rate to runs (attempts / opportunities)"),
                    y=alt.Y("comp_ratio_runs:Q", title="Pass completion to runs"),
                    tooltip=["player", "attempt_rate_runs", "comp_ratio_runs", "threat_completed_pm", "completed_runs_pm"],
                )
            st.download_button("Download HTML — Attempt vs completion", data=_chart_a.to_html(), file_name="runs_attempt_vs_completion.html", mime="text/html")

            _chart_b = alt.Chart(df_view.dropna(subset=["threat_completed_pm", "completed_runs_pm"])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X("threat_completed_pm:Q", title="Threat of completed runs per match"),
                    y=alt.Y("completed_runs_pm:Q", title="Completed passes to runs per match"),
                    tooltip=["player", "threat_completed_pm", "completed_runs_pm"],
                )
            st.download_button("Download HTML — Threat vs volume", data=_chart_b.to_html(), file_name="runs_threat_vs_volume.html", mime="text/html")

            with st.expander("Preview raw data (mapped)"):
                st.dataframe(df.head(25), use_container_width=True)

# ==========================
# UNDER PRESSURE TAB
# ==========================
with TAB_PRESS:
    st.subheader("Under Pressure — KPIs, Leaderboards & Visualizations")

    if press_df_raw.empty:
        st.info("Upload an UNDER PRESSURE CSV to use this tab.")
    else:
        st.success(f"Loaded UNDER PRESSURE data: {press_df_raw.shape[0]} rows × {press_df_raw.shape[1]} columns")

        st.markdown("### Column mapping")
        norm_cols_p = {normalize_colname(c): c for c in press_df_raw.columns}
        auto_map_p: Dict[str, Optional[str]] = {}
        for canon, aliases in PRESS_EXPECTED.items():
            found = None
            for alias in aliases + [canon]:
                alias_norm = normalize_colname(alias)
                if alias_norm in norm_cols_p:
                    found = norm_cols_p[alias_norm]
                    break
                for k_norm, orig in norm_cols_p.items():
                    if alias_norm in k_norm:
                        found = orig
                        break
                if found:
                    break
            auto_map_p[canon] = found

        mapping_p: Dict[str, Optional[str]] = {}
        cols = st.columns(3)
        items = list(PRESS_EXPECTED.keys())
        for i, canon in enumerate(items):
            with cols[i % 3]:
                options = [None] + list(press_df_raw.columns)
                mapping_p[canon] = st.selectbox(
                    f"{canon}", options=options, index=(options.index(auto_map_p[canon]) if auto_map_p[canon] in options else 0), key=f"map_press_{canon}"
                )

        required_p = [
            "player", "pressures_pm", "pass_att_press_pm", "comp_ratio_press",
            "retentions_press_pm", "forced_losses_press_pm",
        ]
        missing_p = [c for c in required_p if mapping_p.get(c) is None]
        if missing_p:
            st.error("Missing required columns: " + ", ".join(missing_p))
        else:
            cols_to_take_p = [c for c in mapping_p.values() if c is not None]
            dfp = press_df_raw[cols_to_take_p].copy()
            dfp.columns = [k for k, v in mapping_p.items() if v is not None]

            for opt in ["third", "channel", "minutes_pm", "comp_ratio_press_danger", "comp_ratio_press_difficult"]:
                if opt not in dfp.columns:
                    dfp[opt] = np.nan

            eps = 1e-9
            dfp['press_load_pm'] = dfp['pressures_pm']
            dfp['pass_share_under_pressure'] = dfp['pass_att_press_pm'] / (dfp['press_load_pm'] + eps)
            dfp['retention_rate_under_pressure'] = dfp['retentions_press_pm'] / (dfp['press_load_pm'] + eps)
            dfp['forced_loss_rate_under_pressure'] = dfp['forced_losses_press_pm'] / (dfp['press_load_pm'] + eps)

            dfp['succ_pass_press_pm'] = dfp['comp_ratio_press'] * dfp['pass_att_press_pm']
            dfp['succ_pass_press_danger_pm'] = dfp['comp_ratio_press_danger'] * dfp['pass_att_press_pm']
            dfp['succ_pass_press_difficult_pm'] = dfp['comp_ratio_press_difficult'] * dfp['pass_att_press_pm']

            dfp['OPR'] = (dfp['retentions_press_pm'] + dfp['succ_pass_press_pm']) / (dfp['press_load_pm'] + eps)
            dfp['OPR_danger'] = (dfp['retentions_press_pm'] + dfp['succ_pass_press_danger_pm']) / (dfp['press_load_pm'] + eps)
            dfp['OPR_difficult'] = (dfp['retentions_press_pm'] + dfp['succ_pass_press_difficult_pm']) / (dfp['press_load_pm'] + eps)
            dfp['risk_index'] = 1.0 - dfp['forced_loss_rate_under_pressure']

            # Percentiles (for benchmarking)
            pct = lambda s: s.rank(pct=True, na_option="keep")
            for col in ['OPR','OPR_danger','OPR_difficult','risk_index','comp_ratio_press','comp_ratio_press_danger','comp_ratio_press_difficult','pass_share_under_pressure']:
                dfp[f'p_{col}'] = pct(dfp[col])

            # Filters
            st.markdown("### Filters")
            min_minutes_p = st.number_input("Min minutes per match", 0.0, value=0.0, step=1.0, key="press_minmins")
            sel_third_p = st.multiselect("Third(s)", options=sorted([x for x in dfp["third"].dropna().unique()]), key="press_third")
            sel_channel_p = st.multiselect("Channel(s)", options=sorted([x for x in dfp["channel"].dropna().unique()]), key="press_channel")

            maskp = pd.Series(True, index=dfp.index)
            if not pd.isna(dfp["minutes_pm"]).all():
                maskp &= (dfp["minutes_pm"].fillna(0) >= min_minutes_p)
            if sel_third_p:
                maskp &= dfp["third"].isin(sel_third_p)
            if sel_channel_p:
                maskp &= dfp["channel"].isin(sel_channel_p)
            dfp_view = dfp.loc[maskp].copy()

            # Leaderboards
            st.markdown("### Leaderboards")
            n_top_p = st.slider("Top N", 5, 100, 20, step=5, key="press_topn")

            cols_press = [
                'player','third','channel','minutes_pm','pressures_pm','pass_att_press_pm','pass_share_under_pressure',
                'retentions_press_pm','forced_losses_press_pm','retention_rate_under_pressure','forced_loss_rate_under_pressure',
                'comp_ratio_press','comp_ratio_press_danger','comp_ratio_press_difficult','OPR','OPR_danger','OPR_difficult','risk_index'
            ]
            leader_press = dfp_view.sort_values(['OPR','risk_index'], ascending=[False, False])[cols_press].head(n_top_p)

            st.dataframe(leader_press.round(3), use_container_width=True)
            st.download_button(
                "Download CSV — Under-pressure metrics",
                data=leader_press.to_csv(index=False).encode("utf-8"),
                file_name="under_pressure_metrics_top.csv",
                mime="text/csv",
            )

            # Visualizations (user choices)
            st.markdown("### Visualizations")
            vis_choice = st.selectbox("Choose a visualization", [
                "OPR vs Forced-loss rate",
                "Pass-share under pressure vs Completion",
                "OPR (danger) vs OPR (difficult)",
                "Bar: Top OPR",
            ], key="press_vis")

            if vis_choice == "OPR vs Forced-loss rate":
                chart = alt.Chart(dfp_view.dropna(subset=['OPR','forced_loss_rate_under_pressure'])) \
                    .mark_circle(size=70) \
                    .encode(
                        x=alt.X('forced_loss_rate_under_pressure:Q', title='Forced-loss rate under pressure (↓ better)'),
                        y=alt.Y('OPR:Q', title='Overcome Pressure Rate (↑ better)'),
                        tooltip=['player','OPR','forced_loss_rate_under_pressure','pressures_pm','pass_att_press_pm']
                    )
                color_by = st.selectbox("Color by", ["None","third","channel"], index=0, key="press_color1")
                if color_by != "None" and color_by in dfp_view.columns:
                    chart = chart.encode(color=alt.Color(f"{color_by}:N"))
                st.altair_chart(chart.properties(title="Pressure resilience"), use_container_width=True)

            elif vis_choice == "Pass-share under pressure vs Completion":
                chart = alt.Chart(dfp_view.dropna(subset=['pass_share_under_pressure','comp_ratio_press'])) \
                    .mark_circle(size=70) \
                    .encode(
                        x=alt.X('pass_share_under_pressure:Q', title='Pass share under pressure'),
                        y=alt.Y('comp_ratio_press:Q', title='Pass completion under pressure'),
                        tooltip=['player','pass_share_under_pressure','comp_ratio_press','OPR']
                    )
                color_by = st.selectbox("Color by", ["None","third","channel"], index=0, key="press_color2")
                if color_by != "None" and color_by in dfp_view.columns:
                    chart = chart.encode(color=alt.Color(f"{color_by}:N"))
                st.altair_chart(chart.properties(title="Style vs execution under pressure"), use_container_width=True)

            elif vis_choice == "OPR (danger) vs OPR (difficult)":
                chart = alt.Chart(dfp_view.dropna(subset=['OPR_danger','OPR_difficult'])) \
                    .mark_circle(size=70) \
                    .encode(
                        x=alt.X('OPR_danger:Q', title='OPR (danger)'),
                        y=alt.Y('OPR_difficult:Q', title='OPR (difficult)'),
                        tooltip=['player','OPR_danger','OPR_difficult','pressures_pm']
                    )
                color_by = st.selectbox("Color by", ["None","third","channel"], index=0, key="press_color3")
                if color_by != "None" and color_by in dfp_view.columns:
                    chart = chart.encode(color=alt.Color(f"{color_by}:N"))
                st.altair_chart(chart.properties(title="Value vs technical difficulty (under pressure)"), use_container_width=True)

            elif vis_choice == "Bar: Top OPR":
                topn = st.slider("Bar chart — Top N", 5, 30, 15, key="press_bar_topn")
                df_bar = dfp_view.sort_values('OPR', ascending=False).head(topn)
                chart = alt.Chart(df_bar).mark_bar() \
                    .encode(
                        x=alt.X('OPR:Q', title='Overcome Pressure Rate'),
                        y=alt.Y('player:N', sort='-x', title='Player'),
                        tooltip=['player','OPR','pressures_pm','pass_att_press_pm']
                    )
                st.altair_chart(chart.properties(title="Top OPR"), use_container_width=True)

            # --- Exports: XLSX & HTML charts (Under Pressure)
            st.markdown("### Exports")
            import io as _io
            _bufp = _io.BytesIO()
            try:
                with pd.ExcelWriter(_bufp, engine="xlsxwriter") as xw:
                    leader_press.to_excel(xw, sheet_name="Under pressure — Top", index=False)
                    dfp_view.to_excel(xw, sheet_name="Mapped data", index=False)
                st.download_button("Download XLSX — Under Pressure module", data=_bufp.getvalue(), file_name="under_pressure_leaderboards.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"XLSX export unavailable: {e}")

            # Offer HTML exports for all four visual variants regardless of the current selection
            _p1 = alt.Chart(dfp_view.dropna(subset=['OPR','forced_loss_rate_under_pressure'])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X('forced_loss_rate_under_pressure:Q', title='Forced-loss rate under pressure (↓ better)'),
                    y=alt.Y('OPR:Q', title='Overcome Pressure Rate (↑ better)'),
                    tooltip=['player','OPR','forced_loss_rate_under_pressure','pressures_pm','pass_att_press_pm']
                )
            st.download_button("Download HTML — OPR vs forced-loss", data=_p1.to_html(), file_name="press_opr_vs_forcedloss.html", mime="text/html")

            _p2 = alt.Chart(dfp_view.dropna(subset=['pass_share_under_pressure','comp_ratio_press'])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X('pass_share_under_pressure:Q', title='Pass share under pressure'),
                    y=alt.Y('comp_ratio_press:Q', title='Pass completion under pressure'),
                    tooltip=['player','pass_share_under_pressure','comp_ratio_press','OPR']
                )
            st.download_button("Download HTML — Style vs execution", data=_p2.to_html(), file_name="press_style_vs_execution.html", mime="text/html")

            _p3 = alt.Chart(dfp_view.dropna(subset=['OPR_danger','OPR_difficult'])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X('OPR_danger:Q', title='OPR (danger)'),
                    y=alt.Y('OPR_difficult:Q', title='OPR (difficult)'),
                    tooltip=['player','OPR_danger','OPR_difficult','pressures_pm']
                )
            st.download_button("Download HTML — OPR(danger) vs OPR(difficult)", data=_p3.to_html(), file_name="press_opr_danger_vs_difficult.html", mime="text/html")

            _topn = min(15, len(dfp_view))
            _df_bar = dfp_view.sort_values('OPR', ascending=False).head(_topn)
            _p4 = alt.Chart(_df_bar).mark_bar().encode(
                x=alt.X('OPR:Q', title='Overcome Pressure Rate'),
                y=alt.Y('player:N', sort='-x', title='Player'),
                tooltip=['player','OPR','pressures_pm','pass_att_press_pm']
            )
            st.download_button("Download HTML — Top OPR (bar)", data=_p4.to_html(), file_name="press_top_opr_bar.html", mime="text/html")

            with st.expander("Preview raw data (mapped)"):
                st.dataframe(dfp.head(25), use_container_width=True)

# ==========================
# ABOUT & METHODS TAB
# ==========================
with TAB_NOTES:
    st.subheader("About this app & methods")

    st.markdown(
        """
        **What this app does**
        - **Runs module**: evaluates creation/execution when passing to teammate runs using per-match rates, completion to runs, and the threat associated with completed runs.
        - **Under Pressure module**: evaluates possession outcomes when the player is pressed, separating retention, forced losses, and successful passes (including *dangerous* and *difficult* flavors when available).

        **Key metrics**
        - *Attempt rate to runs* = attempts / opportunities.
        - *Completion to runs* = share of successful passes to runs.
        - *Threat of completed runs per match* = average threat on runs that actually received the pass.
        - *Composite runs creator* = percentile-weighted blend of threat, productivity, completion, attempt rate, and conversion to shot (weights adjustable in the UI).
        - *Press load per match* = pressures received.
        - *Pass share under pressure* = among pressured actions, how often the player chooses to pass.
        - *Retention rate under pressure* = share of pressured actions ending with possession kept (without losing it).
        - *Forced-loss rate under pressure* = losses caused by pressure / pressures.
        - *OPR — Overcome Pressure Rate* = (retentions + successful passes under pressure) / pressures.
        - *OPR (danger)* and *OPR (difficult)* variants swap generic successful passes for the dangerous/difficult completion ratios.

        **Why these metrics**
        - Passing into runs captures *progressive intent* and *value added* beyond raw completion.
        - Under-pressure outcomes capture *press-resistance* — a key differentiator at higher levels.

        **Background & references (non-exhaustive)**
        - Merlin et al. (2022), *Classification and determinants of passing difficulty in soccer* — identifies determinants of pass difficulty (receiver context, ball trajectory, field zones, passer context). These ideas inform how we interpret *dangerous/difficult* passes and contextual filters.
        - Anzer & Bauer (2022), *Expected passes* — probability models for pass success using positional data; motivates separating completion from *value*.
        - Fernández, Bornn & Cervone (2019/2021), *Expected Possession Value (EPV)* — valuing actions by impact on possession value; motivates the focus on *threat* of completed runs.

        **Extending with coordinates**
        - If you upload event/tracking data with start/end coordinates, we can add distance-binned xPass curves, value-vs-distance plots, and leaderboards conditioned on long vs short passes.
        """
    )

    st.info("Have ideas or different definitions? Adjust weights and filters, or export CSVs to continue analysis in Python/R.")
