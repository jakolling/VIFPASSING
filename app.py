# Create full Streamlit app and radar module, and a requirements.txt for download
from textwrap import dedent
import io, os, json

app_code = dedent(r"""
# Streamlit app — SkillCorner Analyzer (Runs + Under Pressure + About)
# Author: GPT-5 Thinking
# Purpose: Upload CSVs exported from SkillCorner, map columns, compute KPIs for
# (A) passes to teammate runs and (B) actions under pressure; rank players,
# visualize trade-offs (Altair with hover), export HTML of charts, and export XLSX.
#
# How to run locally:
#   1) pip install -r requirements.txt
#   2) streamlit run streamlit_app_skillcorner_analyzer.py

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
st.caption("Upload CSVs, map columns if needed, explore KPIs, visualize, and download leaderboards. All charts include hover tooltips and can be exported to self-contained HTML.")

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
            # Export HTML with hover
            html1 = chart.properties(title="Attempt vs. completion").to_html()
            st.download_button("Download HTML — Attempt vs completion", data=html1, file_name="runs_attempt_vs_completion.html", mime="text/html")

            chart2 = alt.Chart(df_view.dropna(subset=["threat_completed_pm", "completed_runs_pm"])) \
                .mark_circle(size=70) \
                .encode(
                    x=alt.X("threat_completed_pm:Q", title="Threat of completed runs per match"),
                    y=alt.Y("completed_runs_pm:Q", title="Completed passes to runs per match"),
                    tooltip=["player", "threat_completed_pm", "completed_runs_pm"],
                )
            st.altair_chart(chart2.properties(title="Value (threat) vs. volume (completed runs)"), use_container_width=True)
            html2 = chart2.properties(title="Value (threat) vs. volume (completed runs)").to_html()
            st.download_button("Download HTML — Threat vs volume", data=html2, file_name="runs_threat_vs_volume.html", mime="text/html")

            # Exports: XLSX & HTML charts (Runs)
            st.markdown("### Exports")
            _buf = io.BytesIO()
            try:
                with pd.ExcelWriter(_buf, engine="xlsxwriter") as xw:
                    leader_creator.to_excel(xw, sheet_name="Top creators", index=False)
                    leader_danger.to_excel(xw, sheet_name="Dangerous runs", index=False)
                    df_view.to_excel(xw, sheet_name="Mapped data", index=False)
                st.download_button("Download XLSX — Runs module", data=_buf.getvalue(), file_name="runs_module_leaderboards.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"XLSX export unavailable: {e}")

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
                htmlp1 = chart.properties(title="Pressure resilience").to_html()
                st.download_button("Download HTML — OPR vs forced-loss", data=htmlp1, file_name="press_opr_vs_forcedloss.html", mime="text/html")

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
                htmlp2 = chart.properties(title="Style vs execution under pressure").to_html()
                st.download_button("Download HTML — Style vs execution", data=htmlp2, file_name="press_style_vs_execution.html", mime="text/html")

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
                htmlp3 = chart.properties(title="Value vs technical difficulty (under pressure)").to_html()
                st.download_button("Download HTML — OPR(danger) vs OPR(difficult)", data=htmlp3, file_name="press_opr_danger_vs_difficult.html", mime="text/html")

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
                htmlp4 = chart.properties(title="Top OPR").to_html()
                st.download_button("Download HTML — Top OPR (bar)", data=htmlp4, file_name="press_top_opr_bar.html", mime="text/html")

            # --- Exports: XLSX & HTML charts (Under Pressure)
            st.markdown("### Exports")
            _bufp = io.BytesIO()
            try:
                with pd.ExcelWriter(_bufp, engine="xlsxwriter") as xw:
                    leader_press.to_excel(xw, sheet_name="Under pressure — Top", index=False)
                    dfp_view.to_excel(xw, sheet_name="Mapped data", index=False)
                st.download_button("Download XLSX — Under Pressure module", data=_bufp.getvalue(), file_name="under_pressure_leaderboards.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as e:
                st.warning(f"XLSX export unavailable: {e}")

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
        - Merlin et al. (2022), *Classification and determinants of passing difficulty in soccer* — determinants of pass difficulty (receiver context, ball trajectory, field zones, passer context).
        - Anzer & Bauer (2022), *Expected passes* — probability models for pass success using positional data; motivates separating completion from *value*.
        - Fernández, Bornn & Cervone (2019/2021), *Expected Possession Value (EPV)* — valuing actions by impact on possession value; motivates the focus on *threat* of completed runs.

        **Extending with coordinates**
        - If you upload event/tracking data with start/end coordinates, we can add distance-binned xPass curves, value-vs-distance plots, and leaderboards conditioned on long vs short passes.
        """
    )
    st.info("Have ideas or different definitions? Adjust weights and filters, or export CSVs to continue analysis in Python/R.")
""")

radar_code = dedent(r"""
# Player Radar Tab — drop-in module for your current Streamlit app
# Usage in your app:
#   from player_radar_tab import render_player_radar_tab
#   ... after loading runs_df_raw and press_df_raw ...
#   render_player_radar_tab(runs_df_raw, press_df_raw)

from __future__ import annotations
import io
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

# We import plotting libs inside the render to avoid top-level import edits
def _ensure_plot_libs():
    import importlib
    globals()['plt'] = importlib.import_module('matplotlib.pyplot')
    Radar = getattr(importlib.import_module('mplsoccer'), 'Radar')
    globals()['Radar'] = Radar


def _normalize_0_100(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        lo, hi = np.nanmin(s), np.nanmax(s)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def _aggregate_per_player(df: pd.DataFrame, agg_map: Dict[str, str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if 'Player' not in df.columns:
        return pd.DataFrame()
    g = df.groupby('Player', dropna=False)
    out = g.agg(agg_map)
    out.reset_index(inplace=True)
    return out


def render_player_radar_tab(runs_df_raw: Optional[pd.DataFrame], press_df_raw: Optional[pd.DataFrame]):
    st.subheader("Player Radar (like the reference app)")
    st.caption("Pick a player (and optional comparison), choose metrics, export PNG/PDF. Metrics are normalized to 0–100 using robust percentiles.")

    if (runs_df_raw is None or runs_df_raw.empty) and (press_df_raw is None or press_df_raw.empty):
        st.info("Upload at least one CSV (Runs or Under Pressure) to use this tab.")
        return

    # Build a merged per-player table from both modules
    pieces = []
    if runs_df_raw is not None and not runs_df_raw.empty:
        rcols = {
            'Player': 'first',
            'Count pass attempts for Runs per Match': 'sum',
            'Pass completion ratio to Runs': 'mean',
            'Threat of Runs to which a pass was completed per Match': 'mean',
            'Count completed passes for Runs per Match': 'sum',
            'Count completed passes leading to shot for Runs per Match': 'sum',
            'Count completed passes leading to goal for Runs per Match': 'sum',
            'Count opportunities to pass to dangerous Runs per Match': 'sum',
            'Count pass attempts for dangerous Runs per Match': 'sum',
            'Count completed passes for dangerous Runs per Match': 'sum',
        }
        rcols = {k: v for k, v in rcols.items() if k in runs_df_raw.columns}
        if rcols:
            runs_pp = _aggregate_per_player(runs_df_raw, rcols)
            runs_pp.rename(columns={
                'Count pass attempts for Runs per Match': 'att_runs_pm',
                'Pass completion ratio to Runs': 'comp_runs',
                'Threat of Runs to which a pass was completed per Match': 'threat_runs',
                'Count completed passes for Runs per Match': 'comp_runs_pm',
                'Count completed passes leading to shot for Runs per Match': 'comp_runs_shot_pm',
                'Count completed passes leading to goal for Runs per Match': 'comp_runs_goal_pm',
                'Count opportunities to pass to dangerous Runs per Match': 'opp_danger_runs_pm',
                'Count pass attempts for dangerous Runs per Match': 'att_danger_runs_pm',
                'Count completed passes for dangerous Runs per Match': 'comp_danger_runs_pm',
            }, inplace=True)
            pieces.append(runs_pp)

    if press_df_raw is not None and not press_df_raw.empty:
        pcols = {
            'Player': 'first',
            'Count Pressures received per Match': 'sum',
            'Count pass attempts under Pressure per Match': 'sum',
            'Pass completion ratio under Pressure': 'mean',
            'Dangerous pass completion ratio under Pressure': 'mean',
            'Difficult pass completion ratio under Pressure': 'mean',
            'Count ball retentions under Pressure per Match': 'sum',
            'Count forced losses under Pressure per Match': 'sum',
        }
        pcols = {k: v for k, v in pcols.items() if k in press_df_raw.columns}
        if pcols:
            press_pp = _aggregate_per_player(press_df_raw, pcols)
            press_pp.rename(columns={
                'Count Pressures received per Match': 'pressures_pm',
                'Count pass attempts under Pressure per Match': 'pass_att_press_pm',
                'Pass completion ratio under Pressure': 'comp_ratio_press',
                'Dangerous pass completion ratio under Pressure': 'comp_ratio_press_danger',
                'Difficult pass completion ratio under Pressure': 'comp_ratio_press_difficult',
                'Count ball retentions under Pressure per Match': 'retentions_press_pm',
                'Count forced losses under Pressure per Match': 'forced_losses_press_pm',
            }, inplace=True)
            pieces.append(press_pp)

    if not pieces:
        st.warning("No known columns found to build the radar. Check your CSVs.")
        return

    df = pieces[0]
    for p in pieces[1:]:
        df = pd.merge(df, p, on='Player', how='outer')

    # Derived under-pressure KPIs (same as earlier logic, but per-player)
    eps = 1e-9
    if 'pressures_pm' in df.columns:
        df['OPR'] = (df.get('retentions_press_pm', 0) + df.get('comp_ratio_press', 0) * df.get('pass_att_press_pm', 0)) / (df['pressures_pm'] + eps)
        df['OPR_danger'] = (df.get('retentions_press_pm', 0) + df.get('comp_ratio_press_danger', 0) * df.get('pass_att_press_pm', 0)) / (df['pressures_pm'] + eps)
        df['OPR_difficult'] = (df.get('retentions_press_pm', 0) + df.get('comp_ratio_press_difficult', 0) * df.get('pass_att_press_pm', 0)) / (df['pressures_pm'] + eps)
        df['forced_loss_rate_under_pressure'] = df.get('forced_losses_press_pm', 0) / (df['pressures_pm'] + eps)
        df['risk_index'] = 1.0 - df['forced_loss_rate_under_pressure']

    # Runs attempt rate
    if 'att_runs_pm' in df.columns and 'comp_runs_pm' in df.columns:
        df['completion_runs_rate'] = df.get('comp_runs', np.nan)

    # Candidate metrics for radar (normalized 0–100)
    candidates = []
    for col in [
        'comp_runs_pm','comp_runs_shot_pm','comp_runs_goal_pm','comp_danger_runs_pm','threat_runs',
        'completion_runs_rate','OPR','OPR_danger','OPR_difficult','risk_index','comp_ratio_press',
        'comp_ratio_press_danger','comp_ratio_press_difficult'
    ]:
        if col in df.columns:
            df[col + '_n'] = _normalize_0_100(df[col])
            candidates.append(col + '_n')

    if not candidates:
        st.warning("Could not form any normalized metrics for the radar.")
        return

    st.markdown("### Choose players and metrics")
    players = sorted(df['Player'].dropna().unique().tolist())
    if not players:
        st.warning("No players found.")
        return
    c1, c2 = st.columns([1,1])
    with c1:
        p1 = st.selectbox('Player A', players)
    with c2:
        p2 = st.selectbox('Player B (optional)', ['—'] + players)
        p2 = None if p2 == '—' else p2

    metrics_sel = st.multiselect(
        'Radar metrics (pick up to 12)',
        options=candidates,
        default=[m for m in candidates if any(k in m for k in ['OPR','risk','threat','comp_runs'])][:8]
    )
    if len(metrics_sel) < 3:
        st.info("Select at least 3 metrics.")
        return
    metrics_sel = metrics_sel[:12]

    # Plot radar
    _ensure_plot_libs()
    Radar = globals()['Radar']
    plt = globals()['plt']

    lowers, uppers = [], []
    for m in metrics_sel:
        s = pd.to_numeric(df[m], errors='coerce')
        lo = np.nanpercentile(s, 5); hi = np.nanpercentile(s, 95)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            lo, hi = np.nanmin(s), np.nanmax(s)
            if hi == lo:
                hi = lo + 1e-6
        lowers.append(float(lo)); uppers.append(float(hi))

    radar = Radar(metrics_sel, lowers, uppers, num_rings=4)
    row_a = df[df['Player'] == p1].iloc[0]
    v_a = [float(row_a[m]) if pd.notna(row_a[m]) else np.nan for m in metrics_sel]
    v_b = None
    if p2 is not None:
        row_b = df[df['Player'] == p2].iloc[0]
        v_b = [float(row_b[m]) if pd.notna(row_b[m]) else np.nan for m in metrics_sel]

    fig, ax = plt.subplots(figsize=(8, 8))
    radar.setup_axis(ax=ax)
    radar.draw_circles(ax=ax, facecolor="#f3f3f3", edgecolor="#c9c9c9", alpha=0.18)
    try:
        radar.spoke(ax=ax, color="#c9c9c9", linestyle="--", alpha=0.18)
    except Exception:
        pass
    radar.draw_radar(v_a, ax=ax, kwargs_radar={"facecolor": "#2A9D8F33", "edgecolor": "#2A9D8F", "linewidth": 2})
    if v_b is not None:
        radar.draw_radar(v_b, ax=ax, kwargs_radar={"facecolor": "#E76F5133", "edgecolor": "#E76F51", "linewidth": 2})
    radar.draw_range_labels(ax=ax, fontsize=9)
    radar.draw_param_labels(ax=ax, fontsize=10)
    title = p1 if p2 is None else f"{p1} vs {p2}"
    ax.set_title(title, fontsize=16, pad=18)
    st.pyplot(fig, use_container_width=True)

    # Exports PNG/PDF
    cexp1, cexp2 = st.columns(2)
    with cexp1:
        buf_png = io.BytesIO()
        fig.savefig(buf_png, format='png', dpi=220, bbox_inches='tight')
        buf_png.seek(0)
        st.download_button('Download Radar (PNG)', data=buf_png.getvalue(), file_name='player_radar.png', mime='image/png')
    with cexp2:
        buf_pdf = io.BytesIO()
        fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
        buf_pdf.seek(0)
        st.download_button('Download Radar (PDF)', data=buf_pdf.getvalue(), file_name='player_radar.pdf', mime='application/pdf')

    st.markdown("#### Tips")
    st.write("Use this radar alongside the interactive Altair scatter plots (exportable to HTML with hover) from the Runs and Under Pressure tabs.")
""")

reqs = dedent(r"""
streamlit>=1.36
pandas>=2.0
numpy>=1.24
altair>=5.0
xlsxwriter>=3.1
matplotlib>=3.8
mplsoccer>=1.1
""").strip() + "\n"

# Write files
app_path = "/mnt/data/streamlit_app_skillcorner_analyzer.py"
radar_path = "/mnt/data/player_radar_tab.py"
req_path = "/mnt/data/requirements.txt"

with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_code)

with open(radar_path, "w", encoding="utf-8") as f:
    f.write(radar_code)

with open(req_path, "w", encoding="utf-8") as f:
    f.write(reqs)

(app_path, radar_path, req_path)
