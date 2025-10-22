# Streamlit app — SkillCorner Analyzer (Runs + Under Pressure + Player Radar + About)
# Author: GPT-5 Thinking
# How to run:
#   1) pip install -r requirements.txt
#   2) streamlit run app_skillcorner_integrated.py

from __future__ import annotations
import io
import re
import unicodedata
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Lazy imports for radar
import importlib

st.set_page_config(page_title="SkillCorner Analyzer", page_icon="⚽", layout="wide")
st.title("⚽ SkillCorner Analyzer — Integrated")
st.caption("Upload CSVs → map columns → compute KPIs → visualize (interactive with hover) → export HTML & XLSX. Includes a Player Radar tab.")

# ----------------------------------
# Utilities
# ----------------------------------

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
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
        return pd.read_csv(io.StringIO(text), sep=sep)
    else:
        file.seek(0)
        return pd.read_csv(file, sep=sep)

# ----------------------------------
# Canonical columns (TEAM/Team/team supported)
# ----------------------------------
RUNS_EXPECTED: Dict[str, list[str]] = {
    "player": ["player", "name", "Player"],
    "team": ["team", "Team", "TEAM", "club", "squad"],
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
}

PRESS_EXPECTED: Dict[str, list[str]] = {
    "player": ["player", "name", "Player"],
    "team": ["team", "Team", "TEAM", "club", "squad"],
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

TEAM_COLOR_SCHEME = alt.Scale(scheme="tableau20")

# ----------------------------------
# Sidebar — input (unique keys to avoid duplicates)
# ----------------------------------

st.sidebar.header("1) Upload CSVs")
runs_file = st.sidebar.file_uploader("Upload RUNS CSV (player-level)", type=["csv"], key="runs_uploader")
press_file = st.sidebar.file_uploader("Upload UNDER PRESSURE CSV (player-level)", type=["csv"], key="press_uploader")

st.sidebar.subheader("Delimiter & decimal")
use_semicolon = st.sidebar.checkbox("CSV uses semicolon (;) as delimiter", value=True, key="delim_semicolon")
use_comma_decimal = st.sidebar.checkbox("Numbers use comma as decimal (e.g., 12,3)", value=False, key="decimal_comma")

runs_df_raw = load_csv(runs_file, use_semicolon, use_comma_decimal)
press_df_raw = load_csv(press_file, use_semicolon, use_comma_decimal)

TAB_RUNS, TAB_PRESS, TAB_RADAR, TAB_NOTES = st.tabs(["Runs", "Under Pressure", "Player Radar", "About & Methods"])

# ----------------------------------
# Helper for highlightable scatter — with PLAYER legend when highlighted
# ----------------------------------
def highlight_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    tooltip: list[str],
    title: str,
    width: int,
    height: int,
    point_size: int,
    color_choice: str,
    highlights: list[str],
):
    # Greyed background (all points)
    background = (
        alt.Chart(df)
        .mark_circle(size=point_size)
        .encode(
            x=alt.X(f"{x}:Q"),
            y=alt.Y(f"{y}:Q"),
            color=alt.value("#bdbdbd"),
            opacity=alt.value(0.25),
            tooltip=tooltip,
        )
    )

    # Foreground: only highlighted players
    if highlights:
        # Color & legend by PLAYER (only the selected names)
        fg = (
            alt.Chart(df)
            .transform_filter(alt.FieldOneOfPredicate(field="player", oneOf=highlights))
            .mark_circle(size=int(point_size * 1.35), stroke="black", strokeWidth=1)
            .encode(
                x=alt.X(f"{x}:Q"),
                y=alt.Y(f"{y}:Q"),
                color=alt.Color(
                    "player:N",
                    title="Highlighted player(s)",
                    scale=alt.Scale(scheme="tableau10", domain=highlights),
                    legend=alt.Legend(title="Highlighted player(s)"),
                ),
                tooltip=tooltip,
            )
        )
        chart = (background + fg).properties(width=width, height=height, title=title)
    else:
        # No highlights → use normal coloring choice
        base = alt.Chart(df).mark_circle(size=point_size).encode(
            x=alt.X(f"{x}:Q"),
            y=alt.Y(f"{y}:Q"),
            tooltip=tooltip,
        )
        if color_choice != "None":
            if color_choice == "team":
                base = base.encode(color=alt.Color("team:N", scale=TEAM_COLOR_SCHEME, legend=alt.Legend(title="Team")))
            else:
                base = base.encode(color=alt.Color(f"{color_choice}:N"))
        chart = base.properties(width=width, height=height, title=title)

    return chart

# ==================================
# RUNS TAB
# ==================================
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

            for opt in ["team", "third", "channel", "minutes_pm"]:
                if opt not in df.columns:
                    df[opt] = np.nan

            # ---- KPIs
            nz = lambda s: s.replace({0: np.nan})
            df["attempt_rate_runs"] = df["att_runs_pm"] / nz(df["opp_runs_pm"])  # A1
            df["attempt_rate_danger"] = df["att_danger_runs_pm"] / nz(df["opp_danger_runs_pm"])  # A2
            df["productivity_runs"] = df["completed_runs_pm"]  # V1
            df["conversion_to_shot"] = df["completed_to_shot_pm"] / nz(df["completed_runs_pm"])  # C1
            df["conversion_to_goal"] = df["completed_to_goal_pm"] / nz(df["completed_runs_pm"])  # C2
            df["exp_threat_per_attempt"] = df["threat_completed_pm"] / nz(df["att_runs_pm"])  # V2
            df["dangerous_completion"] = df["completed_danger_runs_pm"] / nz(df["att_danger_runs_pm"])  # C3
            df["dangerous_attempt_share"] = df["att_danger_runs_pm"] / nz(df["att_runs_pm"])  # S1

            # Percentiles
            pct = lambda s: s.rank(pct=True, na_option="keep")
            rank_cols = [
                "attempt_rate_runs","comp_ratio_runs","threat_completed_pm","productivity_runs",
                "conversion_to_shot","conversion_to_goal","exp_threat_per_attempt","dangerous_completion","dangerous_attempt_share"
            ]
            for col in rank_cols:
                df[f"p_{col}"] = pct(df[col])

            # Composite indices
            df["creator_index"] = (
                0.30*df["p_threat_completed_pm"] + 0.20*df["p_productivity_runs"] + 0.20*df["p_comp_ratio_runs"]
                + 0.20*df["p_attempt_rate_runs"] + 0.10*df["p_conversion_to_shot"]
            )
            df["creator_index_v2"] = (
                0.35*df["p_exp_threat_per_attempt"] + 0.25*df["p_threat_completed_pm"] + 0.15*df["p_comp_ratio_runs"]
                + 0.15*df["p_attempt_rate_runs"] + 0.10*df["p_dangerous_completion"]
            )

            # ---- Filters
            st.markdown("### Filters")
            min_minutes = st.number_input("Min minutes per match", 0.0, value=0.0, step=1.0, key="runs_min_minutes")
            teams = sorted([x for x in df["team"].dropna().unique()])
            sel_teams = st.multiselect("Team(s)", options=teams, default=teams, key="runs_team_multiselect")
            sel_third = st.multiselect("Third(s)", options=sorted([x for x in df["third"].dropna().unique()]), key="runs_third_multiselect")
            sel_channel = st.multiselect("Channel(s)", options=sorted([x for x in df["channel"].dropna().unique()]), key="runs_channel_multiselect")

            mask = pd.Series(True, index=df.index)
            if not pd.isna(df["minutes_pm"]).all():
                mask &= (df["minutes_pm"].fillna(0) >= min_minutes)
            if sel_teams:
                mask &= df["team"].isin(sel_teams)
            if sel_third:
                mask &= df["third"].isin(sel_third)
            if sel_channel:
                mask &= df["channel"].isin(sel_channel)
            df_view = df.loc[mask].copy()

            # ---- Leaderboards
            st.markdown("### Leaderboards")
            n_top = st.slider("Top N", 5, 100, 20, step=5, key="runs_topn")

            cols_creator = [
                "player","team","third","channel","opp_runs_pm","att_runs_pm","attempt_rate_runs","comp_ratio_runs",
                "threat_completed_pm","exp_threat_per_attempt","completed_runs_pm","completed_to_shot_pm","completed_to_goal_pm",
                "dangerous_attempt_share","dangerous_completion","creator_index","creator_index_v2"
            ]
            leader_creator = df_view.sort_values("creator_index_v2", ascending=False)[cols_creator].head(n_top)
            st.dataframe(leader_creator.round(3), use_container_width=True)
            st.download_button("Download CSV — Runs creators (v2)", leader_creator.to_csv(index=False).encode("utf-8"), file_name="runs_creators_v2.csv")

            # ---- Visuals with highlight (legend = players when highlighted)
            st.markdown("### Visualizations")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                chart_w = st.number_input("Chart width (px)", 800, 4000, 1600, step=100, key="runs_w")
            with c2:
                chart_h = st.number_input("Chart height (px)", 400, 2400, 900, step=50, key="runs_h")
            with c3:
                point_size = st.number_input("Point size", 20, 300, 90, step=5, key="runs_pts")

            color_choice = st.selectbox("Color points by (when not highlighting)", ["team","third","channel","None"], index=0, key="runs_color")
            highlight_players = st.multiselect("Highlight player(s)", options=sorted(df_view["player"].dropna().unique()), key="runs_highlight_players")

            chart1 = highlight_scatter(
                df_view.dropna(subset=["attempt_rate_runs","comp_ratio_runs"]),
                x="attempt_rate_runs",
                y="comp_ratio_runs",
                tooltip=["player","team","attempt_rate_runs","comp_ratio_runs","threat_completed_pm","completed_runs_pm"],
                title="Attempt vs Completion",
                width=chart_w, height=chart_h, point_size=point_size,
                color_choice=color_choice, highlights=highlight_players
            )
            st.altair_chart(chart1, use_container_width=True)
            st.download_button("Download HTML — Attempt vs Completion", chart1.to_html(), file_name="runs_attempt_vs_completion.html")

            chart2 = highlight_scatter(
                df_view.dropna(subset=["exp_threat_per_attempt","attempt_rate_runs"]),
                x="attempt_rate_runs",
                y="exp_threat_per_attempt",
                tooltip=["player","team","exp_threat_per_attempt","attempt_rate_runs","threat_completed_pm"],
                title="Risk–Reward: Attempts vs Value per Attempt",
                width=chart_w, height=chart_h, point_size=point_size,
                color_choice="None", highlights=highlight_players,
            ).encode(color=alt.Color("dangerous_attempt_share:Q", title="Dangerous share", scale=alt.Scale(scheme='blues')))
            st.altair_chart(chart2, use_container_width=True)
            st.download_button("Download HTML — Attempts vs Value per Attempt", chart2.to_html(), file_name="runs_attempts_vs_value_per_attempt.html")

            chart3 = highlight_scatter(
                df_view.dropna(subset=["dangerous_attempt_share","dangerous_completion"]),
                x="dangerous_attempt_share",
                y="dangerous_completion",
                tooltip=["player","team","dangerous_attempt_share","dangerous_completion"],
                title="Dangerous tendency vs execution",
                width=chart_w, height=chart_h, point_size=point_size,
                color_choice=color_choice, highlights=highlight_players
            )
            st.altair_chart(chart3, use_container_width=True)
            st.download_button("Download HTML — Dangerous tendency vs execution", chart3.to_html(), file_name="runs_dangerous_tendency_vs_execution.html")

            chart4 = alt.Chart(df_view.dropna(subset=["comp_ratio_runs"])) \
                .transform_density('comp_ratio_runs', as_=['comp_ratio_runs','density']) \
                .mark_area(opacity=0.5) \
                .encode(x=alt.X('comp_ratio_runs:Q', title='Completion to runs'), y='density:Q') \
                .properties(width=chart_w, height=int(chart_h*0.6), title="Distribution — Pass completion to runs")
            st.altair_chart(chart4, use_container_width=True)
            st.download_button("Download HTML — Completion distribution", chart4.to_html(), file_name="runs_completion_distribution.html")

            # ---- Exports
            st.markdown("### Exports")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                leader_creator.to_excel(xw, sheet_name="Runs — creators v2", index=False)
                df_view.to_excel(xw, sheet_name="Runs — mapped", index=False)
            st.download_button("Download XLSX — Runs module", buf.getvalue(), file_name="runs_module.xlsx")

# ==================================
# UNDER PRESSURE TAB
# ==================================
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
        cols2 = st.columns(3)
        items2 = list(PRESS_EXPECTED.keys())
        for i, canon in enumerate(items2):
            with cols2[i % 3]:
                options = [None] + list(press_df_raw.columns)
                mapping_p[canon] = st.selectbox(
                    f"{canon}", options=options, index=(options.index(auto_map_p[canon]) if auto_map_p[canon] in options else 0), key=f"map_press_{canon}"
                )

        required_p = ["player","pressures_pm","pass_att_press_pm","comp_ratio_press","retentions_press_pm","forced_losses_press_pm"]
        missing_p = [c for c in required_p if mapping_p.get(c) is None]
        if missing_p:
            st.error("Missing required columns: " + ", ".join(missing_p))
        else:
            cols_to_take_p = [c for c in mapping_p.values() if c is not None]
            dfp = press_df_raw[cols_to_take_p].copy()
            dfp.columns = [k for k, v in mapping_p.items() if v is not None]

            for opt in ["team","third","channel","minutes_pm","comp_ratio_press_danger","comp_ratio_press_difficult"]:
                if opt not in dfp.columns:
                    dfp[opt] = np.nan

            eps = 1e-9
            dfp['press_load_pm'] = dfp['pressures_pm']
            dfp['pass_share_under_pressure'] = dfp['pass_att_press_pm'] / (dfp['press_load_pm'] + eps)
            dfp['retention_rate_under_pressure'] = dfp['retentions_press_pm'] / (dfp['press_load_pm'] + eps)
            dfp['forced_loss_rate_under_pressure'] = dfp['forced_losses_press_pm'] / (dfp['press_load_pm'] + eps)

            # Successful pass volumes under pressure
            dfp['succ_pass_press_pm'] = dfp['comp_ratio_press'] * dfp['pass_att_press_pm']
            dfp['succ_pass_press_danger_pm'] = dfp['comp_ratio_press_danger'] * dfp['pass_att_press_pm']
            dfp['succ_pass_press_difficult_pm'] = dfp['comp_ratio_press_difficult'] * dfp['pass_att_press_pm']

            # Core indices
            dfp['OPR'] = (dfp['retentions_press_pm'] + dfp['succ_pass_press_pm']) / (dfp['press_load_pm'] + eps)
            dfp['OPR_danger'] = (dfp['retentions_press_pm'] + dfp['succ_pass_press_danger_pm']) / (dfp['press_load_pm'] + eps)
            dfp['OPR_difficult'] = (dfp['retentions_press_pm'] + dfp['succ_pass_press_difficult_pm']) / (dfp['press_load_pm'] + eps)
            dfp['risk_index'] = 1.0 - dfp['forced_loss_rate_under_pressure']
            dfp['throughput_succ_pass_pm'] = dfp['succ_pass_press_pm']
            dfp['safe_action_rate'] = dfp['retention_rate_under_pressure'] * 0.6 + (1 - dfp['forced_loss_rate_under_pressure']) * 0.4

            # Percentiles for benchmarking
            pct = lambda s: s.rank(pct=True, na_option="keep")
            bench_cols = ['OPR','OPR_danger','OPR_difficult','risk_index','comp_ratio_press','comp_ratio_press_danger','comp_ratio_press_difficult','pass_share_under_pressure','safe_action_rate']
            for col in bench_cols:
                dfp[f'p_{col}'] = pct(dfp[col])

            # ---- Filters
            st.markdown("### Filters")
            min_minutes_p = st.number_input("Min minutes per match", 0.0, value=0.0, step=1.0, key="press_minmins")
            teams_p = sorted([x for x in dfp["team"].dropna().unique()])
            sel_teams_p = st.multiselect("Team(s)", options=teams_p, default=teams_p, key="press_team_multiselect")
            sel_third_p = st.multiselect("Third(s)", options=sorted([x for x in dfp["third"].dropna().unique()]), key="press_third_multiselect")
            sel_channel_p = st.multiselect("Channel(s)", options=sorted([x for x in dfp["channel"].dropna().unique()]), key="press_channel_multiselect")

            maskp = pd.Series(True, index=dfp.index)
            if not pd.isna(dfp["minutes_pm"]).all():
                maskp &= (dfp["minutes_pm"].fillna(0) >= min_minutes_p)
            if sel_teams_p:
                maskp &= dfp["team"].isin(sel_teams_p)
            if sel_third_p:
                maskp &= dfp["third"].isin(sel_third_p)
            if sel_channel_p:
                maskp &= dfp["channel"].isin(sel_channel_p)
            dfp_view = dfp.loc[maskp].copy()

            # ---- Leaderboard
            st.markdown("### Leaderboards")
            n_top_p = st.slider("Top N", 5, 100, 20, step=5, key="press_topn")
            cols_press = [
                'player','team','third','channel','minutes_pm','pressures_pm','pass_att_press_pm','pass_share_under_pressure',
                'retentions_press_pm','forced_losses_press_pm','retention_rate_under_pressure','forced_loss_rate_under_pressure',
                'comp_ratio_press','comp_ratio_press_danger','comp_ratio_press_difficult','OPR','OPR_danger','OPR_difficult','risk_index','throughput_succ_pass_pm','safe_action_rate'
            ]
            leader_press = dfp_view.sort_values(['OPR','risk_index','safe_action_rate'], ascending=[False, False, False])[cols_press].head(n_top_p)
            st.dataframe(leader_press.round(3), use_container_width=True)
            st.download_button("Download CSV — Under-pressure metrics", leader_press.to_csv(index=False).encode("utf-8"), file_name="under_pressure_metrics.csv")

            # ---- Visualizations with highlight (legend = players when highlighted)
            st.markdown("### Visualizations")
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                chart_w = st.number_input("Chart width (px)", 800, 4000, 1600, step=100, key="press_w")
            with c2:
                chart_h = st.number_input("Chart height (px)", 400, 2400, 900, step=50, key="press_h")
            with c3:
                point_size = st.number_input("Point size", 20, 300, 90, step=5, key="press_pts")

            color_choice_p = st.selectbox("Color points by (when not highlighting)", ["team","third","channel","None"], index=0, key="press_color_choice")
            highlight_players_p = st.multiselect("Highlight player(s)", options=sorted(dfp_view["player"].dropna().unique()), key="press_highlight_players")

            def _save_html(chart, name):
                return st.download_button(name, chart.to_html(), file_name=name.replace(" ", "_").lower())

            choice = st.selectbox("Choose a visualization", [
                "OPR vs Forced-loss rate",
                "Pass-share under pressure vs Completion",
                "OPR (danger) vs OPR (difficult)",
                "Bar: Top OPR",
                "Histogram: Forced-loss rate",
                "Heatmap: Decision vs Completion",
            ], key="press_vis")

            if choice == "OPR vs Forced-loss rate":
                c = highlight_scatter(
                    dfp_view.dropna(subset=['OPR','forced_loss_rate_under_pressure']),
                    x="forced_loss_rate_under_pressure",
                    y="OPR",
                    tooltip=['player','team','OPR','forced_loss_rate_under_pressure','pressures_pm','pass_att_press_pm'],
                    title="Pressure resilience",
                    width=chart_w, height=chart_h, point_size=point_size,
                    color_choice=color_choice_p, highlights=highlight_players_p
                )
                st.altair_chart(c, use_container_width=True)
                _save_html(c, "press_opr_vs_forcedloss.html")

            elif choice == "Pass-share under pressure vs Completion":
                c = highlight_scatter(
                    dfp_view.dropna(subset=['pass_share_under_pressure','comp_ratio_press']),
                    x="pass_share_under_pressure",
                    y="comp_ratio_press",
                    tooltip=['player','team','pass_share_under_pressure','comp_ratio_press','OPR'],
                    title="Style vs execution under pressure",
                    width=chart_w, height=chart_h, point_size=point_size,
                    color_choice=color_choice_p, highlights=highlight_players_p
                )
                st.altair_chart(c, use_container_width=True)
                _save_html(c, "press_style_vs_execution.html")

            elif choice == "OPR (danger) vs OPR (difficult)":
                c = highlight_scatter(
                    dfp_view.dropna(subset=['OPR_danger','OPR_difficult']),
                    x="OPR_danger",
                    y="OPR_difficult",
                    tooltip=['player','team','OPR_danger','OPR_difficult','pressures_pm'],
                    title="Value vs technical difficulty (under pressure)",
                    width=chart_w, height=chart_h, point_size=point_size,
                    color_choice=color_choice_p, highlights=highlight_players_p
                )
                st.altair_chart(c, use_container_width=True)
                _save_html(c, "press_opr_danger_vs_difficult.html")

            elif choice == "Bar: Top OPR":
                df_bar = dfp_view.sort_values('OPR', ascending=False).head(15)
                c = alt.Chart(df_bar).mark_bar().encode(
                    x=alt.X('OPR:Q', title='Overcome Pressure Rate'),
                    y=alt.Y('player:N', sort='-x', title='Player'),
                    color=alt.Color('team:N', scale=TEAM_COLOR_SCHEME, legend=alt.Legend(title='Team')),
                    tooltip=['player','team','OPR','pressures_pm','pass_att_press_pm']
                ).properties(width=chart_w, height=int(chart_h*0.8), title="Top OPR")
                st.altair_chart(c, use_container_width=True)
                _save_html(c, "press_top_opr_bar.html")

            elif choice == "Histogram: Forced-loss rate":
                c = alt.Chart(dfp_view.dropna(subset=['forced_loss_rate_under_pressure']))\
                    .mark_bar(opacity=0.85).encode(
                        x=alt.X('forced_loss_rate_under_pressure:Q', bin=alt.Bin(maxbins=25), title='Forced-loss rate'),
                        y=alt.Y('count()', title='Count'),
                        color=alt.Color('team:N', scale=TEAM_COLOR_SCHEME, legend=alt.Legend(title='Team'))
                    ).properties(width=chart_w, height=int(chart_h*0.8), title="Distribution — Forced-loss rate")
                st.altair_chart(c, use_container_width=True)
                _save_html(c, "press_forcedloss_hist.html")

            elif choice == "Heatmap: Decision vs Completion":
                c = alt.Chart(dfp_view.dropna(subset=['pass_share_under_pressure','comp_ratio_press']))\
                    .mark_rect().encode(
                        x=alt.X('pass_share_under_pressure:Q', bin=alt.Bin(maxbins=14), title='Pass share'),
                        y=alt.Y('comp_ratio_press:Q', bin=alt.Bin(maxbins=14), title='Completion'),
                        color=alt.Color('count():Q', title='Count')
                    ).properties(width=chart_w, height=chart_h, title="Decision vs completion (binned)")
                st.altair_chart(c, use_container_width=True)
                _save_html(c, "press_heatmap_decision_completion.html")

            # Exports
            st.markdown("### Exports")
            bufp = io.BytesIO()
            with pd.ExcelWriter(bufp, engine="xlsxwriter") as xw:
                leader_press.to_excel(xw, sheet_name="Under Pressure — top", index=False)
                dfp_view.to_excel(xw, sheet_name="Under Pressure — mapped", index=False)
            st.download_button("Download XLSX — Under Pressure module", bufp.getvalue(), file_name="under_pressure_module.xlsx")

# ==================================
# PLAYER RADAR TAB (integrated) — unchanged here
# ==================================
with TAB_RADAR:
    st.subheader("Player Radar")
    st.caption("Compare two players. Metrics normalized 0–100 (robust percentiles). Export PNG/PDF.")

    if runs_df_raw.empty and press_df_raw.empty:
        st.info("Upload at least one CSV (Runs or Under Pressure) to use this tab.")
    else:
        pieces = []
        # ---- RUNS aggregate (numeric-only aggregation to avoid TypeError)
        if not runs_df_raw.empty:
            take = [c for c in [
                'Player','Team','team','player',
                'Count pass attempts for Runs per Match','Pass completion ratio to Runs','Threat of Runs to which a pass was completed per Match',
                'Count completed passes for Runs per Match','Count completed passes leading to shot for Runs per Match','Count completed passes leading to goal for Runs per Match',
                'Count opportunities to pass to dangerous Runs per Match','Count pass attempts for dangerous Runs per Match','Count completed passes for dangerous Runs per Match'] if c in runs_df_raw.columns]
            if any(c in take for c in ['Player','player']):
                tcol = 'Team' if 'Team' in runs_df_raw.columns else ('team' if 'team' in runs_df_raw.columns else None)
                pcol = 'Player' if 'Player' in runs_df_raw.columns else 'player'
                g = runs_df_raw[take].copy()
                # standardize team label
                if tcol and tcol not in g.columns and 'team' in g.columns:
                    tcol = 'team'
                g['team'] = g[tcol] if (tcol in g.columns) else np.nan
                # numeric-only columns for aggregation
                num_cols = g.select_dtypes(include='number').columns.tolist()
                if len(num_cols) > 0:
                    g_agg = g.groupby([pcol,'team'])[num_cols].agg(['sum','mean']).reset_index()
                    g_agg.columns = [" ".join(col).strip() if isinstance(col, tuple) else col for col in g_agg.columns.to_flat_index()]
                    g_agg = g_agg.rename(columns={pcol:'player'})
                    pieces.append(g_agg)
        # ---- PRESS aggregate (numeric-only)
        if not press_df_raw.empty:
            takep = [c for c in [
                'Player','Team','team','player',
                'Count Pressures received per Match','Count pass attempts under Pressure per Match','Pass completion ratio under Pressure',
                'Dangerous pass completion ratio under Pressure','Difficult pass completion ratio under Pressure','Count ball retentions under Pressure per Match','Count forced losses under Pressure per Match'] if c in press_df_raw.columns]
            if any(c in takep for c in ['Player','player']):
                tcol = 'Team' if 'Team' in press_df_raw.columns else ('team' if 'team' in press_df_raw.columns else None)
                pcol = 'Player' if 'Player' in press_df_raw.columns else 'player'
                gp = press_df_raw[takep].copy()
                gp['team'] = gp[tcol] if (tcol in gp.columns) else np.nan
                num_cols_p = gp.select_dtypes(include='number').columns.tolist()
                if len(num_cols_p) > 0:
                    gp_agg = gp.groupby([pcol,'team'])[num_cols_p].agg(['sum','mean']).reset_index()
                    gp_agg.columns = [" ".join(col).strip() if isinstance(col, tuple) else col for col in gp_agg.columns.to_flat_index()]
                    gp_agg = gp_agg.rename(columns={pcol:'player'})
                    pieces.append(gp_agg)

        if not pieces:
            st.warning("Could not form per-player aggregates from your CSVs. Ensure 'Player' and numeric metrics exist.")
        else:
            dfp = pieces[0]
            for p in pieces[1:]:
                dfp = pd.merge(dfp, p, on=['player','team'], how='outer')

            # Normalize selected columns
            def norm01(s: pd.Series) -> pd.Series:
                s = pd.to_numeric(s, errors='coerce')
                lo, hi = np.nanpercentile(s, 5), np.nanpercentile(s, 95)
                if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                    lo, hi = np.nanmin(s), np.nanmax(s)
                    if hi == lo:
                        hi = lo + 1e-6
                return (s - lo) / (hi - lo) * 100

            candidate_cols = [c for c in dfp.columns if any(k in c.lower() for k in ['opr','threat','completion','completed passes for runs per match sum','pass attempts under pressure per match sum'])]
            for c in candidate_cols:
                try:
                    dfp[c+"_n"] = norm01(dfp[c])
                except Exception:
                    pass

            teams_all = [t for t in sorted(dfp['team'].dropna().unique())]
            team_sel = st.selectbox('Team filter (optional)', ['— All —'] + teams_all, key="radar_team_filter")
            dfp_view = dfp if team_sel == '— All —' else dfp[dfp['team'] == team_sel]

            players = sorted(dfp_view['player'].dropna().unique().tolist())
            if len(players) == 0:
                st.warning("No players available after aggregation.")
            else:
                p1 = st.selectbox('Player A', players, key="radar_player_a")
                p2 = st.selectbox('Player B (optional)', ['—'] + players, key="radar_player_b")
                p2 = None if p2 == '—' else p2

                metric_options = [c for c in dfp_view.columns if c.endswith('_n')]
                default_metrics = metric_options[:8] if len(metric_options) >= 3 else metric_options
                metrics_sel = st.multiselect('Radar metrics (3–12)', options=metric_options,
                                             default=default_metrics, key="radar_metrics_sel")
                if len(metrics_sel) < 3:
                    st.info("Select at least 3 metrics.")
                else:
                    try:
                        plt = importlib.import_module('matplotlib.pyplot')
                        Radar = getattr(importlib.import_module('mplsoccer'), 'Radar')
                        lowers, uppers = [], []
                        for m in metrics_sel:
                            s = pd.to_numeric(dfp_view[m], errors='coerce')
                            lo = float(np.nanpercentile(s, 5)); hi = float(np.nanpercentile(s, 95))
                            if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
                                lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
                                if hi == lo:
                                    hi = lo + 1e-6
                            lowers.append(lo); uppers.append(hi)
                        radar = Radar(metrics_sel, lowers, uppers, num_rings=4)
                        v_a = [float(dfp_view.loc[dfp_view['player']==p1, m].iloc[0]) for m in metrics_sel]
                        v_b = [float(dfp_view.loc[dfp_view['player']==p2, m].iloc[0]) for m in metrics_sel] if p2 else None
                        fig, ax = plt.subplots(figsize=(9.5,9.5))
                        radar.setup_axis(ax=ax)
                        radar.draw_circles(ax=ax, facecolor="#f3f3f3", edgecolor="#c9c9c9", alpha=0.18)
                        try:
                            radar.spoke(ax=ax, color="#c9c9c9", linestyle="--", alpha=0.18)
                        except Exception:
                            pass
                        radar.draw_radar(v_a, ax=ax, kwargs_radar={"facecolor":"#2A9D8F33","edgecolor":"#2A9D8F","linewidth":2})
                        if v_b is not None:
                            radar.draw_radar(v_b, ax=ax, kwargs_radar={"facecolor":"#E76F5133","edgecolor":"#E76F51","linewidth":2})
                        radar.draw_range_labels(ax=ax, fontsize=10)
                        radar.draw_param_labels(ax=ax, fontsize=11)
                        subtitle = team_sel if team_sel != '— All —' else 'All teams'
                        ax.set_title((p1 if p2 is None else f"{p1} vs {p2}") + f" — {subtitle}", fontsize=16, pad=18)
                        st.pyplot(fig, use_container_width=True)

                        colp1, colp2 = st.columns(2)
                        with colp1:
                            buf_png = io.BytesIO(); fig.savefig(buf_png, format='png', dpi=300, bbox_inches='tight'); buf_png.seek(0)
                            st.download_button('Download Radar (PNG)', buf_png.getvalue(), file_name='player_radar.png', key="radar_png_dl")
                        with colp2:
                            buf_pdf = io.BytesIO(); fig.savefig(buf_pdf, format='pdf', bbox_inches='tight'); buf_pdf.seek(0)
                            st.download_button('Download Radar (PDF)', buf_pdf.getvalue(), file_name='player_radar.pdf', key="radar_pdf_dl")
                    except Exception as e:
                        st.error(f"Radar plotting requires matplotlib + mplsoccer. Error: {e}")

# ==================================
# ABOUT
# ==================================
with TAB_NOTES:
    st.subheader("About this app & methods")
    st.markdown(
        """
        **Runs module** adds: *expected threat per attempt*, *dangerous attempt share*, and an updated **Creator Index v2**.
        **Under Pressure module** adds: *throughput of successful passes* and a **safe action rate**.

        **Highlighting**: when you select **Highlight player(s)**, the legend switches to **player names** (only for the highlighted ones) and colors follow those players. Non-highlighted points are greyed.

        **Quality exports**: every Altair chart has **width/height controls** for larger, sharper standalone HTML. Team coloring uses a Tableau palette when not highlighting.

        **References (non-exhaustive)**
        - Merlin et al. (2022) — determinants of pass difficulty (receiver, trajectory, zones, passer).
        - Anzer & Bauer (2022) — Expected Passes models.
        - Fernández, Bornn & Cervone (2019/2021) — EPV concepts.

        All charts are interactive (hover) and downloadable as standalone HTML; tables and leaderboards export to XLSX.
        """
    )
