"""Streamlit Research & Simulation Platform for Crash Game Machine Learning.

Interactive research dashboard inspecting dataset properties, model benchmark comparisons,
pre-outcome predictions, calibration, walk-forward stability, and offline backtesting.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Aviator ML Research & Simulation Platform",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Aviator ML Research & Simulation Platform")
st.markdown(
    "*Offline research system empirically evaluating whether historical crash multipliers "
    "contain predictive signal under chronological, out-of-sample constraints.*"
)

# Sidebar dataset selection
st.sidebar.title("Benchmark Selection")
dataset_choice = st.sidebar.radio(
    "Evaluation Mode",
    ["Real Aviator Historical (15,962 rounds)", "Synthetic Verification (10,000 rounds)"],
    index=0,
)
is_real = dataset_choice.startswith("Real")

# Tabs
tab_overview, tab_comparison, tab_predictions, tab_calibration, tab_walkforward, tab_backtest, tab_experiments = st.tabs([
    "📊 Dataset Overview",
    "🏆 Model Comparison",
    "🔮 Pre-Outcome Predictions",
    "🎯 Probability Calibration",
    "⏳ Walk-Forward Validation",
    "📈 Strategy Backtest",
    "📝 Experiment Tracking",
])

# -----------------------------------------------------------------------------
# TAB 1: DATASET OVERVIEW
# -----------------------------------------------------------------------------
with tab_overview:
    st.header(f"Dataset Distribution & Provenance: {'Real Aviator' if is_real else 'Synthetic'}")
    meta_path = Path("data/processed/real_dataset_v001_metadata.json" if is_real else "data/processed/dataset_v001_metadata.json")
    data_path = Path("data/processed/real_dataset_v001.csv" if is_real else "data/processed/dataset_v001.csv")

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset Version", meta.get("dataset_version", "N/A"))
        col2.metric("Total Rows", f"{meta.get('row_count', 0):,}")
        col3.metric("Invalid Dropped", meta.get("dropped_invalid_multipliers", 0))
        col4.metric("Duplicates Dropped", meta.get("dropped_duplicate_records", 0))

        st.caption(f"SHA-256 Provenance Checksum: `{meta.get('sha256', 'N/A')}`")

    if data_path.exists():
        df = pd.read_csv(data_path)
        mults = df["multiplier"]

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Min Multiplier", f"{mults.min():.2f}x")
        col_b.metric("Median Multiplier", f"{mults.median():.2f}x")
        col_c.metric("Mean Multiplier", f"{mults.mean():.2f}x")
        col_d.metric("Max Multiplier", f"{mults.max():.2f}x")

        st.subheader("Multiplier Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(np.clip(mults, 1.0, 15.0), bins=40, color="#1f77b4", edgecolor="black", alpha=0.7)
        ax1.set_title("Multiplier Frequency (Capped at 15x)")
        ax1.set_xlabel("Multiplier")
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.hist(mults, bins=np.logspace(0, np.log10(max(10.0, float(mults.max()))), 40),
                 color="#ff7f0e", edgecolor="black", alpha=0.7)
        ax2.set_xscale("log")
        ax2.set_title("Multiplier Frequency (Log Scale)")
        ax2.set_xlabel("Multiplier (log scale)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------------------------------------------------------
# TAB 2: MODEL COMPARISON
# -----------------------------------------------------------------------------
with tab_comparison:
    st.header(f"Benchmark: All 6 ML Models vs. Statistical Baselines ({'Real Aviator' if is_real else 'Synthetic'})")
    comp_path = Path("results/reports/real_data_model_comparison.csv" if is_real else "results/reports/model_comparison.csv")

    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        threshold_filter = st.selectbox("Filter Target Threshold", ["All"] + sorted(list(comp_df["threshold"].unique())))

        disp_df = comp_df if threshold_filter == "All" else comp_df[comp_df["threshold"] == threshold_filter]
        st.dataframe(disp_df, use_container_width=True)

        st.subheader("ROC-AUC vs. Baseline by Threshold")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
        for model_name, grp in comp_df.groupby("model"):
            if "Baseline" not in model_name:
                ax_bar.plot(grp["threshold"], grp["roc_auc"], marker="o", label=model_name)
        ax_bar.axhline(0.5, color="red", linestyle="--", label="Random Baseline (0.50)")
        ax_bar.set_ylabel("ROC-AUC")
        ax_bar.set_xlabel("Threshold")
        ax_bar.grid(True, linestyle="--", alpha=0.5)
        ax_bar.legend(loc="lower left")
        st.pyplot(fig_bar)
        plt.close(fig_bar)
        if not is_real:
            st.subheader("Weighted Blend Ensemble Benchmark (Test Partition)")
            ens_path = Path("results/reports/ensemble_comparison.csv")
            if ens_path.exists():
                ens_df = pd.read_csv(ens_path)
                st.dataframe(ens_df, use_container_width=True)
    else:
        st.warning("Model comparison report not found. Run benchmark first.")

# -----------------------------------------------------------------------------
# TAB 3: PRE-OUTCOME PREDICTIONS
# -----------------------------------------------------------------------------
with tab_predictions:
    st.header("Pre-Outcome Round Probability Inspection")
    st.markdown("Examine predictions generated **strictly before** round *t*'s outcome is revealed.")

    preds_xgb_path = Path("results/predictions/val_predictions_xgboost.csv")
    preds_mlp_path = Path("results/predictions/val_predictions_mlp.csv")
    base_preds_path = Path("results/predictions/val_predictions_baseline.csv")

    if preds_xgb_path.exists() and base_preds_path.exists():
        df_xgb = pd.read_csv(preds_xgb_path)
        df_base = pd.read_csv(base_preds_path)

        round_idx = st.slider("Select Historical Validation Round Index", 0, len(df_xgb) - 1, 100)

        st.markdown(f"### Predicted Probabilities for Round `#{round_idx}` (Pre-Outcome):")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("P(>= 1.5x)", f"{df_xgb.loc[round_idx, 'xgb_prob_1.5x']:.1%}",
                    delta=f"{df_xgb.loc[round_idx, 'xgb_prob_1.5x'] - df_base.loc[round_idx, 'hist_prob_1.5x']:+.1%}")
        col2.metric("P(>= 2.0x)", f"{df_xgb.loc[round_idx, 'xgb_prob_2.0x']:.1%}",
                    delta=f"{df_xgb.loc[round_idx, 'xgb_prob_2.0x'] - df_base.loc[round_idx, 'hist_prob_2.0x']:+.1%}")
        col3.metric("P(>= 3.0x)", f"{df_xgb.loc[round_idx, 'xgb_prob_3.0x']:.1%}",
                    delta=f"{df_xgb.loc[round_idx, 'xgb_prob_3.0x'] - df_base.loc[round_idx, 'hist_prob_3.0x']:+.1%}")
        col4.metric("P(>= 5.0x)", f"{df_xgb.loc[round_idx, 'xgb_prob_5.0x']:.1%}",
                    delta=f"{df_xgb.loc[round_idx, 'xgb_prob_5.0x'] - df_base.loc[round_idx, 'hist_prob_5.0x']:+.1%}")
        col5.metric("P(>= 10.0x)", f"{df_xgb.loc[round_idx, 'xgb_prob_10.0x']:.1%}",
                    delta=f"{df_xgb.loc[round_idx, 'xgb_prob_10.0x'] - df_base.loc[round_idx, 'hist_prob_10.0x']:+.1%}")

        if "xgb_reg_pred" in df_xgb.columns:
            st.metric("Continuous Multiplier Predicted (XGB Regressor)", f"{df_xgb.loc[round_idx, 'xgb_reg_pred']:.2f}x")
    else:
        st.info("Run `python main.py train --model all` to generate prediction logs.")

# -----------------------------------------------------------------------------
# TAB 4: PROBABILITY CALIBRATION
# -----------------------------------------------------------------------------
with tab_calibration:
    st.header(f"Probability Calibration & Reliability Diagrams ({'Real Aviator' if is_real else 'Synthetic'})")
    st.markdown("Verifying whether predicted confidence matches empirical outcomes ($P(\\hat{y}=0.70) \\approx 70\\%$).")

    calib_csv_path = Path("results/reports/real_data_calibration_evaluation.csv" if is_real else "results/reports/calibration_evaluation.csv")
    if calib_csv_path.exists():
        st.subheader("Out-of-Sample Calibration Metrics (Brier, Log Loss, ECE)")
        cal_df = pd.read_csv(calib_csv_path)
        st.dataframe(cal_df, use_container_width=True)

    calib_plot_path = Path("results/plots/calibration_curve_xgboost.png")
    if calib_plot_path.exists():
        st.subheader("Reliability Curve")
        st.image(str(calib_plot_path), caption="Reliability Diagram: Uncalibrated vs. Calibrated (Platt/Isotonic)")
    else:
        st.info("Run `python main.py calibrate` to generate calibration curves.")

# -----------------------------------------------------------------------------
# TAB 5: WALK-FORWARD VALIDATION
# -----------------------------------------------------------------------------
with tab_walkforward:
    st.header(f"Walk-Forward Temporal Cross-Validation ({'Real Aviator' if is_real else 'Synthetic'})")
    st.markdown("Expanding-window evaluation across sequential time periods with zero lookahead.")

    wf_sum_path = Path("results/reports/real_data_walk_forward_summary.csv" if is_real else "results/reports/walk_forward_summary.csv")
    if wf_sum_path.exists():
        wf_sum = pd.read_csv(wf_sum_path)
        st.dataframe(wf_sum, use_container_width=True)
    else:
        st.info("Run walk-forward validation to generate results.")

# -----------------------------------------------------------------------------
# TAB 6: STRATEGY BACKTEST
# -----------------------------------------------------------------------------
with tab_backtest:
    st.header(f"Offline Historical Strategy Simulation ({'Real Aviator' if is_real else 'Synthetic'})")
    st.markdown(
        "**Strict Research & Simulation Mode**: Evaluates hypothetical fixed stakes against historical round results. "
        "No live betting or automated wagers."
    )

    bt_res_path = Path("results/reports/real_data_backtest_results.csv" if is_real else "results/reports/backtest_results.csv")
    if bt_res_path.exists():
        st.subheader("Performance & Risk Metrics Table")
        bt_df = pd.read_csv(bt_res_path)
        st.dataframe(bt_df, use_container_width=True)

    backtest_plot_path = Path("results/plots/real_data_backtest_equity_curves_2x.png" if is_real else "results/plots/backtest_equity_curves_2x.png")
    if not backtest_plot_path.exists():
        backtest_plot_path = Path("results/plots/backtest_equity_curves.png")

    if backtest_plot_path.exists():
        st.subheader("Equity & Drawdown Curves")
        st.image(str(backtest_plot_path), caption="Simulated Balance and Drawdown Curves")
    else:
        st.info("Run backtest to execute offline historical simulations.")

# -----------------------------------------------------------------------------
# TAB 7: EXPERIMENT TRACKING
# -----------------------------------------------------------------------------
with tab_experiments:
    st.header("Reproducible Experiment Tracking History")
    exp_log_path = Path("results/metrics/experiment_history.jsonl")

    if exp_log_path.exists():
        records = []
        with open(exp_log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))
        if records:
            exp_df = pd.DataFrame(records)
            st.dataframe(exp_df[["experiment_id", "model_name", "threshold", "training_time_sec", "timestamp"]], use_container_width=True)
            st.caption(f"Total experiments recorded: {len(records)}")
    else:
        st.info("No experiment logs found.")
