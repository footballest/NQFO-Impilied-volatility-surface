"""Locked final inference pipeline for the NQFO implied-volatility task."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


FINAL_MODEL_NAME = "hybrid_slice_no_hard_case_override_pruned"

# Locked final recipe from the notebooks.
LOOKBACK_DATES = 20
MIN_HISTORY_DATES = 20
MASK_SEED = "nqfo-val-v1"
SHRINK_ALPHA = 0.25

PRIMARY_MASK_CONFIG = {
    "base_hide_rate": 0.10,
    "node_weight": 0.65,
    "support_weight": 0.35,
}

RIDGE_ALPHA = 10.0
HISTGB_CONFIG = {
    "learning_rate": 0.05,
    "max_depth": 3,
    "max_iter": 300,
    "min_samples_leaf": 10,
}

BUCKET_COLS = ["maturity_label", "option_type"]
NODE_COLS = ["maturity_label", "moneyness", "option_type"]


FEATURE_GROUPS = {
    "row_local_core": [
        "moneyness",
        "log_moneyness",
        "maturity_days",
        "tau",
        "is_call",
        "is_center",
        "is_wing",
        "is_edge_maturity",
        "is_interior_maturity",
        "option_type",
        "maturity_label",
        "wing_center_bucket",
    ],
    "structured_predictions": [
        "structured_winner_pred",
        "structured_region_blend_pred",
        "tv_maturity_interp_pred",
        "quadratic_smile_logm_pred",
        "same_date_linear_interp_pred",
        "structured_winner_source",
        "structured_region_blend_source",
    ],
    "structured_gaps": [
        "smile_minus_linear",
        "tv_minus_linear",
        "winner_minus_linear",
        "winner_minus_region",
    ],
    "same_date_anchor_values": [
        "opp_visible_iv_same_node",
        "has_opp_same_node_visible",
        "left_anchor_iv",
        "right_anchor_iv",
        "left_anchor_dist",
        "right_anchor_dist",
        "same_maturity_visible_anchor_count",
    ],
    "support_geometry": [
        "opp_option_visible",
        "same_maturity_adj_visible_count",
        "same_moneyness_adj_visible_count",
        "true_local_support_score",
        "mask_local_support_score",
        "support_score_gap",
        "hard_case",
    ],
    "historical_priors": [
        "recent_node_pred",
        "full_node_pred",
    ],
    "date_regime_proxies": [
        "date_atm_iv_proxy",
        "date_skew_proxy",
        "date_term_slope_proxy",
        "date_visible_anchor_ratio",
        "date_visible_iv_dispersion",
        "date_visible_iv_mean",
        "date_visible_anchor_count",
    ],
}

FULL_FEATURE_COLUMNS = sorted(
    {column for columns in FEATURE_GROUPS.values() for column in columns}
)
PRUNED_HISTGB_FEATURE_COLUMNS = [
    column
    for column in FULL_FEATURE_COLUMNS
    if column not in set(FEATURE_GROUPS["same_date_anchor_values"])
    and column not in set(FEATURE_GROUPS["date_regime_proxies"])
]


RuntimeContext = dict[str, Any]


def _log(message: str, verbose: bool) -> None:
    """Print a progress message when verbose logging is enabled."""
    if verbose:
        print(message, flush=True)


def run_locked_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_prediction: float = 5.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the frozen final recipe and return predictions for missing test rows.

    Returns a dataframe with:
    - row_id
    - iv_predicted
    """
    _log("Preparing train/test frames...", verbose)
    train = _ensure_tau(train_df)
    test = _ensure_tau(test_df)

    _log("Building runtime context...", verbose)
    context = _build_runtime_context(train, test)

    _log("Building final pseudo-training table...", verbose)
    final_train_table = _build_final_training_table(train, context)

    _log("Building final test feature table...", verbose)
    actual_test_rows = _build_actual_test_rows(test, context)
    full_test_feature_table = _build_feature_table(
        train_history=train,
        masked_target_rows=actual_test_rows,
        context=context,
    )
    test_missing_feature_table = (
        full_test_feature_table.loc[full_test_feature_table["is_scored_target"]]
        .copy()
        .reset_index(drop=True)
    )

    _log("Checking feature availability...", verbose)
    _assert_feature_availability(final_train_table, test_missing_feature_table)

    _log("Preparing structured branch...", verbose)
    structured_pred_df = _predict_structured_branch(test_missing_feature_table)

    _log("Fitting Ridge branch...", verbose)
    ridge_pred_df = _predict_ridge_branch(final_train_table, test_missing_feature_table)

    _log("Fitting HistGB branch...", verbose)
    histgb_pred_df = _predict_pruned_histgb_branch(
        final_train_table,
        test_missing_feature_table,
    )

    _log("Applying locked hybrid routing...", verbose)
    hybrid_pred_df = _apply_locked_hybrid_routing(
        test_missing_feature_table,
        structured_pred_df=structured_pred_df,
        ridge_pred_df=ridge_pred_df,
        pruned_histgb_pred_df=histgb_pred_df,
        min_prediction=min_prediction,
    )

    _log("Final predictions ready.", verbose)
    return (
        hybrid_pred_df[["row_id", "iv_predicted"]]
        .sort_values("row_id")
        .reset_index(drop=True)
    )


def _ensure_tau(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tau exists and return a copy of the input dataframe."""
    out = df.copy()
    if "tau" not in out.columns:
        out["tau"] = out["maturity_days"] / 365.0
    return out


def _build_runtime_context(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> RuntimeContext:
    """Build static lookup tables used across masking and feature construction."""
    overall_test_missing_rate = test_df["iv_observed"].isna().mean()

    test_bucket_pattern = (
        test_df.assign(is_missing=test_df["iv_observed"].isna())
        .groupby(BUCKET_COLS)["is_missing"]
        .mean()
        .rename("test_bucket_missing_rate")
        .reset_index()
    )

    test_node_pattern = (
        test_df.assign(is_missing=test_df["iv_observed"].isna())
        .groupby(NODE_COLS)["is_missing"]
        .mean()
        .rename("test_node_missing_rate")
        .reset_index()
    )

    surface_levels = pd.concat(
        [
            train_df[["moneyness", "maturity_label", "maturity_days"]],
            test_df[["moneyness", "maturity_label", "maturity_days"]],
        ],
        ignore_index=True,
    )

    moneyness_levels = sorted(surface_levels["moneyness"].dropna().unique().tolist())
    maturity_levels = (
        surface_levels[["maturity_label", "maturity_days"]]
        .drop_duplicates()
        .sort_values("maturity_days")["maturity_label"]
        .tolist()
    )

    return {
        "overall_test_missing_rate": overall_test_missing_rate,
        "test_bucket_pattern": test_bucket_pattern,
        "test_node_pattern": test_node_pattern,
        "moneyness_levels": moneyness_levels,
        "maturity_levels": maturity_levels,
        "m_idx": {value: index for index, value in enumerate(moneyness_levels)},
        "t_idx": {value: index for index, value in enumerate(maturity_levels)},
    }


def _stable_uniform(key: str) -> float:
    """Map a string key to a deterministic U[0,1] value."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _opposite_option(option_type: str) -> str:
    """Return the opposite option type."""
    return "put" if option_type == "call" else "call"


def _local_support_profile(
    target_rows: pd.DataFrame,
    visible_rows: pd.DataFrame,
    context: RuntimeContext,
) -> pd.DataFrame:
    """Compute same-date local support around each target node."""
    profile = target_rows.copy()

    visible_key_set = set(
        zip(
            visible_rows["date"],
            visible_rows["moneyness"],
            visible_rows["maturity_label"],
            visible_rows["option_type"],
        )
    )

    moneyness_levels = context["moneyness_levels"]
    maturity_levels = context["maturity_levels"]
    m_idx = context["m_idx"]
    t_idx = context["t_idx"]

    opposite_visible = []
    same_maturity_adj_count = []
    same_moneyness_adj_count = []

    for date, moneyness, maturity_label, option_type in zip(
        profile["date"],
        profile["moneyness"],
        profile["maturity_label"],
        profile["option_type"],
    ):
        opposite_visible.append(
            (date, moneyness, maturity_label, _opposite_option(option_type))
            in visible_key_set
        )

        i = m_idx[moneyness]
        j = t_idx[maturity_label]

        same_maturity_candidates = []
        if i - 1 >= 0:
            same_maturity_candidates.append(
                (date, moneyness_levels[i - 1], maturity_label, option_type)
            )
        if i + 1 < len(moneyness_levels):
            same_maturity_candidates.append(
                (date, moneyness_levels[i + 1], maturity_label, option_type)
            )

        same_moneyness_candidates = []
        if j - 1 >= 0:
            same_moneyness_candidates.append(
                (date, moneyness, maturity_levels[j - 1], option_type)
            )
        if j + 1 < len(maturity_levels):
            same_moneyness_candidates.append(
                (date, moneyness, maturity_levels[j + 1], option_type)
            )

        same_maturity_adj_count.append(
            sum(candidate in visible_key_set for candidate in same_maturity_candidates)
        )
        same_moneyness_adj_count.append(
            sum(candidate in visible_key_set for candidate in same_moneyness_candidates)
        )

    profile["opp_option_visible"] = opposite_visible
    profile["same_maturity_adj_visible_count"] = same_maturity_adj_count
    profile["same_moneyness_adj_visible_count"] = same_moneyness_adj_count
    profile["local_support_score"] = (
        profile["opp_option_visible"].astype(int)
        + profile["same_maturity_adj_visible_count"]
        + profile["same_moneyness_adj_visible_count"]
    )
    return profile


def _build_masked_rows_for_primary_protocol(
    full_df: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    context: RuntimeContext,
) -> pd.DataFrame:
    """Build pseudo-masked rows using the locked primary_realistic protocol."""
    cfg = PRIMARY_MASK_CONFIG

    out = full_df.loc[full_df["date"].isin(target_dates)].copy()
    out["is_orig_observed"] = out["iv_observed"].notna()
    out["is_orig_missing"] = ~out["is_orig_observed"]

    out = out.merge(context["test_bucket_pattern"], on=BUCKET_COLS, how="left")
    out = out.merge(context["test_node_pattern"], on=NODE_COLS, how="left")

    out["bucket_hide_rate_on_observed"] = (
        cfg["base_hide_rate"]
        * out["test_bucket_missing_rate"]
        / context["overall_test_missing_rate"]
    )

    out["priority_noise"] = out.apply(
        lambda row: _stable_uniform(
            f"{MASK_SEED}|primary_realistic|{row['date'].date()}|"
            f"{row['maturity_label']}|{row['moneyness']:.4f}|{row['option_type']}"
        ),
        axis=1,
    )

    observed_pool = out.loc[out["is_orig_observed"]].copy()
    observed_support = _local_support_profile(
        observed_pool,
        observed_pool,
        context,
    )[["row_id", "local_support_score"]]
    out = out.merge(observed_support, on="row_id", how="left")
    out["local_support_score"] = out["local_support_score"].fillna(0).astype(int)

    out["is_pseudo_hidden"] = False

    observed_pool = out.loc[out["is_orig_observed"]].copy()
    for _, group in observed_pool.groupby(["date", *BUCKET_COLS], sort=False):
        n_observed = len(group)
        n_hide = int(
            np.round(group["bucket_hide_rate_on_observed"].iloc[0] * n_observed)
        )
        if n_hide <= 0:
            continue

        node_rank = group["test_node_missing_rate"].rank(method="average", pct=True)
        support_rank = group["local_support_score"].rank(method="average", pct=True)

        selection_priority = (
            cfg["node_weight"] * node_rank
            + cfg["support_weight"] * support_rank
            + 1e-6 * group["priority_noise"]
        )

        chosen_index = (
            group.assign(selection_priority=selection_priority)
            .sort_values(["selection_priority", "row_id"], ascending=[False, True])
            .head(n_hide)
            .index
        )
        out.loc[chosen_index, "is_pseudo_hidden"] = True

    out["is_scored_target"] = out["is_pseudo_hidden"]
    out["is_visible_anchor"] = out["is_orig_observed"] & ~out["is_pseudo_hidden"]
    out["is_effectively_missing"] = out["is_orig_missing"] | out["is_pseudo_hidden"]
    out["mask_protocol"] = "primary_realistic"

    return out.sort_values(
        ["date", "maturity_days", "option_type", "moneyness"]
    ).reset_index(drop=True)


def _build_node_lookup(observed_df: pd.DataFrame, pred_name: str) -> pd.DataFrame:
    """Aggregate observed IVs at the surface node level."""
    return (
        observed_df.groupby(NODE_COLS)["iv_observed"]
        .median()
        .rename(pred_name)
        .reset_index()
    )


def _predict_recent_node_median(
    train_history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Predict using recent-node median with full-history fallback."""
    out = target_rows.copy()

    observed_train = train_history.loc[train_history["iv_observed"].notna()].copy()
    global_median = observed_train["iv_observed"].median()

    recent_dates = sorted(observed_train["date"].unique())[-LOOKBACK_DATES:]
    recent_obs = observed_train.loc[observed_train["date"].isin(recent_dates)].copy()

    recent_lookup = _build_node_lookup(recent_obs, "recent_node_pred")
    full_lookup = _build_node_lookup(observed_train, "full_node_pred")

    out = out.merge(recent_lookup, on=NODE_COLS, how="left")
    out = out.merge(full_lookup, on=NODE_COLS, how="left")

    out["iv_pred"] = (
        out["recent_node_pred"].fillna(out["full_node_pred"]).fillna(global_median)
    )
    out["pred_source"] = np.select(
        [
            out["recent_node_pred"].notna(),
            out["full_node_pred"].notna(),
        ],
        [
            "recent_node_median",
            "full_node_median",
        ],
        default="global_median",
    )
    return out


def _predict_same_date_linear_interp(
    train_history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate same-date IVs within maturity/option slices when anchors exist."""
    out = _predict_recent_node_median(train_history, target_rows).copy()
    out["interp_pred"] = np.nan
    out["interp_source"] = pd.Series(index=out.index, dtype="object")

    for _, group_index in out.groupby(
        ["date", "maturity_label", "option_type"], sort=False
    ).groups.items():
        group = out.loc[group_index].copy()

        anchors = (
            group.loc[group["is_visible_anchor"], ["moneyness", "iv_observed"]]
            .dropna()
            .sort_values("moneyness")
        )

        if len(anchors) == 0:
            continue

        x = anchors["moneyness"].to_numpy()
        y = anchors["iv_observed"].to_numpy()

        if len(anchors) == 1:
            interp_values = np.repeat(y[0], len(group))
            interp_label = "same_date_single_anchor"
        else:
            interp_values = np.interp(group["moneyness"].to_numpy(), x, y)
            interp_label = "same_date_linear_interp"

        out.loc[group.index, "interp_pred"] = interp_values
        out.loc[group.index, "interp_source"] = interp_label

    use_interp = out["interp_pred"].notna()
    out.loc[use_interp, "iv_pred"] = out.loc[use_interp, "interp_pred"]
    out["pred_source"] = np.where(use_interp, out["interp_source"], out["pred_source"])
    return out


def _predict_quadratic_smile_logm(
    train_history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Fit a quadratic smile in log-moneyness using same-date anchors."""
    out = _predict_same_date_linear_interp(train_history, target_rows).copy()

    out["log_moneyness"] = np.log(out["moneyness"])
    out["smile_pred"] = np.nan
    out["pred_source_smile"] = pd.Series(index=out.index, dtype="object")

    observed_train = train_history.loc[
        train_history["iv_observed"].notna(), "iv_observed"
    ]
    pred_lo = observed_train.quantile(0.001)
    pred_hi = observed_train.quantile(0.999)

    for _, group_index in out.groupby(
        ["date", "maturity_label", "option_type"], sort=False
    ).groups.items():
        group = out.loc[group_index].copy()

        anchors = (
            group.loc[group["is_visible_anchor"], ["log_moneyness", "iv_observed"]]
            .dropna()
            .sort_values("log_moneyness")
        )

        if len(anchors) < 5:
            continue
        if anchors["log_moneyness"].nunique() < 3:
            continue

        x = anchors["log_moneyness"].to_numpy()
        y = anchors["iv_observed"].to_numpy()

        x_center = x.mean()
        coeffs = np.polyfit(x - x_center, y, deg=2)

        target_x = group["log_moneyness"].to_numpy()
        pred = np.polyval(coeffs, target_x - x_center)

        in_range = (target_x >= x.min()) & (target_x <= x.max())
        pred = np.where(in_range, pred, np.nan)
        pred = np.clip(pred, pred_lo, pred_hi)

        out.loc[group.index, "smile_pred"] = pred
        out.loc[group.index, "pred_source_smile"] = np.where(
            in_range,
            "quadratic_smile_logm",
            pd.NA,
        )

    use_smile = out["smile_pred"].notna()
    out.loc[use_smile, "iv_pred"] = out.loc[use_smile, "smile_pred"]
    out["pred_source"] = np.where(
        use_smile, out["pred_source_smile"], out["pred_source"]
    )
    return out


def _predict_total_variance_maturity_interp(
    train_history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Interpolate total variance across maturities for fixed moneyness/option nodes."""
    out = _predict_same_date_linear_interp(train_history, target_rows).copy()

    out["anchor_total_variance"] = np.where(
        out["iv_observed"].notna(),
        (out["iv_observed"] / 100.0) ** 2 * out["tau"],
        np.nan,
    )
    out["tv_pred"] = np.nan
    out["pred_source_tv"] = pd.Series(index=out.index, dtype="object")

    observed_train = train_history.loc[
        train_history["iv_observed"].notna(), "iv_observed"
    ]
    pred_lo = observed_train.quantile(0.001)
    pred_hi = observed_train.quantile(0.999)

    for _, group_index in out.groupby(
        ["date", "moneyness", "option_type"], sort=False
    ).groups.items():
        group = out.loc[group_index].copy()

        anchors = (
            group.loc[group["is_visible_anchor"], ["tau", "anchor_total_variance"]]
            .dropna()
            .sort_values("tau")
        )

        if len(anchors) < 2:
            continue

        x = anchors["tau"].to_numpy()
        y = anchors["anchor_total_variance"].to_numpy()
        target_tau = group["tau"].to_numpy()

        in_range = (target_tau >= x.min()) & (target_tau <= x.max())
        pred_tv = np.interp(target_tau, x, y)
        pred_tv = np.where(in_range, pred_tv, np.nan)
        pred_tv = np.where(pred_tv > 0, pred_tv, np.nan)

        pred_iv = 100.0 * np.sqrt(pred_tv / target_tau)
        pred_iv = np.clip(pred_iv, pred_lo, pred_hi)

        out.loc[group.index, "tv_pred"] = pred_iv
        out.loc[group.index, "pred_source_tv"] = np.where(
            in_range,
            "tv_maturity_interp",
            pd.NA,
        )

    use_tv = out["tv_pred"].notna()
    out.loc[use_tv, "iv_pred"] = out.loc[use_tv, "tv_pred"]
    out["pred_source"] = np.where(use_tv, out["pred_source_tv"], out["pred_source"])
    return out


def _predict_structured_region_blend(
    train_history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Blend same-date smile and total-variance interpolation into a structured predictor."""
    base = _predict_same_date_linear_interp(train_history, target_rows).copy()

    smile = _predict_quadratic_smile_logm(train_history, target_rows)[
        ["row_id", "smile_pred"]
    ]
    tv = _predict_total_variance_maturity_interp(train_history, target_rows)[
        ["row_id", "tv_pred"]
    ]

    out = base.merge(smile, on="row_id", how="left")
    out = out.merge(tv, on="row_id", how="left")

    out["wing_center_bucket"] = np.where(
        out["moneyness"].between(0.9, 1.1, inclusive="both"),
        "center",
        "wing",
    )

    both_available = out["smile_pred"].notna() & out["tv_pred"].notna()
    only_smile = out["smile_pred"].notna() & out["tv_pred"].isna()
    only_tv = out["tv_pred"].notna() & out["smile_pred"].isna()

    center_mask = both_available & (out["wing_center_bucket"] == "center")
    wing_mask = both_available & (out["wing_center_bucket"] == "wing")

    out.loc[center_mask, "iv_pred"] = (
        0.65 * out.loc[center_mask, "smile_pred"]
        + 0.35 * out.loc[center_mask, "tv_pred"]
    )
    out.loc[wing_mask, "iv_pred"] = (
        0.35 * out.loc[wing_mask, "smile_pred"] + 0.65 * out.loc[wing_mask, "tv_pred"]
    )
    out.loc[only_smile, "iv_pred"] = out.loc[only_smile, "smile_pred"]
    out.loc[only_tv, "iv_pred"] = out.loc[only_tv, "tv_pred"]

    out["pred_source"] = np.select(
        [
            center_mask,
            wing_mask,
            only_smile,
            only_tv,
        ],
        [
            "structured_region_blend_center",
            "structured_region_blend_wing",
            "quadratic_smile_only",
            "tv_maturity_only",
        ],
        default=out["pred_source"],
    )
    return out


def _predict_structured_winner(
    train_history: pd.DataFrame,
    target_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the locked call-put shrink winner on top of the structured blend."""
    out = _predict_structured_region_blend(train_history, target_rows).copy()

    opposite_visible = out.loc[
        out["is_visible_anchor"],
        ["date", "moneyness", "maturity_label", "option_type", "iv_observed"],
    ].copy()
    opposite_visible["option_type"] = opposite_visible["option_type"].map(
        _opposite_option
    )
    opposite_visible = opposite_visible.rename(
        columns={"iv_observed": "opp_visible_iv"}
    )

    out = out.merge(
        opposite_visible,
        on=["date", "moneyness", "maturity_label", "option_type"],
        how="left",
    )

    use_shrink = out["opp_visible_iv"].notna()
    out.loc[use_shrink, "iv_pred"] = (1.0 - SHRINK_ALPHA) * out.loc[
        use_shrink, "iv_pred"
    ] + SHRINK_ALPHA * out.loc[use_shrink, "opp_visible_iv"]

    out["pred_source"] = np.where(
        use_shrink,
        "structured_region_blend_callput_shrink",
        out["pred_source"],
    )
    return out


def _add_true_support_columns(
    scored_df: pd.DataFrame,
    context: RuntimeContext,
) -> pd.DataFrame:
    """Add support geometry features used by the locked models."""
    out = scored_df.copy()

    if "local_support_score" in out.columns:
        out = out.rename(columns={"local_support_score": "mask_local_support_score"})

    scored_targets = out.loc[out["is_scored_target"]].copy()
    visible_anchors = out.loc[out["is_visible_anchor"]].copy()

    support = _local_support_profile(scored_targets, visible_anchors, context)[
        [
            "row_id",
            "opp_option_visible",
            "same_maturity_adj_visible_count",
            "same_moneyness_adj_visible_count",
            "local_support_score",
        ]
    ].copy()

    support = support.rename(
        columns={"local_support_score": "true_local_support_score"}
    )
    support["any_local_same_date_support"] = (
        support["opp_option_visible"]
        | (support["same_maturity_adj_visible_count"] > 0)
        | (support["same_moneyness_adj_visible_count"] > 0)
    )
    support["hard_case"] = ~support["any_local_same_date_support"]

    out = out.merge(support, on="row_id", how="left")

    out["opp_option_visible"] = (
        out["opp_option_visible"].astype("boolean").fillna(False).astype(bool)
    )
    out["same_maturity_adj_visible_count"] = (
        out["same_maturity_adj_visible_count"].fillna(0).astype(int)
    )
    out["same_moneyness_adj_visible_count"] = (
        out["same_moneyness_adj_visible_count"].fillna(0).astype(int)
    )
    out["true_local_support_score"] = (
        out["true_local_support_score"].fillna(0).astype(int)
    )
    out["any_local_same_date_support"] = (
        out["any_local_same_date_support"].astype("boolean").fillna(False).astype(bool)
    )
    out["hard_case"] = out["hard_case"].astype("boolean").fillna(False).astype(bool)

    if "mask_local_support_score" in out.columns:
        out["mask_local_support_score"] = (
            out["mask_local_support_score"].fillna(0).astype(int)
        )

    return out


def _add_same_maturity_anchor_features(masked_df: pd.DataFrame) -> pd.DataFrame:
    """Add nearest left/right same-maturity anchor IVs and distances."""
    out = masked_df.copy()

    out["left_anchor_iv"] = np.nan
    out["right_anchor_iv"] = np.nan
    out["left_anchor_dist"] = np.nan
    out["right_anchor_dist"] = np.nan
    out["same_maturity_visible_anchor_count"] = 0

    for _, group_index in out.groupby(
        ["date", "maturity_label", "option_type"], sort=False
    ).groups.items():
        group = out.loc[group_index].copy()

        anchors = (
            group.loc[group["is_visible_anchor"], ["moneyness", "iv_observed"]]
            .dropna()
            .sort_values("moneyness")
        )

        if len(anchors) == 0:
            continue

        anchor_x = anchors["moneyness"].to_numpy()
        anchor_y = anchors["iv_observed"].to_numpy()
        target_x = group["moneyness"].to_numpy()

        left_iv = []
        right_iv = []
        left_dist = []
        right_dist = []

        for x in target_x:
            left_mask = anchor_x <= x
            right_mask = anchor_x >= x

            if left_mask.any():
                lx = anchor_x[left_mask][-1]
                ly = anchor_y[left_mask][-1]
                left_iv.append(ly)
                left_dist.append(abs(x - lx))
            else:
                left_iv.append(np.nan)
                left_dist.append(np.nan)

            if right_mask.any():
                rx = anchor_x[right_mask][0]
                ry = anchor_y[right_mask][0]
                right_iv.append(ry)
                right_dist.append(abs(rx - x))
            else:
                right_iv.append(np.nan)
                right_dist.append(np.nan)

        out.loc[group.index, "left_anchor_iv"] = left_iv
        out.loc[group.index, "right_anchor_iv"] = right_iv
        out.loc[group.index, "left_anchor_dist"] = left_dist
        out.loc[group.index, "right_anchor_dist"] = right_dist
        out.loc[group.index, "same_maturity_visible_anchor_count"] = len(anchors)

    return out


def _add_same_node_opposite_option_feature(masked_df: pd.DataFrame) -> pd.DataFrame:
    """Add opposite-option same-node IV if visible."""
    out = masked_df.copy()

    opposite_visible = out.loc[
        out["is_visible_anchor"],
        ["date", "moneyness", "maturity_label", "option_type", "iv_observed"],
    ].copy()
    opposite_visible["option_type"] = opposite_visible["option_type"].map(
        _opposite_option
    )
    opposite_visible = opposite_visible.rename(
        columns={"iv_observed": "opp_visible_iv_same_node"}
    )

    return out.merge(
        opposite_visible,
        on=["date", "moneyness", "maturity_label", "option_type"],
        how="left",
    )


def _nearest_moneyness_rows(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """Return rows nearest to a requested moneyness level on each date."""
    if df.empty:
        return df.copy()

    distances = (df["moneyness"] - target).abs()
    min_distance = distances.groupby(df["date"]).transform("min")
    return df.loc[min_distance.eq(distances)].copy()


def _add_date_level_regime_proxy_features(masked_df: pd.DataFrame) -> pd.DataFrame:
    """Add date-level visible-anchor proxies used by the locked Ridge branch."""
    out = masked_df.copy()
    visible = out.loc[out["is_visible_anchor"] & out["iv_observed"].notna()].copy()

    total_rows = out.groupby("date").size().rename("date_total_row_count")
    visible_count = visible.groupby("date").size().rename("date_visible_anchor_count")
    visible_ratio = (visible_count / total_rows).rename("date_visible_anchor_ratio")
    visible_mean = (
        visible.groupby("date")["iv_observed"].mean().rename("date_visible_iv_mean")
    )
    visible_dispersion = (
        visible.groupby("date")["iv_observed"]
        .std()
        .rename("date_visible_iv_dispersion")
    )

    atm = (
        _nearest_moneyness_rows(visible, 1.0)
        .groupby("date")["iv_observed"]
        .mean()
        .rename("date_atm_iv_proxy")
    )
    left = (
        _nearest_moneyness_rows(visible, 0.9)
        .groupby("date")["iv_observed"]
        .mean()
        .rename("date_iv_0p9_proxy")
    )
    right = (
        _nearest_moneyness_rows(visible, 1.1)
        .groupby("date")["iv_observed"]
        .mean()
        .rename("date_iv_1p1_proxy")
    )

    maturity_means = (
        visible.groupby(["date", "maturity_days"])["iv_observed"]
        .mean()
        .reset_index()
        .sort_values(["date", "maturity_days"])
    )
    short_end = (
        maturity_means.groupby("date")
        .first()["iv_observed"]
        .rename("date_short_end_iv_proxy")
    )
    long_end = (
        maturity_means.groupby("date")
        .last()["iv_observed"]
        .rename("date_long_end_iv_proxy")
    )

    proxy_df = pd.concat(
        [
            total_rows,
            visible_count,
            visible_ratio,
            visible_mean,
            visible_dispersion,
            atm,
            left,
            right,
            short_end,
            long_end,
        ],
        axis=1,
    ).reset_index()

    proxy_df["date_visible_anchor_count"] = (
        proxy_df["date_visible_anchor_count"].fillna(0).astype(int)
    )
    proxy_df["date_visible_anchor_ratio"] = proxy_df[
        "date_visible_anchor_ratio"
    ].fillna(0.0)
    proxy_df["date_skew_proxy"] = (
        proxy_df["date_iv_0p9_proxy"] - proxy_df["date_iv_1p1_proxy"]
    )
    proxy_df["date_term_slope_proxy"] = (
        proxy_df["date_short_end_iv_proxy"] - proxy_df["date_long_end_iv_proxy"]
    )

    return out.merge(proxy_df, on="date", how="left")


def _build_feature_table(
    train_history: pd.DataFrame,
    masked_target_rows: pd.DataFrame,
    context: RuntimeContext,
) -> pd.DataFrame:
    """Build the locked feature table for pseudo-training or final test inference."""
    base = masked_target_rows.copy()

    linear_pred = _predict_same_date_linear_interp(train_history, base)[
        ["row_id", "iv_pred"]
    ].rename(columns={"iv_pred": "same_date_linear_interp_pred"})
    smile_pred = _predict_quadratic_smile_logm(train_history, base)[
        ["row_id", "smile_pred"]
    ].rename(columns={"smile_pred": "quadratic_smile_logm_pred"})
    tv_pred = _predict_total_variance_maturity_interp(train_history, base)[
        ["row_id", "tv_pred"]
    ].rename(columns={"tv_pred": "tv_maturity_interp_pred"})
    region_pred = _predict_structured_region_blend(train_history, base)[
        ["row_id", "iv_pred", "pred_source"]
    ].rename(
        columns={
            "iv_pred": "structured_region_blend_pred",
            "pred_source": "structured_region_blend_source",
        }
    )
    winner_pred = _predict_structured_winner(train_history, base)[
        ["row_id", "iv_pred", "pred_source"]
    ].rename(
        columns={
            "iv_pred": "structured_winner_pred",
            "pred_source": "structured_winner_source",
        }
    )
    node_pred = _predict_recent_node_median(train_history, base)[
        ["row_id", "recent_node_pred", "full_node_pred"]
    ]

    feat = base.merge(linear_pred, on="row_id", how="left")
    feat = feat.merge(smile_pred, on="row_id", how="left")
    feat = feat.merge(tv_pred, on="row_id", how="left")
    feat = feat.merge(region_pred, on="row_id", how="left")
    feat = feat.merge(winner_pred, on="row_id", how="left")
    feat = feat.merge(node_pred, on="row_id", how="left")

    feat = _add_true_support_columns(feat, context)
    feat = _add_same_maturity_anchor_features(feat)
    feat = _add_same_node_opposite_option_feature(feat)
    feat = _add_date_level_regime_proxy_features(feat)

    feat["support_score_gap"] = feat["true_local_support_score"] - feat.get(
        "mask_local_support_score", 0
    )
    feat["has_opp_same_node_visible"] = (
        feat["opp_visible_iv_same_node"].notna().astype(int)
    )

    feat["log_moneyness"] = np.log(feat["moneyness"])
    feat["is_call"] = (feat["option_type"] == "call").astype(int)
    feat["wing_center_bucket"] = np.where(
        feat["moneyness"].between(0.9, 1.1, inclusive="both"),
        "center",
        "wing",
    )
    feat["is_center"] = (feat["wing_center_bucket"] == "center").astype(int)
    feat["is_wing"] = (feat["wing_center_bucket"] == "wing").astype(int)
    feat["is_edge_maturity"] = feat["maturity_label"].isin(["1M", "6M"]).astype(int)
    feat["is_interior_maturity"] = 1 - feat["is_edge_maturity"]

    feat["smile_minus_linear"] = (
        feat["quadratic_smile_logm_pred"] - feat["same_date_linear_interp_pred"]
    )
    feat["tv_minus_linear"] = (
        feat["tv_maturity_interp_pred"] - feat["same_date_linear_interp_pred"]
    )
    feat["winner_minus_linear"] = (
        feat["structured_winner_pred"] - feat["same_date_linear_interp_pred"]
    )
    feat["winner_minus_region"] = (
        feat["structured_winner_pred"] - feat["structured_region_blend_pred"]
    )

    feat["target_iv"] = feat["iv_observed"]
    feat["target_residual"] = feat["target_iv"] - feat["structured_winner_pred"]

    return feat


def _build_final_training_table(
    train_df: pd.DataFrame,
    context: RuntimeContext,
) -> pd.DataFrame:
    """Build the final pseudo-supervised training table under the locked primary_only policy."""
    train_dates = sorted(train_df["date"].drop_duplicates().tolist())
    target_dates = train_dates[MIN_HISTORY_DATES:]

    tables = []
    for target_date in target_dates:
        history_df = train_df.loc[train_df["date"] < target_date].copy()
        masked_target_rows = _build_masked_rows_for_primary_protocol(
            train_df,
            [target_date],
            context,
        )

        feat = _build_feature_table(history_df, masked_target_rows, context)
        scored = feat.loc[feat["is_scored_target"]].copy()

        if not scored["is_orig_observed"].all():
            raise ValueError(
                "Pseudo-training rows must come from originally observed train rows."
            )

        tables.append(scored)

    return pd.concat(tables, ignore_index=True)


def _build_actual_test_rows(
    test_df: pd.DataFrame,
    context: RuntimeContext,
) -> pd.DataFrame:
    """Build final inference rows using true observed test anchors and true missing targets."""
    out = test_df.copy()

    out["is_orig_observed"] = out["iv_observed"].notna()
    out["is_orig_missing"] = ~out["is_orig_observed"]
    out["is_pseudo_hidden"] = False
    out["is_scored_target"] = out["is_orig_missing"]
    out["is_visible_anchor"] = out["is_orig_observed"]
    out["is_effectively_missing"] = out["is_orig_missing"]
    out["mask_protocol"] = "actual_test_missing"

    scored_targets = out.loc[out["is_scored_target"]].copy()
    visible_anchors = out.loc[out["is_visible_anchor"]].copy()

    if len(scored_targets) > 0:
        support = _local_support_profile(scored_targets, visible_anchors, context)[
            ["row_id", "local_support_score"]
        ].copy()
        out = out.merge(support, on="row_id", how="left")
    else:
        out["local_support_score"] = 0

    out["local_support_score"] = out["local_support_score"].fillna(0).astype(int)

    return out.sort_values(
        ["date", "maturity_days", "option_type", "moneyness"]
    ).reset_index(drop=True)


def _assert_feature_availability(
    train_table: pd.DataFrame,
    test_feature_table: pd.DataFrame,
) -> None:
    """Check that locked train/test feature schemas support the final branches."""
    missing_full = sorted(set(FULL_FEATURE_COLUMNS) - set(test_feature_table.columns))
    missing_pruned = sorted(
        set(PRUNED_HISTGB_FEATURE_COLUMNS) - set(test_feature_table.columns)
    )

    if missing_full:
        raise ValueError(
            f"Missing full feature columns on test-time table: {missing_full}"
        )
    if missing_pruned:
        raise ValueError(
            f"Missing pruned HistGB feature columns on test-time table: {missing_pruned}"
        )

    missing_full_train = sorted(set(FULL_FEATURE_COLUMNS) - set(train_table.columns))
    missing_pruned_train = sorted(
        set(PRUNED_HISTGB_FEATURE_COLUMNS) - set(train_table.columns)
    )

    if missing_full_train:
        raise ValueError(
            f"Missing full feature columns on training table: {missing_full_train}"
        )
    if missing_pruned_train:
        raise ValueError(
            f"Missing pruned HistGB feature columns on training table: {missing_pruned_train}"
        )


def _prepare_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare numeric design matrices with train-based imputations and aligned dummies."""
    x_train = train_df.loc[:, feature_columns].copy()
    x_test = test_df.loc[:, feature_columns].copy()

    for frame in (x_train, x_test):
        bool_columns = frame.select_dtypes(include=["bool"]).columns.tolist()
        for column in bool_columns:
            frame[column] = frame[column].astype(int)

    categorical_columns = x_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_columns = [
        column for column in x_train.columns if column not in categorical_columns
    ]

    if numeric_columns:
        train_num = x_train[numeric_columns].copy()
        test_num = x_test[numeric_columns].copy()

        medians = train_num.median(numeric_only=True)
        train_num = train_num.fillna(medians)
        test_num = test_num.fillna(medians)
    else:
        train_num = pd.DataFrame(index=x_train.index)
        test_num = pd.DataFrame(index=x_test.index)

    if categorical_columns:
        train_cat = pd.get_dummies(x_train[categorical_columns], dummy_na=True)
        test_cat = pd.get_dummies(x_test[categorical_columns], dummy_na=True)
        train_cat, test_cat = train_cat.align(
            test_cat, join="outer", axis=1, fill_value=0
        )
    else:
        train_cat = pd.DataFrame(index=x_train.index)
        test_cat = pd.DataFrame(index=x_test.index)

    return (
        pd.concat([train_num, train_cat], axis=1),
        pd.concat([test_num, test_cat], axis=1),
    )


def _predict_ridge_branch(
    train_table: pd.DataFrame,
    test_feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the locked Ridge branch and return direct IV predictions."""
    x_train, x_test = _prepare_design_matrices(
        train_table, test_feature_table, FULL_FEATURE_COLUMNS
    )

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(x_train, train_table["target_iv"])

    out = test_feature_table[["row_id"]].copy()
    out["iv_pred"] = model.predict(x_test)
    return out


def _predict_pruned_histgb_branch(
    train_table: pd.DataFrame,
    test_feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Fit the locked pruned HistGB residual branch and return final IV predictions."""
    x_train, x_test = _prepare_design_matrices(
        train_table,
        test_feature_table,
        PRUNED_HISTGB_FEATURE_COLUMNS,
    )

    model = HistGradientBoostingRegressor(
        learning_rate=HISTGB_CONFIG["learning_rate"],
        max_depth=HISTGB_CONFIG["max_depth"],
        max_iter=HISTGB_CONFIG["max_iter"],
        min_samples_leaf=HISTGB_CONFIG["min_samples_leaf"],
        random_state=42,
    )
    model.fit(x_train, train_table["target_residual"])

    out = test_feature_table[["row_id", "structured_winner_pred"]].copy()
    out["ml_raw_pred"] = model.predict(x_test)
    out["iv_pred"] = out["structured_winner_pred"] + out["ml_raw_pred"]
    return out


def _predict_structured_branch(test_feature_table: pd.DataFrame) -> pd.DataFrame:
    """Return the locked structured winner predictions."""
    return test_feature_table[["row_id", "structured_winner_pred"]].rename(
        columns={"structured_winner_pred": "iv_pred"}
    )


def _apply_locked_hybrid_routing(
    test_feature_table: pd.DataFrame,
    structured_pred_df: pd.DataFrame,
    ridge_pred_df: pd.DataFrame,
    pruned_histgb_pred_df: pd.DataFrame,
    min_prediction: float,
) -> pd.DataFrame:
    """Apply the frozen hybrid routing rule and clip final predictions to the minimum floor."""
    out = test_feature_table[
        ["row_id", "wing_center_bucket", "structured_winner_source"]
    ].copy()

    out = out.merge(
        structured_pred_df.rename(columns={"iv_pred": "structured_winner__iv_pred"}),
        on="row_id",
        how="left",
    )
    out = out.merge(
        ridge_pred_df.rename(columns={"iv_pred": "ridge_direct_full__iv_pred"}),
        on="row_id",
        how="left",
    )
    out = out.merge(
        pruned_histgb_pred_df[["row_id", "iv_pred"]].rename(
            columns={"iv_pred": "histgb_pruned_v1_no_anchor_no_regime__iv_pred"}
        ),
        on="row_id",
        how="left",
    )

    out["hybrid_route"] = "histgb_pruned_v1_no_anchor_no_regime"
    out["iv_predicted"] = out["histgb_pruned_v1_no_anchor_no_regime__iv_pred"]

    ridge_mask = out["structured_winner_source"] == "tv_maturity_only"
    out.loc[ridge_mask, "hybrid_route"] = "ridge_direct_full"
    out.loc[ridge_mask, "iv_predicted"] = out.loc[
        ridge_mask, "ridge_direct_full__iv_pred"
    ]

    structured_mask = (out["wing_center_bucket"] == "center") | (
        out["structured_winner_source"] == "quadratic_smile_only"
    )
    out.loc[structured_mask, "hybrid_route"] = "structured_winner"
    out.loc[structured_mask, "iv_predicted"] = out.loc[
        structured_mask, "structured_winner__iv_pred"
    ]

    out["iv_predicted"] = out["iv_predicted"].clip(lower=min_prediction)
    return out[["row_id", "hybrid_route", "iv_predicted"]]
