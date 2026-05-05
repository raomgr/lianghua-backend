from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.services.factor_engine import FEATURE_COLUMNS, build_factor_table_from_histories, build_training_dataset_from_histories
from app.services.storage import MarketRepository

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "name": "ridge-alpha-v1",
        "model_type": "Ridge",
        "fit": "ridge",
        "alpha": 2.0,
        "note": "Linear benchmark trained on standardized daily bar factors.",
    },
    {
        "name": "gbdt-alpha-v1",
        "model_type": "GradientBoostingRegressor",
        "fit": "gbdt",
        "learning_rate": 0.03,
        "n_estimators": 220,
        "max_depth": 3,
        "min_samples_leaf": 8,
        "note": "Tree benchmark trained on daily bar factors.",
    },
]


ProgressCallback = Callable[[str], None]


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _compute_directional_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) == 0:
        return 0.0
    return float((np.sign(predicted) == np.sign(actual)).mean())


def _compute_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 2:
        return 0.0
    pred_std = float(np.std(predicted))
    actual_std = float(np.std(actual))
    if pred_std == 0 or actual_std == 0:
        return 0.0
    return float(np.corrcoef(predicted, actual)[0, 1])


def _safe_float(value: float | np.floating | int) -> float:
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value


def _split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(dataset["trade_date"].unique())
    if len(unique_dates) < 8:
        split_index = max(len(unique_dates) - 1, 1)
    else:
        split_index = max(int(len(unique_dates) * 0.8), 1)

    train_dates = set(unique_dates[:split_index])
    valid_dates = set(unique_dates[split_index:])
    train_df = dataset[dataset["trade_date"].isin(train_dates)].copy()
    valid_df = dataset[dataset["trade_date"].isin(valid_dates)].copy()
    if valid_df.empty:
        valid_df = train_df.tail(min(len(train_df), 20)).copy()
        train_df = train_df.iloc[:-len(valid_df)] if len(train_df) > len(valid_df) else train_df
    return train_df, valid_df


def _fit_model(train_df: pd.DataFrame, spec: dict[str, Any]) -> Any:
    feature_matrix = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    target = train_df["future_return_5d"].to_numpy(dtype=float)

    if spec["fit"] == "ridge":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(spec["alpha"]))),
            ]
        )
    else:
        model = GradientBoostingRegressor(
            loss="huber",
            learning_rate=float(spec["learning_rate"]),
            n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]),
            min_samples_leaf=int(spec["min_samples_leaf"]),
            random_state=42,
        )

    model.fit(feature_matrix, target)
    return model


def _predict(frame: pd.DataFrame, model: Any) -> np.ndarray:
    feature_matrix = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    return model.predict(feature_matrix)


def _extract_feature_importance(model: Any, spec: dict[str, Any]) -> dict[str, float]:
    if spec["fit"] == "ridge":
        ridge = model.named_steps["ridge"]
        values = np.abs(ridge.coef_)
    else:
        values = model.feature_importances_

    return {
        feature: float(weight)
        for feature, weight in sorted(zip(FEATURE_COLUMNS, values), key=lambda item: item[1], reverse=True)
    }


def _build_model_metrics(
    spec: dict[str, Any],
    model: Any,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    validation_mse: float,
    walk_forward_metrics: dict[str, Any],
    selection_score: float,
) -> dict[str, Any]:
    metadata = {
        "model_type": spec["model_type"],
        "selection_score": selection_score,
        "train_start": str(train_df["trade_date"].min()),
        "train_end": str(train_df["trade_date"].max()),
        "validation_start": str(valid_df["trade_date"].min()),
        "validation_end": str(valid_df["trade_date"].max()),
        "validation_mse": validation_mse,
    }

    if spec["fit"] == "ridge":
        metadata["alpha"] = float(spec["alpha"])
    else:
        metadata["n_estimators"] = int(model.n_estimators)
        metadata["learning_rate"] = float(model.learning_rate)
        metadata["max_depth"] = int(spec["max_depth"])
        metadata["min_samples_leaf"] = int(spec["min_samples_leaf"])

    return {**metadata, **walk_forward_metrics}


def _run_walk_forward_validation(
    dataset: pd.DataFrame,
    spec: dict[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    unique_dates = sorted(dataset["trade_date"].unique())
    if len(unique_dates) < 12:
        return {
            "walk_forward_windows": 0,
            "walk_forward_mean_ic": 0.0,
            "walk_forward_median_ic": 0.0,
            "walk_forward_positive_ic_ratio": 0.0,
            "walk_forward_mean_directional_accuracy": 0.0,
            "walk_forward_mean_long_short_return": 0.0,
            "walk_forward_start": "",
            "walk_forward_end": "",
        }

    min_train_dates = max(min(40, len(unique_dates) - 2), max(6, len(unique_dates) // 2))
    eval_dates = unique_dates[min_train_dates:]
    ic_values: list[float] = []
    acc_values: list[float] = []
    long_short_values: list[float] = []
    total_windows = len(eval_dates)

    for index, current_date in enumerate(eval_dates, start=1):
        train_df = dataset[dataset["trade_date"] < current_date].copy()
        eval_df = dataset[dataset["trade_date"] == current_date].copy()
        if train_df.empty or eval_df.empty or len(eval_df) < 2:
            continue
        if len(train_df["trade_date"].unique()) < min_train_dates:
            continue

        model = _fit_model(train_df, spec)
        pred = _predict(eval_df, model)
        actual = eval_df["future_return_5d"].to_numpy(dtype=float)

        ic_values.append(_compute_ic(pred, actual))
        acc_values.append(_compute_directional_accuracy(pred, actual))

        order = np.argsort(pred)
        bucket_size = max(1, len(order) // 2)
        bottom_actual = actual[order[:bucket_size]]
        top_actual = actual[order[-bucket_size:]]
        long_short_values.append(float(np.mean(top_actual) - np.mean(bottom_actual)))

        if index == 1 or index == total_windows or index % 10 == 0:
            _emit_progress(
                progress_callback,
                f"[{spec['name']}] walk-forward {index}/{total_windows} "
                f"({current_date})",
            )

    if not ic_values:
        return {
            "walk_forward_windows": 0,
            "walk_forward_mean_ic": 0.0,
            "walk_forward_median_ic": 0.0,
            "walk_forward_positive_ic_ratio": 0.0,
            "walk_forward_mean_directional_accuracy": 0.0,
            "walk_forward_mean_long_short_return": 0.0,
            "walk_forward_start": "",
            "walk_forward_end": "",
        }

    return {
        "walk_forward_windows": len(ic_values),
        "walk_forward_mean_ic": _safe_float(np.mean(ic_values)),
        "walk_forward_median_ic": _safe_float(np.median(ic_values)),
        "walk_forward_positive_ic_ratio": _safe_float(np.mean(np.array(ic_values) > 0)),
        "walk_forward_mean_directional_accuracy": _safe_float(np.mean(acc_values)),
        "walk_forward_mean_long_short_return": _safe_float(np.mean(long_short_values)),
        "walk_forward_start": str(eval_dates[0]),
        "walk_forward_end": str(eval_dates[-1]),
    }


def _build_predictions(latest_factor_table: pd.DataFrame, model: Any, latest_trade_date: str) -> list[dict[str, Any]]:
    table = latest_factor_table.copy()
    table["predicted_return_5d"] = _predict(table, model)
    table["score"] = table["predicted_return_5d"].rank(pct=True)
    table = table.sort_values("predicted_return_5d", ascending=False).reset_index(drop=True)
    table["rank"] = table.index + 1
    return [
        {
            "symbol": row["symbol"],
            "name": row["name"],
            "trade_date": latest_trade_date,
            "predicted_return_5d": float(row["predicted_return_5d"]),
            "score": float(row["score"]),
            "rank": int(row["rank"]),
        }
        for _, row in table.head(20).iterrows()
    ]


def _selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = result["metrics"]
    return (
        float(metrics.get("walk_forward_mean_ic", 0.0)),
        float(result["validation_ic"]),
        float(metrics.get("walk_forward_mean_long_short_return", 0.0)),
        float(result["validation_directional_accuracy"]),
    )


def train_local_models(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str],
    *,
    provider_name: str | None = None,
    configured_provider: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    dataset = build_training_dataset_from_histories(histories, names=names)
    if dataset.empty or len(dataset) < 30:
        raise RuntimeError("Not enough historical rows to train a local model.")

    _emit_progress(
        progress_callback,
        f"training dataset ready: rows={len(dataset)} symbols={dataset['symbol'].nunique()}",
    )

    train_df, valid_df = _split_dataset(dataset)
    if train_df.empty or valid_df.empty:
        raise RuntimeError("Not enough train/validation rows after dataset split.")

    _emit_progress(
        progress_callback,
        f"dataset split complete: train_rows={len(train_df)} validation_rows={len(valid_df)}",
    )

    latest_factor_table = build_factor_table_from_histories(histories, names=names)
    if latest_factor_table.empty:
        raise RuntimeError("No latest factor snapshot is available for prediction.")
    latest_trade_date = str(max(frame["trade_date"].max() for frame in histories.values() if not frame.empty))

    evaluated_results: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        _emit_progress(progress_callback, f"start fitting model: {spec['name']}")
        model = _fit_model(train_df, spec)
        _emit_progress(progress_callback, f"fitted model: {spec['name']}")
        valid_pred = _predict(valid_df, model)
        valid_actual = valid_df["future_return_5d"].to_numpy(dtype=float)

        validation_ic = _compute_ic(valid_pred, valid_actual)
        validation_directional_accuracy = _compute_directional_accuracy(valid_pred, valid_actual)
        validation_mse = float(np.mean((valid_pred - valid_actual) ** 2))
        _emit_progress(progress_callback, f"start walk-forward validation: {spec['name']}")
        walk_forward_metrics = _run_walk_forward_validation(
            dataset,
            spec,
            progress_callback=progress_callback,
        )
        _emit_progress(progress_callback, f"finished walk-forward validation: {spec['name']}")
        selection_score = float(walk_forward_metrics.get("walk_forward_mean_ic", 0.0) or validation_ic)

        evaluated_results.append(
            {
                "model_name": spec["name"],
                "model": model,
                "spec": spec,
                "validation_ic": validation_ic,
                "validation_directional_accuracy": validation_directional_accuracy,
                "metrics": _build_model_metrics(
                    spec,
                    model,
                    train_df,
                    valid_df,
                    validation_mse,
                    walk_forward_metrics,
                    selection_score,
                ),
                "coefficients": {"feature_importance": _extract_feature_importance(model, spec)},
                "predictions": _build_predictions(latest_factor_table, model, latest_trade_date),
            }
        )

    ranked_results = sorted(evaluated_results, key=_selection_key, reverse=True)
    champion_name = ranked_results[0]["model_name"]
    _emit_progress(progress_callback, f"champion selected: {champion_name}")

    settings = get_settings()
    repo = MarketRepository()
    active_provider = provider_name or settings.data_provider
    configured_name = configured_provider or settings.data_provider
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for result in sorted(evaluated_results, key=lambda item: item["model_name"] == champion_name):
        is_champion = result["model_name"] == champion_name
        fallback_note = (
            ""
            if active_provider == configured_name
            else f" Configured provider {configured_name} was unavailable, so {active_provider} data was used."
        )
        result["metrics"]["is_champion"] = is_champion
        run_id = repo.record_model_run(
            model_name=result["model_name"],
            provider=active_provider,
            run_at=run_at,
            train_rows=len(train_df),
            validation_rows=len(valid_df),
            validation_ic=result["validation_ic"],
            validation_directional_accuracy=result["validation_directional_accuracy"],
            metrics=result["metrics"],
            coefficients=result["coefficients"],
            feature_stats={
                "mean": {feature: float(train_df[feature].mean()) for feature in FEATURE_COLUMNS},
                "std": {feature: float(train_df[feature].std()) for feature in FEATURE_COLUMNS},
            },
            note=(
                f"{result['spec']['note']} Selected as current champion model.{fallback_note}"
                if is_champion
                else f"{result['spec']['note']} Stored as comparison baseline.{fallback_note}"
            ),
        )
        repo.replace_predictions(run_id, result["predictions"], created_at=run_at)
        result["run_at"] = run_at

    champion = next(item for item in ranked_results if item["model_name"] == champion_name)
    comparison = [
        {
            "model_name": item["model_name"],
            "provider": active_provider,
            "run_at": run_at,
            "validation_ic": item["validation_ic"],
            "validation_directional_accuracy": item["validation_directional_accuracy"],
            "walk_forward_mean_ic": float(item["metrics"].get("walk_forward_mean_ic", 0.0)),
            "walk_forward_positive_ic_ratio": float(item["metrics"].get("walk_forward_positive_ic_ratio", 0.0)),
            "walk_forward_mean_long_short_return": float(item["metrics"].get("walk_forward_mean_long_short_return", 0.0)),
            "note": (
                "Current champion model."
                if item["model_name"] == champion_name
                else "Comparison baseline for side-by-side evaluation."
            ),
            "is_champion": item["model_name"] == champion_name,
        }
        for item in ranked_results
    ]

    return {
        "status": "success",
        "message": f"Models trained locally and {champion_name} is now active.",
        "model_name": champion_name,
        "train_rows": len(train_df),
        "validation_rows": len(valid_df),
        "validation_ic": champion["validation_ic"],
        "validation_directional_accuracy": champion["validation_directional_accuracy"],
        "comparison": comparison,
        "top_predictions": champion["predictions"][:5],
    }


def train_local_model(
    histories: dict[str, pd.DataFrame],
    names: dict[str, str],
    *,
    provider_name: str | None = None,
    configured_provider: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    return train_local_models(
        histories,
        names,
        provider_name=provider_name,
        configured_provider=configured_provider,
        progress_callback=progress_callback,
    )
