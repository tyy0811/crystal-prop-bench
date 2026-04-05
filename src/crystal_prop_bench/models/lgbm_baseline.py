"""LightGBM baseline model for Tier 1 and Tier 2."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np

logger = logging.getLogger(__name__)


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int = 42,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 127,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 50,
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    """Train LightGBM regressor and compute calibration residuals.

    Parameters
    ----------
    X_train, y_train : Training data.
    X_val, y_val : Validation data for early stopping only.
    X_cal, y_cal : Calibration data for conformal residuals only.
        NOT used during training — preserves exchangeability.
    seed : Random seed.
    Other params : LightGBM hyperparameters.

    Returns
    -------
    (model, cal_residuals) where cal_residuals = |y_cal - y_pred_cal|.
    """
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mae",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )

    cal_preds = model.predict(X_cal)
    cal_residuals = np.abs(y_cal - cal_preds)

    logger.info(
        "Trained LightGBM (seed=%d): best_iteration=%d, cal_MAE=%.4f",
        seed,
        model.best_iteration_,
        np.mean(cal_residuals),
    )

    return model, cal_residuals
