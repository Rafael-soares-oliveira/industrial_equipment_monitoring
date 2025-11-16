from datetime import datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.exceptions import DuplicatedStudyError, OptunaError
from optuna.trial import Trial
from optuna.visualization import (
    plot_contour,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from plotly.graph_objects import Figure
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost.callback import EarlyStopping
from xgboost.core import Booster

from industrial_equipment_monitoring.utils.logging_config import setup_logger

logger = setup_logger("ModelTraining_Optimize")


def create_objective(
    x: pd.DataFrame,
    y: pd.Series,
    xgb_params: dict[str, Any],
):
    """
    Function to optimize hyperparameters using Optuna.

    Args:
        x (pd.DataFrame): Features data
        y (pd.Series): Target data
        xgb_params (dict): Parameters and hyperparameters
    """

    def objective(trial: Trial) -> float:
        trial_params: dict[str, int | float | str] = {
            "objective": xgb_params["objective"],
            "eval_metric": xgb_params["eval_metric"],
            "tree_method": xgb_params["tree_method"],
            "device": xgb_params["device"],
            "max_depth": trial.suggest_int(
                "max_depth", xgb_params["max_depth"][0], xgb_params["max_depth"][1]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                xgb_params["learning_rate"][0],
                xgb_params["learning_rate"][1],
                log=xgb_params["learning_rate_log"],
            ),
            "subsample": trial.suggest_float(
                "subsample", xgb_params["subsample"][0], xgb_params["subsample"][1]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                xgb_params["colsample_bytree"][0],
                xgb_params["colsample_bytree"][1],
            ),
            "gamma": trial.suggest_float(
                "gamma", xgb_params["gamma"][0], xgb_params["gamma"][1]
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                xgb_params["min_child_weight"][0],
                xgb_params["min_child_weight"][1],
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                xgb_params["reg_alpha"][0],
                xgb_params["reg_alpha"][1],
                log=xgb_params["reg_alpha_log"],
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                xgb_params["reg_lambda"][0],
                xgb_params["reg_lambda"][1],
                log=xgb_params["reg_lambda_log"],
            ),
        }

        cv = StratifiedKFold(
            n_splits=xgb_params["cv_folds"],
            shuffle=xgb_params["cv_shuffle"],
            random_state=42,
        )
        scores: list[float] = []

        # Manual Cross-Validation
        for fold, (train_idx, val_idx) in enumerate(cv.split(x, y), 1):
            X_tr = x.iloc[train_idx]
            X_val = x.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            try:
                dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=x.columns.tolist())
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=x.columns.tolist())

                # Stop trial when there is no improvement
                early_stopping = EarlyStopping(
                    rounds=xgb_params["early_stopping_boost"],
                    metric_name=xgb_params["eval_metric"],
                    data_name="validation",
                    min_delta=xgb_params["min_delta_boost"],
                    save_best=True,
                )

                # Using Native XGBoost API with EarlyStopping
                booster: Booster = xgb.train(
                    trial_params,
                    dtrain,
                    num_boost_round=xgb_params["num_boost_round"],
                    evals=[(dval, "validation")],
                    callbacks=[early_stopping],
                    verbose_eval=False,
                )

                preds: np.ndarray = booster.predict(dval)
                score: float = float(roc_auc_score(y_val, preds))
                scores.append(score)

            except xgb.core.XGBoostError as e:
                logger.error(
                    f"Erro do XGBoost no Trial {trial.number}, Fold {fold}: {e}"
                )
            except Exception as e:
                logger.error(f"Trial {trial.number}, Fold {fold} falhou: {e}")

        if not scores:
            logger.error(f"Trial {trial.number} não produziu scores válidos.")
            raise ValueError(f"Trial {trial.number} não produziu scores válidos.")

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        logger.info(
            f"Trial {trial.number} finalizado: AUC = {mean_score:.4f} ± {std_score:.4f}"
        )

        return mean_score

    return objective


def setup_optuna_study(study_name: str, direction: str) -> optuna.Study:
    """
    Setup Optuna Study.

    Args:
        study_name (str): Study name
        direction (str): Direction to optimize the objective function

    Returns:
        optuna.Study: Study ready to optimize.
    """
    today: str = datetime.today().strftime("%Y%m%d")
    base_name: str = f"{study_name}_{today}"  # add datetime to the study name
    suffix: int = 0

    # When the study name already exists, it will add a number suffix
    while True:
        unique_name: str = f"{base_name}" if suffix == 0 else f"{base_name}_{suffix}"
        try:
            study: optuna.Study = optuna.create_study(
                direction=direction,
                study_name=unique_name,
                pruner=optuna.pruners.HyperbandPruner(),
                sampler=optuna.samplers.TPESampler(seed=42),
            )

            logger.info(f"Estudo Optuna criado: {unique_name}")
            return study
        except DuplicatedStudyError:
            suffix += 1


def optimized_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, xgb_params: dict
) -> tuple[dict[str, str | int | float], dict[str, Figure]]:
    """
    Initiate study to optimize hyperparameters.

    Args:
        X_train (pd.DataFrame): Features
        y_train (pd.Series): Target
        xgb_params (dict): Parameters and hyperparameters.

    Raises:
        ValueError: DataFrame is empty
        Optuna.Error: When the study fails.
        Exception: When an unexpected error occurs.

    Returns:
        tuple[dict[str, str | int | float], dict[str, Figure]]:
        - Best params
        - Study plots
    """
    logger.info("Iniciando otimização de hiperparâmetros com Optuna")
    if X_train.empty:
        raise ValueError("DataFrame X_train vazio")
    if y_train.empty:
        raise ValueError("Dataframe y_train vazio")

    # Function objective for Optuna
    objective_func = create_objective(X_train, y_train, xgb_params)

    # Config study for Optuna
    study = setup_optuna_study(xgb_params["study_name"], xgb_params["direction"])

    # Execute otimization
    logger.info(f"Executando {xgb_params['n_trials']} trials de otimização")

    best_score = None
    no_improvement_count = 0  # Count when the training has no improvement
    patience = xgb_params["early_stopping_rounds"]  # How many trials until stops.
    min_delta = xgb_params["min_delta"]  # % to consider as 'no improvement'.

    # Function to stop training when has no improvement
    def callback_early_stopping(study, trial):
        nonlocal best_score, no_improvement_count  # Get variable outside function

        if best_score is None:
            best_score = study.best_value
            return

        improvement = study.best_value - best_score

        if improvement > min_delta:
            best_score = study.best_value
            no_improvement_count = 0
            logger.info(f"Melhoria significativa: {improvement:.4f}")
        else:
            no_improvement_count += 1
            logger.info(f"Sem melhoria por {no_improvement_count}/{patience} trials.")

        if no_improvement_count >= patience:
            logger.info(f"Early stopping acionado após {trial.number + 1} trials")
            study.stop()

    try:
        # Start study
        study.optimize(
            objective_func,
            n_trials=xgb_params["n_trials"],
            callbacks=[callback_early_stopping],  # How many trials until stop
            show_progress_bar=True,
        )
    except OptunaError as e:  # Study fails
        logger.error(f"Estudo falhou: {e}")
        raise e
    except Exception as e:  # Unexpected error
        logger.error(f"Erro inesperado: {e}")
        raise e

    best_trial: optuna.trial.FrozenTrial = study.best_trial

    logger.info(
        "Otimização concluída:"
        f"Melhor trial: {best_trial.number}; "
        f"Melhor ROC AUC: {best_trial.value:.4f}; "
        f"Trials completados: {len(study.trials)}"
    )
    logger.info("Melhores parâmetros:")
    for key, value in best_trial.params.items():
        logger.info(f"{key}: {value:.4f}")

    # Log da distribuição dos trials
    if len(study.trials) > 1:
        all_scores = [trial.value for trial in study.trials if trial.value is not None]
        logger.info(f"AUC range: {np.min(all_scores):.4f} - {np.max(all_scores):.4f}")
        logger.info(f"AUC médio: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")

    # Export plots
    plots = {
        "optimization_history": plot_optimization_history(study),
        "param_importance": plot_param_importances(study),
        "contour": plot_contour(study),
        "parallel_coordinate": plot_parallel_coordinate(study),
        "slice": plot_slice(study),
        "intermediate_values": plot_intermediate_values(study),
    }

    return best_trial.params, plots
