# TODO: registrar docstring
"""_summary_"""


def get_xgb_params(params: dict) -> dict:
    # TODO: registrar docstring
    """_summary_

    Args:
        params (dict): _description_

    Returns:
        dict: _description_
    """
    target_col: str = params["featured_data"]["target_column"]
    model_training: dict = params["model_training"]

    # Optuna study config
    study_name: str = model_training["study_name"]
    optuna_params: dict = model_training["xgb_params"]
    n_trials: int = model_training["n_trials"]

    # Cross-Validation
    cv_folds: int = model_training["cv_folds"]
    cv_shuffle: bool = model_training["cv_shuffle"]

    # Model parameters
    early_stopping_rounds: int = model_training["early_stopping_rounds"]
    min_delta: float = model_training["min_delta"]
    num_boost_round: int = model_training["num_boost_round"]
    early_stopping_boost: int = model_training["early_stopping_boost"]
    min_delta_boost: float = model_training["min_delta_boost"]

    # Model hyperparameters
    objective: str = optuna_params["objective"]
    eval_metric: str = optuna_params["eval_metric"]
    tree_method: str = optuna_params["tree_method"]
    device: str = optuna_params["device"]
    max_depth: list[int] = optuna_params["max_depth"]
    learning_rate: list[float] = optuna_params["learning_rate"]
    learning_rate_log: bool = optuna_params["learning_rate_log"]
    subsample: list[float] = optuna_params["subsample"]
    colsample_bytree: list[float] = optuna_params["colsample_bytree"]
    gamma: list[float] = optuna_params["gamma"]
    min_child_weight: list[float] = optuna_params["min_child_weight"]
    reg_alpha: list[float] = optuna_params["reg_alpha"]
    reg_alpha_log: bool = optuna_params["reg_alpha_log"]
    reg_lambda: list[float] = optuna_params["reg_lambda"]
    reg_lambda_log: bool = optuna_params["reg_lambda_log"]

    return {
        "study_name": study_name,
        "n_trials": n_trials,
        "target_col": target_col,
        "cv_folds": cv_folds,
        "cv_shuffle": cv_shuffle,
        "early_stopping_rounds": early_stopping_rounds,
        "min_delta": min_delta,
        "num_boost_round": num_boost_round,
        "early_stopping_boost": early_stopping_boost,
        "min_delta_boost": min_delta_boost,
        "objective": objective,
        "eval_metric": eval_metric,
        "tree_method": tree_method,
        "device": device,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "learning_rate_log": learning_rate_log,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "gamma": gamma,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_alpha_log": reg_alpha_log,
        "reg_lambda": reg_lambda,
        "reg_lambda_log": reg_lambda_log,
    }
