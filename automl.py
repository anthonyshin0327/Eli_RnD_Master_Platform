# automl.py
import pandas as pd
from autogluon.tabular import TabularPredictor

def run_automl_pipeline(df, target, task_type="regression"):
    df = df.copy()
    # Drop NA in target for simplicity
    df = df[df[target].notnull()]
    if task_type == "regression":
        problem = "regression"
        eval_metric = "r2"
    else:
        problem = "classification"
        eval_metric = "accuracy"
    predictor = TabularPredictor(label=target, problem_type=problem, eval_metric=eval_metric)
    predictor.fit(df, presets="best_quality", time_limit=180)
    leaderboard = predictor.leaderboard(silent=True)
    # Return best model name and leaderboard DataFrame
    return predictor, leaderboard
