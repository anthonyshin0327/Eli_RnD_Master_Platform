# insights.py
def generate_insights(task_type, leaderboard):
    top_model = leaderboard.iloc[0]
    if task_type == "regression":
        return (f"AutoGluon selected **{top_model['model']}** as the best regression model with validation R² = {top_model['score_val']:.2f}.\n"
                f"Inference time: {top_model['pred_time_val']:.2f}s, Training time: {top_model['fit_time']:.2f}s.")
    else:
        return (f"AutoGluon selected **{top_model['model']}** as the best classification model with validation accuracy = {top_model['score_val']:.2%}.\n"
                f"Inference time: {top_model['pred_time_val']:.2f}s, Training time: {top_model['fit_time']:.2f}s.")
