import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import time # For simulating long processes
import json # For LLM interaction (simulated)
import traceback # For detailed error logging
from itertools import product # Added this import for product

# Import actual ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap # For feature importance explanations
import plotly.graph_objects as go # For advanced plotting
import plotly.express as px # For scatter plots

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for matplotlib for Dash
import matplotlib.pyplot as plt


# --- Embedded Demo Data ---
DEMO_DATA_CSV_STRING = """MEMBRANE_TYPE,ANTIBODY_CONCENTRATION,CONJUGATE_TYPE,CONJUGATE_VOLUME,BLOCKING_AGENT,INCUBATION_TIME,STRIP_WIDTH,SAMPLE_VOLUME,DETECTION_LIMIT,ASSAY_ID
NC-180,0.861051855,Latex,0.393210974,Casein,13.69568098,2.932179499,117.1909017,2.180900702,LFA_001
NC-170,2.860301253,Latex,0.426512115,Fish Gelatin,28.99795353,4.641443046,110.0016491,1.52726941,LFA_002
NC-180,2.346785721,Latex,1.311966476,Fish Gelatin,23.90483071,4.880932093,100.0098005,0.822443531,LFA_003
NC-180,4.920703776,Carbon,1.507864376,BSA,26.08073285,4.063860937,112.0393952,0.156169898,LFA_004
NC-170,1.00417506,Carbon,1.106983894,BSA,23.90968613,2.790860932,110.0628033,0.797853837,LFA_005
NC-170,2.290350196,Latex,1.675133692,Casein,19.0785093,3.905855854,113.0846002,0.365687036,LFA_006
NC-180,4.86261695,AuNP,1.983400131,Casein,17.05750406,4.833986015,102.0017857,0.158130004,LFA_007
NC-135,4.394782067,Carbon,1.658023995,Fish Gelatin,21.72983603,4.186377011,105.6673993,0.224781743,LFA_010
NC-135,0.507112437,AuNP,0.336896941,BSA,10.04706046,4.548467498,118.8220356,3.410028396,LFA_011
NC-180,1.60071016,Latex,1.886968467,Casein,20.65676008,3.583710104,100.3150414,0.507518061,LFA_012
NC-170,3.850400978,Carbon,1.040810131,Fish Gelatin,15.29056854,3.680979905,111.7969989,0.304357826,LFA_013
NC-135,2.010543046,AuNP,1.871712083,BSA,27.36607749,4.96689883,100.0001501,0.402195807,LFA_014
NC-180,3.011145048,Carbon,0.516603071,Casein,12.04570701,2.500187113,119.9045579,0.760493034,LFA_015
"""

# --- Model & Hyperparameter Explanations ---
MODEL_EXPLANATIONS = {
    "RandomForestRegressor": {
        "description": "Builds multiple decision trees and merges them together to get a more accurate and stable prediction. Good for capturing complex non-linear relationships.",
        "pros": "Robust to outliers, handles categorical and numerical data well, generally high accuracy.",
        "cons": "Can be slower to train, less interpretable than single decision trees.",
        "hyperparameters": {
            "n_estimators": "The number of trees in the forest. More trees can improve performance but increase computation time.",
            "max_depth": "The maximum depth of each tree. Deeper trees can model more complex patterns but risk overfitting."
        }
    },
    "GradientBoostingRegressor": {
        "description": "Builds trees one at a time, where each new tree helps to correct errors made by previously trained trees. Powerful and often provides high accuracy.",
        "pros": "High accuracy, handles different types of data.",
        "cons": "Sensitive to hyperparameters, can overfit if not tuned carefully, can be computationally intensive.",
        "hyperparameters": {
            "n_estimators": "The number of boosting stages (trees) to perform. More stages can lead to better performance but also overfitting.",
            "learning_rate": "Shrinks the contribution of each tree. Lower values require more trees but can improve generalization."
        }
    },
    "SVR": {
        "description": "Support Vector Regression tries to find a function that deviates from y by a value no greater than ε for each training point x, and at the same time is as flat as possible.",
        "pros": "Effective in high dimensional spaces, memory efficient.",
        "cons": "Does not perform well with large datasets, sensitive to feature scaling and kernel choice.",
        "hyperparameters": {
            "C": "Regularization parameter. Strength of the regularization is inversely proportional to C. Must be strictly positive.",
            "kernel": "Specifies the kernel type to be used in the algorithm (e.g., 'linear', 'rbf')."
        }
    },
    "LinearRegression": {
        "description": "Fits a linear model with coefficients w = (w1, ..., wp) to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.",
        "pros": "Simple to understand, interpretable, computationally fast.",
        "cons": "Assumes linear relationships, sensitive to outliers.",
        "hyperparameters": {}
    },
    "Ridge": {
        "description": "Linear least squares with L2 regularization. Adds a penalty equal to the square of the magnitude of coefficients to address multicollinearity and prevent overfitting.",
        "pros": "Reduces model complexity, helps with multicollinearity.",
        "cons": "Includes all features (doesn't perform feature selection).",
        "hyperparameters": {
            "alpha": "Regularization strength; must be a positive float. Larger values specify stronger regularization."
        }
    },
    "Lasso": {
        "description": "Linear Model trained with L1 prior as regularizer. Adds a penalty equal to the absolute value of the magnitude of coefficients, which can lead to some coefficients being zero (feature selection).",
        "pros": "Performs feature selection, helps with multicollinearity.",
        "cons": "Can be unstable with highly correlated features.",
        "hyperparameters": {
            "alpha": "Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square."
        }
    },
    "ElasticNet": {
        "description": "A linear regression model trained with L1 and L2 prior as regularizer. This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, while still maintaining the regularization properties of Ridge.",
        "pros": "Combines benefits of Lasso and Ridge, good for highly correlated features.",
        "cons": "Two hyperparameters to tune.",
        "hyperparameters": {
            "alpha": "Constant that multiplies the penalty terms. Defaults to 1.0.",
            "l1_ratio": "The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1."
        }
    },
    "KNeighborsRegressor": {
        "description": "Prediction based on the k-nearest neighbors of each point. The target is predicted by local interpolation of the targets associated of the k nearest neighbors in the training set.",
        "pros": "Simple to understand, no assumptions about data distribution.",
        "cons": "Computationally expensive for large datasets, sensitive to irrelevant features and feature scaling.",
        "hyperparameters": {
            "n_neighbors": "Number of neighbors to use by default for kneighbors queries.",
            "weights": "Weight function used in prediction. Possible values: 'uniform', 'distance'."
        }
    },
    "DecisionTreeRegressor": {
        "description": "A non-parametric supervised learning method used for regression. It predicts the value of a target variable by learning simple decision rules inferred from the data features.",
        "pros": "Easy to understand and interpret, handles both numerical and categorical data.",
        "cons": "Can be unstable, prone to overfitting.",
        "hyperparameters": {
            "max_depth": "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples."
        }
    }
}


# --- Real AutoML Functions ---

def create_preprocessing_pipeline(numerical_features, categorical_features, numerical_imputer_strategy):
    transformers = []
    if numerical_features:
        num_pipeline_steps = []
        if numerical_imputer_strategy is not None:
            num_pipeline_steps.append(('imputer', SimpleImputer(strategy=numerical_imputer_strategy)))
        num_pipeline_steps.append(('scaler', StandardScaler()))
        numerical_transformer = Pipeline(steps=num_pipeline_steps)
        transformers.append(('num', numerical_transformer, numerical_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    return preprocessor

MODELS_TO_EVALUATE = []
default_param_grids_from_user_code = { 
    'RandomForestRegressor': {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [None, 10]}, 
    'GradientBoostingRegressor': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.05, 0.1]}, 
    'SVR': {'regressor__kernel': ['rbf', 'linear'], 'regressor__C': [0.5, 1, 5]}, 
    'LinearRegression': {},
    'Ridge': {'regressor__alpha': [0.5, 1.0, 5.0]}, 
    'Lasso': {'regressor__alpha': [0.01, 0.1, 1.0]}, 
    'ElasticNet': {'regressor__alpha': [0.1, 1.0], 'regressor__l1_ratio': [0.3, 0.7]}, 
    'KNeighborsRegressor': {'regressor__n_neighbors': [3, 5, 7], 'regressor__weights': ['uniform', 'distance']},
    'DecisionTreeRegressor': {'regressor__max_depth': [None, 5, 10]} 
}

for key in MODEL_EXPLANATIONS.keys(): 
    estimator_instance = None
    if key == "RandomForestRegressor": estimator_instance = RandomForestRegressor(random_state=42, n_jobs=-1)
    elif key == "GradientBoostingRegressor": estimator_instance = GradientBoostingRegressor(random_state=42)
    elif key == "SVR": estimator_instance = SVR()
    elif key == "LinearRegression": estimator_instance = LinearRegression()
    elif key == "Ridge": estimator_instance = Ridge(random_state=42)
    elif key == "Lasso": estimator_instance = Lasso(random_state=42)
    elif key == "ElasticNet": estimator_instance = ElasticNet(random_state=42)
    elif key == "KNeighborsRegressor": estimator_instance = KNeighborsRegressor()
    elif key == "DecisionTreeRegressor": estimator_instance = DecisionTreeRegressor(random_state=42)

    if estimator_instance is not None: 
        param_grid_for_model = default_param_grids_from_user_code.get(key, {})
        MODELS_TO_EVALUATE.append({
            'name': key,
            'estimator': estimator_instance,
            'param_grid': param_grid_for_model 
        })


def run_automl_pipeline(data_df, target_column, feature_columns, missing_value_strategy, progress_callback, analysis_type, optimization_goal=None, feature_ranges=None):
    print(f"Starting AutoML pipeline for {analysis_type} analysis on target: {target_column}")
    # Initialize all return values to ensure function signature is met even if errors occur early
    all_model_results = []
    best_model_info_dict = {} 
    importances_df = pd.DataFrame()
    shap_beeswarm_plot_src = None
    actual_vs_predicted_fig = go.Figure()
    residuals_vs_predicted_fig = go.Figure()
    
    # Optimization specific return values
    optimal_settings = {}
    predicted_target, predicted_target_lower, predicted_target_upper = np.nan, np.nan, np.nan
    response_surface_fig = {}
    surrogate_tree_text = "Not generated."
    model_info_summary = {}
    surrogate_tree_plot_src = None

    current_progress_steps = [f"Validating Inputs & Data..."] # Start with a validation step
    current_progress_steps.append(f"Preprocessing Data...")
    current_progress_steps.extend([f"Training {model['name']} ({i+1}/{len(MODELS_TO_EVALUATE)})" for i, model in enumerate(MODELS_TO_EVALUATE)])
    current_progress_steps.extend(["Evaluating Best Model...", "Calculating SHAP Values for Best Model..."])
    
    optimization_specific_steps = []
    if analysis_type == 'optimization':
        optimization_specific_steps = ["Generating Surrogate Tree...", "Optimizing (Grid Search over Parameter Space)...", "Evaluating Optimal Settings...", "Generating Optimization Visualizations..."]
        current_progress_steps.extend(optimization_specific_steps)
    current_progress_steps.append("Finalizing Results...")
    
    progress_callback(current_progress_steps, 0) # Initial call with all defined steps

    try:
        if data_df is None or data_df.empty: raise ValueError("Input data is empty.")
        if not target_column or not feature_columns: raise ValueError("Target column or feature columns not specified.")
        progress_callback(current_progress_steps, 0) # Validating Inputs

        best_r2 = -float('inf') 
        total_models = len(MODELS_TO_EVALUATE)
        df_processed = data_df.copy() 
        imputer_strategy_for_pipeline = None 

        if missing_value_strategy == 'drop_rows':
            cols_for_dropna = feature_columns + [target_column] 
            df_processed = data_df[cols_for_dropna].dropna().reset_index(drop=True)
        elif missing_value_strategy in ['mean', 'median', 'most_frequent']:
            imputer_strategy_for_pipeline = missing_value_strategy 
        else:
            raise ValueError(f"Invalid missing value strategy: {missing_value_strategy}")

        X = df_processed[feature_columns]
        y = df_processed[target_column]

        if X.empty or y.empty: raise ValueError("Feature set (X) or target (y) is empty after handling missing values.")

        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if X_train.empty or X_test.empty: raise ValueError("Training or testing data is empty after split.")

        base_preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features, imputer_strategy_for_pipeline if missing_value_strategy != 'drop_rows' else None)
        
        progress_callback(current_progress_steps, 1) # "Preprocessing Data..."
        X_train_processed = base_preprocessor.fit_transform(X_train)
        X_test_processed = base_preprocessor.transform(X_test)
        
        try:
            feature_names_out = base_preprocessor.get_feature_names_out()
        except AttributeError: 
            ohe_feature_names = []
            if 'cat' in base_preprocessor.named_transformers_:
                cat_pipeline = base_preprocessor.named_transformers_['cat']
                if 'onehot' in cat_pipeline.named_steps and categorical_features:
                    try: ohe_feature_names = cat_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
                    except: ohe_feature_names = [f"cat_feat_{i}" for i in range(len(categorical_features))] 
            feature_names_out = numerical_features + ohe_feature_names
            if not feature_names_out or len(feature_names_out) != X_train_processed.shape[1]:
                feature_names_out = [f"feature_{j}" for j in range(X_train_processed.shape[1])]
        
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out)
        
        best_model_pipeline_obj = None # Use a different name to avoid conflict with a potential function name

        for i, model_config in enumerate(MODELS_TO_EVALUATE):
            model_name = model_config['name']
            estimator = model_config['estimator']
            param_grid_for_estimator = {k.replace('regressor__', ''): v for k, v in model_config['param_grid'].items()}
            
            progress_callback(current_progress_steps, i + 2) # +2 for "Validating" and "Preprocessing"
            
            start_time = time.time()
            current_best_estimator_for_model = None; best_params_for_model = {}; best_cv_score_for_model = -float('inf')
            cv_folds = min(5, len(X_train_processed_df) -1 if len(X_train_processed_df) > 1 else 1)

            if len(X_train_processed_df) <= cv_folds or cv_folds < 2 :
                current_best_estimator_for_model = estimator
                current_best_estimator_for_model.fit(X_train_processed_df, y_train)
                y_train_pred_cv = current_best_estimator_for_model.predict(X_train_processed_df)
                best_cv_score_for_model = r2_score(y_train, y_train_pred_cv)
            elif param_grid_for_estimator:
                grid_search = GridSearchCV(estimator, param_grid_for_estimator, cv=cv_folds, scoring='r2', n_jobs=-1, error_score='raise')
                grid_search.fit(X_train_processed_df, y_train)
                current_best_estimator_for_model = grid_search.best_estimator_
                best_params_for_model = grid_search.best_params_
                best_cv_score_for_model = grid_search.best_score_
            else:
                current_best_estimator_for_model = estimator
                current_best_estimator_for_model.fit(X_train_processed_df, y_train)
                y_train_pred_cv = current_best_estimator_for_model.predict(X_train_processed_df)
                best_cv_score_for_model = r2_score(y_train, y_train_pred_cv)
            
            training_time = time.time() - start_time
            y_pred_on_test = current_best_estimator_for_model.predict(X_test_processed_df)
            r_squared = r2_score(y_test, y_pred_on_test)
            mae = mean_absolute_error(y_test, y_pred_on_test)
            mse = mean_squared_error(y_test, y_pred_on_test)
            y_test_np = np.array(y_test)
            non_zero_mask = y_test_np != 0
            mape = np.mean(np.abs((y_test_np[non_zero_mask] - y_pred_on_test[non_zero_mask]) / y_test_np[non_zero_mask])) * 100 if np.any(non_zero_mask) else np.nan
            mape_serializable = None if np.isnan(mape) else mape
            final_pipeline_for_model = Pipeline(steps=[('preprocessor', base_preprocessor), ('regressor', current_best_estimator_for_model)])

            model_result = {
                "Model Type": model_name, "R-squared": r_squared, "MAE": mae, "RMSE": np.sqrt(mse),
                "MAPE": mape_serializable, "Best Hyperparameters": best_params_for_model,
                "Cross-Validation R2": best_cv_score_for_model, "Training Time (s)": training_time,
                "Pipeline": final_pipeline_for_model, 
                "Hyperparameter_Definitions": MODEL_EXPLANATIONS.get(model_name, {}).get('hyperparameters', {})
            }
            all_model_results.append(model_result)

            if r_squared > best_r2:
                best_r2 = r_squared
                best_model_info_dict = model_result
                best_model_pipeline_obj = final_pipeline_for_model
        
        current_progress_idx_offset = 2 + total_models # Validating + Preprocessing + All models
        progress_callback(current_progress_steps, current_progress_idx_offset) # "Evaluating Best Model..."
        
        if best_model_pipeline_obj and not X_train_processed_df.empty and best_model_info_dict:
            best_regressor = best_model_pipeline_obj.named_steps['regressor']
            y_pred_best_model_test = best_regressor.predict(X_test_processed_df)
            
            try:
                actual_vs_predicted_fig = px.scatter(x=y_test, y=y_pred_best_model_test, labels={'x': 'Actual Values', 'y': 'Predicted Values'}, title=f'Actual vs. Predicted ({best_model_info_dict.get("Model Type", "N/A")})')
                actual_vs_predicted_fig.add_trace(go.Scatter(x=[min(y_test.min(), y_pred_best_model_test.min()), max(y_test.max(), y_pred_best_model_test.max())], y=[min(y_test.min(), y_pred_best_model_test.min()), max(y_test.max(), y_pred_best_model_test.max())], mode='lines', name='Ideal Fit', line=dict(dash='dash', color='grey')))
                actual_vs_predicted_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            except Exception as e_avp: print(f"Error Actual vs Predicted plot: {e_avp}")

            try:
                residuals = y_test - y_pred_best_model_test
                residuals_vs_predicted_fig = px.scatter(x=y_pred_best_model_test, y=residuals, labels={'x': 'Predicted Values', 'y': 'Residuals'}, title=f'Residuals vs. Predicted ({best_model_info_dict.get("Model Type", "N/A")})')
                residuals_vs_predicted_fig.add_hline(y=0, line_dash="dash", line_color="grey")
                residuals_vs_predicted_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400)
            except Exception as e_resid: print(f"Error Residuals plot: {e_resid}")

            progress_callback(current_progress_steps, current_progress_idx_offset + 1) # "Calculating SHAP..."
            
            shap_values_for_plot = None; X_shap_for_plot = None
            n_shap_samples = min(100, X_train_processed_df.shape[0])
            if n_shap_samples > 0:
                X_shap_for_plot = shap.sample(X_train_processed_df, n_shap_samples, random_state=42) 
                explainer = None
                try:
                    if isinstance(best_regressor, (RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor)): explainer = shap.TreeExplainer(best_regressor, X_shap_for_plot)
                    else: explainer = shap.KernelExplainer(best_regressor.predict, X_shap_for_plot) 
                    if explainer: shap_values_for_plot = explainer.shap_values(X_shap_for_plot) 
                except Exception as shap_e: print(f"SHAP calculation failed: {shap_e}")
            
            if shap_values_for_plot is not None and X_shap_for_plot is not None and not X_shap_for_plot.empty:
                try:
                    plt.figure() 
                    shap.summary_plot(shap_values_for_plot, X_shap_for_plot, plot_type="dot", show=False, feature_names=X_train_processed_df.columns.tolist())
                    fig_shap = plt.gcf(); fig_shap.set_size_inches(10, max(6, len(X_train_processed_df.columns) * 0.3)) 
                    plt.title(f'SHAP Feature Importance ({best_model_info_dict.get("Model Type", "N/A")})', fontsize=12); plt.xlabel("SHAP value (impact on model output)", fontsize=10); plt.tight_layout()
                    img_buffer_shap = io.BytesIO(); fig_shap.savefig(img_buffer_shap, format="png", bbox_inches="tight"); plt.close(fig_shap)
                    img_buffer_shap.seek(0); img_base64_shap = base64.b64encode(img_buffer_shap.read()).decode()
                    shap_beeswarm_plot_src = f"data:image/png;base64,{img_base64_shap}"
                except Exception as e_shap_plot: print(f"Error SHAP beeswarm plot: {e_shap_plot}")
                try:
                    mean_abs_shap = np.abs(shap_values_for_plot).mean(axis=0)
                    importances_df = pd.DataFrame({'feature': X_shap_for_plot.columns, 'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)
                except Exception as e_imp_df: print(f"Error SHAP importances_df: {e_imp_df}")
        
        current_progress_idx_offset += 2 # For "Evaluating..." and "Calculating SHAP..."

        if analysis_type == 'exploration':
            progress_callback(current_progress_steps, current_progress_idx_offset) # "Finalizing..."
            return all_model_results, best_model_info_dict, importances_df, shap_beeswarm_plot_src, current_progress_steps, actual_vs_predicted_fig, residuals_vs_predicted_fig

        elif analysis_type == 'optimization':
            progress_callback(current_progress_steps, current_progress_idx_offset) # "Generating Surrogate Tree..."
            if not X_train_processed_df.empty and best_model_pipeline_obj:
                best_regressor_for_surrogate = best_model_pipeline_obj.named_steps['regressor']
                y_hat_train_surrogate = best_regressor_for_surrogate.predict(X_train_processed_df)
                interpretable_tree = DecisionTreeRegressor(max_depth=3, random_state=42)
                try:
                    interpretable_tree.fit(X_train_processed_df, y_hat_train_surrogate) 
                    surrogate_tree_text = export_text(interpretable_tree, feature_names=X_train_processed_df.columns.tolist())
                    fig_tree, ax_tree = plt.subplots(figsize=(min(20, 3*len(X_train_processed_df.columns)),10), dpi=100)
                    plot_tree(interpretable_tree, feature_names=X_train_processed_df.columns.tolist(), filled=True, rounded=True, fontsize=min(9, 100/len(X_train_processed_df.columns) if len(X_train_processed_df.columns)>0 else 9), ax=ax_tree, max_depth=3, label='all', impurity=False, proportion=True)
                    plt.title("Surrogate Decision Tree", fontsize=12); img_buffer = io.BytesIO(); fig_tree.savefig(img_buffer, format="png", bbox_inches="tight"); plt.close(fig_tree)
                    img_buffer.seek(0); img_base64 = base64.b64encode(img_buffer.read()).decode(); surrogate_tree_plot_src = f"data:image/png;base64,{img_base64}"
                except Exception as e_tree_plot: print(f"Error surrogate tree plot: {e_tree_plot}")
            
            current_progress_idx_offset +=1 
            progress_callback(current_progress_steps, current_progress_idx_offset) # "Optimizing (Grid Search)..."
            optimal_settings = {col: "N/A" for col in feature_columns} 

            grid_points_num_orig = {col: np.linspace(feature_ranges[col]['min'], feature_ranges[col]['max'], 5) for col in numerical_features if col in feature_ranges and feature_ranges[col]['min'] != feature_ranges[col]['max']}
            grid_points_cat_orig = {col: data_df[col].unique().tolist() for col in categorical_features}
            iterables_for_product_orig = []
            for col in feature_columns:
                if col in grid_points_num_orig: iterables_for_product_orig.append((col, grid_points_num_orig[col]))
                elif col in grid_points_cat_orig: iterables_for_product_orig.append((col, grid_points_cat_orig[col]))
                elif col in numerical_features: iterables_for_product_orig.append((col, [data_df[col].mean()]))
                elif col in categorical_features and not data_df[col].mode().empty: iterables_for_product_orig.append((col, [data_df[col].mode()[0]]))
                else: iterables_for_product_orig.append((col, [None]))
            current_iterables_values_orig = [item[1] for item in iterables_for_product_orig]; current_feature_names_ordered_orig = [item[0] for item in iterables_for_product_orig]
            all_combinations_list_orig = [dict(zip(current_feature_names_ordered_orig, combo_values_orig)) for combo_values_orig in product(*current_iterables_values_orig)]
            
            if not all_combinations_list_orig and not data_df.empty:
                 default_row = {col: data_df[col].mean() if col in numerical_features else (data_df[col].mode()[0] if col in categorical_features and not data_df[col].mode().empty else None) for col in feature_columns}
                 all_combinations_list_orig.append(default_row)

            if all_combinations_list_orig:
                optimization_df_orig_features = pd.DataFrame(all_combinations_list_orig, columns=feature_columns)
                try:
                    optimization_df_processed = base_preprocessor.transform(optimization_df_orig_features)
                    optimization_df_processed = pd.DataFrame(optimization_df_processed, columns=feature_names_out)
                    best_regressor_for_opt = best_model_pipeline_obj.named_steps['regressor']
                    optimization_predictions = best_regressor_for_opt.predict(optimization_df_processed)
                    optimal_idx = np.argmax(optimization_predictions) if optimization_goal == 'maximize' else np.argmin(optimization_predictions)
                    optimal_settings = optimization_df_orig_features.iloc[optimal_idx].to_dict()
                    predicted_target = optimization_predictions[optimal_idx]
                    if not X_test_processed_df.empty:
                        test_predictions_for_std = best_regressor_for_opt.predict(X_test_processed_df)
                        prediction_std = np.std(test_predictions_for_std)
                        predicted_target_lower = predicted_target - 1.96 * prediction_std; predicted_target_upper = predicted_target + 1.96 * prediction_std
                    else: predicted_target_lower, predicted_target_upper = predicted_target, predicted_target
                except Exception as e_opt_pred: print(f"Error optimization prediction: {e_opt_pred}")

            current_progress_idx_offset +=1
            progress_callback(current_progress_steps, current_progress_idx_offset) # "Evaluating Optimal Settings..."
            
            try:
                varied_numerical_features_for_plot_orig = [col for col in numerical_features if col in feature_columns and col in grid_points_num_orig and len(grid_points_num_orig[col]) > 1]
                plot_points_surface = 15
                if len(varied_numerical_features_for_plot_orig) >= 2:
                    x_f, y_f = varied_numerical_features_for_plot_orig[0], varied_numerical_features_for_plot_orig[1]
                    x_r, y_r = np.linspace(feature_ranges[x_f]['min'], feature_ranges[x_f]['max'], plot_points_surface), np.linspace(feature_ranges[y_f]['min'], feature_ranges[y_f]['max'], plot_points_surface)
                    X_g, Y_g = np.meshgrid(x_r, y_r); p_list = []
                    for i0 in range(X_g.shape[0]):
                        for j0 in range(X_g.shape[1]):
                            r0 = {x_f: X_g[i0, j0], y_f: Y_g[i0, j0]}
                            for cf in feature_columns:
                                if cf not in r0: r0[cf] = optimal_settings.get(cf, data_df[cf].mean() if cf in numerical_features else (data_df[cf].mode()[0] if cf in categorical_features and not data_df[cf].mode().empty else None))
                            p_list.append(r0)
                    p_df_s = pd.DataFrame(p_list, columns=feature_columns)
                    if not p_df_s.empty:
                        p_df_s_proc = base_preprocessor.transform(p_df_s); p_df_s_proc = pd.DataFrame(p_df_s_proc, columns=feature_names_out)
                        Z_g_flat = best_model_pipeline_obj.named_steps['regressor'].predict(p_df_s_proc); Z_g = Z_g_flat.reshape(X_g.shape)
                        response_surface_fig = {'data': [go.Contour(x=x_r, y=y_r, z=Z_g, colorscale='Viridis', contours_coloring='heatmap', colorbar_title=target_column)], 'layout': {'title': f'Response Surface for {target_column}', 'xaxis': {'title': x_f}, 'yaxis': {'title': y_f}, 'height': 500, 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}}
                elif len(varied_numerical_features_for_plot_orig) == 1:
                    x_f1 = varied_numerical_features_for_plot_orig[0]; x_r1 = np.linspace(feature_ranges[x_f1]['min'], feature_ranges[x_f1]['max'], plot_points_surface * 2); p_r1 = []
                    for vx1 in x_r1:
                        r1 = {x_f1: vx1}
                        for cf in feature_columns:
                            if cf not in r1: r1[cf] = optimal_settings.get(cf, data_df[cf].mean() if cf in numerical_features else (data_df[cf].mode()[0] if cf in categorical_features and not data_df[cf].mode().empty else None))
                        p_r1.append(r1)
                    p_df1 = pd.DataFrame(p_r1, columns=feature_columns)
                    if not p_df1.empty:
                        p_df1_proc = base_preprocessor.transform(p_df1); p_df1_proc = pd.DataFrame(p_df1_proc, columns=feature_names_out)
                        y_pv = best_model_pipeline_obj.named_steps['regressor'].predict(p_df1_proc)
                        response_surface_fig = {'data': [go.Scatter(x=x_r1, y=y_pv, mode='lines', line_color='#28a745')], 'layout': {'title': f'Response: {target_column} vs {x_f1}', 'xaxis': {'title': x_f1}, 'yaxis': {'title': target_column}, 'height': 500, 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}}
            except Exception as e_resp_surf: print(f"Error response surface plot: {e_resp_surf}")

            current_progress_idx_offset +=1
            progress_callback(current_progress_steps, current_progress_idx_offset) # "Generating Optimization Visualizations..."

            model_info_summary = {}
            if best_model_info_dict: 
                regressor_params = best_model_info_dict['Pipeline'].named_steps['regressor'].get_params()
                hyperparams_to_show = {k_orig: v for k_orig, v in regressor_params.items() if k_orig.split('__')[-1] in MODEL_EXPLANATIONS.get(best_model_info_dict['Model Type'], {}).get('hyperparameters', {})}
                if not hyperparams_to_show and best_model_info_dict.get("Best Hyperparameters"): hyperparams_to_show = best_model_info_dict.get("Best Hyperparameters", {})
                model_info_summary = {
                    "Model Type": best_model_info_dict.get('Model Type', "N/A"), "Hyperparameters": hyperparams_to_show,
                    "Training Data Shape (Original)": X_train.shape if not X_train.empty else "N/A", "Test Data Shape (Original)": X_test.shape if not X_test.empty else "N/A",
                    "Features Used (Original)": feature_columns, "Target Column": target_column, "Missing Value Strategy": missing_value_strategy, 
                    "Optimization Goal": optimization_goal, "Hyperparameter_Definitions": MODEL_EXPLANATIONS.get(best_model_info_dict.get('Model Type', ''), {}).get('hyperparameters', {})}
            
            current_progress_idx_offset +=1
            progress_callback(current_progress_steps, current_progress_idx_offset) # "Finalizing Results..."
            return all_model_results, best_model_info_dict, importances_df, shap_beeswarm_plot_src, current_progress_steps, \
                   optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, \
                   response_surface_fig, surrogate_tree_text, model_info_summary, surrogate_tree_plot_src, \
                   actual_vs_predicted_fig, residuals_vs_predicted_fig
        
    except Exception as main_pipeline_error:
        print(f"CRITICAL ERROR in AutoML pipeline: {main_pipeline_error}\n{traceback.format_exc()}")
        progress_callback(current_progress_steps, len(current_progress_steps) -1) # Mark as finished, possibly error state
        # Ensure all expected return values are provided, even if they are defaults/placeholders
        if analysis_type == 'exploration':
            return all_model_results, best_model_info_dict, importances_df, shap_beeswarm_plot_src, current_progress_steps, actual_vs_predicted_fig, residuals_vs_predicted_fig
        elif analysis_type == 'optimization':
            return all_model_results, best_model_info_dict, importances_df, shap_beeswarm_plot_src, current_progress_steps, \
                   optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, \
                   response_surface_fig, surrogate_tree_text, model_info_summary, surrogate_tree_plot_src, \
                   actual_vs_predicted_fig, residuals_vs_predicted_fig
        else: # Should not happen
            raise main_pipeline_error


def run_exploration_automl(data_df, target_column, feature_columns, missing_value_strategy, progress_callback):
    return run_automl_pipeline(data_df, target_column, feature_columns, missing_value_strategy, progress_callback, 'exploration')

def run_optimization_automl(data_df, target_column, feature_columns, optimization_goal, feature_ranges, missing_value_strategy, progress_callback):
    return run_automl_pipeline(data_df, target_column, feature_columns, missing_value_strategy,
                                 progress_callback, 'optimization',
                                 optimization_goal=optimization_goal, feature_ranges=feature_ranges)


def generate_explanation_llm(analysis_type, results_data, custom_prompt_addition=""):
    prompt_data = {}
    best_model_name = "N/A" 

    if analysis_type == "Exploration":
        best_model_name = results_data.get('best_model_info', {}).get('Model Type', 'N/A')
        prompt_data = {
            'target_column': results_data.get('target_column'),
            'best_model_type': best_model_name,
            'best_model_r_squared': results_data.get('best_model_info', {}).get('R-squared'),
            'top_features_shap': [f"{f['feature']} (Importance: {f['importance']:.3f})" for f in results_data.get('importances', [])[:3]], # Changed to include importance
            'missing_strategy': results_data.get('missing_strategy', 'default')
        }
    elif analysis_type == "Optimization":
        best_model_name = results_data.get('model_info', {}).get('Model Type', 'N/A') # Surrogate model
        prompt_data = {
            'target_column': results_data.get('target_column'),
            'optimization_goal': results_data.get('goal'),
            'surrogate_model_type': best_model_name, 
            'optimal_settings': results_data.get('optimal_settings'),
            'predicted_target_at_optimum': results_data.get('predicted_target'),
            'top_features_in_surrogate_shap': [f"{f['feature']} (Importance: {f['importance']:.3f})" for f in results_data.get('importances', [])[:3]],
            'missing_strategy': results_data.get('missing_strategy', 'default')
        }

    model_desc_for_prompt = MODEL_EXPLANATIONS.get(best_model_name, {}).get('description', 'Standard machine learning model.')
    base_prompt = f"""
    As an expert data scientist analyzing R&D experimental data, provide a detailed explanation of the '{analysis_type}' analysis.
    Key results summary: {json.dumps(prompt_data)}.
    The primary model involved was '{best_model_name}', which is described as: '{model_desc_for_prompt}'.

    For a scientist audience (e.g., synthetic biology, protein engineering), explain:
    1.  The overall objective and methodology of this '{analysis_type}' analysis.
    2.  The significance of the best model ('{best_model_name}') chosen and its performance metrics (R-squared, MAE, RMSE, MAPE). Interpret what these metrics mean in practical terms for this experiment.
    3.  Detailed insights from the SHAP feature importance analysis. Explain what the SHAP plot (typically a beeswarm plot showing feature impact direction and magnitude) reveals about the top features. How do these features influence the target variable?
    4.  Guidance on interpreting the 'Actual vs. Predicted' and 'Residuals vs. Predicted' plots. What should a scientist look for in these plots to assess model performance and trustworthiness?
    5.  If '{analysis_type}' is 'Optimization':
        a.  Explain the role of the surrogate model ('{best_model_name}') in the optimization process.
        b.  Interpret the suggested optimal settings and the predicted target value at these settings, including any confidence intervals.
        c.  Explain how the surrogate decision tree (if provided) and response surface plots help in understanding the optimization landscape.
    6.  Provide clear, actionable insights and next steps for the scientist based on these results. What should they investigate further or try in the lab?

    Focus on clarity, practical implications, and scientific relevance.
    {custom_prompt_addition}
    """.strip()
    print(f"LLM Prompt (simulated): {base_prompt[:800]}...") 
    time.sleep(1) 

    if analysis_type == "Exploration":
        binfo = results_data.get('best_model_info', {})
        bname = binfo.get('Model Type', 'N/A')
        bprops = MODEL_EXPLANATIONS.get(bname, {})
        imp_df = pd.DataFrame(results_data.get('importances', []))
        top_feats = ", ".join(imp_df.head(3)['feature'].tolist()) if not imp_df.empty else "Not available"
        r2, mae, rmse, mape = binfo.get('R-squared',0), binfo.get('MAE',0), binfo.get('RMSE',0), binfo.get('MAPE')
        
        return f"""
        #### **Advanced Exploration Analysis: Unveiling Key Experimental Drivers**

        **Objective & Methodology:**
        The primary goal of this Exploration Analysis was to dissect the experimental data concerning '{results_data.get('target_column', 'N/A')}' and identify the most influential input variables (features). We employed an automated machine learning (AutoML) pipeline that rigorously evaluated a diverse set of regression models. Each model underwent hyperparameter optimization using Grid Search with cross-validation to ensure robustness and prevent overfitting. Data preprocessing was a critical first step, involving a '{str(results_data.get('missing_strategy', 'default')).replace('_', ' ').title()}' strategy for handling missing values, One-Hot Encoding for categorical features, and Standard Scaling for numerical features to bring them to a comparable range.

        **Best Performing Model: {bname}**
        * **Rationale & Nature:** The {bname} model emerged as the top performer. {bprops.get('description', 'It is a powerful and flexible algorithm.')} Its strengths, such as '{bprops.get('pros', 'good general performance and ability to capture complex patterns')}', likely contributed to its success in this dataset.
        * **Performance Deep Dive:**
            * **R-squared (R²): {r2:.3f}**. This indicates that approximately **{r2*100:.1f}%** of the variability in '{results_data.get('target_column', 'N/A')}' can be explained by the input features using this model. A higher R² (closer to 1) suggests a better fit.
            * **Mean Absolute Error (MAE): {mae:.3f}**. On average, the model's predictions for '{results_data.get('target_column', 'N/A')}' are off by this amount. Lower is better.
            * **Root Mean Squared Error (RMSE): {rmse:.3f}**. This is similar to MAE but penalizes larger errors more heavily. It's in the same units as the target variable.
            * **Mean Absolute Percentage Error (MAPE): {f'{mape:.2f}%' if mape is not None else 'N/A'}**. This expresses the average prediction error as a percentage of the actual value, useful for relative error assessment (interpret with caution if actual values are close to zero).

        **Interpreting Diagnostic Plots:**
        * **Actual vs. Predicted Plot:** Ideally, points should cluster tightly around the diagonal (y=x) line. Deviations indicate prediction errors. Look for systematic biases (e.g., consistent over or under-prediction).
        * **Residuals vs. Predicted Plot:** Residuals (actual - predicted) should be randomly scattered around the horizontal zero line. Patterns (e.g., a funnel shape, a curve) suggest issues like heteroscedasticity (non-constant variance of errors) or non-linearity not captured by the model.

        **Key Feature Insights (from SHAP Analysis):**
        The SHAP (SHapley Additive exPlanations) analysis provides a sophisticated view of feature importance. The beeswarm plot (if generated) shows:
        * **Feature Importance:** Features are typically ranked by their mean absolute SHAP value.
        * **Impact Direction & Magnitude:** For each data point (represented by a dot), its color often indicates the original feature value (high/low), and its position on the x-axis shows how much that feature value pushed the prediction higher (positive SHAP value) or lower (negative SHAP value) for that specific point.
        * **Distribution:** The spread of dots for a feature shows the variability of its impact.
        Based on this, the most critical factors influencing '{results_data.get('target_column', 'N/A')}' appear to be: **{top_feats}**. Examine the SHAP plot to understand if high values of these features tend to increase or decrease the target.

        **Actionable Scientific Insights & Next Steps:**
        1.  **Focus on {top_feats}:** These variables warrant the most attention in subsequent experimental design and mechanistic investigation.
        2.  **Validate SHAP Directions:** Correlate the SHAP impact directions with existing scientific knowledge. Do high concentrations of a reagent increasing the target (as per SHAP) make sense biochemically?
        3.  **Refine Experiments:** Use these insights to design more targeted experiments. If a feature has a strong positive impact, explore its upper range further (if feasible).
        4.  **Data Quality for Key Features:** Ensure precise and accurate measurements for these high-impact features, as noise here can significantly affect model reliability.
        5.  **Model Limitations:** While '{bname}' performed best, remember all models are simplifications. The R² value gives a sense of unexplained variance. Consider if unmeasured factors might play a role.
        {custom_prompt_addition}
        """

    elif analysis_type == "Optimization":
        s_info = results_data.get('model_info', {}) 
        s_name = s_info.get('Model Type', 'N/A')
        s_props = MODEL_EXPLANATIONS.get(s_name, {})
        s_perf = results_data.get('best_model_info', {}) 
        s_r2 = s_perf.get('R-squared', 0.0)
        
        opt_set = results_data.get('optimal_settings', {})
        opt_str = ", ".join([f"**{k}**: {v:.3f}" if isinstance(v, (float, np.number)) and not np.isnan(v) else f"**{k}**: {v}" for k,v in opt_set.items()])
        goal_v = "maximize" if results_data.get('goal') == "maximize" else "minimize"
        pred_val, pred_low, pred_upp = results_data.get('predicted_target', np.nan), results_data.get('predicted_target_lower', np.nan), results_data.get('predicted_target_upper', np.nan)
        
        imp_opt_df = pd.DataFrame(results_data.get('importances', []))
        top_shap_opt = ", ".join(imp_opt_df.head(3)['feature'].tolist()) if not imp_opt_df.empty else "N/A"

        return f"""
        #### **Advanced Optimization Analysis: Pinpointing Optimal Experimental Conditions**

        **Objective & Methodology:**
        The aim of this Optimization Analysis was to identify experimental settings that **{goal_v}** your target variable: '{results_data.get('target_column', 'N/A')}'. This was achieved by first selecting a high-performing surrogate model, the **{s_name}** (R²: {s_r2:.3f} during its initial evaluation), to represent the complex relationships in your data. This surrogate model, chosen for its predictive accuracy ({s_props.get('description', 'It is a robust modeling technique.')}), then enabled an efficient search across a grid of possible input parameter combinations. This in-silico experimentation predicts outcomes without the immediate need for extensive lab work.

        **Suggested Optimal Settings & Predicted Outcome:**
        * **Optimal Conditions:** {opt_str}
        * **Predicted Target Value:** Under these settings, the model predicts '{results_data.get('target_column', 'N/A')}' to be approximately **{pred_val:.3f}**.
        * **Confidence Interval (95%):** The likely range for this prediction is between {pred_low:.3f} and {pred_upp:.3f}. This interval provides a measure of uncertainty around the prediction.

        **Interpreting Surrogate Model Insights & Visualizations:**
        * **Surrogate Model ({s_name}):** This model's performance (Actual vs. Predicted, Residuals plots shown for its evaluation run) gives confidence in its ability to guide the optimization.
        * **SHAP Analysis on Surrogate:** The SHAP beeswarm plot for the surrogate model (if shown) indicates that **{top_shap_opt}** were key drivers of its predictions within the explored optimization space. Understanding these can help rationalize why the optimal settings work.
        * **Surrogate Decision Tree:** The visual decision tree (if provided) offers a simplified, interpretable approximation of the surrogate model's logic, highlighting key decision rules based on (transformed) feature values.
        * **Response Surface Plot:** This plot (if applicable, for 1 or 2 varied numerical inputs) visually maps how the predicted target changes as key inputs are varied, helping to understand the sensitivity and landscape around the optimum.

        **Actionable Scientific Insights & Next Steps:**
        1.  **Crucial Lab Validation:** The **most important next step** is to experimentally validate these predicted optimal settings in your laboratory. Models guide, experiments confirm.
        2.  **Sensitivity & Robustness:** Perform a few experiments slightly varying the conditions around the suggested optimum (e.g., +/- 10% for key numerical inputs). This helps assess how sensitive the outcome is to small changes and identifies a robust operating window.
        3.  **Consider Practicalities:** Always evaluate the suggested settings against practical laboratory constraints, safety, cost, and time. Minor adjustments might be needed.
        4.  **Iterative Refinement:** If validation is promising, these results can inform the design of a new, more focused experimental set. The model can be retrained with new data for further iterative optimization.
        5.  **Mechanism Exploration:** Why do these settings work? Use the SHAP insights and your scientific expertise to hypothesize underlying mechanisms.
        {custom_prompt_addition}
        """
    return "Placeholder explanation: LLM analysis type or data missing."


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME],
                  meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
                  suppress_callback_exceptions=True)
app.title = "R&D Experiment Analysis Platform"

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename: df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename: df = pd.read_excel(io.BytesIO(decoded))
        else: return html.Div(['Invalid file type. Please upload CSV or Excel.'])
        return df
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return html.Div([f'There was an error processing this file: {str(e)}'])

def process_dataframe_for_ui(df, filename_display):
    display_df = df.head(50) if len(df) > 50 else df
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in display_df.columns],
        page_size=8, style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 'minWidth': '100px', 'width': '150px', 'maxWidth': '200px', 'whiteSpace': 'normal', 'textOverflow': 'ellipsis'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'},
        style_data={'borderBottom': '1px solid #dee2e6'}, sort_action='native', filter_action='native',
        tooltip_data=[{column: {'value': str(value), 'type': 'markdown'} for column, value in row.items()} for row in display_df.to_dict('records')],
        tooltip_duration=None
    )
    column_options = [{'label': col, 'value': col} for col in df.columns]
    role_assignment_ui = html.Div([
        dbc.Label("Input Variables (X):", html_for='dropdown-input-vars', className="fw-bold mt-3"),
        dcc.Dropdown(id='dropdown-input-vars', options=column_options, multi=True, placeholder="Select features/factors", className="mb-2"),
        dbc.Label("Target Output Variable (Y):", html_for='dropdown-output-var', className="fw-bold"),
        dcc.Dropdown(id='dropdown-output-var', options=column_options, multi=False, placeholder="Select the single output to analyze/optimize", className="mb-2"),
        dbc.Label("Missing Value Strategy:", html_for='dropdown-missing-strategy', className="fw-bold"),
        dcc.Dropdown(id='dropdown-missing-strategy', options=[
            {'label': 'Drop Rows with Missing Data', 'value': 'drop_rows'},
            {'label': 'Impute with Mean (Numeric only)', 'value': 'mean'},
            {'label': 'Impute with Median (Numeric only)', 'value': 'median'},
            {'label': 'Impute with Mode (Numeric & Categorical)', 'value': 'most_frequent'},
        ], value='drop_rows', clearable=False, className="mb-3"),
        dbc.Label("Ignore Columns (Optional):", html_for='dropdown-ignore-vars', className="fw-bold"),
        dcc.Dropdown(id='dropdown-ignore-vars', options=column_options, multi=True, placeholder="Select columns to exclude", className="mb-3"),
        dbc.Button(children=[html.I(className="fas fa-cogs me-2"), "Confirm Setup & Proceed"], id="btn-confirm-setup", color="primary", className="mt-3 w-100 btn-lg")
    ])
    status_message = dbc.Alert(f"Loaded: {filename_display}. Displaying first {len(display_df)} of {len(df)} rows.", color="success", duration=4000) if len(df) > 50 else dbc.Alert(f"Loaded: {filename_display}", color="success", duration=4000)
    stored_data = df.to_json(date_format='iso', orient='split')
    return status_message, table, stored_data, role_assignment_ui

app.layout = dbc.Container(fluid=True, className="bg-light min-vh-100", children=[
    dcc.Store(id='store-raw-data'), dcc.Store(id='store-column-roles'),
    dcc.Store(id='store-exploration-results'), dcc.Store(id='store-optimization-results'),
    dcc.Store(id='store-current-analysis-type'), dcc.Store(id='store-progress-text'), 
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0, disabled=True), 
    dcc.Store(id='store-progress-steps', data=[]), 
    dcc.Store(id='store-current-step-index', data=0), 
    dcc.Store(id='active-tab-store', data='tab-data-upload'),

    dbc.Row(dbc.Col(html.H1(children=[html.I(className="fas fa-flask me-2"), "R&D Experiment Analysis Platform"], className="text-center my-4 display-5 text-primary fw-bold"), width=12)),
    dbc.Row(dbc.Col(dbc.Nav([
        dbc.NavItem(dbc.NavLink("1. Data & Setup", active=True, href="#", id="nav-data-setup", className="fw-medium fs-5 p-2")),
        dbc.NavItem(dbc.NavLink("2. Analysis & Results", active=False, href="#", id="nav-analysis", disabled=True, className="fw-medium fs-5 p-2")),
        dbc.NavItem(dbc.NavLink("3. AI Insights & Suggestions", active=False, href="#", id="nav-suggestions", disabled=True, className="fw-medium fs-5 p-2")),
    ], pills=True, className="nav-pills justify-content-center mb-4 shadow-sm bg-white rounded p-2"), width=12)),
    dbc.Row(dbc.Col(html.Div(id='global-progress-message'), width=12)), # Initially empty
    dbc.Card(dbc.CardBody([
        html.Div(id="tab-content", className="p-3"),
        dbc.Button(children=[html.I(className="fas fa-lightbulb me-2"),"Proceed to AI Insights (Exploration)"], id="btn-goto-suggestions-expl", color="info", className="mt-3 d-block mx-auto", style={'display': 'none', 'width':'fit-content'}),
        dbc.Button(children=[html.I(className="fas fa-lightbulb me-2"),"Proceed to AI Insights (Optimization)"], id="btn-goto-suggestions-opt", color="success", className="mt-3 d-block mx-auto", style={'display': 'none', 'width':'fit-content'})
    ]), className="mt-3 shadow-lg rounded"),
    dbc.Modal([dbc.ModalHeader(id="modal-model-name"), dbc.ModalBody(id="modal-model-desc"), dbc.ModalFooter(dbc.Button("Close", id="close-model-modal", className="ms-auto", n_clicks=0))], id="model-explanation-modal", is_open=False, size="lg", scrollable=True),
    dbc.Row(dbc.Col(html.P("Powered by AutoML and Generative AI", className="text-center text-muted mt-5 small"), width=12))
])

@app.callback(
    Output('tab-content', 'children'),
    Input('active-tab-store', 'data'),
    State('store-raw-data', 'data'),
    State('store-column-roles', 'data'),
    State('store-current-analysis-type', 'data'),
    State('store-exploration-results', 'data'),
    State('store-optimization-results', 'data'),
)
def render_tab_content(active_tab, raw_data_json, column_roles, analysis_type, exploration_data, optimization_data):
    data_setup_content = dbc.Card(dbc.CardBody([dbc.Row([
        dbc.Col([
            html.H4(children=[html.I(className="fas fa-upload me-2"), "Upload Experiment Data"], className="mb-3 text-secondary"),
            dcc.Upload(id='upload-data', children=html.Div(['Drag & Drop or ', html.A('Select Files')]), style={'width': '100%', 'height': '80px', 'lineHeight': '80px', 'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px', 'textAlign': 'center', 'margin': '10px 0', 'backgroundColor': '#f8f9fa', 'borderColor': '#adb5bd'}, multiple=False, className="mb-2"),
            html.Div("Or", className="text-center my-2 small text-muted"),
            dbc.Button(children=[html.I(className="fas fa-database me-2"),"Load Demo Data"], id="btn-load-demo", color="info", outline=True, className="w-100 mb-3"),
            html.Div(id='output-data-upload-status', className="mt-2"),
            html.Div(id='output-datatable-div', className="mt-3", style={'maxHeight': '350px', 'overflowY': 'auto', 'overflowX': 'auto', 'border': '1px solid #dee2e6', 'borderRadius': '5px'})
        ], md=7, className="p-3 border-end"),
        dbc.Col([
            html.H4(children=[html.I(className="fas fa-sliders-h me-2"), "Experiment Configuration"], className="mb-3 text-secondary"),
            dbc.Label("Experiment Type:", html_for='dropdown-experiment-type', className="fw-bold"),
            dcc.Dropdown(id='dropdown-experiment-type',
                         options=[
                             {'label': 'Exploration (Identify Key Factors)', 'value': 'exploration'},
                             {'label': 'Optimization (Find Best Settings)', 'value': 'optimization'}
                         ],
                         value='exploration', clearable=False, className="mb-3"),
            html.Div(id='column-role-assignment-div')
        ], md=5, className="p-3")])]), className="mt-0 shadow-sm")

    if active_tab == 'tab-data-upload':
        return data_setup_content
    elif active_tab == 'tab-analysis':
        return render_analysis_tab_content(active_tab, analysis_type, column_roles, raw_data_json)
    elif active_tab == 'tab-suggestions':
        return render_suggestions_tab_content(active_tab, 0, analysis_type, exploration_data, optimization_data, None)
    return html.Div("Select a tab")

@app.callback(
    [Output('nav-data-setup', 'active'), Output('nav-analysis', 'active'), Output('nav-suggestions', 'active'),
     Output('nav-analysis', 'disabled'), Output('nav-suggestions', 'disabled')],
    [Input('active-tab-store', 'data')],
    [State('nav-analysis', 'disabled'), State('nav-suggestions', 'disabled')]
)
def update_nav_links(active_main_tab, nav_analysis_disabled, nav_suggestions_disabled):
    return active_main_tab == 'tab-data-upload', \
           active_main_tab == 'tab-analysis', \
           active_main_tab == 'tab-suggestions', \
           nav_analysis_disabled, \
           nav_suggestions_disabled

@app.callback(
    Output('active-tab-store', 'data', allow_duplicate=True),
    [Input('nav-data-setup', 'n_clicks'), Input('nav-analysis', 'n_clicks'), Input('nav-suggestions', 'n_clicks')],
    [State('nav-analysis', 'disabled'), State('nav-suggestions', 'disabled')],
    prevent_initial_call=True
)
def switch_tabs_from_nav(n_data, n_analysis, n_suggestions, analysis_nav_disabled, suggestions_nav_disabled):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'nav-data-setup': return 'tab-data-upload'
    elif button_id == 'nav-analysis' and not analysis_nav_disabled: return 'tab-analysis'
    elif button_id == 'nav-suggestions' and not suggestions_nav_disabled: return 'tab-suggestions'
    return dash.no_update

@app.callback(
    Output('global-progress-message', 'children'),
    [Input('progress-interval', 'n_intervals'), 
     Input('store-progress-steps', 'data'),      
     Input('store-current-step-index', 'data')], 
    [State('progress-interval', 'disabled'), 
     State('store-current-analysis-type', 'data')] 
)
def update_progress_display(n_intervals, progress_steps, current_step_index, interval_disabled, analysis_type):
    if not progress_steps: 
        return html.Div() # Initially empty, or a very minimal placeholder if preferred

    log_display_elements = []
    is_complete = current_step_index >= (len(progress_steps) - 1) if progress_steps else True
    analysis_name = str(analysis_type).capitalize() if analysis_type else "Analysis"

    for i, step_text in enumerate(progress_steps):
        if i < current_step_index:
            prefix = html.I(className="fas fa-check-circle text-success me-2")
            text_class = "text-muted"
        elif i == current_step_index and not is_complete:
            prefix = dbc.Spinner(size="sm", color="primary", className="me-2")
            text_class = "text-primary fw-bold"
        elif i == current_step_index and is_complete: 
            prefix = html.I(className="fas fa-check-circle text-success me-2")
            text_class = "text-success fw-bold"
        elif i > current_step_index : 
            prefix = html.I(className="far fa-circle text-secondary me-2") 
            text_class = "text-secondary"
        else: 
            prefix = html.I(className="fas fa-check-circle text-success me-2") 
            text_class = "text-muted"
        log_display_elements.append(html.Div([prefix, html.Span(step_text)], className=f"d-flex align-items-center mb-1 small {text_class}"))

    progress_percentage = ((current_step_index + 1) / len(progress_steps)) * 100 if len(progress_steps) > 0 else 0
    if is_complete:
        progress_percentage = 100
        if log_display_elements and current_step_index == len(progress_steps) -1 :
             log_display_elements[-1] = html.Div([html.I(className="fas fa-check-circle text-success me-2"), html.Span(progress_steps[-1])], className="d-flex align-items-center mb-1 small text-success fw-bold")

    header_icon = html.I(className="fas fa-check-circle text-success me-2") if is_complete else dbc.Spinner(size="sm", color="primary", className="me-2 align-middle")
    header_text = f"{analysis_name} Complete!" if is_complete else f"{analysis_name} in Progress..."
    progress_bar_label = f"Step {min(current_step_index + 1, len(progress_steps))}/{len(progress_steps)}" if progress_steps else "0/0"
    if is_complete and progress_steps:
        progress_bar_label = f"Completed {len(progress_steps)}/{len(progress_steps)} steps"

    return dbc.Card(dbc.CardBody([
        html.Div([header_icon, html.Span(header_text, className="fw-bold h5", style={'color': '#495057'})], className="d-flex align-items-center mb-2 justify-content-center"),
        dbc.Progress(value=progress_percentage, striped=not is_complete, animated=not is_complete, label=progress_bar_label, style={'height': '25px', 'fontSize': '0.9rem'}, className="mb-3"),
        html.Div(log_display_elements, className="mt-2 text-start", style={'maxHeight': '250px', 'overflowY': 'auto', 'fontSize': '0.8rem', 'fontFamily': 'monospace', 'backgroundColor': '#f8f9fa', 'border': '1px solid #eee', 'padding': '10px', 'borderRadius':'5px'})
    ]), className="shadow-sm")


@app.callback(
    [Output('output-data-upload-status', 'children'), Output('output-datatable-div', 'children'),
     Output('store-raw-data', 'data'), Output('column-role-assignment-div', 'children')],
    [Input('upload-data', 'contents'), Input('btn-load-demo', 'n_clicks')],
    State('upload-data', 'filename'), prevent_initial_call=True
)
def handle_data_input(contents, n_clicks_demo, filename):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df, filename_display = None, None
    if triggered_id == 'upload-data' and contents:
        df_or_error = parse_contents(contents, filename)
        if isinstance(df_or_error, html.Div): return dbc.Alert(df_or_error.children[0], color="danger", duration=4000), "", None, ""
        df, filename_display = df_or_error, filename
    elif triggered_id == 'btn-load-demo' and n_clicks_demo:
        try:
            df = pd.read_csv(io.StringIO(DEMO_DATA_CSV_STRING))
            filename_display = "lfa_random_data.csv (Demo)"
        except Exception as e: return dbc.Alert(f"Error loading demo data: {e}", color="danger", duration=4000), "", None, ""
    if df is None: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    return process_dataframe_for_ui(df, filename_display)

@app.callback(
    [Output('store-column-roles', 'data'), Output('active-tab-store', 'data', allow_duplicate=True),
     Output('nav-analysis', 'disabled', allow_duplicate=True),
     Output('nav-suggestions', 'disabled', allow_duplicate=True),
     Output('output-data-upload-status', 'children', allow_duplicate=True), Output('store-current-analysis-type', 'data')],
    Input('btn-confirm-setup', 'n_clicks'),
    [State('dropdown-experiment-type', 'value'), State('dropdown-input-vars', 'value'), State('dropdown-output-var', 'value'),
     State('dropdown-ignore-vars', 'value'), State('dropdown-missing-strategy', 'value'), State('store-raw-data', 'data')],
    prevent_initial_call=True
)
def confirm_setup_and_proceed(n_clicks, exp_type, inputs, output_var, ignored, missing_strategy, raw_data_json):
    if not n_clicks or not raw_data_json: return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    alert_msg = None
    if not exp_type: alert_msg = dbc.Alert("⚠️ Please select an Experiment Type.", color="warning", duration=4000)
    elif not inputs or not output_var: alert_msg = dbc.Alert("⚠️ Select input(s) and target.", color="warning", duration=4000)
    elif output_var in inputs: alert_msg = dbc.Alert("⚠️ Target cannot be an Input.", color="danger", duration=4000)
    
    if alert_msg:
        return dash.no_update, dash.no_update, True, True, alert_msg, dash.no_update

    column_roles = {'inputs': inputs, 'target_for_analysis': output_var, 'ignored': ignored or [], 'missing_strategy': missing_strategy}
    return column_roles, "tab-analysis", False, True, dbc.Alert(f"Setup Confirmed for {exp_type.capitalize()}.", color="success", duration=3000), exp_type

@app.callback(
    Output('analysis-content-div', 'children'), 
    Input('active-tab-store', 'data'), 
    [State('store-current-analysis-type', 'data'), State('store-column-roles', 'data'), State('store-raw-data', 'data')],
)
def render_analysis_tab_content(active_tab, analysis_type, column_roles, raw_data_json): 
    if active_tab != 'tab-analysis': 
        return html.Div([
            # Ensure placeholders for tables exist if they are inputs to global callbacks
            html.Div(id='exploration-results-area', style={'display':'none'}),
            html.Div(id='optimization-results-area', style={'display':'none'}),
            # Add empty tables with IDs if they are direct inputs to callbacks
            # This ensures their IDs are in the layout from the start.
            html.Div(dash_table.DataTable(id='model-comparison-table', data=[], columns=[]), style={'display': 'none'}),
            html.Div(dash_table.DataTable(id='surrogate-candidates-table', data=[], columns=[]), style={'display': 'none'})
        ])


    if not raw_data_json:
        return dbc.Alert("No data uploaded. Please go back to 'Data & Setup' tab.", color="warning")
    
    if not column_roles or not isinstance(column_roles, dict) or \
       not column_roles.get('target_for_analysis') or not column_roles.get('inputs') or not analysis_type:
        return dbc.Alert("Column roles or analysis type not defined. Please complete setup on 'Data & Setup' tab.", color="warning")

    target_column = column_roles.get('target_for_analysis')
    
    try:
        df = pd.read_json(raw_data_json, orient='split')
        if target_column not in df.columns:
            return dbc.Alert(f"Error: Target column '{target_column}' not found in the uploaded data.", color="danger")
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            return dbc.Alert(f"Error: Target column '{target_column}' must be numeric for regression analysis.", color="danger")
        
        for col in column_roles.get('inputs', []):
            if col not in df.columns:
                 return dbc.Alert(f"Error: Input column '{col}' not found in the uploaded data.", color="danger")
            if not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
                return dbc.Alert(f"Error: Input column '{col}' has an unsupported data type ({df[col].dtype}). Please ensure features are numeric, string/object (for categorical), or boolean.", color="danger")
    except Exception as e_df_check:
        return dbc.Alert(f"Error processing data columns: {str(e_df_check)}", color="danger")


    exploration_ui = html.Div([
        html.H3(f"Exploration Analysis: Understanding '{target_column}'", className="mb-3 text-info"),
        html.P("This analysis will evaluate multiple machine learning models to identify which input variables (features) have the most significant impact on your selected target output. It helps in understanding key drivers and relationships within your data."),
        dbc.Button(children=[html.I(className="fas fa-rocket me-2"), "Run Exploration AutoML"], id="btn-run-exploration-automl", color="info", className="my-3 btn-lg w-100 shadow"),
        dcc.Loading(id="loading-exploration", type="default", children=[html.Div(id="exploration-results-area")]) # Populated by callback
    ])
    
    optimization_ui = html.Div([
        html.H3(f"Optimization Analysis: Targeting '{target_column}'", className="mb-3 text-success"),
        html.P(f"This analysis aims to find the optimal combination of input variable settings to either maximize or minimize your selected target output: '{target_column}'. It uses the best model identified from an initial evaluation to predict outcomes over a range of settings."),
        dbc.Label("Optimization Goal:", className="fw-bold"),
        dcc.Dropdown(id='dropdown-optimization-goal', options=[{'label': 'Maximize Target', 'value': 'maximize'}, {'label': 'Minimize Target', 'value': 'minimize'}], value='maximize', clearable=False, className="mb-3"),
        html.P("Note: For numerical inputs, ranges for optimization are inferred from your data's min/max values. Categorical inputs are tested using their unique values present in the dataset.", className="small text-muted"),
        dbc.Button(children=[html.I(className="fas fa-bullseye me-2"), "Run Optimization AutoML"], id="btn-run-optimization-automl", color="success", className="my-3 btn-lg w-100 shadow"),
        dcc.Loading(id="loading-optimization", type="default", children=[html.Div(id="optimization-results-area")]) # Populated by callback
    ])

    # Always render placeholders for the DataTables so their IDs exist in the layout for the modal callback
    # These will be overwritten by the actual tables when results are generated.
    placeholder_expl_table = html.Div(dash_table.DataTable(id='model-comparison-table', data=[], columns=[]), style={'display': 'none'})
    placeholder_opt_table = html.Div(dash_table.DataTable(id='surrogate-candidates-table', data=[], columns=[]), style={'display': 'none'})


    if analysis_type == 'exploration':
        return html.Div([exploration_ui, placeholder_opt_table])
    elif analysis_type == 'optimization':
        return html.Div([optimization_ui, placeholder_expl_table])
    
    return dbc.Alert("Selected analysis type not recognized or setup incomplete.", color="warning")


@app.callback(
    [Output('exploration-results-area', 'children'), Output('store-exploration-results', 'data'),
     Output('nav-suggestions', 'disabled', allow_duplicate=True),
     Output('store-progress-text', 'data', allow_duplicate=True), 
     Output('progress-interval', 'disabled', allow_duplicate=True), 
     Output('store-progress-steps', 'data', allow_duplicate=True), 
     Output('store-current-step-index', 'data', allow_duplicate=True), 
     Output('btn-goto-suggestions-expl', 'style', allow_duplicate=True)],
    Input('btn-run-exploration-automl', 'n_clicks'),
    [State('store-raw-data', 'data'), State('store-column-roles', 'data')],
    prevent_initial_call=True
)
def run_exploration_analysis_callback(n_clicks, raw_data_json, column_roles):
    if not n_clicks:
        return dash.no_update, dash.no_update, True, None, True, [], 0, {'display': 'none'}

    initial_progress_steps = [f"Validating Inputs & Data..."]
    initial_progress_steps.append(f"Preprocessing Data...")
    initial_progress_steps.extend([f"Training {model['name']} ({i+1}/{len(MODELS_TO_EVALUATE)})" for i, model in enumerate(MODELS_TO_EVALUATE)])
    initial_progress_steps.extend(["Evaluating Best Model...", "Calculating SHAP Values for Best Model...", "Finalizing Results..."])
    
    _captured_progress_steps_list = list(initial_progress_steps) 
    _captured_current_idx = 0

    def _progress_updater_local(steps_list, current_idx_val):
        nonlocal _captured_progress_steps_list, _captured_current_idx
        _captured_progress_steps_list = list(steps_list) 
        _captured_current_idx = current_idx_val
        if steps_list and current_idx_val < len(steps_list):
             print(f"Expl Progress (Internal): Step {current_idx_val + 1}/{len(steps_list)}: {steps_list[current_idx_val]}")
    
    if raw_data_json is None:
        return dbc.Alert("No data uploaded.", color="danger"), None, True, "Error: No data", True, initial_progress_steps, 0, {'display': 'none'}
    if column_roles is None or not isinstance(column_roles, dict) or \
       not column_roles.get('target_for_analysis') or not column_roles.get('inputs'):
        return dbc.Alert("Column roles not defined.", color="danger"), None, True, "Error: Setup incomplete", True, initial_progress_steps, 0, {'display': 'none'}

    df = pd.read_json(raw_data_json, orient='split')
    target_column = column_roles['target_for_analysis']
    feature_columns = column_roles['inputs']
    missing_strategy = column_roles.get('missing_strategy', 'drop_rows')
    current_step_text_for_store = "Initiating Exploration Analysis..." 

    try:
        all_model_results, best_model_info_dict, importances, shap_beeswarm_plot_src, \
        final_progress_steps_from_pipeline, actual_vs_pred_fig, resid_fig = \
            run_exploration_automl(df, target_column, feature_columns, missing_strategy, _progress_updater_local)
        
        _captured_progress_steps_list = list(final_progress_steps_from_pipeline)
        _captured_current_idx = len(_captured_progress_steps_list) - 1 
        current_step_text_for_store = "Exploration Complete!"
        
    except Exception as e:
        print(f"Error Exploration AutoML: {e}\n{traceback.format_exc()}")
        error_message = f"Exploration Error: {str(e)}"
        current_step_text_for_store = f"Error: {str(e)}"
        # Ensure _captured_current_idx is valid for _captured_progress_steps_list
        if not _captured_progress_steps_list: _captured_progress_steps_list = ["Error occurred"]
        _captured_current_idx = min(_captured_current_idx, len(_captured_progress_steps_list) - 1)

        return dbc.Alert(error_message, color="danger"), \
               None, True, current_step_text_for_store, True, \
               _captured_progress_steps_list, _captured_current_idx, {'display': 'none'}

    best_model_name = best_model_info_dict.get('Model Type', 'N/A')
    hyperparam_defs = best_model_info_dict.get('Hyperparameter_Definitions', {})

    model_def_accordion = dbc.Accordion([
        dbc.AccordionItem([
            html.P(MODEL_EXPLANATIONS.get(best_model_name, {}).get('description', 'No description available.')),
            html.H6("Key Strengths:", className="mt-2"), html.P(MODEL_EXPLANATIONS.get(best_model_name, {}).get('pros', 'N/A')),
            html.H6("Potential Weaknesses:", className="mt-2"), html.P(MODEL_EXPLANATIONS.get(best_model_name, {}).get('cons', 'N/A')),
        ], title=f"About the {best_model_name} Model")
    ], start_collapsed=True, className="mb-3")

    hyperparam_explanations_div = [html.H6(f"Key Hyperparameters for {best_model_name}:", className="mt-3 mb-2")]
    best_hyperparams_values = best_model_info_dict.get("Best Hyperparameters", {})
    if best_hyperparams_values:
        for p_name_full, p_val in best_hyperparams_values.items():
            clean_p_name = p_name_full.split('__')[-1]
            explanation = hyperparam_defs.get(clean_p_name, "No specific explanation available.")
            hyperparam_explanations_div.append(html.P([html.Strong(f"{clean_p_name}:"), f" {p_val} - {explanation}"], className="small"))
    else:
        hyperparam_explanations_div.append(html.P("No hyperparameters tuned or available for this model.", className="small"))

    best_model_details_card = dbc.Card(dbc.CardBody([
        html.H5(f"Details for Best Model: {best_model_name}", className="card-title text-info"),
        model_def_accordion,
        html.P(f"Target: {target_column} | Missing Value Strategy: {missing_strategy.replace('_', ' ').title()}"),
        *hyperparam_explanations_div
    ]), className="mb-4 shadow-sm bg-light")

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Best Model R²", className="card-title text-muted small"), html.P(f"{best_model_info_dict.get('R-squared', 0):.3f}", className="card-text fs-4 text-info fw-bold")])), md=3, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Best Model MAE", className="card-title text-muted small"), html.P(f"{best_model_info_dict.get('MAE', 0):.3f}", className="card-text fs-4 text-info fw-bold")])), md=3, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Best Model RMSE", className="card-title text-muted small"), html.P(f"{best_model_info_dict.get('RMSE', 0):.3f}", className="card-text fs-4 text-info fw-bold")])), md=3, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Top Feature (SHAP)", className="card-title text-muted small"), html.P(f"{importances['feature'].iloc[0] if not importances.empty else 'N/A'}", className="card-text fs-5 text-info fw-bold text-truncate")])), md=3, className="mb-2")
    ], className="mb-3 g-3")

    comparison_table_data = []
    for model_res in all_model_results:
        model_type = model_res["Model Type"]
        row = {
            "Model": model_type, "R²": f"{model_res.get('R-squared', 0):.3f}", "MAE": f"{model_res.get('MAE', 0):.3f}",
            "RMSE": f"{model_res.get('RMSE', 0):.3f}", "MAPE": f"{model_res.get('MAPE', float('nan')):.2f}%" if model_res.get('MAPE') is not None else "N/A",
            "CV R²": f"{model_res.get('Cross-Validation R2', 0):.3f}", "Time (s)": f"{model_res.get('Training Time (s)', 0):.2f}",
            "Details_action": "View" 
        }
        comparison_table_data.append(row)

    model_comparison_table_component = html.Div([
        html.H5("All Models Evaluated", className="mt-4 mb-2 text-secondary"),
        html.P(f"Best model: **{best_model_name}**. Click 'View' in the Details column for model information.", className="mb-3 small"),
        dash_table.DataTable(id='model-comparison-table',
                             columns=[ {"name": col_name, "id": col_id} for col_id, col_name in 
                                        [("Model","Model"), ("R²","R²"), ("MAE","MAE"), ("RMSE","RMSE"), ("MAPE","MAPE"), 
                                         ("CV R²","CV R²"), ("Time (s)","Time (s)"), ("Details_action","Details")]],
                             data=comparison_table_data, sort_action='native', page_size=len(MODELS_TO_EVALUATE)+1 if MODELS_TO_EVALUATE else 1,
                             style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '0.85rem'},
                             style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
                             style_data_conditional=[
                                 {'if': {'filter_query': '{Model} = "' + best_model_name + '"'}, 'backgroundColor': '#d1ecf1', 'fontWeight': 'bold'},
                                 {'if': {'column_id': 'Details_action'}, 'cursor': 'pointer', 'color': 'blue', 'textDecoration': 'underline'}
                             ]),
    ], className="mb-4")
    
    shap_plot_display = html.Div([
        html.H5("SHAP Feature Importance & Impact", className="mt-4 mb-2 text-secondary"),
        html.P("This plot shows how much each feature contributes to the model's predictions. Dots to the right increase the prediction, left decrease. Color often indicates feature value (red=high, blue=low).", className="small text-muted"),
        html.Img(src=shap_beeswarm_plot_src, style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'}) if shap_beeswarm_plot_src else dbc.Alert("SHAP plot could not be generated.", color="warning")
    ], className="mb-4 text-center")

    results_layout = html.Div([
        dbc.Alert(f"Exploration AutoML complete for '{target_column}'.", color="info", className="mt-2"),
        kpi_cards, 
        model_comparison_table_component, 
        best_model_details_card,
        dbc.Row([
            dbc.Col(dcc.Graph(id='actual-vs-predicted-plot', figure=actual_vs_pred_fig if actual_vs_pred_fig else go.Figure()), md=6),
            dbc.Col(dcc.Graph(id='residuals-vs-predicted-plot', figure=resid_fig if resid_fig else go.Figure()), md=6)
        ], className="mb-4"),
        shap_plot_display
    ])

    cleaned_best_model_info = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in best_model_info_dict.items() if k != 'Pipeline'}
    cleaned_all_model_results = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items() if k != 'Pipeline'} for m in all_model_results]
    results_data_for_store = {
        'all_model_results': cleaned_all_model_results, 'best_model_info': cleaned_best_model_info,
        'importances': importances.to_dict('records') if not importances.empty else [],
        'target_column': target_column, 'missing_strategy': missing_strategy,
        'actual_vs_predicted_fig': actual_vs_pred_fig.to_json() if actual_vs_pred_fig and isinstance(actual_vs_pred_fig, go.Figure) else None,
        'residuals_vs_predicted_fig': resid_fig.to_json() if resid_fig and isinstance(resid_fig, go.Figure) else None,
        'shap_beeswarm_plot_src': shap_beeswarm_plot_src 
    }
    # Set progress-interval to disabled=True as the callback is finishing
    return results_layout, results_data_for_store, False, current_step_text_for_store, True, _captured_progress_steps_list, _captured_current_idx, {'display': 'block', 'width':'fit-content', 'margin': 'auto'}

# ... (rest of the callbacks, including run_optimization_analysis_callback, toggle_model_explanation_modal, etc.)
# Ensure run_optimization_analysis_callback also has robust error handling and correct progress step management.

@app.callback(
    [Output("model-explanation-modal", "is_open"), Output("modal-model-name", "children"), Output("modal-model-desc", "children")],
    [Input('model-comparison-table', 'active_cell'), Input('surrogate-candidates-table', 'active_cell'), Input("close-model-modal", "n_clicks")],
    [State('model-comparison-table', 'data'), State('surrogate-candidates-table', 'data'), State("model-explanation-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_model_explanation_modal(active_cell_expl, active_cell_opt, close_click, expl_table_data, opt_table_data, is_open):
    ctx = callback_context
    if not ctx.triggered:
        return is_open, dash.no_update, dash.no_update

    triggered_prop_id = ctx.triggered[0]['prop_id']
    active_cell = ctx.triggered[0]['value'] 

    model_name_to_show = None

    if triggered_prop_id == "close-model-modal.n_clicks":
        return False, dash.no_update, dash.no_update

    if "model-comparison-table.active_cell" in triggered_prop_id and active_cell:
        if active_cell['column_id'] == 'Details_action' and expl_table_data and active_cell['row'] < len(expl_table_data): 
            model_name_to_show = expl_table_data[active_cell['row']]['Model']
    elif "surrogate-candidates-table.active_cell" in triggered_prop_id and active_cell:
        if active_cell['column_id'] == 'Details_action' and opt_table_data and active_cell['row'] < len(opt_table_data): 
            model_name_to_show = opt_table_data[active_cell['row']]['Model']
            
    if model_name_to_show:
        expl = MODEL_EXPLANATIONS.get(model_name_to_show, {})
        desc_children = [
            html.P(expl.get("description", "No description available.")),
            html.H6("Key Strengths:", className="mt-3"), html.P(expl.get("pros", "N/A")),
            html.H6("Potential Weaknesses:", className="mt-3"), html.P(expl.get("cons", "N/A"))
        ]
        if expl.get("hyperparameters"):
            desc_children.append(html.H6("Key Hyperparameters (General):", className="mt-3"))
            for hp, hp_desc in expl["hyperparameters"].items():
                desc_children.append(html.P([html.Strong(f"{hp}: "), hp_desc], className="small"))
        return True, f"{model_name_to_show} Details", desc_children

    return is_open, dash.no_update, dash.no_update


@app.callback(
    [Output('optimization-results-area', 'children'), Output('store-optimization-results', 'data'),
     Output('nav-suggestions', 'disabled', allow_duplicate=True),
     Output('store-progress-text', 'data', allow_duplicate=True),
     Output('progress-interval', 'disabled', allow_duplicate=True), 
     Output('store-progress-steps', 'data', allow_duplicate=True),
     Output('store-current-step-index', 'data', allow_duplicate=True), 
     Output('btn-goto-suggestions-opt', 'style', allow_duplicate=True)],
    Input('btn-run-optimization-automl', 'n_clicks'),
    [State('store-raw-data', 'data'), State('store-column-roles', 'data'), State('dropdown-optimization-goal', 'value')],
    prevent_initial_call=True
)
def run_optimization_analysis_callback(n_clicks, raw_data_json, column_roles, opt_goal):
    if not n_clicks:
        return dash.no_update, dash.no_update, True, None, True, [], 0, {'display': 'none'}

    initial_progress_steps_opt = [f"Validating Inputs & Data..."]
    initial_progress_steps_opt.append(f"Preprocessing Data...")
    initial_progress_steps_opt.extend([f"Training {model['name']} ({i+1}/{len(MODELS_TO_EVALUATE)})" for i, model in enumerate(MODELS_TO_EVALUATE)])
    initial_progress_steps_opt.extend([
        "Evaluating Best Model...", "Calculating SHAP Values for Best Model...",
        "Generating Surrogate Tree...", "Optimizing (Grid Search over Parameter Space)...", 
        "Evaluating Optimal Settings...", "Generating Optimization Visualizations...", "Finalizing Results..."
    ])

    _captured_progress_steps_list_opt = list(initial_progress_steps_opt)
    _captured_current_idx_opt = 0
    
    def _progress_updater_local_opt(steps_list, current_idx_val):
        nonlocal _captured_progress_steps_list_opt, _captured_current_idx_opt
        _captured_progress_steps_list_opt = list(steps_list)
        _captured_current_idx_opt = current_idx_val
        if steps_list and current_idx_val < len(steps_list):
            print(f"Opt Progress (Internal): Step {current_idx_val + 1}/{len(steps_list)}: {steps_list[current_idx_val]}")

    if raw_data_json is None: return dbc.Alert("No data uploaded.", color="danger"), None, True, "Error: No data", True, initial_progress_steps_opt, 0, {'display': 'none'}
    if column_roles is None or not opt_goal: return dbc.Alert("Column roles or optimization goal not set.", color="danger"), None, True, "Error: Setup incomplete", True, initial_progress_steps_opt, 0, {'display': 'none'}

    df = pd.read_json(raw_data_json, orient='split')
    target_column = column_roles['target_for_analysis']
    feature_columns = column_roles['inputs']
    missing_strategy = column_roles.get('missing_strategy', 'drop_rows')
    current_step_text_for_store_opt = "Initiating Optimization Analysis..."

    try:
        feature_ranges = {col: {'min': df[col].min(), 'max': df[col].max()} for col in feature_columns if pd.api.types.is_numeric_dtype(df[col]) and col in df.columns and df[col].nunique() > 1} 
        
        all_model_results_opt, best_model_info_opt_dict, importances_opt, shap_beeswarm_plot_src_opt, \
        final_progress_steps_from_pipeline_opt, optimal_settings, predicted_target, \
        predicted_target_lower, predicted_target_upper, response_fig, surrogate_tree_text, \
        model_info_opt_summary, surrogate_tree_plot_src, actual_vs_pred_fig_opt, resid_fig_opt = \
            run_optimization_automl(df, target_column, feature_columns, opt_goal, feature_ranges, missing_strategy, _progress_updater_local_opt)
        
        _captured_progress_steps_list_opt = list(final_progress_steps_from_pipeline_opt)
        _captured_current_idx_opt = len(_captured_progress_steps_list_opt) - 1
        current_step_text_for_store_opt = "Optimization Complete!"

    except Exception as e:
        print(f"Error Optimization AutoML: {e}\n{traceback.format_exc()}")
        error_message_opt = f"Optimization Error: {str(e)}"
        current_step_text_for_store_opt = f"Error: {str(e)}"
        if not _captured_progress_steps_list_opt: _captured_progress_steps_list_opt = ["Error occurred"]
        _captured_current_idx_opt = min(_captured_current_idx_opt, len(_captured_progress_steps_list_opt)-1)

        return dbc.Alert(error_message_opt, color="danger"), \
               None, True, current_step_text_for_store_opt, True, \
               _captured_progress_steps_list_opt, _captured_current_idx_opt, {'display': 'none'}

    optimal_settings_kpi_cards = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6(f"Optimal {k}", className="card-title text-muted small text-truncate", style={'fontSize': '0.8rem'}),
            html.P(f"{v:.3f}" if isinstance(v, (float, np.number)) and not np.isnan(v) else str(v), className="card-text fs-5 text-success fw-bold")
        ])), width=6, lg=3, className="mb-2") for k, v in optimal_settings.items()
    ]
    predicted_target_kpi = dbc.Col(dbc.Card(dbc.CardBody([
        html.H5(f"Predicted {target_column} ({opt_goal}d)", className="card-title text-muted", style={'fontSize': '0.9rem'}),
        html.P(f"{predicted_target:.3f}" if not np.isnan(predicted_target) else "N/A", className="card-text fs-3 fw-bold text-success"),
        html.Small(f"95% CI: {predicted_target_lower:.3f} - {predicted_target_upper:.3f}" if not np.isnan(predicted_target_lower) and not np.isnan(predicted_target_upper) else "CI N/A", className="text-muted")
    ])), width=12, lg=6, className="mb-2 align-self-stretch")

    surrogate_model_name = model_info_opt_summary.get('Model Type', 'N/A')
    surrogate_hyperparam_defs = model_info_opt_summary.get('Hyperparameter_Definitions', {})
    surrogate_hyperparam_expl_div = [html.H6(f"Key Hyperparameters for Surrogate Model ({surrogate_model_name}):", className="mt-3 mb-2")]
    
    sur_hyperparams_values = model_info_opt_summary.get("Hyperparameters", {})
    if sur_hyperparams_values:
        for p_name_full, p_val in sur_hyperparams_values.items():
            clean_p_name = p_name_full.split('__')[-1]
            explanation = surrogate_hyperparam_defs.get(clean_p_name, "No specific explanation.")
            surrogate_hyperparam_expl_div.append(html.P([html.Strong(f"{clean_p_name}:"), f" {p_val} - {explanation}"], className="small"))
    else:
        surrogate_hyperparam_expl_div.append(html.P("No hyperparameters tuned or available for this model.", className="small"))

    surrogate_model_details_card = dbc.Card(dbc.CardBody([
        html.H5(f"Details for Surrogate Model: {surrogate_model_name}", className="card-title text-success"),
        dbc.Accordion([dbc.AccordionItem([
            html.P(MODEL_EXPLANATIONS.get(surrogate_model_name, {}).get('description', 'N/A')),
            html.H6("Strengths:", className="mt-2"), html.P(MODEL_EXPLANATIONS.get(surrogate_model_name, {}).get('pros', 'N/A')),
            html.H6("Weaknesses:", className="mt-2"), html.P(MODEL_EXPLANATIONS.get(surrogate_model_name, {}).get('cons', 'N/A')),
        ], title=f"About the {surrogate_model_name} Model")], start_collapsed=True, className="mb-3"),
        *surrogate_hyperparam_expl_div,
        html.H6("Surrogate Model Performance (on Test Set):", className="mt-3"),
        dbc.Row([
            dbc.Col(html.P(f"R-squared: {best_model_info_opt_dict.get('R-squared', 0):.3f}", className="small"), width=6),
            dbc.Col(html.P(f"MAE: {best_model_info_opt_dict.get('MAE', 0):.3f}", className="small"), width=6)
        ])
    ]), className="mb-4 shadow-sm bg-light")

    comparison_table_data_opt = []
    for model_res in all_model_results_opt: 
        model_type_opt = model_res["Model Type"]
        row_opt = {
            "Model": model_type_opt, "R²": f"{model_res.get('R-squared', 0):.3f}", "MAE": f"{model_res.get('MAE', 0):.3f}",
            "RMSE": f"{model_res.get('RMSE', 0):.3f}", "MAPE": f"{model_res.get('MAPE', float('nan')):.2f}%" if model_res.get('MAPE') is not None else "N/A",
            "CV R²": f"{model_res.get('Cross-Validation R2', 0):.3f}", "Time (s)": f"{model_res.get('Training Time (s)', 0):.2f}",
            "Details_action": "View"
        }
        comparison_table_data_opt.append(row_opt)

    surrogate_candidates_table_component = html.Div([
        html.H5("All Models Evaluated (for Surrogate Selection)", className="mt-4 mb-2 text-secondary"),
        html.P(f"The model chosen as surrogate was **{surrogate_model_name}**. Click 'View' for model details.", className="mb-3 small"),
        dash_table.DataTable(id='surrogate-candidates-table',
                             columns=[ {"name": col_name, "id": col_id} for col_id, col_name in 
                                        [("Model","Model"), ("R²","R²"), ("MAE","MAE"), ("RMSE","RMSE"), ("MAPE","MAPE"), 
                                         ("CV R²","CV R²"), ("Time (s)","Time (s)"), ("Details_action","Details")]],
                             data=comparison_table_data_opt, sort_action='native', page_size=len(MODELS_TO_EVALUATE)+1 if MODELS_TO_EVALUATE else 1,
                             style_table={'overflowX': 'auto'}, style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '0.85rem'},
                             style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
                             style_data_conditional=[
                                 {'if': {'filter_query': '{Model} = "' + surrogate_model_name + '"'},'backgroundColor': '#d4edda', 'fontWeight': 'bold'},
                                 {'if': {'column_id': 'Details_action'}, 'cursor': 'pointer', 'color': 'blue', 'textDecoration': 'underline'}
                              ]),
    ], className="mb-4")

    surrogate_tree_plot_layout = html.Div()
    if surrogate_tree_plot_src:
        surrogate_tree_plot_layout = html.Div([
            html.H5("Visual Surrogate Decision Tree (Simplified Logic)", className="mt-4 mb-2 text-secondary"),
            html.P("This tree approximates the surrogate model's behavior for easier interpretation. It shows key decision paths based on transformed features.", className="small text-muted"),
            html.Img(src=surrogate_tree_plot_src, style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': 'white', 'padding':'10px', 'display':'block', 'marginLeft':'auto', 'marginRight':'auto'})
        ], className="mb-4 text-center")
    else:
        surrogate_tree_plot_layout = html.Div([
            html.H5("Surrogate Tree Logic (Textual)", className="mt-4 mb-2 text-secondary"),
            html.P("Visual plot not available or generation failed. Textual rules based on transformed features:", className="small text-muted"),
            dbc.Card(dbc.CardBody(html.Pre(surrogate_tree_text if surrogate_tree_text else "Tree text not available.", style={'whiteSpace': 'pre-wrap', 'maxHeight': '200px', 'overflowY':'auto', 'backgroundColor': '#f8f9fa', 'border': '1px solid #eee', 'padding': '10px'})), className="mb-4")
        ])
        
    shap_plot_opt_display = html.Div([
        html.H5("SHAP Feature Importance & Impact (Surrogate Model)", className="mt-4 mb-2 text-secondary"),
        html.P("This plot shows feature impacts for the surrogate model used in optimization.", className="small text-muted"),
        html.Img(src=shap_beeswarm_plot_src_opt, style={'maxWidth': '100%', 'height': 'auto', 'border': '1px solid #ddd', 'borderRadius': '5px'}) if shap_beeswarm_plot_src_opt else dbc.Alert("SHAP plot for surrogate model could not be generated.", color="warning")
    ], className="mb-4 text-center")

    results_layout = html.Div([
        dbc.Alert(f"Optimization AutoML complete for '{target_column}'. Goal: {opt_goal.capitalize()}.", color="success", className="mt-2"),
        dbc.Row(optimal_settings_kpi_cards + [predicted_target_kpi], className="mb-3 g-3 justify-content-center"),
        surrogate_candidates_table_component,
        surrogate_model_details_card,
        dbc.Row([
            dbc.Col(dcc.Graph(id='actual-vs-predicted-plot-opt', figure=actual_vs_pred_fig_opt if actual_vs_pred_fig_opt else go.Figure()), md=6),
            dbc.Col(dcc.Graph(id='residuals-vs-predicted-plot-opt', figure=resid_fig_opt if resid_fig_opt else go.Figure()), md=6)
        ], className="mb-4"),
        html.Div(dcc.Graph(figure=response_fig), className="mb-3") if response_fig and response_fig.get('data') else html.P("Response surface plot not available (requires at least one varied numerical feature).", className="text-muted small text-center"),
        shap_plot_opt_display,
        surrogate_tree_plot_layout
    ])

    cleaned_best_model_info_opt = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in best_model_info_opt_dict.items() if k != 'Pipeline'}
    cleaned_all_model_results_opt = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items() if k != 'Pipeline'} for m in all_model_results_opt]
    cleaned_model_info_opt_summary = {k: (None if isinstance(v, float) and np.isnan(v) else v if not isinstance(v, tuple) else str(v)) for k,v in model_info_opt_summary.items()}
    results_data_for_store = {
        'all_model_results': cleaned_all_model_results_opt, 'best_model_info': cleaned_best_model_info_opt, 
        'optimal_settings': {k: (None if isinstance(v, float) and np.isnan(v) else v) for k,v in optimal_settings.items()},
        'predicted_target': None if isinstance(predicted_target, float) and np.isnan(predicted_target) else predicted_target,
        'predicted_target_lower': None if isinstance(predicted_target_lower, float) and np.isnan(predicted_target_lower) else predicted_target_lower,
        'predicted_target_upper': None if isinstance(predicted_target_upper, float) and np.isnan(predicted_target_upper) else predicted_target_upper,
        'target_column': target_column, 'goal': opt_goal, 'feature_columns': feature_columns, 'missing_strategy': missing_strategy,
        'importances': importances_opt.to_dict('records') if not importances_opt.empty else [], 
        'model_info': cleaned_model_info_opt_summary, 
        'surrogate_tree_text': surrogate_tree_text,
        'surrogate_tree_plot_src': surrogate_tree_plot_src,
        'actual_vs_predicted_fig_opt': actual_vs_pred_fig_opt.to_json() if actual_vs_pred_fig_opt and isinstance(actual_vs_pred_fig_opt, go.Figure) else None,
        'residuals_vs_predicted_fig_opt': resid_fig_opt.to_json() if resid_fig_opt and isinstance(resid_fig_opt, go.Figure) else None,
        'shap_beeswarm_plot_src_opt': shap_beeswarm_plot_src_opt
    }
    return results_layout, results_data_for_store, False, current_step_text_for_store_opt, True, _captured_progress_steps_list_opt, _captured_current_idx_opt, {'display': 'block', 'width':'fit-content', 'margin': 'auto'}


@app.callback(Output('active-tab-store', 'data', allow_duplicate=True),
              [Input('btn-goto-suggestions-expl', 'n_clicks'), Input('btn-goto-suggestions-opt', 'n_clicks')],
              prevent_initial_call=True)
def switch_to_suggestions_tab_from_analysis(n1, n2):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    return "tab-suggestions"

@app.callback(Output('suggestions-content-div', 'children'), 
              [Input('active-tab-store', 'data'),Input('btn-regenerate-llm', 'n_clicks')],
              [State('store-current-analysis-type', 'data'), State('store-exploration-results', 'data'),
               State('store-optimization-results', 'data'), State('custom-llm-prompt-input', 'value')],
)
def render_suggestions_tab_content(active_tab_id, n_clicks_regenerate, analysis_type, exploration_data, optimization_data, custom_prompt_value):
    ctx = callback_context
    triggered_by_tab_switch = active_tab_id == 'tab-suggestions' and ctx.triggered and ctx.triggered[0]['prop_id'] == 'active-tab-store.data'
    triggered_by_button = ctx.triggered and ctx.triggered[0]['prop_id'] == 'btn-regenerate-llm.n_clicks'

    if not (triggered_by_tab_switch or triggered_by_button):
        return html.Div() 

    if not analysis_type:
        return dbc.Alert("Analysis type not set. Please complete an analysis on Tab 2 first.", color="warning")

    header_text, llm_input_data = "", None
    current_analysis_type_for_llm = analysis_type 

    if current_analysis_type_for_llm == 'exploration':
        if not exploration_data or not isinstance(exploration_data, dict) or not exploration_data.get('target_column'):
            return dbc.Alert("Exploration analysis results are not available. Please run the analysis on Tab 2.", color="warning")
        header_text = f"AI Insights for Exploration: {exploration_data.get('target_column', 'N/A')}"
        llm_input_data = exploration_data 
    elif current_analysis_type_for_llm == 'optimization':
        if not optimization_data or not isinstance(optimization_data, dict) or not optimization_data.get('target_column'):
            return dbc.Alert("Optimization analysis results are not available. Please run the analysis on Tab 2.", color="warning")
        header_text = f"AI Insights for Optimization: {optimization_data.get('target_column', 'N/A')} ({optimization_data.get('goal', 'N/A')})"
        llm_input_data = optimization_data 
    else:
        return dbc.Alert("No analysis results available or analysis type not recognized.", color="danger")

    if llm_input_data is None: 
        return dbc.Alert("Failed to prepare data for AI insights.", color="danger")

    try:
        llm_explanation_markdown = generate_explanation_llm(current_analysis_type_for_llm.capitalize(), llm_input_data, custom_prompt_value)
    except Exception as e:
        print(f"Error generating AI insights: {e}\n{traceback.format_exc()}")
        return dbc.Alert(f"Error generating AI insights: {str(e)}", color="danger")

    return html.Div([
        html.H3(children=[html.I(className="fas fa-brain me-2"), header_text], className="mb-3 text-primary"),
        dcc.Markdown(llm_explanation_markdown, className="border p-3 bg-white rounded shadow-sm", style={'lineHeight': '1.7', 'fontSize': '0.95rem'})
    ], className="p-2")


if __name__ == '__main__':
    app.run(debug=True)
