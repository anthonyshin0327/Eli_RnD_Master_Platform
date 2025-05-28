import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import time # For simulating long processes
import json # For LLM interaction (simulated)

# Import actual ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap # For feature importance explanations
import plotly.graph_objects as go # For advanced plotting

# --- Embedded Demo Data (Replaced with content from lfa_random_data.csv) ---
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

# --- Real AutoML Functions ---

def create_preprocessing_pipeline(numerical_features, categorical_features, numerical_imputer_strategy):
    """
    Creates a preprocessing pipeline for numerical and categorical features.
    Handles missing values and one-hot encoding.
    numerical_imputer_strategy: must be a valid SimpleImputer strategy for numerical features, or None.
    """
    transformers = []

    if numerical_features:
        if numerical_imputer_strategy is not None:
            numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy=numerical_imputer_strategy))])
            transformers.append(('num', numerical_transformer, numerical_features))
        else:
            # If no numerical imputation is needed (e.g., rows dropped), just passthrough numerical features
            transformers.append(('num', 'passthrough', numerical_features))

    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Always impute categorical with most_frequent
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough' # Keep other columns (like Assay ID)
    )
    return preprocessor

def run_exploration_automl(data_df, target_column, feature_columns, missing_value_strategy):
    """
    Performs actual exploration analysis using RandomForestRegressor and SHAP.
    """
    print(f"Starting real Exploration AutoML for target: {target_column}")
    
    progress_steps = [
        "Handling Missing Values...", # New step
        "Data Preprocessing (Encoding)...", # Renamed from Imputation & Encoding
        "Splitting Data (Train/Test)...",
        "Training RandomForest Model...",
        "Calculating SHAP Values...",
        "Evaluating Model Performance...",
        "Generating Exploration Results..."
    ]

    df_processed = data_df.copy()
    imputer_strategy_for_pipeline = None # Default: no imputer for numericals in pipeline

    # Handle 'drop_rows' missing value strategy first
    if missing_value_strategy == 'drop_rows':
        # Drop rows with any NaN in the relevant columns (features + target)
        df_processed = data_df[feature_columns + [target_column]].dropna()
        # After dropping, the SimpleImputer in the pipeline won't have NaNs for numerical features.
        # So we don't need to pass an imputation strategy to create_preprocessing_pipeline.
        imputer_strategy_for_pipeline = None # Explicitly set to None
    elif missing_value_strategy == 'impute_mode':
        imputer_strategy_for_pipeline = 'most_frequent'
    else: # 'impute_mean', 'impute_median'
        imputer_strategy_for_pipeline = missing_value_strategy
    
    # Separate features (X) and target (y) from the potentially processed DataFrame
    X = df_processed[feature_columns]
    y = df_processed[target_column]

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Create preprocessing pipeline, passing the determined imputer strategy
    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features, imputer_strategy_for_pipeline)

    # Create the full pipeline with a RandomForestRegressor
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = model_pipeline.predict(X_test)

    # Evaluate performance
    r_squared = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # SHAP Feature Importance
    # For SHAP, we need the preprocessed data and the regressor itself
    explainer = shap.TreeExplainer(model_pipeline.named_steps['regressor'])
    
    # Get preprocessed data for SHAP explanation
    X_train_transformed = model_pipeline.named_steps['preprocessor'].transform(X_train)
    
    # Get feature names after one-hot encoding
    feature_names = []
    # ColumnTransformer's get_feature_names_out() is preferred for sklearn >= 0.23
    # Fallback for older versions or more complex cases
    try:
        feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        # Manual collection if get_feature_names_out is not available or for older sklearn
        for name, transformer, features in model_pipeline.named_steps['preprocessor'].transformers_:
            if name == 'num': # numerical features retain their names
                feature_names.extend(numerical_features)
            elif name == 'cat': # one-hot encoded features need their new names
                # This is a bit of a hack without get_feature_names_out on the encoder directly
                # For robust feature naming post-onehot, ensure the encoder itself is accessible or mock it.
                # For now, let's assume one-hot encoder provides feature names if it's not None
                if isinstance(transformer, Pipeline) and 'onehot' in transformer.named_steps:
                    onehot_encoder = transformer.named_steps['onehot']
                    if hasattr(onehot_encoder, 'get_feature_names_out'):
                         feature_names.extend(onehot_encoder.get_feature_names_out(features))
                    else: # Fallback: append original categorical feature names + "_category"
                        # This fallback is less precise but prevents errors.
                        for f in features:
                            for cat in X[f].unique():
                                feature_names.append(f"{f}_{cat}")
            else: # remainder passthrough
                feature_names.extend([f for f in features if f not in numerical_features and f not in categorical_features])


    # Create a DataFrame for SHAP values with correct feature names
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    # If X_train_transformed_df is empty, handle it
    if X_train_transformed_df.empty:
        print("Warning: X_train_transformed_df is empty, cannot compute SHAP values.")
        # Create a placeholder for importances_df
        importances_df = pd.DataFrame({'feature': feature_columns, 'importance': np.zeros(len(feature_columns))})
        shap_summary_plot = {} # Empty plot
    else:
        # Limit the number of samples for SHAP to avoid long computation times in demo
        if X_train_transformed_df.shape[0] > 100:
            X_shap = X_train_transformed_df.sample(n=100, random_state=42)
        else:
            X_shap = X_train_transformed_df
            
        shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list): # For multi-output models, take the first output's SHAP values
            shap_values_abs_mean = np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

        importances_df = pd.DataFrame({
            'feature': X_shap.columns,
            'importance': shap_values_abs_mean
        }).sort_values(by='importance', ascending=False)
        
        # Create a simple bar chart for feature importance
        shap_summary_plot = {
            'data': [go.Bar(x=importances_df['feature'], y=importances_df['importance'], marker_color='#17A2B8')],
            'layout': {
                'title': 'Feature Importance (Mean Absolute SHAP Value)',
                'xaxis': {'title': 'Feature', 'tickangle': 45},
                'yaxis': {'title': 'Mean Absolute SHAP Value'},
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100},
                'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#495057'}
            }
        }

    performance_metrics = {"R-squared": r_squared, "MAE": mae}
    
    # Capture model details and hyperparameters
    model_params = model_pipeline.named_steps['regressor'].get_params()
    model_info = {
        "Model Type": "RandomForestRegressor",
        "Hyperparameters": {k: v for k, v in model_params.items() if '_' not in k or k.startswith('n_estimators') or k.startswith('random_state') or k.startswith('max_depth')}, # Example: filter some common ones
        "Training Data Shape": X_train.shape,
        "Test Data Shape": X_test.shape,
        "Features Used": feature_columns,
        "Target Column": target_column,
        "Missing Value Strategy": missing_value_strategy
    }

    print("Real Exploration AutoML complete.")
    return importances_df, performance_metrics, shap_summary_plot, progress_steps, model_info

def run_optimization_automl(data_df, target_column, feature_columns, optimization_goal, feature_ranges, missing_value_strategy):
    """
    Performs actual optimization analysis using RandomForestRegressor as surrogate model
    and a basic grid search for optimization.
    """
    print(f"Starting real Optimization AutoML for target: {target_column}")

    progress_steps = [
        "Handling Missing Values...", # New step
        "Data Preprocessing (Encoding)...", # Renamed
        "Splitting Data (Train/Test)...",
        "Training Surrogate Model (RandomForest)...",
        "Performing Optimization (Grid Search)...",
        "Evaluating Optimal Settings...",
        "Generating Optimization Results..."
    ]

    df_processed = data_df.copy()
    imputer_strategy_for_pipeline = None # Default: no imputer for numericals in pipeline

    # Handle 'drop_rows' missing value strategy first
    if missing_value_strategy == 'drop_rows':
        # Drop rows with any NaN in the relevant columns (features + target)
        df_processed = data_df[feature_columns + [target_column]].dropna()
        imputer_strategy_for_pipeline = None # Explicitly set to None
    elif missing_value_strategy == 'impute_mode':
        imputer_strategy_for_pipeline = 'most_frequent'
    else: # 'impute_mean', 'impute_median'
        imputer_strategy_for_pipeline = missing_value_strategy

    X = df_processed[feature_columns]
    y = df_processed[target_column]

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features, imputer_strategy_for_pipeline)
    
    # Surrogate model pipeline
    surrogate_model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)) # Fewer estimators for speed
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    surrogate_model_pipeline.fit(X_train, y_train)

    # Simulate optimization: Grid search over feature ranges
    # For a real optimization, one would use Bayesian Optimization or other advanced techniques.
    # Here, we'll create a simple grid based on min/max ranges and predict.
    
    # Create a grid for numerical features
    grid_points_num = {}
    for col in numerical_features:
        min_val = feature_ranges[col]['min']
        max_val = feature_ranges[col]['max']
        grid_points_num[col] = np.linspace(min_val, max_val, 10) # 10 points per numerical feature

    # For categorical features, use unique values
    grid_points_cat = {}
    for col in categorical_features:
        grid_points_cat[col] = df_processed[col].unique().tolist() # Use df_processed for unique values

    # Create all combinations of grid points
    from itertools import product
    
    all_combinations = []
    # Prepare a list of iterables for product, maintaining original feature order
    iterables_for_product = []
    for col in feature_columns:
        if col in numerical_features:
            iterables_for_product.append(grid_points_num[col])
        elif col in categorical_features:
            iterables_for_product.append(grid_points_cat[col])

    if not iterables_for_product:
        # Handle case where no combinations can be formed (e.g., no features selected)
        optimal_settings = {col: "N/A" for col in feature_columns}
        predicted_target = 0.0
        predicted_target_lower = 0.0
        predicted_target_upper = 0.0
        surrogate_model_info_str = "No features selected for optimization."
        response_surface_fig = {}
        print("No combinations for optimization.")
        # Return model_info as well, even if empty
        model_info = {
            "Model Type": "N/A",
            "Hyperparameters": {},
            "Training Data Shape": (0,0),
            "Test Data Shape": (0,0),
            "Features Used": feature_columns,
            "Target Column": target_column,
            "Missing Value Strategy": missing_value_strategy
        }
        return optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, surrogate_model_info_str, response_surface_fig, progress_steps, model_info

    for combo in product(*iterables_for_product):
        row_dict = {feature_columns[i]: combo[i] for i in range(len(feature_columns))}
        all_combinations.append(row_dict)

    optimization_df = pd.DataFrame(all_combinations)
    
    # Predict on the optimization grid
    optimization_predictions = surrogate_model_pipeline.predict(optimization_df)

    # Find optimal settings based on goal
    if optimization_goal == 'maximize':
        optimal_idx = np.argmax(optimization_predictions)
    else: # minimize
        optimal_idx = np.argmin(optimization_predictions)

    optimal_settings_series = optimization_df.iloc[optimal_idx]
    optimal_settings = optimal_settings_series.to_dict()
    predicted_target = optimization_predictions[optimal_idx]

    # Simple uncertainty estimation (e.g., using std dev of predictions on test set)
    test_predictions = surrogate_model_pipeline.predict(X_test)
    prediction_std = np.std(test_predictions)
    predicted_target_lower = predicted_target - 1.96 * prediction_std # 95% CI
    predicted_target_upper = predicted_target + 1.96 * prediction_std # 95% CI

    surrogate_model_info_str = f"RandomForestRegressor (R-squared on test: {r2_score(y_test, test_predictions):.2f})"
    
    response_surface_fig = {}
    # Only plot if there are exactly two numerical features, as requested by user's feedback
    if len(numerical_features) == 2:
        x_feat, y_feat = numerical_features[0], numerical_features[1]
        
        # Create a finer grid for plotting the surface
        x_plot = np.linspace(feature_ranges[x_feat]['min'], feature_ranges[x_feat]['max'], 30)
        y_plot = np.linspace(feature_ranges[y_feat]['min'], feature_ranges[y_feat]['max'], 30)
        X_grid, Y_grid = np.meshgrid(x_plot, y_plot)

        # Create a DataFrame for predictions on this grid
        plot_df_rows = []
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                row = {x_feat: X_grid[i, j], y_feat: Y_grid[i, j]}
                # For other features (numerical or categorical), use their optimal settings
                for col in feature_columns:
                    if col not in [x_feat, y_feat]:
                        row[col] = optimal_settings[col] # Use optimal setting for other features
                plot_df_rows.append(row)
        
        plot_df = pd.DataFrame(plot_df_rows)
        
        # Ensure plot_df has all the original feature columns, even if they are not in the grid
        # Fill missing columns with their optimal values
        for col in feature_columns:
            if col not in plot_df.columns:
                plot_df[col] = optimal_settings[col]

        # Predict Z values
        Z_grid_flat = surrogate_model_pipeline.predict(plot_df[feature_columns])
        Z_grid = Z_grid_flat.reshape(X_grid.shape)

        response_surface_fig = {
            'data': [go.Contour(x=x_plot, y=y_plot, z=Z_grid, colorscale='Viridis',
                                contours_coloring='heatmap', line_width=0)],
            'layout': {
                'title': f'Predicted Response Surface for {target_column}',
                'xaxis': {'title': x_feat},
                'yaxis': {'title': y_feat},
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100},
                'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#495057'},
                'height': 500
            }
        }
    elif len(numerical_features) == 1 and not categorical_features:
        # 1D plot for a single numerical feature
        x_feat = numerical_features[0]
        x_plot = np.linspace(feature_ranges[x_feat]['min'], feature_ranges[x_feat]['max'], 100)
        plot_df_rows = [{x_feat: val} for val in x_plot]
        # For other features (if any, though none in this case), use their optimal settings
        # This part is crucial for making predictions on a single feature while others are fixed
        for row in plot_df_rows:
            for col in feature_columns:
                if col not in row: # If it's not the x_feat, use its optimal value
                    row[col] = optimal_settings[col]
        
        plot_df = pd.DataFrame(plot_df_rows)
        
        y_plot = surrogate_model_pipeline.predict(plot_df[feature_columns])
        
        response_surface_fig = {
            'data': [go.Scatter(x=x_plot, y=y_plot, mode='lines', line_color='#28a745')],
            'layout': {
                'title': f'Predicted Response Curve for {target_column} vs {x_feat}',
                'xaxis': {'title': x_feat},
                'yaxis': {'title': target_column},
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100},
                'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#495057'},
                'height': 500
            }
        }
    else:
        response_surface_fig = {} # No plot for >2 numerical features or complex categorical combinations

    # Capture model details and hyperparameters
    model_params = surrogate_model_pipeline.named_steps['regressor'].get_params()
    model_info = {
        "Model Type": "RandomForestRegressor (Surrogate Model)",
        "Hyperparameters": {k: v for k, v in model_params.items() if '_' not in k or k.startswith('n_estimators') or k.startswith('random_state') or k.startswith('max_depth')},
        "Training Data Shape": X_train.shape,
        "Test Data Shape": X_test.shape,
        "Features Used": feature_columns,
        "Target Column": target_column,
        "Missing Value Strategy": missing_value_strategy,
        "Optimization Goal": optimization_goal
    }


    print("Real Optimization AutoML complete.")
    return optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, surrogate_model_info_str, response_surface_fig, progress_steps, model_info

def generate_explanation_llm(analysis_type, results_data, custom_prompt_addition=""):
    print(f"Generating insights for {analysis_type} (LLM simulation)...")
    base_prompt = f"The R&D team has performed an '{analysis_type}' analysis. Key results are: {json.dumps(results_data)}. Provide a detailed, human-like explanation, focusing on actionable insights for synthetic biology and protein engineering."
    
    # Incorporate custom prompt addition
    final_prompt = f"{base_prompt} {custom_prompt_addition}".strip()

    time.sleep(1) # Simulate API call
    print("LLM explanation generation complete.")
    if analysis_type == "Exploration":
        importances_df = pd.DataFrame(results_data.get('importances', []))
        top_features_str = ", ".join(importances_df.head(3)['feature'].tolist()) if not importances_df.empty else "N/A"
        r_squared = results_data.get('performance_metrics', {}).get('R-squared', 'N/A')
        if isinstance(r_squared, str): r_squared_val = 0.0 
        else: r_squared_val = r_squared

        return f"""
        #### **Exploration Analysis: Uncovering Key Factors**
        **Objective:** Our automated exploration aimed to pinpoint which of your input variables (we call them 'features') have the most significant impact on your chosen output, '{results_data.get('target_column', 'N/A')}'.
        **Methodology:** We employed a **Random Forest Regressor** model, a powerful ensemble learning method, to analyze the relationships within your data. Data preprocessing involved handling missing values using a **{results_data.get('missing_strategy', 'default')}** strategy (e.g., dropping rows or imputation) and applying **One-Hot Encoding** for categorical variables. The dataset was split into training and testing sets to ensure robust model evaluation.
        **What We Found:**
        * **Top Influencers:** The analysis, primarily driven by **SHAP (SHapley Additive exPlanations) values**, suggests that **{top_features_str}** are the most critical variables influencing '{results_data.get('target_column', 'N/A')}'. SHAP values provide a clear and transparent way to understand how each feature contributes to the model's predictions, both positively and negatively.
        * **Model Fit (R-squared):** The model achieved an R-squared of **{r_squared_val:.3f}**, indicating that approximately **{r_squared_val*100:.0f}%** of the variance in your target variable can be explained by the input features.
        * **Mean Absolute Error (MAE):** The Mean Absolute Error (MAE) was **{results_data.get('performance_metrics', {}).get('MAE', 'N/A'):.3f}**, representing the average magnitude of the errors in a set of predictions, without considering their direction.
        **Actionable Insights & Suggestions:**
        1.  **Prioritize Investigation:** Focus your future experimental efforts and deeper scientific investigation on **{top_features_str}**. These are the factors most likely to yield significant changes in your target output.
        2.  **Mechanistic Understanding:** Delve into the underlying biochemical or synthetic biology mechanisms that might explain why these top features are so influential. This could lead to novel hypotheses.
        3.  **Targeted Optimization:** Consider these high-impact features as prime candidates for fine-tuning in subsequent optimization experiments.
        4.  **Data Quality:** Review the quality and variability of data collected for these key features. High importance coupled with noisy data might indicate a need for more precise measurement techniques.
        """
    elif analysis_type == "Optimization":
        optimal_settings_str = ", ".join([f"**{k}**: {v:.3f}" if isinstance(v, (int, float)) else f"**{k}**: {v}" for k, v in results_data.get('optimal_settings', {}).items()])
        goal_verb = "maximize" if results_data.get('goal') == "maximize" else "minimize"
        predicted_val = results_data.get('predicted_target', 'N/A')
        predicted_lower = results_data.get('predicted_target_lower', 'N/A')
        predicted_upper = results_data.get('predicted_target_upper', 'N/A')

        if isinstance(predicted_val, str): predicted_val_num = 0.0
        else: predicted_val_num = predicted_val
        
        if isinstance(predicted_lower, str): predicted_lower_num = 0.0
        else: predicted_lower_num = predicted_lower

        if isinstance(predicted_upper, str): predicted_upper_num = 0.0
        else: predicted_upper_num = predicted_upper


        return f"""
        #### **Optimization Analysis: Finding the Sweet Spot**
        **Objective:** To identify the optimal experimental conditions to **{goal_verb}** '{results_data.get('target_column', 'N/A')}'.
        **Methodology:** We constructed a **Random Forest Regressor** as a surrogate model to learn the complex relationship between your input variables and the target output. Data preprocessing involved handling missing values using a **{results_data.get('missing_strategy', 'default')}** strategy (e.g., dropping rows or imputation) and applying **One-Hot Encoding** for categorical variables. A systematic **grid search** was then performed across the defined ranges of your input variables to predict the target output for various combinations, identifying the conditions that best meet your optimization goal.
        **Optimal Settings Suggested:** Based on the surrogate model's predictions, the following optimal settings are recommended: {optimal_settings_str}
        **Predicted Outcome:** Under these optimal conditions, the model predicts an approximate target value of **{predicted_val_num:.3f}** (with a 95% Confidence Interval: {predicted_lower_num:.3f} - {predicted_upper_num:.3f}). This confidence interval provides an estimate of the uncertainty in the prediction.
        **Under the Hood (Transparency):** The surrogate model used was a {results_data.get('surrogate_model_info', 'N/A')}. This model acts as a fast approximation of your real-world experiment, allowing for efficient exploration of the design space.
        **Actionable Insights & Suggestions:**
        1.  **Experimental Validation:** The most crucial next step is to **experimentally validate** these predicted optimal settings in your lab. Real-world experiments are essential to confirm the model's predictions.
        2.  **Sensitivity Analysis:** Consider performing additional experiments slightly varying the recommended optimal settings to understand the sensitivity of your system. This helps in identifying robust operating ranges.
        3.  **Practical Constraints:** Always consider any practical or safety constraints in your lab when implementing these optimal settings. The model provides a theoretical optimum, but real-world limitations may necessitate minor adjustments.
        4.  **Iterative Optimization:** If initial validation is successful, consider using these results to refine your input ranges and perform another round of optimization for even finer tuning.
        """
    return "Placeholder explanation: LLM integration pending."

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
                suppress_callback_exceptions=True)
app.title = "R&D Experiment Analysis Platform"

# ========= Helper Functions =========
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Invalid file type. Please upload CSV or Excel.'])
        return df
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return html.Div([f'There was an error processing this file: {str(e)}'])

def process_dataframe_for_ui(df, filename_display):
    """Helper function to generate UI components from a DataFrame."""
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=8,
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 'minWidth': '100px', 'width': '150px', 'maxWidth': '200px', 'whiteSpace': 'normal'},
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'},
        style_data={'borderBottom': '1px solid #dee2e6'},
        # Enable client-side sorting and filtering
        sort_action='native',
        filter_action='native',
    )
    column_options = [{'label': col, 'value': col} for col in df.columns]
    role_assignment_ui = html.Div([
        dbc.Label("Input Variables (X):", html_for='dropdown-input-vars', className="fw-bold mt-3"),
        dcc.Dropdown(id='dropdown-input-vars', options=column_options, multi=True, placeholder="Select features/factors", className="mb-2"),
        dbc.Tooltip("These are the independent variables that you control or measure in your experiment.", target="dropdown-input-vars"),

        dbc.Label("Target Output Variable (Y):", html_for='dropdown-output-var', className="fw-bold"),
        dcc.Dropdown(id='dropdown-output-var', options=column_options, multi=False, placeholder="Select the single output to analyze/optimize", className="mb-2"),
        dbc.Tooltip("This is the dependent variable you are trying to predict or optimize.", target="dropdown-output-var"),

        dbc.Label("Missing Value Strategy:", html_for='dropdown-missing-strategy', className="fw-bold"),
        dcc.Dropdown(
            id='dropdown-missing-strategy',
            options=[
                {'label': 'Drop Rows with Missing Data', 'value': 'drop_rows'}, # More descriptive label
                {'label': 'Impute with Mean (Numeric only)', 'value': 'impute_mean'},
                {'label': 'Impute with Median (Numeric only)', 'value': 'impute_median'},
                {'label': 'Impute with Mode (Numeric & Categorical)', 'value': 'impute_mode'},
            ],
            value='drop_rows', clearable=False, className="mb-3",
            placeholder="Select how to handle missing data"
        ),
        dbc.Tooltip("Choose a strategy to handle any missing values in your dataset.", target="dropdown-missing-strategy"),

        dbc.Label("Ignore Columns (Optional):", html_for='dropdown-ignore-vars', className="fw-bold"),
        dcc.Dropdown(id='dropdown-ignore-vars', options=column_options, multi=True, placeholder="Select columns to exclude", className="mb-3"),
        dbc.Tooltip("Columns selected here will be excluded from the analysis.", target="dropdown-ignore-vars"),

        dbc.Button(children=[html.I(className="fas fa-cogs me-2"), "Confirm Setup & Proceed to Analysis"], id="btn-confirm-setup", color="primary", className="mt-3 w-100 btn-lg")
    ])
    status_message = dbc.Alert(f"Successfully loaded: {filename_display}", color="success", duration=4000)
    stored_data = df.to_json(date_format='iso', orient='split')
    return status_message, table, stored_data, role_assignment_ui

# ========= App Layout =========
app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id='store-raw-data'),
    dcc.Store(id='store-column-roles'), 
    dcc.Store(id='store-exploration-results'),
    dcc.Store(id='store-optimization-results'),
    dcc.Store(id='store-current-analysis-type'),
    dcc.Store(id='store-progress-text'), 
    dcc.Interval(id='progress-interval', interval=500, n_intervals=0, disabled=True), # Faster interval for progress updates
    dcc.Store(id='store-progress-steps', data=[]), # To store the list of steps for progress bar
    dcc.Store(id='store-current-step-index', data=0), # To store the current step index

    dbc.Row(dbc.Col(html.H1(children=[html.I(className="fas fa-flask me-2"), "R&D Experiment Analysis Platform"], className="text-center my-4 display-4 text-primary"), width=12)),
    
    # Guided Workflow / Stepper
    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("1. Data & Setup", active=True, href="/#", id="nav-data-setup", className="text-primary")),
                    dbc.NavItem(dbc.NavLink("2. Analysis & Results", active=False, href="/#", id="nav-analysis", disabled=True)),
                    dbc.NavItem(dbc.NavLink("3. AI Insights & Suggestions", active=False, href="/#", id="nav-suggestions", disabled=True)),
                ],
                pills=True,
                className="nav-pills justify-content-center mb-4"
            ), width=12
        )
    ]),

    dbc.Row(dbc.Col(html.Div(id='global-progress-message', className="text-center my-2 bg-dark p-2 rounded", style={'color': 'white'}), width=12)), # Added inline style for white text
    
    dbc.Tabs(id="main-tabs", active_tab="tab-data-upload", style={'display': 'none'}, # Hide default tabs, use Nav for visual
             children=[
                dbc.Tab(label="1. Data & Setup", tab_id="tab-data-upload", children=[
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H4("Upload Experiment Data", className="mb-3"),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(['Drag and Drop or ', html.A('Select Files (.csv, .xlsx)')]),
                                    style={
                                        'width': '100%', 'height': '80px', 'lineHeight': '80px',
                                        'borderWidth': '2px', 'borderStyle': 'dashed',
                                        'borderRadius': '8px', 'textAlign': 'center', 'margin': '10px 0',
                                        'backgroundColor': '#f8f9fa'
                                    },
                                    multiple=False
                                ),
                                html.Div("Or", className="text-center my-2 small text-muted"),
                                dbc.Button(children=[html.I(className="fas fa-database me-2"),"Load Demo Dataset (LFA Data)"], id="btn-load-demo", color="secondary", outline=True, className="w-100 mb-3"),
                                html.Div(id='output-data-upload-status', className="mt-2"),
                                html.Div(id='output-datatable-div', className="mt-3", style={'maxHeight': '400px', 'overflowY': 'auto', 'overflowX': 'auto'})
                            ], md=7, className="p-3 border-end"),
                            dbc.Col([
                                html.H4("Experiment Configuration", className="mb-3"),
                                dbc.Label("Select Experiment Type:", html_for='dropdown-experiment-type', className="fw-bold"),
                                dcc.Dropdown(
                                    id='dropdown-experiment-type',
                                    options=[
                                        {'label': 'Exploration (Screening - Which inputs matter?)', 'value': 'exploration'},
                                        {'label': 'Optimization (RSM-like - What are the best settings?)', 'value': 'optimization'}
                                    ],
                                    value='exploration', clearable=False, className="mb-3"
                                ),
                                dbc.Tooltip("Choose the type of analysis you want to perform on your data.", target="dropdown-experiment-type"),
                                html.Div(id='column-role-assignment-div') 
                            ], md=5, className="p-3")
                        ])
                    ]), className="mt-3")
                ]), # End of Tab 1
                dbc.Tab(label="2. Analysis & Results", tab_id="tab-analysis", id="tab-analysis", disabled=True, children=[
                    dbc.Card(dbc.CardBody([
                        html.Div(id='analysis-content-div', className="p-3"),
                        # These buttons are now always present but hidden
                        dbc.Button(children=[html.I(className="fas fa-lightbulb me-2"),"Proceed to AI Insights (Exploration)"], id="btn-goto-suggestions-expl", color="link", className="mt-3 d-block text-center", style={'display': 'none'}),
                        dbc.Button(children=[html.I(className="fas fa-lightbulb me-2"),"Proceed to AI Insights (Optimization)"], id="btn-goto-suggestions-opt", color="link", className="mt-3 d-block text-center", style={'display': 'none'})
                    ]), className="mt-3")
                ]), # End of Tab 2
                dbc.Tab(label="3. AI Insights & Suggestions", tab_id="tab-suggestions", id="tab-suggestions", disabled=True, children=[
                    dbc.Card(dbc.CardBody([
                        html.Div(id='suggestions-content-div', className="p-3"),
                        html.Hr(),
                        html.H5("Customize AI Prompt (Optional)", className="mt-4"),
                        html.P("Refine the instructions for the AI to get more tailored insights."),
                        dcc.Textarea(
                            id='custom-llm-prompt-input',
                            value='Focus on the implications for protein stability and synthetic biology applications.',
                            style={'width': '100%', 'height': 120, 'borderRadius': '8px', 'border': '1px solid #ced4da', 'padding': '10px'},
                            className="mb-3"
                        ),
                        dbc.Tooltip("Add specific instructions or questions for the AI to consider when generating insights.", target="custom-llm-prompt-input"),
                        dbc.Button(children=[html.I(className="fas fa-sync-alt me-2"), "Regenerate AI Insights"], id="btn-regenerate-llm", color="secondary", className="w-100")
                    ]), className="mt-3")
                ]), # End of Tab 3
            ]), # End of Tabs
    dbc.Row(dbc.Col(html.P("Powered by AutoML and Generative AI", className="text-center text-muted mt-5"), width=12))
])

# ========= Callbacks =========

# Callback to update NavLinks based on active tab
@app.callback(
    [Output('nav-data-setup', 'active'),
     Output('nav-analysis', 'active'),
     Output('nav-suggestions', 'active'),
     Output('nav-analysis', 'disabled'), # Keep disabled status in sync with tab
     Output('nav-suggestions', 'disabled')], # Keep disabled status in sync with tab
    [Input('main-tabs', 'active_tab'),
     Input('tab-analysis', 'disabled'), # Listen to disabled status of tabs
     Input('tab-suggestions', 'disabled')]
)
def update_nav_links(active_tab, analysis_tab_disabled, suggestions_tab_disabled):
    return (
        active_tab == 'tab-data-upload',
        active_tab == 'tab-analysis',
        active_tab == 'tab-suggestions',
        analysis_tab_disabled,
        suggestions_tab_disabled
    )

# Callback to switch tabs when NavLink is clicked
@app.callback(
    Output('main-tabs', 'active_tab', allow_duplicate=True),
    [Input('nav-data-setup', 'n_clicks'),
     Input('nav-analysis', 'n_clicks'),
     Input('nav-suggestions', 'n_clicks')],
    prevent_initial_call=True
)
def switch_tabs_from_nav(n_data, n_analysis, n_suggestions):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'nav-data-setup':
        return 'tab-data-upload'
    elif button_id == 'nav-analysis':
        return 'tab-analysis'
    elif button_id == 'nav-suggestions':
        return 'tab-suggestions'
    return dash.no_update


# Callback to update global progress message
@app.callback(
    Output('global-progress-message', 'children'),
    Input('progress-interval', 'n_intervals'),
    State('store-progress-text', 'data'),
    State('progress-interval', 'disabled'),
    State('store-progress-steps', 'data'),
    State('store-current-step-index', 'data')
)
def update_progress_message(n, progress_text, interval_disabled, progress_steps, current_step_index):
    if progress_text and not interval_disabled:
        if progress_steps and current_step_index < len(progress_steps):
            current_step_text = progress_steps[current_step_index]
            progress_percentage = ((current_step_index + 1) / len(progress_steps)) * 100
            return html.Div([
                dbc.Spinner(size="sm", color="primary", className="me-2"), 
                html.Span(f"{current_step_text} ({current_step_index + 1}/{len(progress_steps)})", style={'color': 'white'}), 
                dbc.Progress(value=progress_percentage, className="ms-3 w-25", style={'height': '15px'})
            ])
        return html.Div([dbc.Spinner(size="sm", color="primary", className="me-2"), html.Span(progress_text, style={'color': 'white'})]) 
    return ""

# Callback to advance progress bar steps
@app.callback(
    Output('store-current-step-index', 'data'),
    Input('progress-interval', 'n_intervals'),
    State('store-current-step-index', 'data'),
    State('store-progress-steps', 'data'),
    State('progress-interval', 'disabled'),
    prevent_initial_call=True
)
def advance_progress_step(n_intervals, current_step_index, progress_steps, interval_disabled):
    if not interval_disabled and progress_steps:
        if current_step_index < len(progress_steps) - 1:
            return current_step_index + 1
    return current_step_index # Keep current index if disabled or at end


@app.callback(
    [Output('output-data-upload-status', 'children'),
     Output('output-datatable-div', 'children'),
     Output('store-raw-data', 'data'),
     Output('column-role-assignment-div', 'children')],
    [Input('upload-data', 'contents'),
     Input('btn-load-demo', 'n_clicks')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def handle_data_input(contents, n_clicks_demo, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    df = None
    filename_display = None

    if triggered_id == 'upload-data' and contents:
        df_or_error = parse_contents(contents, filename)
        if isinstance(df_or_error, html.Div): 
            return dbc.Alert(df_or_error.children[0], color="danger"), "", None, ""
        df = df_or_error
        filename_display = filename
    elif triggered_id == 'btn-load-demo' and n_clicks_demo:
        try:
            df = pd.read_csv(io.StringIO(DEMO_DATA_CSV_STRING))
            filename_display = "lfa_random_data.csv" # Updated filename display
        except Exception as e:
            print(f"Error loading demo data: {e}")
            return dbc.Alert(f"Error loading demo data: {str(e)}", color="danger"), "", None, ""
    
    if df is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return process_dataframe_for_ui(df, filename_display)

@app.callback(
    [Output('store-column-roles', 'data'),
     Output('main-tabs', 'active_tab'),
     Output('tab-analysis', 'disabled'),
     Output('tab-suggestions', 'disabled'), 
     Output('output-data-upload-status', 'children', allow_duplicate=True),
     Output('store-current-analysis-type', 'data')],
    [Input('btn-confirm-setup', 'n_clicks')],
    [State('dropdown-experiment-type', 'value'), State('dropdown-input-vars', 'value'),
     State('dropdown-output-var', 'value'), State('dropdown-ignore-vars', 'value'),
     State('dropdown-missing-strategy', 'value'), # New input for missing value strategy
     State('store-raw-data', 'data')],
    prevent_initial_call=True
)
def confirm_setup_and_proceed(n_clicks, exp_type, inputs, output_var, ignored, missing_strategy, raw_data_json):
    if not n_clicks or not raw_data_json:
        return dash.no_update, dash.no_update, True, True, dash.no_update, dash.no_update
    if not inputs or not output_var:
        return dash.no_update, dash.no_update, True, True, dbc.Alert("⚠️ Please select input(s) and a target output.", color="warning", duration=5000, className="mt-2"), dash.no_update
    if output_var in inputs:
        return dash.no_update, dash.no_update, True, True, dbc.Alert("⚠️ Target Output cannot be an Input Variable.", color="danger", duration=5000, className="mt-2"), dash.no_update
    column_roles = {
        'inputs': inputs, 
        'target_for_analysis': output_var, 
        'ignored': ignored or [],
        'missing_strategy': missing_strategy # Store the missing value strategy
    }
    return column_roles, "tab-analysis", False, True, dbc.Alert(f"Setup Confirmed for {exp_type.capitalize()}. Proceed to 'Analysis & Results'.", color="info", duration=4000, className="mt-2"), exp_type

@app.callback(
    Output('analysis-content-div', 'children'),
    Input('main-tabs', 'active_tab'),
    State('store-current-analysis-type', 'data'), State('store-column-roles', 'data')
)
def render_analysis_tab_content(active_tab, analysis_type, column_roles):
    if active_tab != 'tab-analysis' or not analysis_type or not column_roles:
        return html.P("Complete data upload and setup on Tab 1, then click 'Confirm Setup'.")
    target_column = column_roles.get('target_for_analysis')
    if analysis_type == 'exploration':
        return html.Div([
            html.H3(f"Exploration Analysis for Target: {target_column}", className="mb-3 text-info"),
            html.P("Identify key input variables impacting the selected output using AutoML."),
            dbc.Button(children=[html.I(className="fas fa-rocket me-2"), "Run AutoML for Exploration"], id="btn-run-exploration-automl", color="info", className="my-3 btn-lg w-100"),
            dbc.Tooltip("Initiate the AutoML process to discover the most influential factors.", target="btn-run-exploration-automl"),
            dcc.Loading(id="loading-exploration", type="default", children=[html.Div(id="exploration-results-area", className="mt-4")]) # Results area wrapped in loading
        ])
    elif analysis_type == 'optimization':
        return html.Div([
            html.H3(f"Optimization Analysis for Target: {target_column}", className="mb-3 text-success"),
            html.P(f"Find optimal settings for inputs to maximize or minimize '{target_column}'."),
            dbc.Label("Optimization Goal:", className="fw-bold"),
            dcc.Dropdown(id='dropdown-optimization-goal', options=[{'label': 'Maximize Target', 'value': 'maximize'}, {'label': 'Minimize Target', 'value': 'minimize'}], value='maximize', clearable=False, className="mb-3"),
            dbc.Tooltip("Select whether you want to maximize or minimize the target output.", target="dropdown-optimization-goal"),
            html.P("Note: Input variable ranges inferred from data (can be manually set in advanced configs - demo only).", className="small text-muted"),
            dbc.Button(children=[html.I(className="fas fa-bullseye me-2"), "Run AutoML for Optimization"], id="btn-run-optimization-automl", color="success", className="my-3 btn-lg w-100"),
            dbc.Tooltip("Start the AutoML process to find the ideal experimental conditions.", target="btn-run-optimization-automl"),
            dcc.Loading(id="loading-optimization", type="default", children=[html.Div(id="optimization-results-area", className="mt-4")]) # Results area wrapped in loading
        ])
    return "Analysis type not recognized."

# Callback to run Exploration AutoML and display results
@app.callback(
    [Output('exploration-results-area', 'children'),
     Output('store-exploration-results', 'data'),
     Output('tab-suggestions', 'disabled', allow_duplicate=True), 
     Output('store-progress-text', 'data'),
     Output('progress-interval', 'disabled'),
     Output('store-progress-steps', 'data'),
     Output('store-current-step-index', 'data', allow_duplicate=True),
     Output('btn-goto-suggestions-expl', 'style')], # Output to control visibility of the button
    Input('btn-run-exploration-automl', 'n_clicks'),
    State('store-raw-data', 'data'),
    State('store-column-roles', 'data'),
    prevent_initial_call=True
)
def run_exploration_analysis_callback(n_clicks, raw_data_json, column_roles):
    if not n_clicks or not raw_data_json or not column_roles:
        return dash.no_update, dash.no_update, True, None, True, dash.no_update, dash.no_update, {'display': 'none'} # Keep button hidden
    
    df = pd.read_json(raw_data_json, orient='split')
    target_column = column_roles['target_for_analysis']
    feature_columns = column_roles['inputs']
    missing_strategy = column_roles.get('missing_strategy', 'drop_rows')

    # Validate if target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        error_message = f"Error: The target output variable '{target_column}' is not numeric. Please select a numeric target for analysis."
        return dbc.Alert(error_message, color="danger", className="mt-2"), None, True, None, True, [], 0, {'display': 'none'}

    # Check for features that are neither numeric nor clearly categorical (object type but not suitable for OHE)
    problematic_features = []
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            problematic_features.append(col)
    
    if problematic_features:
        error_message = f"Error: The following input variables have data types unsuitable for analysis (neither numeric nor categorical): {', '.join(problematic_features)}. Please ensure inputs are numeric or categorical strings."
        return dbc.Alert(error_message, color="danger", className="mt-2"), None, True, None, True, [], 0, {'display': 'none'}

    try:
        importances, performance_metrics, shap_plot_fig, progress_steps, model_info = run_exploration_automl(df, target_column, feature_columns, missing_strategy)
    except Exception as e:
        error_message = f"An error occurred during AutoML Exploration: {str(e)}. Please check your data and selections."
        print(f"Error in run_exploration_analysis_callback: {e}")
        return dbc.Alert(error_message, color="danger", className="mt-2"), None, True, None, True, [], 0, {'display': 'none'}
    
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("R-squared", className="card-title text-muted"), html.P(f"{performance_metrics.get('R-squared', 0):.3f}", className="card-text fs-3 text-info")])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("MAE (Error)", className="card-title text-muted"), html.P(f"{performance_metrics.get('MAE', 0):.3f}", className="card-text fs-3 text-info")])), md=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Top Feature", className="card-title text-muted"), html.P(f"{importances['feature'].iloc[0] if not importances.empty else 'N/A'}", className="card-text fs-4 text-info text-truncate")])), md=4)
    ], className="mb-4")
    
    results_data_for_store = {
        'importances': importances.to_dict('records'), 
        'performance_metrics': performance_metrics, 
        'target_column': target_column,
        'missing_strategy': missing_strategy, # Pass strategy to LLM for explanation
        'model_info': model_info # Store model info for LLM explanation
    }

    model_details_layout = html.Div([
        html.H5("Model Details", className="mt-4 mb-2"),
        dbc.Card(dbc.CardBody([
            html.P(f"**Model Type:** {model_info.get('Model Type')}"),
            html.P(f"**Training Data Shape:** {model_info.get('Training Data Shape')}"),
            html.P(f"**Test Data Shape:** {model_info.get('Test Data Shape')}"),
            html.P(f"**Features Used:** {', '.join(model_info.get('Features Used', []))}"),
            html.P(f"**Target Column:** {model_info.get('Target Column')}"),
            html.P(f"**Missing Value Strategy:** {model_info.get('Missing Value Strategy')}"),
            html.H6("Hyperparameters:", className="mt-3"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in model_info.get('Hyperparameters', {}).items()])
        ]), className="mb-4 bg-light")
    ])

    results_layout = html.Div([
        dbc.Alert(f"Exploration AutoML complete for '{target_column}'.", color="info", className="mt-2"),
        kpi_cards, 
        dcc.Graph(id='exploration-shap-plot-graph', figure=shap_plot_fig),
        model_details_layout, # Add model details here
        # Removed dynamic button creation here
    ])
    
    return results_layout, results_data_for_store, False, None, True, progress_steps, 0, {'display': 'block'} # Show button, enable tab-suggestions

# Callback to run Optimization AutoML and display results
@app.callback(
    [Output('optimization-results-area', 'children'),
     Output('store-optimization-results', 'data'),
     Output('tab-suggestions', 'disabled', allow_duplicate=True), 
     Output('store-progress-text', 'data', allow_duplicate=True),
     Output('progress-interval', 'disabled', allow_duplicate=True),
     Output('store-progress-steps', 'data', allow_duplicate=True),
     Output('store-current-step-index', 'data', allow_duplicate=True),
     Output('btn-goto-suggestions-opt', 'style')], # Output to control visibility of the button
    Input('btn-run-optimization-automl', 'n_clicks'),
    State('store-raw-data', 'data'),
    State('store-column-roles', 'data'),
    State('dropdown-optimization-goal', 'value'),
    prevent_initial_call=True
)
def run_optimization_analysis_callback(n_clicks, raw_data_json, column_roles, opt_goal):
    if not n_clicks or not raw_data_json or not column_roles:
        return dash.no_update, dash.no_update, True, None, True, dash.no_update, dash.no_update, {'display': 'none'} # Keep button hidden
        
    df = pd.read_json(raw_data_json, orient='split')
    target_column = column_roles['target_for_analysis']
    feature_columns = column_roles['inputs']
    missing_strategy = column_roles.get('missing_strategy', 'drop_rows')

    # Validate if target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        error_message = f"Error: The target output variable '{target_column}' is not numeric. Please select a numeric target for analysis."
        return dbc.Alert(error_message, color="danger", className="mt-2"), None, True, None, True, [], 0, {'display': 'none'}

    # Check for features that are neither numeric nor clearly categorical (object type but not suitable for OHE)
    problematic_features = []
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            problematic_features.append(col)
    
    if problematic_features:
        error_message = f"Error: The following input variables have data types unsuitable for analysis (neither numeric nor categorical): {', '.join(problematic_features)}. Please ensure inputs are numeric or categorical strings."
        return dbc.Alert(error_message, color="danger", className="mt-2"), None, True, None, True, [], 0, {'display': 'none'}

    try:
        feature_ranges = {col: {'min': df[col].min(), 'max': df[col].max()} for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])}
        optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, surrogate_info_str, response_fig, progress_steps, model_info = run_optimization_automl(df, target_column, feature_columns, opt_goal, feature_ranges, missing_strategy)
    except Exception as e:
        error_message = f"An error occurred during AutoML Optimization: {str(e)}. Please check your data and selections."
        print(f"Error in run_optimization_analysis_callback: {e}")
        return dbc.Alert(error_message, color="danger", className="mt-2"), None, True, None, True, [], 0, {'display': 'none'}
    
    kpi_cards_list = []
    for k, v in optimal_settings.items():
        # Conditionally format based on type
        formatted_v = f"{v:.3f}" if isinstance(v, (int, float, np.number)) else str(v)
        kpi_cards_list.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5(f"Optimal {k}", className="card-title text-muted text-truncate", style={'fontSize': '0.9rem'}), html.P(formatted_v, className="card-text fs-5 text-success")])), width=6, lg=3, className="mb-2"))
    
    kpi_cards_list.append(
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(f"Predicted {target_column} ({opt_goal}d)", className="card-title text-muted", style={'fontSize': '0.9rem'}), 
            html.P(f"{predicted_target:.3f}", className="card-text fs-3 fw-bold text-success"),
            html.Small(f"95% CI: {predicted_target_lower:.3f} - {predicted_target_upper:.3f}", className="text-muted")
        ])), width=12, lg=6, className="mb-2")
    )
    kpi_cards_display = dbc.Row(kpi_cards_list, className="mb-4 align-items-stretch")
    
    results_data_for_store = {
        'optimal_settings': optimal_settings, 
        'predicted_target': predicted_target, 
        'predicted_target_lower': predicted_target_lower, 
        'predicted_target_upper': predicted_target_upper, 
        'surrogate_model_info': surrogate_info_str, 
        'target_column': target_column, 
        'goal': opt_goal, 
        'feature_columns': feature_columns,
        'missing_strategy': missing_strategy, # Pass strategy to LLM for explanation
        'model_info': model_info # Store model info for LLM explanation
    }

    model_details_layout = html.Div([
        html.H5("Model Details", className="mt-4 mb-2"),
        dbc.Card(dbc.CardBody([
            html.P(f"**Model Type:** {model_info.get('Model Type')}"),
            html.P(f"**Training Data Shape:** {model_info.get('Training Data Shape')}"),
            html.P(f"**Test Data Shape:** {model_info.get('Test Data Shape')}"),
            html.P(f"**Features Used:** {', '.join(model_info.get('Features Used', []))}"),
            html.P(f"**Target Column:** {model_info.get('Target Column')}"),
            html.P(f"**Missing Value Strategy:** {model_info.get('Missing Value Strategy')}"),
            html.P(f"**Optimization Goal:** {model_info.get('Optimization Goal')}"),
            html.H6("Hyperparameters:", className="mt-3"),
            html.Ul([html.Li(f"{k}: {v}") for k, v in model_info.get('Hyperparameters', {}).items()])
        ]), className="mb-4 bg-light")
    ])

    response_surface_explanation = html.Div([
        html.P(
            """
            **Response Surface Plot Explanation:** This plot visualizes the predicted relationship between the two most impactful numerical features (or selected ones) and the target output, holding all other features constant at their optimal values.
            For experiments with more than two numerical input variables, a single 2D response surface cannot capture all interactions. You can interpret this plot as a slice through the multi-dimensional response surface at the optimal settings of other variables.
            """,
            className="small text-muted mt-3"
        )
    ])


    plot_div = [dcc.Graph(id='optimization-response-surface-plot-graph', figure=response_fig), response_surface_explanation] if response_fig else []
    results_layout = html.Div([
        dbc.Alert(f"Optimization AutoML complete for '{target_column}'. Goal: {opt_goal}.", color="success", className="mt-2"),
        kpi_cards_display, 
        *plot_div, # Unpack the plot and its explanation
        model_details_layout, # Add model details here
        # Removed dynamic button creation here
    ])
    
    return results_layout, results_data_for_store, False, None, True, progress_steps, 0, {'display': 'block'} # Show button, enable tab-suggestions

# Callback to handle explicit tab switching via buttons (now listening to dynamically created buttons)
@app.callback(
    Output('main-tabs', 'active_tab', allow_duplicate=True),
    [Input('btn-goto-suggestions-expl', 'n_clicks'),
     Input('btn-goto-suggestions-opt', 'n_clicks')],
    [State('tab-suggestions', 'disabled')], # Add state to check if it's already enabled
    prevent_initial_call=True
)
def switch_to_suggestions_tab(n1, n2, suggestions_tab_disabled):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Only switch if the button was clicked and the tab is currently disabled
    if (triggered_id == 'btn-goto-suggestions-expl' or triggered_id == 'btn-goto-suggestions-opt'):
        return "tab-suggestions"
    return dash.no_update


@app.callback(
    Output('suggestions-content-div', 'children', allow_duplicate=True),
    [Input('main-tabs', 'active_tab'),
     Input('btn-regenerate-llm', 'n_clicks')],
    [State('store-current-analysis-type', 'data'), 
     State('store-exploration-results', 'data'), 
     State('store-optimization-results', 'data'),
     State('custom-llm-prompt-input', 'value')],
    prevent_initial_call=True
)
def render_suggestions_tab_content(active_tab, n_clicks_regenerate, analysis_type, exploration_data, optimization_data, custom_prompt_value):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if active_tab != 'tab-suggestions' and triggered_id != 'btn-regenerate-llm':
        return dash.no_update 
    
    if not analysis_type: return dbc.Alert("Complete an analysis on Tab 2 first.", color="warning")
    
    header_text, llm_input_data = "", None
    if analysis_type == 'exploration' and exploration_data:
        header_text = f"AI Insights for Exploration: {exploration_data.get('target_column')}"
        llm_input_data = exploration_data
    elif analysis_type == 'optimization' and optimization_data:
        header_text = f"AI Insights for Optimization: {optimization_data.get('target_column')} ({optimization_data.get('goal')})"
        llm_input_data = optimization_data
    else:
        return dbc.Alert("No analysis results available for suggestions. Run analysis on Tab 2.", color="warning")
    
    llm_explanation_markdown = generate_explanation_llm(analysis_type, llm_input_data, custom_prompt_value)
    
    return html.Div([
        html.H3(header_text, className="mb-3 text-primary"),
        dcc.Loading(type="default", children=[ 
            dcc.Markdown(llm_explanation_markdown, className="border p-3 bg-light rounded shadow-sm", style={'lineHeight': '1.6'})
        ])
    ])

if __name__ == '__main__':
    app.run(debug=True)
