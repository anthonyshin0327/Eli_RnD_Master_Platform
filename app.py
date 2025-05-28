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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, export_text # Added export_text
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # Removed mean_absolute_percentage_error for now to simplify NaN handling
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap # For feature importance explanations
import plotly.graph_objects as go # For advanced plotting
import plotly.express as px # For scatter plots

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
        num_pipeline_steps = []
        if numerical_imputer_strategy is not None:
            num_pipeline_steps.append(('imputer', SimpleImputer(strategy=numerical_imputer_strategy)))
        num_pipeline_steps.append(('scaler', StandardScaler())) # Add StandardScaler for numerical features
        numerical_transformer = Pipeline(steps=num_pipeline_steps)
        transformers.append(('num', numerical_transformer, numerical_features))
        
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

# Define models and their hyperparameter grids for GridSearchCV
MODELS_TO_EVALUATE = [
    {
        'name': 'RandomForestRegressor',
        'estimator': RandomForestRegressor(random_state=42, n_jobs=-1),
        'param_grid': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20],
        }
    },
    {
        'name': 'GradientBoostingRegressor',
        'estimator': GradientBoostingRegressor(random_state=42),
        'param_grid': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
        }
    },
    {
        'name': 'SVR',
        'estimator': SVR(),
        'param_grid': {
            'regressor__kernel': ['rbf', 'linear'],
            'regressor__C': [0.1, 1, 10],
        }
    },
    {
        'name': 'LinearRegression',
        'estimator': LinearRegression(),
        'param_grid': {} # No hyperparameters to tune for basic Linear Regression
    },
    {
        'name': 'Ridge',
        'estimator': Ridge(random_state=42),
        'param_grid': {
            'regressor__alpha': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'Lasso',
        'estimator': Lasso(random_state=42),
        'param_grid': {
            'regressor__alpha': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'ElasticNet',
        'estimator': ElasticNet(random_state=42),
        'param_grid': {
            'regressor__alpha': [0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.1, 0.5, 0.9]
        }
    },
    {
        'name': 'KNeighborsRegressor',
        'estimator': KNeighborsRegressor(),
        'param_grid': {
            'regressor__n_neighbors': [3, 5, 7],
            'regressor__weights': ['uniform', 'distance']
        }
    },
    {
        'name': 'DecisionTreeRegressor',
        'estimator': DecisionTreeRegressor(random_state=42),
        'param_grid': {
            'regressor__max_depth': [None, 5, 10, 15]
        }
    }
]


def run_automl_pipeline(data_df, target_column, feature_columns, missing_value_strategy, progress_callback, analysis_type, optimization_goal=None, feature_ranges=None):
    """
    Generalized function to run AutoML for both exploration and optimization.
    """
    print(f"Starting AutoML pipeline for {analysis_type} analysis on target: {target_column}")
    
    all_model_results = []
    best_model_info = None
    best_r2 = -float('inf')
    total_models = len(MODELS_TO_EVALUATE)
    
    df_processed = data_df.copy()
    imputer_strategy_for_pipeline = None

    # Handle missing values based on strategy
    if missing_value_strategy == 'drop_rows':
        # Ensure only feature_columns and target_column are used for dropna, to avoid dropping rows due to NaNs in ignored columns
        cols_for_dropna = feature_columns + [target_column]
        df_processed = data_df[cols_for_dropna].dropna()
        # After dropna, df_processed might have fewer rows. X and y should be derived from this df_processed.
        imputer_strategy_for_pipeline = None 
    elif missing_value_strategy == 'impute_mode':
        imputer_strategy_for_pipeline = 'most_frequent' # This will be used for numerical if selected
    else: # 'impute_mean', 'impute_median'
        imputer_strategy_for_pipeline = missing_value_strategy
    
    # Select features (X) and target (y) from the (potentially row-reduced) df_processed
    X = df_processed[feature_columns]
    y = df_processed[target_column]

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Check if X or y is empty after potential dropna
    if X.empty or y.empty:
        raise ValueError("After handling missing values, the feature set (X) or target (y) is empty. Please check your data or missing value strategy.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the base preprocessor. For 'drop_rows', numerical_imputer_strategy will be None.
    # For other strategies, it will be 'mean', 'median', or 'most_frequent'.
    base_preprocessor = create_preprocessing_pipeline(numerical_features, categorical_features, 
                                                      imputer_strategy_for_pipeline if missing_value_strategy != 'drop_rows' else None)

    progress_steps = [f"Training and optimizing {model['name']} ({i+1}/{total_models})" for i, model in enumerate(MODELS_TO_EVALUATE)]
    progress_steps.append("Calculating SHAP values for the best model...")
    if analysis_type == 'optimization':
        progress_steps.append("Generating Surrogate Tree Logic...")
        progress_steps.append("Performing Optimization (Grid Search)...")
        progress_steps.append("Evaluating Optimal Settings...")
        progress_steps.append("Generating Optimization Results...")
    progress_steps.append("Finalizing results and plots...")

    progress_callback(progress_steps, 0)

    best_model_pipeline = None # Initialize best_model_pipeline

    for i, model_config in enumerate(MODELS_TO_EVALUATE):
        model_name = model_config['name']
        estimator = model_config['estimator']
        param_grid = model_config['param_grid']

        progress_callback(progress_steps, i)

        full_pipeline = Pipeline(steps=[
            ('preprocessor', base_preprocessor),
            ('regressor', estimator)
        ])
        
        start_time = time.time()
        
        if param_grid:
            grid_search = GridSearchCV(full_pipeline, param_grid, cv=min(5, len(X_train) if len(X_train) > 1 else 2), scoring='r2', n_jobs=-1) # Ensure cv is not > n_samples
            grid_search.fit(X_train, y_train)
            current_best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
        else:
            current_best_pipeline = full_pipeline
            current_best_pipeline.fit(X_train, y_train)
            best_params = {}
            # Calculate R2 on training data if no CV
            y_train_pred = current_best_pipeline.predict(X_train)
            best_cv_score = r2_score(y_train, y_train_pred)


        training_time = time.time() - start_time

        y_pred = current_best_pipeline.predict(X_test)

        r_squared = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate MAPE carefully, avoiding division by zero
        # Ensure y_test is a numpy array for boolean indexing
        y_test_np = np.array(y_test)
        non_zero_mask = y_test_np != 0
        if np.any(non_zero_mask): # Check if there are any non-zero values in y_test
            mape = np.mean(np.abs((y_test_np[non_zero_mask] - y_pred[non_zero_mask]) / y_test_np[non_zero_mask])) * 100
        else:
            mape = np.nan # Or some other indicator like 'N/A' or None

        mape_serializable = None if np.isnan(mape) else mape


        model_result = {
            "Model Type": model_name,
            "R-squared": r_squared,
            "MAE": mae,
            "RMSE": np.sqrt(mse),
            "MAPE": mape_serializable,
            "Best Hyperparameters": best_params,
            "Cross-Validation R2": best_cv_score,
            "Training Time (s)": training_time,
            "Pipeline": current_best_pipeline # Keep pipeline for potential future use
        }
        all_model_results.append(model_result)

        if r_squared > best_r2:
            best_r2 = r_squared
            best_model_info = model_result
            best_model_pipeline = current_best_pipeline # Update the best pipeline

    progress_callback(progress_steps, total_models) # SHAP calculation step

    # --- SHAP Feature Importance for the BEST Model ---
    shap_summary_plot = {}
    importances_df = pd.DataFrame()
    X_train_transformed_df = pd.DataFrame() # Initialize for scope

    if best_model_pipeline:
        # Ensure X_train is not empty for transformation
        if X_train.empty:
            print("Warning: X_train is empty, skipping SHAP value calculation.")
        else:
            X_train_transformed = best_model_pipeline.named_steps['preprocessor'].transform(X_train)
            
            feature_names_out = []
            try:
                # Attempt to get feature names from the preprocessor
                feature_names_out = best_model_pipeline.named_steps['preprocessor'].get_feature_names_out()
            except AttributeError:
                # Fallback for older sklearn or complex cases: reconstruct manually
                ohe_feature_names = []
                if 'cat' in best_model_pipeline.named_steps['preprocessor'].named_transformers_:
                    cat_transformer_pipeline = best_model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
                    if 'onehot' in cat_transformer_pipeline.named_steps:
                        ohe_transformer = cat_transformer_pipeline.named_steps['onehot']
                        if categorical_features: # Ensure categorical_features is not empty
                             try:
                                ohe_feature_names = ohe_transformer.get_feature_names_out(categorical_features).tolist()
                             except Exception as e:
                                print(f"Warning: Could not get OHE feature names: {e}. Using original categorical names.")
                                ohe_feature_names = categorical_features # Fallback
                
                # Combine numerical and OHE feature names
                feature_names_out = numerical_features + ohe_feature_names
                
                # Further fallback if still no names, use original X_train columns if shapes match
                if not feature_names_out or len(feature_names_out) != X_train_transformed.shape[1]:
                    if X_train.shape[1] == X_train_transformed.shape[1]: # If original features match transformed shape (e.g. no OHE)
                         feature_names_out = X_train.columns.tolist()
                    else: # Generic names if all else fails
                        feature_names_out = [f"feature_{j}" for j in range(X_train_transformed.shape[1])]
                print(f"Warning: get_feature_names_out() fallback used. Feature names: {feature_names_out}")


            X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names_out)

            if not X_train_transformed_df.empty:
                # Sample for SHAP to keep it manageable
                n_shap_samples = min(100, X_train_transformed_df.shape[0]) # Reduced sample size
                if n_shap_samples > 0:
                    X_shap = X_train_transformed_df.sample(n=n_shap_samples, random_state=42)
                    
                    regressor_step = best_model_pipeline.named_steps['regressor']
                    # Use appropriate SHAP explainer
                    if isinstance(regressor_step, (RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor)):
                        explainer = shap.TreeExplainer(regressor_step)
                    else:
                        # For KernelExplainer, background data should be small
                        n_background_samples = min(50, X_train_transformed_df.shape[0])
                        if n_background_samples > 0:
                            background_data = shap.sample(X_train_transformed_df, n_background_samples)
                            explainer = shap.KernelExplainer(regressor_step.predict, background_data)
                        else:
                            explainer = None # Not enough data for background
                            print("Warning: Not enough data for KernelExplainer background. Skipping SHAP plot.")
                    
                    if explainer:
                        shap_values = explainer.shap_values(X_shap)

                        # Handle different SHAP value structures (e.g., for multi-output or some explainers)
                        if isinstance(shap_values, list): # Typically for multi-output, take the first output
                            shap_values_abs_mean = np.abs(shap_values[0]).mean(axis=0)
                        else:
                            shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

                        importances_df = pd.DataFrame({
                            'feature': X_shap.columns,
                            'importance': shap_values_abs_mean
                        }).sort_values(by='importance', ascending=False)
                        
                        shap_summary_plot = {
                            'data': [go.Bar(x=importances_df['feature'], y=importances_df['importance'], marker_color='#17A2B8' if analysis_type == 'exploration' else '#28a745')],
                            'layout': {
                                'title': f'Feature Importance (Mean Absolute SHAP Value) for Best Model ({best_model_info["Model Type"]})',
                                'xaxis': {'title': 'Feature', 'tickangle': 45},
                                'yaxis': {'title': 'Mean Absolute SHAP Value'},
                                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 150}, # Increased bottom margin
                                'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#495057'}
                            }
                        }
                else:
                    print("Warning: Not enough samples for SHAP X_shap. Skipping SHAP plot.")
    
    if analysis_type == 'exploration':
        progress_callback(progress_steps, total_models + 1) # Finalizing results
        print("Exploration AutoML complete.")
        return all_model_results, best_model_info, importances_df, shap_summary_plot, progress_steps

    elif analysis_type == 'optimization':
        progress_callback(progress_steps, total_models + 1) # Surrogate Tree Logic step

        # --- Surrogate Tree Logic (Textual Representation) ---
        surrogate_tree_text = "Not applicable (no transformed data or features for tree visualization)."
        if not X_train_transformed_df.empty and X_train_transformed_df.shape[0] > 0:
            # Ensure y_hat_best_model is not empty
            if best_model_pipeline and not X_train.empty: # Use original X_train for predict
                y_hat_best_model = best_model_pipeline.predict(X_train) 
                interpretable_tree = DecisionTreeRegressor(max_depth=3, random_state=42)  
                # Fit interpretable_tree on transformed data and predictions from original X_train
                interpretable_tree.fit(X_train_transformed_df, y_hat_best_model) 
                
                tree_feature_names = X_train_transformed_df.columns.tolist()
                surrogate_tree_text = export_text(interpretable_tree, feature_names=tree_feature_names)
                print("Surrogate Tree Text:\n", surrogate_tree_text)
        
        progress_callback(progress_steps, total_models + 2) # Performing Optimization (Grid Search)

        optimal_settings = {col: "N/A" for col in feature_columns}
        predicted_target = 0.0
        predicted_target_lower = 0.0
        predicted_target_upper = 0.0
        response_surface_fig = {}

        # Define grid points for numerical features based on provided ranges
        grid_points_num = {}
        for col in numerical_features: # Iterate over numerical features identified from X
            if col in feature_ranges: # Ensure the column has a defined range
                min_val = feature_ranges[col]['min']
                max_val = feature_ranges[col]['max']
                grid_points_num[col] = np.linspace(min_val, max_val, 10) # 10 points for grid search
            else:
                # If a numerical feature has no range, use its mean from original data as a fixed value
                grid_points_num[col] = np.array([X[col].mean()])


        # Define grid points for categorical features from their unique values in X
        grid_points_cat = {}
        for col in categorical_features: # Iterate over categorical features identified from X
             grid_points_cat[col] = X[col].unique().tolist()


        from itertools import product
        
        all_combinations = []
        
        # Create a list of (feature_name, iterable_of_values) for product
        # Only include features that are actually part of the input `feature_columns`
        iterables_for_product_details = []
        for col in feature_columns:
            if col in grid_points_num: # It's a numerical feature we are considering
                iterables_for_product_details.append((col, grid_points_num[col]))
            elif col in grid_points_cat: # It's a categorical feature we are considering
                iterables_for_product_details.append((col, grid_points_cat[col]))
        
        current_iterables = [item[1] for item in iterables_for_product_details]
        current_feature_names_ordered = [item[0] for item in iterables_for_product_details]

        if not current_iterables and feature_columns: 
            # Case: No features to vary (e.g., all are fixed single values), but features were selected.
            # Construct a single row with mean/mode.
            fixed_row_dict = {}
            for col in feature_columns:
                if col in numerical_features:
                    fixed_row_dict[col] = X[col].mean() 
                elif col in categorical_features:
                    fixed_row_dict[col] = X[col].mode()[0] if not X[col].mode().empty else X[col].unique()[0] if X[col].nunique() > 0 else None
            all_combinations.append(fixed_row_dict)
        elif current_iterables:
            for combo_values in product(*current_iterables):
                current_combo_dict = {}
                for i, feature_name in enumerate(current_feature_names_ordered):
                    current_combo_dict[feature_name] = combo_values[i]
                
                # Ensure all original feature_columns are present, filling fixed ones if not varied
                for original_col in feature_columns:
                    if original_col not in current_combo_dict:
                        if original_col in numerical_features:
                             current_combo_dict[original_col] = X[original_col].mean()
                        elif original_col in categorical_features:
                             current_combo_dict[original_col] = X[original_col].mode()[0] if not X[original_col].mode().empty else X[original_col].unique()[0] if X[original_col].nunique() > 0 else None
                all_combinations.append(current_combo_dict)

        if not all_combinations: # If still no combinations (e.g. no features selected)
            # This case should ideally be prevented by UI validation (requiring feature selection)
            # If it occurs, we can't proceed with prediction.
            raise ValueError("No combinations generated for optimization grid search. Ensure features are selected and have valid ranges/values.")

        optimization_df = pd.DataFrame(all_combinations, columns=feature_columns) # Ensure columns are in original order
        
        # --- DEBUG PRINT (Optional: can be removed in production) ---
        print(f"DEBUG (optimization_df columns before predict): {optimization_df.columns.tolist()}")
        print(f"DEBUG (feature_columns expected by pipeline): {feature_columns}")
        print(f"DEBUG (First 5 rows of optimization_df):\n{optimization_df.head()}")
        # --- END DEBUG ---

        # Explicitly use feature_columns for prediction to ensure correct columns and order
        optimization_predictions = best_model_pipeline.predict(optimization_df[feature_columns])


        if optimization_goal == 'maximize':
            optimal_idx = np.argmax(optimization_predictions)
        else:
            optimal_idx = np.argmin(optimization_predictions)

        optimal_settings_series = optimization_df.iloc[optimal_idx]
        optimal_settings = optimal_settings_series.to_dict()
        predicted_target = optimization_predictions[optimal_idx]

        # Estimate confidence interval for the prediction (simple approach using std of test predictions)
        if not X_test.empty:
            test_predictions = best_model_pipeline.predict(X_test[feature_columns]) # Use original X_test features
            prediction_std = np.std(test_predictions)
            predicted_target_lower = predicted_target - 1.96 * prediction_std 
            predicted_target_upper = predicted_target + 1.96 * prediction_std
        else: # Fallback if X_test was empty
            predicted_target_lower = predicted_target 
            predicted_target_upper = predicted_target


        # Plotting response surface if 1 or 2 numerical features are in feature_columns
        # and were part of the optimization grid (i.e., in grid_points_num and varied)
        
        # Identify numerical features that were actually varied (more than 1 point in their grid)
        varied_numerical_features = [col for col in numerical_features if col in feature_columns and col in grid_points_num and len(grid_points_num[col]) > 1]

        if len(varied_numerical_features) == 2:
            x_feat, y_feat = varied_numerical_features[0], varied_numerical_features[1]
            
            x_plot = grid_points_num[x_feat] # Use the actual grid points
            y_plot = grid_points_num[y_feat]
            X_grid, Y_grid = np.meshgrid(x_plot, y_plot)

            plot_df_rows = []
            for i in range(X_grid.shape[0]):
                for j in range(X_grid.shape[1]):
                    row = {x_feat: X_grid[i, j], y_feat: Y_grid[i, j]}
                    # Fill other features from optimal_settings (or their fixed values if not in optimal_settings)
                    for col in feature_columns:
                        if col not in row: # If not one of the varying numerical features
                            row[col] = optimal_settings.get(col, X[col].mean() if col in numerical_features else X[col].mode()[0] if col in categorical_features and not X[col].mode().empty else None)
                    plot_df_rows.append(row)
            
            plot_df = pd.DataFrame(plot_df_rows, columns=feature_columns) # Ensure correct column order
            
            Z_grid_flat = best_model_pipeline.predict(plot_df[feature_columns]) # Explicitly use feature_columns
            Z_grid = Z_grid_flat.reshape(X_grid.shape)

            response_surface_fig = {
                'data': [go.Contour(x=x_plot, y=y_plot, z=Z_grid, colorscale='Viridis',
                                     contours_coloring='heatmap', line_width=0, colorbar_title=target_column)],
                'layout': {
                    'title': f'Predicted Response Surface for {target_column}',
                    'xaxis': {'title': x_feat},
                    'yaxis': {'title': y_feat},
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100},
                    'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#495057'},
                    'height': 500
                }
            }
        elif len(varied_numerical_features) == 1:
            x_feat = varied_numerical_features[0]
            x_plot = grid_points_num[x_feat] # Use actual grid points
            
            plot_df_rows = []
            for val_x in x_plot:
                row = {x_feat: val_x}
                for col in feature_columns:
                     if col not in row:
                        row[col] = optimal_settings.get(col, X[col].mean() if col in numerical_features else X[col].mode()[0] if col in categorical_features and not X[col].mode().empty else None)
                plot_df_rows.append(row)

            plot_df = pd.DataFrame(plot_df_rows, columns=feature_columns)
            y_plot_vals = best_model_pipeline.predict(plot_df[feature_columns]) # Explicitly use feature_columns
            
            response_surface_fig = {
                'data': [go.Scatter(x=x_plot, y=y_plot_vals, mode='lines', line_color='#28a745')],
                'layout': {
                    'title': f'Predicted Response Curve for {target_column} vs {x_feat}',
                    'xaxis': {'title': x_feat},
                    'yaxis': {'title': target_column},
                    'margin': {'l': 50, 'r': 50, 't': 50, 'b': 100},
                    'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)', 'font': {'color': '#495057'},
                    'height': 500
                }
            }
        else: # 0 or >2 varied numerical features
            response_surface_fig = {}


        progress_callback(progress_steps, total_models + 3) # Evaluating Optimal Settings

        # Prepare model_info for the optimization results
        # Ensure best_model_info (from the model selection phase) is available
        if best_model_info:
             # Get params from the regressor step of the best pipeline
            regressor_params = best_model_info['Pipeline'].named_steps['regressor'].get_params()
            # Filter for simpler display if needed, or show all
            hyperparams_to_show = {k: v for k, v in regressor_params.items() if '__' not in k or k.startswith('n_estimators') or k.startswith('random_state') or k.startswith('max_depth') or k.startswith('learning_rate') or k.startswith('C') or k.startswith('kernel') or k.startswith('alpha') or k.startswith('l1_ratio') or k.startswith('n_neighbors') or k.startswith('weights')}


            model_info_for_opt_summary = {
                "Model Type": best_model_info.get('Model Type', "N/A"),
                "Hyperparameters": hyperparams_to_show, # Use filtered hyperparams
                "Training Data Shape": X_train.shape if not X_train.empty else "N/A",
                "Test Data Shape": X_test.shape if not X_test.empty else "N/A",
                "Features Used": feature_columns,
                "Target Column": target_column,
                "Missing Value Strategy": missing_value_strategy,
                "Optimization Goal": optimization_goal
            }
        else: # Fallback if best_model_info wasn't populated (should not happen in normal flow)
            model_info_for_opt_summary = {"Error": "Best model information not found."}


        progress_callback(progress_steps, total_models + 4) # Generating Optimization Results
        print("Optimization AutoML complete.")
        return all_model_results, best_model_info, importances_df, shap_summary_plot, progress_steps, \
               optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, \
               response_surface_fig, surrogate_tree_text, model_info_for_opt_summary


def run_exploration_automl(data_df, target_column, feature_columns, missing_value_strategy, progress_callback):
    return run_automl_pipeline(data_df, target_column, feature_columns, missing_value_strategy, progress_callback, 'exploration')

def run_optimization_automl(data_df, target_column, feature_columns, optimization_goal, feature_ranges, missing_value_strategy, progress_callback):
    return run_automl_pipeline(data_df, target_column, feature_columns, missing_value_strategy, progress_callback, 'optimization', 
                               optimization_goal=optimization_goal, feature_ranges=feature_ranges)


def generate_explanation_llm(analysis_type, results_data, custom_prompt_addition=""):
    print(f"Generating insights for {analysis_type} (LLM simulation)...")
    # Simplified results_data for prompt to avoid excessive length if full model details are included
    prompt_data = {}
    if analysis_type == "Exploration":
        prompt_data = {
            'target_column': results_data.get('target_column'),
            'best_model_type': results_data.get('best_model_info', {}).get('Model Type'),
            'best_model_r_squared': results_data.get('best_model_info', {}).get('R-squared'),
            'top_features': [f['feature'] for f in results_data.get('importances', [])[:3]],
            'missing_strategy': results_data.get('missing_strategy')
        }
    elif analysis_type == "Optimization":
         prompt_data = {
            'target_column': results_data.get('target_column'),
            'optimization_goal': results_data.get('goal'),
            'surrogate_model_type': results_data.get('model_info', {}).get('Model Type'),
            'optimal_settings': results_data.get('optimal_settings'),
            'predicted_target': results_data.get('predicted_target'),
            'top_features_in_surrogate': [f['feature'] for f in results_data.get('importances', [])[:3]],
            'missing_strategy': results_data.get('missing_strategy')
        }

    base_prompt = f"The R&D team has performed an '{analysis_type}' analysis. Key results are: {json.dumps(prompt_data)}. Provide a detailed, human-like explanation, focusing on actionable insights for synthetic biology and protein engineering."
    
    final_prompt = f"{base_prompt} {custom_prompt_addition}".strip()
    print(f"LLM Prompt (simulated): {final_prompt[:500]}...") # Print start of prompt for debugging

    time.sleep(1) # Simulate API call
    print("LLM explanation generation complete.")
    if analysis_type == "Exploration":
        all_model_results = results_data.get('all_model_results', [])
        best_model_info = results_data.get('best_model_info', {})
        importances_df_data = results_data.get('importances', []) # This is a list of dicts
        importances_df = pd.DataFrame(importances_df_data) if importances_df_data else pd.DataFrame()
        
        top_features_str = ", ".join(importances_df.head(3)['feature'].tolist()) if not importances_df.empty else "N/A"
        
        best_model_name = best_model_info.get('Model Type', 'N/A')
        best_r_squared = best_model_info.get('R-squared', 'N/A')
        best_mae = best_model_info.get('MAE', 'N/A')
        best_rmse = best_model_info.get('RMSE', 'N/A')
        best_mape = best_model_info.get('MAPE', 'N/A') # Already serializable (None or float)
        
        best_r_squared_val = best_r_squared if isinstance(best_r_squared, (int, float)) else 0.0

        model_comparison_summary = ""
        if len(all_model_results) > 1:
            model_comparison_summary = "\n\n**Model Comparison Snapshot:**\n"
            for model in all_model_results:
                r2_val = f"{model.get('R-squared', 0):.3f}" if isinstance(model.get('R-squared'), (int, float)) else "N/A"
                mae_val = f"{model.get('MAE', 0):.3f}" if isinstance(model.get('MAE'), (int, float)) else "N/A"
                rmse_val = f"{model.get('RMSE', 0):.3f}" if isinstance(model.get('RMSE'), (int, float)) else "N/A"
                model_comparison_summary += f"- **{model.get('Model Type', 'Unknown')}**: R²={r2_val}, MAE={mae_val}, RMSE={rmse_val}\n"

        mape_display = f"{best_mape:.2f}%" if isinstance(best_mape, (int, float)) and not np.isnan(best_mape) else "N/A"
        r_squared_display = f"{best_r_squared:.3f}" if isinstance(best_r_squared, (int, float)) else "N/A"
        mae_display = f"{best_mae:.3f}" if isinstance(best_mae, (int, float)) else "N/A"
        rmse_display = f"{best_rmse:.3f}" if isinstance(best_rmse, (int, float)) else "N/A"


        return f"""
        #### **Exploration Analysis: Uncovering Key Factors**
        **Objective:** Our automated exploration aimed to pinpoint which of your input variables (we call them 'features') have the most significant impact on your chosen output, '{results_data.get('target_column', 'N/A')}'.
        **Methodology:** We employed an **AutoML pipeline** that systematically evaluated multiple machine learning models including **Random Forest, Gradient Boosting, SVR, Linear Regression, Ridge, Lasso, Elastic Net, KNeighborsRegressor, and Decision Tree Regressor**. For each model, **hyperparameters were optimized using Grid Search with cross-validation** to ensure the best possible performance and robustness. Data preprocessing involved handling missing values using a **'{results_data.get('missing_strategy', 'default')}'** strategy and applying **One-Hot Encoding for categorical features and Standard Scaling for numerical features**. The dataset was split into training and testing sets for robust model evaluation.
        **Why '{best_model_name}' was chosen as the best:**
        The **{best_model_name}** model was selected as the best performer primarily due to its superior **R-squared value of {r_squared_display}** on the test set, indicating it explains the highest proportion of variance in your target variable compared to other models. It also demonstrated strong performance across other metrics like MAE and RMSE.
        **What We Found (Best Model: {best_model_name}):**
        * **Top Influencers:** The analysis, primarily driven by **SHAP (SHapley Additive exPlanations) values**, suggests that **{top_features_str}** are the most critical variables influencing '{results_data.get('target_column', 'N/A')}'. SHAP values provide a clear and transparent way to understand how each feature contributes to the model's predictions.
        * **Model Fit (R-squared):** The best model achieved an R-squared of **{r_squared_display}**, indicating that approximately **{best_r_squared_val*100:.0f}%** of the variance in your target variable can be explained by the input features.
        * **Mean Absolute Error (MAE):** The MAE was **{mae_display}**.
        * **Root Mean Squared Error (RMSE):** The RMSE was **{rmse_display}**.
        * **Mean Absolute Percentage Error (MAPE):** The MAPE was **{mape_display}** (if applicable).
        {model_comparison_summary}
        **Actionable Insights & Suggestions:**
        1.  **Prioritize Investigation:** Focus your future experimental efforts on **{top_features_str}**.
        2.  **Mechanistic Understanding:** Delve into the underlying mechanisms explaining why these top features are influential.
        3.  **Targeted Optimization:** Consider these high-impact features for fine-tuning in optimization experiments.
        4.  **Data Quality Review:** Assess data quality for these key features.
        """
    elif analysis_type == "Optimization":
        optimal_settings_data = results_data.get('optimal_settings', {})
        optimal_settings_str = ", ".join([f"**{k}**: {v:.3f}" if isinstance(v, (int, float, np.number)) and not np.isnan(v) else f"**{k}**: {v}" for k, v in optimal_settings_data.items()])
        goal_verb = "maximize" if results_data.get('goal') == "maximize" else "minimize"
        
        predicted_val = results_data.get('predicted_target', 'N/A')
        predicted_lower = results_data.get('predicted_target_lower', 'N/A')
        predicted_upper = results_data.get('predicted_target_upper', 'N/A')
        
        importances_opt_df_data = results_data.get('importances', [])
        importances_opt_df = pd.DataFrame(importances_opt_df_data) if importances_opt_df_data else pd.DataFrame()
        top_shap_features_str = ", ".join(importances_opt_df.head(3)['feature'].tolist()) if not importances_opt_df.empty else "N/A"

        predicted_val_num = predicted_val if isinstance(predicted_val, (int, float)) and not np.isnan(predicted_val) else 0.0
        predicted_lower_num = predicted_lower if isinstance(predicted_lower, (int, float)) and not np.isnan(predicted_lower) else 0.0
        predicted_upper_num = predicted_upper if isinstance(predicted_upper, (int, float)) and not np.isnan(predicted_upper) else 0.0
        
        surrogate_tree_logic = results_data.get('surrogate_tree_text', 'N/A')
        tree_explanation_part = ""
        if surrogate_tree_logic and surrogate_tree_logic != "Not applicable (no transformed data or features for tree visualization)." and surrogate_tree_logic != "Not applicable (no numerical features or transformed data for tree visualization).": # Added one more check
            tree_explanation_part = f"""
        * **Surrogate Tree Logic:** A simplified decision tree approximated the surrogate model's behavior. Its rules offer insights into decision paths.
        ```
        {surrogate_tree_logic[:1000]}... 
        ```
        (Full tree logic available in detailed results)
        """
        
        best_model_name_opt = results_data.get('model_info', {}).get('Model Type', 'N/A')
        
        model_comparison_summary_opt = ""
        all_opt_models = results_data.get('all_model_results', [])
        if len(all_opt_models) > 1:
            model_comparison_summary_opt = "\n\n**Surrogate Model Candidates Comparison:**\n"
            for model in all_opt_models:
                r2_val = f"{model.get('R-squared',0):.3f}" if isinstance(model.get('R-squared'), (int, float)) else "N/A"
                mae_val = f"{model.get('MAE',0):.3f}" if isinstance(model.get('MAE'), (int, float)) else "N/A"
                rmse_val = f"{model.get('RMSE',0):.3f}" if isinstance(model.get('RMSE'), (int, float)) else "N/A"
                model_comparison_summary_opt += f"- **{model.get('Model Type', 'Unknown')}**: R²={r2_val}, MAE={mae_val}, RMSE={rmse_val}\n"

        return f"""
        #### **Optimization Analysis: Finding the Sweet Spot**
        **Objective:** To identify optimal experimental conditions to **{goal_verb}** '{results_data.get('target_column', 'N/A')}'.
        **Methodology:** An **AutoML pipeline** selected the best predictive model (the **'{best_model_name_opt}'**) from several candidates (including Random Forest, Gradient Boosting, etc.) using hyperparameter optimization and cross-validation. This best model served as a surrogate. Data preprocessing involved a **'{results_data.get('missing_strategy', 'default')}'** strategy for missing values, plus One-Hot Encoding and Standard Scaling. A **grid search** across input variable ranges then identified conditions to meet your optimization goal.
        **Optimal Settings Suggested:** {optimal_settings_str}
        **Predicted Outcome:** Approx. target value of **{predicted_val_num:.3f}** (95% CI: {predicted_lower_num:.3f} - {predicted_upper_num:.3f}).
        **Surrogate Model Insights ({best_model_name_opt}):**
        * **SHAP analysis** indicates **{top_shap_features_str}** were most influential.
        * The **response surface plot** (if applicable) visually shows how key inputs impact the output.
        {tree_explanation_part}
        {model_comparison_summary_opt}
        **Actionable Insights & Suggestions:**
        1.  **Experimental Validation:** Crucially, **validate these settings in your lab**.
        2.  **Sensitivity Analysis:** Experiment by slightly varying settings to understand robustness.
        3.  **Practical Constraints:** Consider lab constraints when implementing.
        4.  **Iterative Optimization:** Use results to refine ranges for further optimization rounds.
        """
    return "Placeholder explanation: LLM analysis type not recognized or data missing."

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
    # Limit display to first N rows for performance if df is large
    display_df = df.head(50) if len(df) > 50 else df
    
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in display_df.columns],
        page_size=8,
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 
                    'minWidth': '100px', 'width': '150px', 'maxWidth': '200px', 
                    'whiteSpace': 'normal', 'textOverflow': 'ellipsis'}, # Added ellipsis
        style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold', 'borderBottom': '2px solid #dee2e6'},
        style_data={'borderBottom': '1px solid #dee2e6'},
        sort_action='native',
        filter_action='native',
        tooltip_data=[ # Add tooltips for truncated cell data
            {
                column: {'value': str(value), 'type': 'markdown'}
                for column, value in row.items()
            } for row in display_df.to_dict('records')
        ],
        tooltip_duration=None
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
                {'label': 'Drop Rows with Missing Data', 'value': 'drop_rows'},
                {'label': 'Impute with Mean (Numeric only)', 'value': 'mean'}, # Changed value
                {'label': 'Impute with Median (Numeric only)', 'value': 'median'}, # Changed value
                {'label': 'Impute with Mode (Numeric & Categorical)', 'value': 'most_frequent'}, # Changed value
            ],
            value='drop_rows', clearable=False, className="mb-3",
            placeholder="Select how to handle missing data"
        ),
        dbc.Tooltip("Choose a strategy to handle any missing values in your dataset. 'Impute with Mode' will use mode for numerical if chosen.", target="dropdown-missing-strategy"),

        dbc.Label("Ignore Columns (Optional):", html_for='dropdown-ignore-vars', className="fw-bold"),
        dcc.Dropdown(id='dropdown-ignore-vars', options=column_options, multi=True, placeholder="Select columns to exclude", className="mb-3"),
        dbc.Tooltip("Columns selected here will be excluded from the analysis.", target="dropdown-ignore-vars"),

        dbc.Button(children=[html.I(className="fas fa-cogs me-2"), "Confirm Setup & Proceed to Analysis"], id="btn-confirm-setup", color="primary", className="mt-3 w-100 btn-lg")
    ])
    status_message = dbc.Alert(f"Successfully loaded: {filename_display}. Displaying first {len(display_df)} of {len(df)} rows.", color="success", duration=4000) if len(df) > 50 else dbc.Alert(f"Successfully loaded: {filename_display}", color="success", duration=4000)
    stored_data = df.to_json(date_format='iso', orient='split')
    return status_message, table, stored_data, role_assignment_ui

# ========= App Layout =========
app.layout = dbc.Container(fluid=True, className="bg-light min-vh-100", children=[ # Added bg-light and min-vh-100
    dcc.Store(id='store-raw-data'),
    dcc.Store(id='store-column-roles'), 
    dcc.Store(id='store-exploration-results'),
    dcc.Store(id='store-optimization-results'),
    dcc.Store(id='store-current-analysis-type'),
    dcc.Store(id='store-progress-text'), 
    dcc.Interval(id='progress-interval', interval=300, n_intervals=0, disabled=True), # Slightly slower interval
    dcc.Store(id='store-progress-steps', data=[]),
    dcc.Store(id='store-current-step-index', data=0),

    dbc.Row(dbc.Col(html.H1(children=[html.I(className="fas fa-flask me-2"), "R&D Experiment Analysis Platform"], className="text-center my-4 display-5 text-primary fw-bold"), width=12)), # Adjusted size and weight
    
    # Guided Workflow / Stepper
    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("1. Data & Setup", active=True, href="#", id="nav-data-setup", className="fw-medium fs-5 p-2")), # Increased padding and font size
                    dbc.NavItem(dbc.NavLink("2. Analysis & Results", active=False, href="#", id="nav-analysis", disabled=True, className="fw-medium fs-5 p-2")),
                    dbc.NavItem(dbc.NavLink("3. AI Insights & Suggestions", active=False, href="#", id="nav-suggestions", disabled=True, className="fw-medium fs-5 p-2")),
                ],
                pills=True, # Changed to pills for better visual separation
                className="nav-pills justify-content-center mb-4 shadow-sm bg-white rounded p-2" # Added shadow, bg, rounded
            ), width=12
        )
    ]),

    dbc.Row(dbc.Col(html.Div(id='global-progress-message', className="text-center my-3"), width=12)), # Removed custom styling, will rely on Progress component
    
    # Main content area using dbc.Card for better structure
    dbc.Card(dbc.CardBody([
        dbc.Tabs(id="main-tabs", active_tab="tab-data-upload", className="mb-3", children=[ # Removed style display none
                dbc.Tab(label="Data & Setup", tab_id="tab-data-upload", children=[
                    dbc.Card(dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H4(children=[html.I(className="fas fa-upload me-2"), "Upload Experiment Data"], className="mb-3 text-secondary"),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(['Drag and Drop or ', html.A('Select Files (.csv, .xlsx)')]),
                                    style={
                                        'width': '100%', 'height': '80px', 'lineHeight': '80px',
                                        'borderWidth': '2px', 'borderStyle': 'dashed',
                                        'borderRadius': '8px', 'textAlign': 'center', 'margin': '10px 0',
                                        'backgroundColor': '#f8f9fa', 'borderColor': '#adb5bd' # Softer border
                                    },
                                    multiple=False, className="mb-2"
                                ),
                                html.Div("Or", className="text-center my-2 small text-muted"),
                                dbc.Button(children=[html.I(className="fas fa-database me-2"),"Load Demo Dataset (LFA Data)"], id="btn-load-demo", color="info", outline=True, className="w-100 mb-3"), # Changed color
                                html.Div(id='output-data-upload-status', className="mt-2"),
                                html.Div(id='output-datatable-div', className="mt-3", style={'maxHeight': '350px', 'overflowY': 'auto', 'overflowX': 'auto', 'border': '1px solid #dee2e6', 'borderRadius': '5px'}) # Added border
                            ], md=7, className="p-3 border-end"),
                            dbc.Col([
                                html.H4(children=[html.I(className="fas fa-sliders-h me-2"), "Experiment Configuration"], className="mb-3 text-secondary"),
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
                    ]), className="mt-0 shadow-sm") # Removed mt-3 to align with tab, added shadow
                ]),
                dbc.Tab(label="Analysis & Results", tab_id="tab-analysis", id="tab-analysis-actual", disabled=True, children=[ # Renamed id to avoid conflict
                    dbc.Card(dbc.CardBody([
                        html.Div(id='analysis-content-div', className="p-3"),
                        dbc.Button(children=[html.I(className="fas fa-lightbulb me-2"),"Proceed to AI Insights"], id="btn-goto-suggestions-expl", color="info", className="mt-3 d-block mx-auto", style={'display': 'none', 'width':'fit-content'}), # Centered button
                        dbc.Button(children=[html.I(className="fas fa-lightbulb me-2"),"Proceed to AI Insights"], id="btn-goto-suggestions-opt", color="success", className="mt-3 d-block mx-auto", style={'display': 'none', 'width':'fit-content'})  # Centered button
                    ]), className="mt-0 shadow-sm")
                ]),
                dbc.Tab(label="AI Insights & Suggestions", tab_id="tab-suggestions", id="tab-suggestions-actual", disabled=True, children=[ # Renamed id
                    dbc.Card(dbc.CardBody([
                        html.Div(id='suggestions-content-div', className="p-3"),
                        html.Hr(),
                        html.H5(children=[html.I(className="fas fa-edit me-2"), "Customize AI Prompt (Optional)"], className="mt-4 text-secondary"),
                        html.P("Refine the instructions for the AI to get more tailored insights.", className="small text-muted"),
                        dcc.Textarea(
                            id='custom-llm-prompt-input',
                            value='Focus on the implications for protein stability and synthetic biology applications.',
                            style={'width': '100%', 'height': 100, 'borderRadius': '8px', 'border': '1px solid #ced4da', 'padding': '10px'}, # Reduced height
                            className="mb-3"
                        ),
                        dbc.Tooltip("Add specific instructions or questions for the AI to consider when generating insights.", target="custom-llm-prompt-input"),
                        dbc.Button(children=[html.I(className="fas fa-sync-alt me-2"), "Regenerate AI Insights"], id="btn-regenerate-llm", color="secondary", className="w-100")
                    ]), className="mt-0 shadow-sm")
                ]),
            ])
    ]), className="mt-3 shadow-lg rounded"), # Added shadow to main card

    dbc.Row(dbc.Col(html.P("Powered by AutoML and Generative AI", className="text-center text-muted mt-5 small"), width=12))
])

# ========= Callbacks =========

# Callback to update NavLinks based on active tab and disabled state of underlying tabs
@app.callback(
    [Output('nav-data-setup', 'active'),
     Output('nav-analysis', 'active'),
     Output('nav-suggestions', 'active'),
     Output('nav-analysis', 'disabled'), # This will control the NavLink's disabled state
     Output('nav-suggestions', 'disabled')],# This will control the NavLink's disabled state
    [Input('main-tabs', 'active_tab'),
     Input('tab-analysis-actual', 'disabled'),  # Listen to the actual tab's disabled state
     Input('tab-suggestions-actual', 'disabled')] # Listen to the actual tab's disabled state
)
def update_nav_links(active_main_tab, analysis_tab_is_disabled, suggestions_tab_is_disabled):
    return (
        active_main_tab == 'tab-data-upload',
        active_main_tab == 'tab-analysis',
        active_main_tab == 'tab-suggestions',
        analysis_tab_is_disabled, # NavLink disabled if corresponding actual tab is disabled
        suggestions_tab_is_disabled # NavLink disabled if corresponding actual tab is disabled
    )


# Callback to switch tabs when NavLink is clicked
@app.callback(
    Output('main-tabs', 'active_tab', allow_duplicate=True),
    [Input('nav-data-setup', 'n_clicks'),
     Input('nav-analysis', 'n_clicks'),
     Input('nav-suggestions', 'n_clicks')],
    [State('nav-analysis', 'disabled'), # Check if the navlink itself is disabled
     State('nav-suggestions', 'disabled')],
    prevent_initial_call=True
)
def switch_tabs_from_nav(n_data, n_analysis, n_suggestions, analysis_nav_disabled, suggestions_nav_disabled):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'nav-data-setup':
        return 'tab-data-upload'
    elif button_id == 'nav-analysis' and not analysis_nav_disabled: # Only switch if not disabled
        return 'tab-analysis'
    elif button_id == 'nav-suggestions' and not suggestions_nav_disabled: # Only switch if not disabled
        return 'tab-suggestions'
    return dash.no_update


# Callback to update global progress message
@app.callback(
    Output('global-progress-message', 'children'),
    Input('progress-interval', 'n_intervals'), # Triggered by interval
    State('store-progress-text', 'data'), # Main text comes from here (e.g. "Running AutoML...")
    State('progress-interval', 'disabled'), # To hide when not active
    State('store-progress-steps', 'data'), # List of detailed steps
    State('store-current-step-index', 'data') # Current index in the detailed steps
)
def update_progress_message(n, progress_text, interval_disabled, progress_steps, current_step_index):
    if interval_disabled or not progress_text: # If interval is off or no base text, show nothing
        return ""

    # If there are detailed steps, show them with a progress bar
    if progress_steps and current_step_index < len(progress_steps):
        current_step_text = progress_steps[current_step_index]
        # Calculate percentage for the detailed step progress
        progress_percentage = ((current_step_index + 1) / len(progress_steps)) * 100 if len(progress_steps) > 0 else 0
        
        return html.Div([
            dbc.Spinner(size="sm", color="primary", className="me-2 align-middle"), 
            html.Span(f"{current_step_text}", className="me-3 align-middle", style={'color': '#495057'}), # Darker text
            dbc.Progress(value=progress_percentage, style={'height': '20px', 'fontSize': '0.75rem'}, className="align-middle d-inline-flex w-50", striped=True, animated=True)
        ], className="d-flex align-items-center justify-content-center")
    
    # Fallback to simple spinner and text if no detailed steps
    return html.Div([
        dbc.Spinner(size="sm", color="primary", className="me-2"), 
        html.Span(progress_text, style={'color': '#495057'})
    ])


# Callback to advance progress bar steps (simulated)
# This callback's sole purpose is to increment the step index when the interval is active.
# It doesn't directly set the text, but allows the update_progress_message callback to show the next step.
@app.callback(
    Output('store-current-step-index', 'data', allow_duplicate=True), # allow_duplicate as it's also an output in run_automl
    Input('progress-interval', 'n_intervals'), # Triggered by interval
    State('store-current-step-index', 'data'), # Current index
    State('store-progress-steps', 'data'), # Total steps available
    State('progress-interval', 'disabled'), # Only advance if interval is enabled
    prevent_initial_call=True
)
def advance_progress_step(n_intervals, current_step_index, progress_steps, interval_disabled):
    if not interval_disabled and progress_steps:
        # Simulate work being done for the current step, then advance.
        # In a real async setup, this would be more complex. Here, we just increment.
        # Let's assume each interval tick means some progress on the current step,
        # and we advance to the next step after a few ticks, or based on external trigger.
        # For simplicity here, we'll just let the main AutoML callback set the index.
        # This interval callback is more for showing continuous activity.
        # The actual step advancement logic is better handled by the AutoML callbacks themselves
        # by updating 'store-current-step-index' directly when a logical step completes.
        # So, this callback might not be strictly needed if AutoML callbacks update the index.
        # However, if we want a "dummy" progress on each step, it could be used.
        # For now, let's make it so it doesn't interfere with AutoML's direct index updates.
        # It will mostly serve to re-trigger the 'update_progress_message'.
        return dash.no_update # Let other callbacks manage the index primarily.
    return dash.no_update


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
            return dbc.Alert(df_or_error.children[0], color="danger", className="mt-2"), "", None, ""
        df = df_or_error
        filename_display = filename
    elif triggered_id == 'btn-load-demo' and n_clicks_demo:
        try:
            df = pd.read_csv(io.StringIO(DEMO_DATA_CSV_STRING))
            filename_display = "lfa_random_data.csv (Demo)"
        except Exception as e:
            print(f"Error loading demo data: {e}")
            return dbc.Alert(f"Error loading demo data: {str(e)}", color="danger", className="mt-2"), "", None, ""
    
    if df is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return process_dataframe_for_ui(df, filename_display)

@app.callback(
    [Output('store-column-roles', 'data'),
     Output('main-tabs', 'active_tab', allow_duplicate=True), # Allow duplicate for main_tabs
     Output('tab-analysis-actual', 'disabled'), # Control actual tab disabled state
     Output('tab-suggestions-actual', 'disabled'), 
     Output('output-data-upload-status', 'children', allow_duplicate=True),
     Output('store-current-analysis-type', 'data')],
    [Input('btn-confirm-setup', 'n_clicks')],
    [State('dropdown-experiment-type', 'value'), State('dropdown-input-vars', 'value'),
     State('dropdown-output-var', 'value'), State('dropdown-ignore-vars', 'value'),
     State('dropdown-missing-strategy', 'value'),
     State('store-raw-data', 'data')],
    prevent_initial_call=True
)
def confirm_setup_and_proceed(n_clicks, exp_type, inputs, output_var, ignored, missing_strategy, raw_data_json):
    if not n_clicks or not raw_data_json:
        # Disable analysis and suggestions tabs, keep current tab, no status update, no analysis type
        return dash.no_update, dash.no_update, True, True, dash.no_update, dash.no_update
    
    if not inputs or not output_var:
        # Validation failed: show warning, keep tabs disabled
        alert_msg = dbc.Alert("⚠️ Please select input(s) and a target output.", color="warning", duration=5000, className="mt-2")
        return dash.no_update, dash.no_update, True, True, alert_msg, dash.no_update

    if output_var in inputs:
        # Validation failed: show error, keep tabs disabled
        alert_msg = dbc.Alert("⚠️ Target Output cannot be an Input Variable.", color="danger", duration=5000, className="mt-2")
        return dash.no_update, dash.no_update, True, True, alert_msg, dash.no_update
    
    # Validation passed
    column_roles = {
        'inputs': inputs, 
        'target_for_analysis': output_var, 
        'ignored': ignored or [],
        'missing_strategy': missing_strategy # e.g. 'drop_rows', 'mean', 'median', 'most_frequent'
    }
    success_msg = dbc.Alert(f"Setup Confirmed for {exp_type.capitalize()}. Proceed to 'Analysis & Results'.", color="info", duration=4000, className="mt-2")
    # Store roles, switch to analysis tab, enable analysis tab, keep suggestions disabled, show success, store type
    return column_roles, "tab-analysis", False, True, success_msg, exp_type


@app.callback(
    Output('analysis-content-div', 'children'),
    Input('main-tabs', 'active_tab'), # Trigger when analysis tab becomes active
    State('store-current-analysis-type', 'data'), 
    State('store-column-roles', 'data'),
    State('store-raw-data', 'data') # Added to check if data exists
)
def render_analysis_tab_content(active_tab, analysis_type, column_roles, raw_data_json):
    if active_tab != 'tab-analysis' or not analysis_type or not column_roles or not raw_data_json:
        # If not on analysis tab, or setup is incomplete, show placeholder
        return html.Div([
            html.P("Complete data upload and experiment configuration on the 'Data & Setup' tab first."),
            html.P("Once confirmed, this section will allow you to run the analysis.")
        ], className="text-center text-muted p-4")

    target_column = column_roles.get('target_for_analysis')
    df = pd.read_json(raw_data_json, orient='split') # Load data to check target type

    # Validate target column type for numeric operations
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return dbc.Alert(f"Error: The selected target output variable '{target_column}' is not numeric. "
                         "AutoML regression models require a numeric target. Please select a different target variable.", 
                         color="danger", className="mt-2")
    
    # Validate input features for appropriate types (numeric or categorical/string)
    problematic_features = []
    if column_roles.get('inputs'):
        for col in column_roles['inputs']:
            # Allow numeric, string/object (categorical). Disallow boolean, datetime unless explicitly handled.
            if not (pd.api.types.is_numeric_dtype(df[col]) or \
                    pd.api.types.is_string_dtype(df[col]) or \
                    pd.api.types.is_object_dtype(df[col])): # object can be categorical
                problematic_features.append(col)
    
    if problematic_features:
        return dbc.Alert(f"Error: The following input variables have data types unsuitable for this analysis (must be numeric or categorical text): "
                         f"{', '.join(problematic_features)}. Please check your data or feature selection.",
                         color="danger", className="mt-2")


    if analysis_type == 'exploration':
        return html.Div([
            html.H3(f"Exploration Analysis for Target: {target_column}", className="mb-3 text-info"),
            html.P("Identify key input variables impacting the selected output using AutoML. This involves training multiple regression models and evaluating their performance and feature importances."),
            dbc.Button(children=[html.I(className="fas fa-rocket me-2"), "Run AutoML for Exploration"], id="btn-run-exploration-automl", color="info", className="my-3 btn-lg w-100 shadow-sm"),
            dbc.Tooltip("Initiate the AutoML process to discover the most influential factors.", target="btn-run-exploration-automl"),
            dcc.Loading(id="loading-exploration", type="default", children=[html.Div(id="exploration-results-area", className="mt-4 p-3 border rounded bg-white shadow-sm")]) # Added styling
        ])
    elif analysis_type == 'optimization':
        # Check if there are any numerical features for optimization ranges
        numerical_inputs = [col for col in column_roles.get('inputs', []) if pd.api.types.is_numeric_dtype(df[col])]

        return html.Div([
            html.H3(f"Optimization Analysis for Target: {target_column}", className="mb-3 text-success"),
            html.P(f"Find optimal settings for inputs to maximize or minimize '{target_column}'. This uses the best model from an AutoML run as a surrogate for a grid search over input ranges."),
            dbc.Label("Optimization Goal:", className="fw-bold"),
            dcc.Dropdown(id='dropdown-optimization-goal', options=[{'label': 'Maximize Target', 'value': 'maximize'}, {'label': 'Minimize Target', 'value': 'minimize'}], value='maximize', clearable=False, className="mb-3"),
            dbc.Tooltip("Select whether you want to maximize or minimize the target output.", target="dropdown-optimization-goal"),
            html.P("Note: For numerical inputs, ranges are inferred from your data's min/max values. Categorical inputs will be tested with their unique values.", className="small text-muted") if numerical_inputs else html.P("Note: Categorical inputs will be tested with their unique values. No numerical inputs detected for range-based optimization.", className="small text-muted"),
            dbc.Button(children=[html.I(className="fas fa-bullseye me-2"), "Run AutoML for Optimization"], id="btn-run-optimization-automl", color="success", className="my-3 btn-lg w-100 shadow-sm"),
            dbc.Tooltip("Start the AutoML process to find the ideal experimental conditions.", target="btn-run-optimization-automl"),
            dcc.Loading(id="loading-optimization", type="default", children=[html.Div(id="optimization-results-area", className="mt-4 p-3 border rounded bg-white shadow-sm")]) # Added styling
        ])
    return "Analysis type not recognized or setup incomplete."


# Progress callback function for AutoML runs (to be called by the AutoML functions)
# This function is a bit conceptual in Dash's context for synchronous operations.
# The actual update to the store components will happen in the main callback that calls the AutoML.
def update_automl_progress_display(progress_steps_list, current_step_idx, analysis_id_trigger):
    """
    This function is intended to be called by the long-running AutoML process.
    In Dash, direct updates from a synchronous function to Store components that trigger other callbacks
    is complex. Instead, the main callback invoking AutoML will update these stores.
    This function signature is a placeholder for how one might structure it if using background callbacks.
    For this app, the main @app.callback for exploration/optimization will set these.
    """
    # This would typically update dcc.Store components if it were a background callback.
    # For now, it's a conceptual link. The actual updates are:
    # 'store-progress-steps' gets progress_steps_list
    # 'store-current-step-index' gets current_step_idx
    # 'progress-interval' disabled = False
    # 'store-progress-text' gets a generic "Running..." message
    # The 'analysis_id_trigger' isn't directly used here but shows it's tied to a specific run.
    pass


# Callback to run Exploration AutoML and display results
@app.callback(
    [Output('exploration-results-area', 'children'),
     Output('store-exploration-results', 'data'),
     Output('tab-suggestions-actual', 'disabled', allow_duplicate=True), 
     Output('store-progress-text', 'data', allow_duplicate=True), # For base message
     Output('progress-interval', 'disabled', allow_duplicate=True), # To enable/disable spinner
     Output('store-progress-steps', 'data', allow_duplicate=True), # For detailed step list
     Output('store-current-step-index', 'data', allow_duplicate=True), # For current step in list
     Output('btn-goto-suggestions-expl', 'style', allow_duplicate=True)],
    Input('btn-run-exploration-automl', 'n_clicks'),
    State('store-raw-data', 'data'),
    State('store-column-roles', 'data'),
    prevent_initial_call=True
)
def run_exploration_analysis_callback(n_clicks, raw_data_json, column_roles):
    if not n_clicks or not raw_data_json or not column_roles:
        return dash.no_update, dash.no_update, True, None, True, [], 0, {'display': 'none'}
    
    # Initial progress state
    initial_progress_steps = ["Initializing Exploration AutoML..."] # Placeholder until real steps are known
    
    # Callback context to manage progress updates from within run_exploration_automl
    # This is a simplified way to pass a "callback" to the synchronous function
    # In a real async setup, you'd use background callbacks.
    
    # This list will be populated by _progress_updater
    # It needs to be mutable and accessible by the nested function.
    # We'll pass it to run_exploration_automl, which will call _progress_updater.
    # The final values will be returned by this main callback.
    # This is a bit of a workaround for Dash's synchronous callback model.
    
    # We will return these values at the end to update the stores
    # This function will be called by run_automl_pipeline to update the step index
    # It doesn't directly trigger Dash updates, but provides the values for the main callback to do so.
    _captured_progress_steps = []
    _captured_current_step_index = 0

    def _progress_updater(steps, current_idx):
        nonlocal _captured_progress_steps, _captured_current_step_index
        _captured_progress_steps = steps
        _captured_current_step_index = current_idx
        # In a true async setup, this would trigger a client-side update.
        # Here, we just capture the state. The main callback will return it.
        print(f"Progress Update: Step {current_idx + 1}/{len(steps)}: {steps[current_idx]}")


    df = pd.read_json(raw_data_json, orient='split')
    target_column = column_roles['target_for_analysis']
    feature_columns = column_roles['inputs']
    missing_strategy = column_roles.get('missing_strategy', 'drop_rows') # Default if not set

    # Basic validation (already in render_analysis_tab_content, but good for direct calls too)
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        error_message = f"Error: Target '{target_column}' is not numeric."
        return dbc.Alert(error_message, color="danger"), None, True, None, True, [], 0, {'display': 'none'}

    try:
        # Start progress indication
        # The run_exploration_automl will call _progress_updater which sets _captured_... vars
        all_model_results, best_model_info, importances, shap_plot_fig, final_progress_steps = \
            run_exploration_automl(df, target_column, feature_columns, missing_strategy, _progress_updater)
        
        # After run_exploration_automl completes, _captured_progress_steps and _captured_current_step_index
        # should reflect the final state of progress from the last call to _progress_updater.
        # We use final_progress_steps directly as it's returned.
        # The index should be the last step.
        final_step_index = len(final_progress_steps) -1 if final_progress_steps else 0


    except ValueError as ve: # Catch specific errors like empty X/y
        error_message = f"Input Data Error for Exploration: {str(ve)}"
        print(f"ValueError in run_exploration_analysis_callback: {ve}")
        return dbc.Alert(error_message, color="danger"), None, True, "Error", False, _captured_progress_steps, _captured_current_step_index, {'display': 'none'}
    except Exception as e:
        error_message = f"An error occurred during AutoML Exploration: {str(e)}. Check console for details."
        print(f"Error in run_exploration_analysis_callback: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert(error_message, color="danger"), None, True, "Error", False, _captured_progress_steps, _captured_current_step_index, {'display': 'none'}
    
    # Extract performance metrics from the best model
    performance_metrics = {
        "R-squared": best_model_info.get('R-squared', 0),
        "MAE": best_model_info.get('MAE', 0),
        "RMSE": best_model_info.get('RMSE', 0),
        "MAPE": best_model_info.get('MAPE', None) 
    }

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Best Model R²", className="card-title text-muted small"), html.P(f"{performance_metrics.get('R-squared', 0):.3f}", className="card-text fs-4 text-info fw-bold")])), md=3, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Best Model MAE", className="card-title text-muted small"), html.P(f"{performance_metrics.get('MAE', 0):.3f}", className="card-text fs-4 text-info fw-bold")])), md=3, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Best Model RMSE", className="card-title text-muted small"), html.P(f"{performance_metrics.get('RMSE', 0):.3f}", className="card-text fs-4 text-info fw-bold")])), md=3, className="mb-2"),
        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Top Feature", className="card-title text-muted small"), html.P(f"{importances['feature'].iloc[0] if not importances.empty else 'N/A'}", className="card-text fs-5 text-info fw-bold text-truncate")])), md=3, className="mb-2")
    ], className="mb-3 g-3") # Added g-3 for gutter
    
    # Ensure all serializable types for JSON store (NaN -> None)
    cleaned_best_model_info = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in best_model_info.items() if k != 'Pipeline'}
    cleaned_all_model_results = []
    for m in all_model_results:
        cleaned_m = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items() if k != 'Pipeline'}
        cleaned_all_model_results.append(cleaned_m)

    results_data_for_store = {
        'all_model_results': cleaned_all_model_results,
        'best_model_info': cleaned_best_model_info,
        'importances': importances.to_dict('records') if not importances.empty else [], 
        'performance_metrics': {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in performance_metrics.items()},
        'target_column': target_column,
        'missing_strategy': missing_strategy,
        'model_info': cleaned_best_model_info # Storing best model info again for consistency with optimization
    }

    comparison_table_data = []
    for model_res in all_model_results: # Use original all_model_results for display formatting
        row = {
            "Model": model_res["Model Type"], # Shorter name
            "R²": f"{model_res.get('R-squared', 0):.3f}",
            "MAE": f"{model_res.get('MAE', 0):.3f}",
            "RMSE": f"{model_res.get('RMSE', 0):.3f}",
            "MAPE": f"{model_res.get('MAPE', float('nan')):.2f}%" if model_res.get('MAPE') is not None else "N/A",
            "CV R²": f"{model_res.get('Cross-Validation R2', 0):.3f}",
            "Time (s)": f"{model_res.get('Training Time (s)', 0):.2f}",
            # "Hyperparameters": json.dumps(model_res.get("Best Hyperparameters", {})) # Can be too long
        }
        comparison_table_data.append(row)

    model_comparison_table = html.Div([
        html.H5("Model Comparison Summary", className="mt-4 mb-2 text-secondary"),
        html.P(f"The best model selected is **{best_model_info.get('Model Type', 'N/A')}**.", className="mb-3 small"),
        dash_table.DataTable(
            id='model-comparison-table',
            columns=[{"name": i, "id": i} for i in comparison_table_data[0].keys()] if comparison_table_data else [],
            data=comparison_table_data,
            sort_action='native',
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '0.85rem'},
            style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Model} = "' + best_model_info.get('Model Type', '') + '"'},
                    'backgroundColor': '#d1ecf1', # Light blue for best model
                    'fontWeight': 'bold'
                }
            ]
        )
    ], className="mb-4")
    
    # Simplified Best Model Details Card
    best_model_details_card = dbc.Card(dbc.CardBody([
        html.H5(f"Details for Best Model: {best_model_info.get('Model Type', 'N/A')}", className="card-title text-info"),
        html.P(f"Target Analyzed: {target_column}", className="card-text"),
        html.P(f"Missing Value Strategy: {missing_strategy.replace('_', ' ').title()}", className="card-text"),
        html.P(f"Hyperparameters: {json.dumps(best_model_info.get('Best Hyperparameters', {}))}", className="card-text small")
    ]), className="mb-4 shadow-sm bg-light")


    results_layout = html.Div([
        dbc.Alert(f"Exploration AutoML complete for '{target_column}'.", color="info", className="mt-2"),
        kpi_cards, 
        model_comparison_table,
        best_model_details_card, # Added simplified details card
        html.H5("Best Model Feature Importance (SHAP Values)", className="mt-4 mb-2 text-secondary"),
        dcc.Graph(id='exploration-shap-plot-graph', figure=shap_plot_fig if shap_plot_fig else go.Figure().update_layout(title="SHAP Plot not available")),
    ])
    
    # Final progress state: text=None (or "Completed"), interval disabled, steps and index at final values
    return results_layout, results_data_for_store, False, "Exploration Complete!", True, final_progress_steps, final_step_index, {'display': 'block', 'width':'fit-content', 'margin': 'auto'}


# Callback to run Optimization AutoML and display results
@app.callback(
    [Output('optimization-results-area', 'children'),
     Output('store-optimization-results', 'data'),
     Output('tab-suggestions-actual', 'disabled', allow_duplicate=True), 
     Output('store-progress-text', 'data', allow_duplicate=True),
     Output('progress-interval', 'disabled', allow_duplicate=True),
     Output('store-progress-steps', 'data', allow_duplicate=True),
     Output('store-current-step-index', 'data', allow_duplicate=True),
     Output('btn-goto-suggestions-opt', 'style', allow_duplicate=True)],
    Input('btn-run-optimization-automl', 'n_clicks'),
    State('store-raw-data', 'data'),
    State('store-column-roles', 'data'),
    State('dropdown-optimization-goal', 'value'),
    prevent_initial_call=True
)
def run_optimization_analysis_callback(n_clicks, raw_data_json, column_roles, opt_goal):
    if not n_clicks or not raw_data_json or not column_roles:
        return dash.no_update, dash.no_update, True, None, True, [], 0, {'display': 'none'}
        
    _captured_progress_steps_opt = []
    _captured_current_step_index_opt = 0
    def _progress_updater_opt(steps, current_idx):
        nonlocal _captured_progress_steps_opt, _captured_current_step_index_opt
        _captured_progress_steps_opt = steps
        _captured_current_step_index_opt = current_idx
        print(f"Opt Progress: Step {current_idx + 1}/{len(steps)}: {steps[current_idx]}")

    df = pd.read_json(raw_data_json, orient='split')
    target_column = column_roles['target_for_analysis']
    feature_columns = column_roles['inputs'] # These are the original selected input features
    missing_strategy = column_roles.get('missing_strategy', 'drop_rows')

    # Basic validation
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        error_message = f"Error: Target '{target_column}' is not numeric for Optimization."
        return dbc.Alert(error_message, color="danger"), None, True, None, True, [], 0, {'display': 'none'}

    try:
        # Define feature_ranges for numerical features based on the original data (df)
        # Ensure only numerical features present in `feature_columns` are included
        feature_ranges = {
            col: {'min': df[col].min(), 'max': df[col].max()} 
            for col in feature_columns 
            if pd.api.types.is_numeric_dtype(df[col]) and col in df.columns # Check col exists in df
        }
        
        all_model_results_opt, best_model_info_opt, importances_opt, shap_plot_fig_opt, final_progress_steps_opt, \
        optimal_settings, predicted_target, predicted_target_lower, predicted_target_upper, \
        response_fig, surrogate_tree_text, model_info_opt_summary = \
            run_optimization_automl(df, target_column, feature_columns, opt_goal, feature_ranges, missing_strategy, _progress_updater_opt)
        
        final_step_index_opt = len(final_progress_steps_opt) -1 if final_progress_steps_opt else 0

    except ValueError as ve:
        error_message = f"Input Data Error for Optimization: {str(ve)}"
        print(f"ValueError in run_optimization_analysis_callback: {ve}")
        return dbc.Alert(error_message, color="danger"), None, True, "Error", False, _captured_progress_steps_opt, _captured_current_step_index_opt, {'display': 'none'}
    except Exception as e:
        error_message = f"An error occurred during AutoML Optimization: {str(e)}. Check console for details."
        print(f"Error in run_optimization_analysis_callback: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert(error_message, color="danger"), None, True, "Error", False, _captured_progress_steps_opt, _captured_current_step_index_opt, {'display': 'none'}
    
    kpi_cards_list = []
    # KPIs for the best surrogate model
    kpi_cards_list.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5("Surrogate Model R²", className="card-title text-muted small"), html.P(f"{best_model_info_opt.get('R-squared', 0):.3f}", className="card-text fs-4 text-success fw-bold")])), md=4, className="mb-2"))
    kpi_cards_list.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5("Surrogate MAE", className="card-title text-muted small"), html.P(f"{best_model_info_opt.get('MAE', 0):.3f}", className="card-text fs-4 text-success fw-bold")])), md=4, className="mb-2"))
    kpi_cards_list.append(dbc.Col(dbc.Card(dbc.CardBody([html.H5("Surrogate Top Feature", className="card-title text-muted small"), html.P(f"{importances_opt['feature'].iloc[0] if not importances_opt.empty else 'N/A'}", className="card-text fs-5 text-success fw-bold text-truncate")])), md=4, className="mb-2"))

    # Optimal settings and predicted target
    optimal_settings_display = dbc.Card(dbc.CardBody([
        html.H5(f"Optimal Settings to {opt_goal.capitalize()} {target_column}", className="card-title text-success"),
        *[html.P(f"{k}: {v:.3f}" if isinstance(v, (float, np.number)) and not np.isnan(v) else f"{k}: {v}", className="card-text mb-1") for k,v in optimal_settings.items()],
        html.Hr(),
        html.H6(f"Predicted {target_column}: {predicted_target:.3f}", className="fw-bold"),
        html.P(f"(95% CI: {predicted_target_lower:.3f} - {predicted_target_upper:.3f})", className="small text-muted")
    ]), className="mb-3 shadow-sm")

    # Ensure all serializable types for JSON store (NaN -> None)
    cleaned_best_model_info_opt = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in best_model_info_opt.items() if k != 'Pipeline'}
    cleaned_all_model_results_opt = []
    for m_opt in all_model_results_opt:
        cleaned_m_opt = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m_opt.items() if k != 'Pipeline'}
        cleaned_all_model_results_opt.append(cleaned_m_opt)
    
    cleaned_model_info_opt_summary = {k: (None if isinstance(v, float) and np.isnan(v) else v if not isinstance(v, tuple) else str(v)) for k,v in model_info_opt_summary.items()}


    results_data_for_store = {
        'all_model_results': cleaned_all_model_results_opt,
        'best_model_info': cleaned_best_model_info_opt, # Info about the best surrogate model
        'optimal_settings': {k: (None if isinstance(v, float) and np.isnan(v) else v) for k,v in optimal_settings.items()},
        'predicted_target': None if isinstance(predicted_target, float) and np.isnan(predicted_target) else predicted_target,
        'predicted_target_lower': None if isinstance(predicted_target_lower, float) and np.isnan(predicted_target_lower) else predicted_target_lower,
        'predicted_target_upper': None if isinstance(predicted_target_upper, float) and np.isnan(predicted_target_upper) else predicted_target_upper,
        'target_column': target_column, 
        'goal': opt_goal, 
        'feature_columns': feature_columns,
        'missing_strategy': missing_strategy,
        'importances': importances_opt.to_dict('records') if not importances_opt.empty else [],
        'model_info': cleaned_model_info_opt_summary, # Summary of the surrogate model used
        'surrogate_tree_text': surrogate_tree_text
    }
    
    # Model comparison table for optimization (surrogate model candidates)
    comparison_table_data_opt = []
    for model_res in all_model_results_opt:
        row = {
            "Model": model_res["Model Type"],
            "R²": f"{model_res.get('R-squared',0):.3f}",
            "MAE": f"{model_res.get('MAE',0):.3f}",
            "RMSE": f"{model_res.get('RMSE',0):.3f}",
            "MAPE": f"{model_res.get('MAPE', float('nan')):.2f}%" if model_res.get('MAPE') is not None else "N/A",
            "CV R²": f"{model_res.get('Cross-Validation R2',0):.3f}",
            "Time (s)": f"{model_res.get('Training Time (s)',0):.2f}",
        }
        comparison_table_data_opt.append(row)

    model_comparison_table_opt_layout = html.Div([
        html.H5("Surrogate Model Candidates Comparison", className="mt-4 mb-2 text-secondary"),
        html.P(f"The best model selected as surrogate was **{best_model_info_opt.get('Model Type', 'N/A')}**.", className="mb-3 small"),
        dash_table.DataTable(
            id='model-comparison-table-opt',
            columns=[{"name": i, "id": i} for i in comparison_table_data_opt[0].keys()] if comparison_table_data_opt else [],
            data=comparison_table_data_opt,
            sort_action='native', page_size=5, style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '8px', 'fontFamily': 'Inter, sans-serif', 'fontSize': '0.85rem'},
            style_header={'backgroundColor': '#e9ecef', 'fontWeight': 'bold'},
            style_data_conditional=[{'if': {'filter_query': '{Model} = "' + best_model_info_opt.get('Model Type', '') + '"'}, 'backgroundColor': '#d4edda', 'fontWeight': 'bold'}]
        )
    ], className="mb-4")
    
    # Surrogate Model Details Card
    surrogate_model_details_card = dbc.Card(dbc.CardBody([
        html.H5(f"Details for Surrogate Model: {model_info_opt_summary.get('Model Type', 'N/A')}", className="card-title text-success"),
        html.P(f"Optimization Goal: {opt_goal.capitalize()} {target_column}", className="card-text"),
        html.P(f"Hyperparameters: {json.dumps(model_info_opt_summary.get('Hyperparameters', {}))}", className="card-text small")
    ]), className="mb-4 shadow-sm bg-light")


    response_surface_explanation = html.P(
        "Response Surface/Curve visualizes the predicted relationship between one or two key numerical inputs and the target, holding other inputs at their optimal or mean values.",
        className="small text-muted mt-1 mb-3"
    )

    plot_div = []
    if response_fig and response_fig.get('data'): # Check if figure has data
        plot_div.extend([
            html.H5("Predicted Response Surface/Curve", className="mt-4 mb-2 text-secondary"),
            response_surface_explanation,
            dcc.Graph(id='optimization-response-surface-plot-graph', figure=response_fig)
        ])
    else:
         plot_div.append(html.P("Response surface plot is not available (e.g., fewer than 1-2 varied numerical features).", className="text-muted small mt-3"))


    shap_plot_div = []
    if shap_plot_fig_opt and shap_plot_fig_opt.get('data'): # Check if figure has data
        shap_plot_div.extend([
            html.H5("Surrogate Model Feature Importance (SHAP)", className="mt-4 mb-2 text-secondary"),
            dcc.Graph(id='optimization-shap-plot-graph', figure=shap_plot_fig_opt)
        ])
    
    surrogate_tree_layout = html.Div()
    if surrogate_tree_text and surrogate_tree_text != "Not applicable (no transformed data or features for tree visualization)." and surrogate_tree_text != "Not applicable (no numerical features or transformed data for tree visualization).":
        surrogate_tree_layout = html.Div([
            html.H5("Surrogate Tree Logic (Simplified Rules)", className="mt-4 mb-2 text-secondary"),
            html.P("This simplified decision tree approximates the surrogate model's predictions, offering interpretable rules.", className="small text-muted"),
            dbc.Card(dbc.CardBody(html.Pre(surrogate_tree_text, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all', 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '8px', 'fontFamily': 'monospace', 'maxHeight': '300px', 'overflowY': 'auto'})), className="mb-4 shadow-sm")
        ])

    results_layout = html.Div([
        dbc.Alert(f"Optimization AutoML complete for '{target_column}'. Goal: {opt_goal.capitalize()}.", color="success", className="mt-2"),
        dbc.Row(kpi_cards_list, className="mb-3 g-3"), 
        optimal_settings_display,
        model_comparison_table_opt_layout,
        surrogate_model_details_card,
        *plot_div,
        *shap_plot_div, 
        surrogate_tree_layout,
    ])
    
    return results_layout, results_data_for_store, False, "Optimization Complete!", True, final_progress_steps_opt, final_step_index_opt, {'display': 'block', 'width':'fit-content', 'margin': 'auto'}


# Callback to handle explicit tab switching via buttons on Analysis tab
@app.callback(
    Output('main-tabs', 'active_tab', allow_duplicate=True),
    [Input('btn-goto-suggestions-expl', 'n_clicks'),
     Input('btn-goto-suggestions-opt', 'n_clicks')],
    prevent_initial_call=True
)
def switch_to_suggestions_tab_from_analysis(n1_expl, n2_opt):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update
    
    # Whichever button was clicked, switch to suggestions tab
    return "tab-suggestions"


@app.callback(
    Output('suggestions-content-div', 'children', allow_duplicate=True),
    [Input('main-tabs', 'active_tab'), # Trigger when suggestions tab is active
     Input('btn-regenerate-llm', 'n_clicks')], # Trigger on regenerate button
    [State('store-current-analysis-type', 'data'), 
     State('store-exploration-results', 'data'), 
     State('store-optimization-results', 'data'),
     State('custom-llm-prompt-input', 'value')],
    prevent_initial_call=True # Important to prevent firing on app load
)
def render_suggestions_tab_content(active_tab_id, n_clicks_regenerate, analysis_type, exploration_data, optimization_data, custom_prompt_value):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Only proceed if the suggestions tab is active OR the regenerate button was clicked
    if not (active_tab_id == 'tab-suggestions' or triggered_id == 'btn-regenerate-llm'):
        return dash.no_update 
    
    if not analysis_type: 
        return dbc.Alert("Complete an analysis on Tab 2 first to generate AI insights.", color="warning", className="m-3")
    
    header_text, llm_input_data = "", None
    if analysis_type == 'exploration' and exploration_data:
        header_text = f"AI Insights for Exploration: {exploration_data.get('target_column', 'N/A')}"
        llm_input_data = exploration_data
    elif analysis_type == 'optimization' and optimization_data:
        header_text = f"AI Insights for Optimization: {optimization_data.get('target_column', 'N/A')} ({optimization_data.get('goal', 'N/A')})"
        llm_input_data = optimization_data
    else:
        return dbc.Alert("No analysis results available for suggestions. Please run an analysis on Tab 2 first.", color="warning", className="m-3")
    
    # Show loading spinner while generating
    loading_spinner = dcc.Loading(type="dots", children=[html.Div(id="llm-output-placeholder")]) # Temporary placeholder for output

    try:
        llm_explanation_markdown = generate_explanation_llm(analysis_type, llm_input_data, custom_prompt_value)
    except Exception as e:
        print(f"Error generating LLM explanation: {e}")
        return dbc.Alert(f"Error generating AI insights: {str(e)}", color="danger", className="m-3")

    return html.Div([
        html.H3(children=[html.I(className="fas fa-brain me-2"), header_text], className="mb-3 text-primary"),
        dcc.Markdown(llm_explanation_markdown, className="border p-3 bg-white rounded shadow-sm", style={'lineHeight': '1.7', 'fontSize': '0.95rem'}) # Enhanced styling
    ], className="p-2")


if __name__ == '__main__':
    app.run(debug=True)
