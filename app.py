import numpy as np
import pandas as pd

def generate_lfa_data(n_rows=1000, random_state=42):
    np.random.seed(random_state)
    data = {
        "membrane_type": np.random.choice(["NC-170", "NC-135", "NC-180"], size=n_rows),
        "antibody_concentration": np.random.uniform(0.5, 5.0, size=n_rows),
        "conjugate_type": np.random.choice(["AuNP", "Latex", "Carbon"], size=n_rows),
        "conjugate_volume": np.random.uniform(0.1, 2.0, size=n_rows),
        "blocking_buffer": np.random.choice(["BSA", "Casein", "FishGel"], size=n_rows),
        "strip_width": np.random.uniform(2.0, 5.0, size=n_rows),
        "sample_volume": np.random.uniform(5, 100, size=n_rows),
        "incubation_time": np.random.uniform(5, 60, size=n_rows),
        "running_buffer_pH": np.random.uniform(6.0, 9.0, size=n_rows),
        "detection_antibody_type": np.random.choice(["Mouse", "Goat", "Rabbit"], size=n_rows),
        "test_line_antigen_density": np.random.uniform(0.1, 5.0, size=n_rows),
        "control_line_antibody": np.random.uniform(0.1, 3.0, size=n_rows),
        "pad_material": np.random.choice(["GlassFiber", "Cellulose", "Polyester"], size=n_rows),
        "strip_length": np.random.uniform(30, 70, size=n_rows),
        "buffer_ionic_strength": np.random.uniform(50, 300, size=n_rows),
        "detergent_concentration": np.random.uniform(0, 1.0, size=n_rows),
        "sucrose": np.random.uniform(0, 5.0, size=n_rows),
        "storage_temp": np.random.uniform(4, 37, size=n_rows),
        "humidity": np.random.uniform(10, 90, size=n_rows),
    }
    # Simulate outputs with random coefficients
    X_df = pd.DataFrame(data)
    rng = np.random.default_rng(random_state)
    coefs = rng.normal(size=(X_df.select_dtypes(include=[np.number]).shape[1], 4))
    X_num = X_df.select_dtypes(include=[np.number]).to_numpy()
    y = X_num @ coefs + rng.normal(0, 0.5, size=(n_rows, 4))
    ic50 = np.abs(y[:, 0] * 10 + 2)      # minimize
    slope = y[:, 1] * 1 + 1              # maximize
    line_intensity = np.clip(y[:, 2]/20 + 0.5, 0, 1)  # maximize
    cv = np.abs(y[:, 3] * 5 + 10)        # minimize
    y_df = pd.DataFrame({
        "IC50": ic50,
        "slope": slope,
        "line_intensity": line_intensity,
        "CV": cv,
    })
    return pd.concat([X_df, y_df], axis=1)

import streamlit as st
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from skopt import gp_minimize
from skopt.space import Real
from jinja2 import Template

st.set_page_config(page_title="Eli Health LFA ML Prototype", layout="wide")

# 1. Sidebar Navigation & Global Settings
st.sidebar.image("elihealth_logo-3.jpeg", width=130)
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Go to page:",
    ["Data", "Model & Metrics", "Interpretability", "Suggest Experiments", "Download", "Help"]
)
# Sidebar Model Settings
st.sidebar.title("⚙️ Model Settings")
n_estimators = st.sidebar.slider("Random Forest Trees", 20, 300, 100, step=10)
test_size = st.sidebar.slider("Test set %", 10, 50, 30, step=5) / 100

# 2. Data Upload/Generation
if page == "Data":
    st.title("📊 LFA Data: Upload, Generate & Explore")
    st.markdown("Upload your LFA CSV or generate demo data.")

    if st.button("Generate Random Demo Data (1000 rows)"):
        df = generate_lfa_data()
        st.session_state['df'] = df
        st.success("Demo data generated!")

    uploaded = st.file_uploader("Upload your LFA CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state['df'] = df

    if 'df' in st.session_state:
        df = st.session_state['df']
        st.data_editor(df.head(50), num_rows="dynamic", use_container_width=True)
        st.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    else:
        st.warning("No data yet. Please upload or generate.")

# 3. Modeling & Metrics
elif page == "Model & Metrics":
    st.title("🤖 Model Training & Metrics Dashboard")
    if 'df' not in st.session_state:
        st.error("Upload or generate data first in the Data tab!")
    else:
        df = st.session_state['df']
        output_cols = ['IC50', 'slope', 'line_intensity', 'CV']
        # --- EXCLUDE composite_score from input_cols always!
        input_cols = [c for c in df.columns if c not in output_cols + ['composite_score']]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_num = df[input_cols].select_dtypes(include=[np.number])
        if categorical_cols:
            X_cat = pd.DataFrame(
                encoder.fit_transform(df[categorical_cols]),
                columns=encoder.get_feature_names_out(categorical_cols)
            )
            X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        else:
            X = X_num
        y = df[output_cols]

        # Composite score
        def compute_composite_score(y_df):
            y_norm = (y_df - y_df.min()) / (y_df.max() - y_df.min() + 1e-9)
            return ((1-y_norm['IC50']) + (1-y_norm['CV']) + y_norm['slope'] + y_norm['line_intensity']) / 4
        df['composite_score'] = compute_composite_score(y)
        y['composite_score'] = df['composite_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1-test_size, random_state=42)
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=3)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        st.session_state['rf'] = rf  # <-- Store in session
        st.session_state['encoder'] = encoder  # Store encoder too for one-hot
        st.session_state['input_cols'] = input_cols
        st.session_state['categorical_cols'] = categorical_cols

        # --- Resulting Performance for Best Formulation ---
        def format_val(val, decimals=2):
            if isinstance(val, float):
                return f"{val:.{decimals}f}"
            return str(val)

        output_units = {
            "IC50": "ng/mL",
            "slope": "",
            "line_intensity": "",
            "CV": "%",
            "composite_score": "",
        }
        output_good_direction = {
            "IC50": "↓ better",
            "slope": "↑ better",
            "line_intensity": "↑ better",
            "CV": "↓ better",
            "composite_score": "↑ better",
        }
        pretty_names = {
            "IC50": "IC50",
            "slope": "Slope",
            "line_intensity": "Line Intensity",
            "CV": "CV",
            "composite_score": "Composite Score",
        }

        # Find the row with the best composite score
        best_idx = df['composite_score'].idxmax()
        best_row = df.loc[best_idx]

        # Precompute dataset stats for context
        perf_cols = ["composite_score", "IC50", "slope", "line_intensity", "CV"]
        output_stats = {
            col: {
                "mean": df[col].mean(),
                "median": df[col].median(),
                "min": df[col].min(),
                "max": df[col].max(),
                "best_direction": output_good_direction[col][0],  # "↑" or "↓"
            }
            for col in perf_cols
        }

        def is_true_best(col, val):
            stats = output_stats[col]
            if stats["best_direction"] == "↑":
                return np.isclose(val, stats["max"])
            else:
                return np.isclose(val, stats["min"])

        st.markdown("### 📈 <u>Resulting Performance for Best Formulation</u>", unsafe_allow_html=True)
        cols = st.columns(len(perf_cols))
        for i, col in enumerate(perf_cols):
            val = best_row[col]
            stats = output_stats[col]
            best = stats["min"] if stats["best_direction"] == "↓" else stats["max"]
            is_best = is_true_best(col, val)
            compare_text = f"{'Lowest' if stats['best_direction']=='↓' else 'Highest'}: {format_val(best)}"
            mean_text = f"Mean: {format_val(stats['mean'])}"
            emoji = "🏆" if is_best else ""
            unit = output_units.get(col, "")
            direction = output_good_direction[col]
            color = "#36B37E" if is_best else "#253858"

            # Card
            cols[i].markdown(
                f"""
                <div style="text-align:center;">
                    <span style="font-size:1.2em;"><b>{pretty_names[col]}</b> {emoji}</span><br>
                    <span style="font-size:1.7em; color:{color};"><b>{format_val(val)}</b></span> <span style="color:gray; font-size:1em;">{unit}</span><br>
                    <span style="font-size:1em; color:#607D8B;">({direction})</span><br>
                    <span style="font-size:0.93em; color:#7A869A;">{compare_text}<br>{mean_text}</span>
                </div>
                """, unsafe_allow_html=True
            )

        # Top summary & breakdown
        st.subheader("🏆 Best LFA Formulation So Far")

        st.markdown(
            f"""
            <div style="background-color:#E3F6FD;padding:15px 18px 8px 18px;border-radius:12px;">
            <b>This experiment achieved the <span style="color:#4A90E2;font-weight:bold">highest composite score</span> (<span style="font-size:1.2em;">{format_val(best_row['composite_score'], 3)}</span>).</b><br>
            <span style="color:#607D8B;">Review both its formulation parameters and resulting performance below.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        colA, colB = st.columns([2, 3])

        with colA:
            st.markdown("#### 🎛️ Input Parameters")
            for col in input_cols:
                display_val = format_val(best_row[col], 3 if isinstance(best_row[col], float) else 0)
                if df[col].dtype == 'object':
                    tag_color = "#B2F2D4"
                    st.markdown(
                        f"<div style='margin-bottom:4px'><b>{col.replace('_', ' ').capitalize()}:</b> <span style='background:{tag_color};padding:3px 8px;border-radius:8px;color:#17493B;font-weight:500'>{display_val}</span></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='margin-bottom:4px'><b>{col.replace('_', ' ').capitalize()}:</b> <span style='color:#36B37E;'>{display_val}</span></div>",
                        unsafe_allow_html=True
                    )

        with colB:
            st.markdown("#### 📈 Resulting Performance")
            for out in perf_cols:
                val = format_val(best_row[out], 3 if out == "composite_score" else 2)
                unit = output_units.get(out, "")
                direction = output_good_direction[out]
                stats = output_stats[out]
                best = stats["min"] if stats["best_direction"] == "↓" else stats["max"]
                is_best = is_true_best(out, best_row[out])
                highlight_color = "#36B37E" if is_best else "#253858"
                st.markdown(
                    f"""
                    <div style="margin-bottom:9px;">
                        <b>{pretty_names[out]}:</b>
                        <span style="font-size:1.15em;color:{highlight_color};font-weight:bold;">{val}</span>
                        <span style="color:gray;font-size:1em">{unit}</span>
                        <span style="color:#607D8B;font-size:1em">({direction})</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with st.expander("Show as Table / Export"):
            st.dataframe(df.loc[[best_idx]])
            st.download_button(
                label="Export this Formulation as CSV",
                data=df.loc[[best_idx]].to_csv(index=False),
                file_name="best_lfa_formulation.csv",
                mime="text/csv"
            )

        st.info(
            "**Note:** These metrics reflect the overall best composite score formulation. Sometimes a different row may have the absolute lowest IC50 or CV, but with tradeoffs in other metrics. Check the comparisons in each card for context."
        )

        # --------------------

        # Metrics Table
        st.subheader("Model Performance (Test Set)")
        metrics = []
        for i, col in enumerate(list(output_cols) + ['composite_score']):
            r2 = r2_score(y_test[col], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[:, i]))
            mae = mean_absolute_error(y_test[col], y_pred[:, i])
            metrics.append({'Output': col, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
        st.dataframe(pd.DataFrame(metrics), use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df[output_cols + ['composite_score']].corr(), text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importances (Plotly)
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        fig = px.bar(
            x=importances[idx],
            y=np.array(X.columns)[idx],
            orientation='h',
            labels={'x':"Importance", 'y':"Feature"},
            title="Top 15 Feature Importances"
        )
        st.plotly_chart(fig, use_container_width=True)

# 4. Interpretability: SHAP + What-If
elif page == "Interpretability":
    st.title("🔎 Model Interpretability & What-If Simulator")
    # Check everything needed is in session_state
    if (
        'df' not in st.session_state or
        'rf' not in st.session_state or
        'encoder' not in st.session_state or
        'input_cols' not in st.session_state or
        'categorical_cols' not in st.session_state
    ):
        st.error("Please load data and train the model first in 'Model & Metrics' tab!")
    else:
        df = st.session_state['df']
        rf = st.session_state['rf']
        encoder = st.session_state['encoder']
        # --- EXCLUDE composite_score every time you set input_cols
        output_cols = ['IC50', 'slope', 'line_intensity', 'CV']
        input_cols = [c for c in df.columns if c not in output_cols + ['composite_score']]
        categorical_cols = st.session_state['categorical_cols']

        # SHAP SUMMARY (Composite Score)
        st.markdown("**SHAP Summary for Composite Score**")
        # Only use *true* input features for SHAP
        X_num = df[input_cols].select_dtypes(include=[np.number])
        if categorical_cols:
            X_cat = pd.DataFrame(
                encoder.transform(df[categorical_cols]),
                columns=encoder.get_feature_names_out(categorical_cols)
            )
            X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
        else:
            X = X_num
        y_composite = df['composite_score']

        # Train a new RF for SHAP explanation
        rf_shap = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
        rf_shap.fit(X, y_composite)
        explainer = shap.TreeExplainer(rf_shap)
        shap_vals = explainer.shap_values(X)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, X, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # WHAT-IF SIMULATOR
        st.markdown("---")
        st.subheader("What-If Experiment Simulator 🧪")

        st.markdown("**Set your experimental parameters:**")
        user_inputs = {}
        for col in input_cols:
            if df[col].dtype == 'object':
                user_inputs[col] = st.selectbox(
                    f"{col.replace('_', ' ').capitalize()}",
                    sorted(df[col].unique())
                )
            else:
                user_inputs[col] = st.slider(
                    f"{col.replace('_', ' ').capitalize()}",
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].mean())
                )

        # Prepare input row
        input_df = pd.DataFrame([user_inputs])
        X_num_ = input_df.select_dtypes(include=[np.number])
        if categorical_cols:
            X_cat_ = pd.DataFrame(
                encoder.transform(input_df[categorical_cols]),
                columns=encoder.get_feature_names_out(categorical_cols)
            )
            X_eval = pd.concat([X_num_.reset_index(drop=True), X_cat_.reset_index(drop=True)], axis=1)
        else:
            X_eval = X_num_

        # Predict ALL outputs (not just composite)
        y_pred_all = rf.predict(X_eval)[0]
        result_dict = dict(zip(output_cols, y_pred_all[:4]))
        # Calculate composite score using same normalization as in your app
        mins = df[output_cols].min().values
        maxs = df[output_cols].max().values
        norm = (y_pred_all[:4] - mins) / (maxs - mins + 1e-9)
        composite_sim = ((1 - norm[0]) + (1 - norm[3]) + norm[1] + norm[2]) / 4

        # Show each metric as a card
        st.markdown("#### Predicted Performance:")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("IC50", f"{result_dict['IC50']:.2f}")
        col2.metric("Slope", f"{result_dict['slope']:.2f}")
        col3.metric("Line Intensity", f"{result_dict['line_intensity']:.2f}")
        col4.metric("CV", f"{result_dict['CV']:.2f}")
        col5.metric("Composite Score", f"{composite_sim:.3f}")

        import pandas as pd

        # 1. Which outputs are dragging the composite down?
        metric_targets = {
            "IC50": "lower",
            "CV": "lower",
            "slope": "higher",
            "line_intensity": "higher"
        }
        norm_metrics = {
            "IC50": 1-norm[0],
            "CV": 1-norm[3],
            "slope": norm[1],
            "line_intensity": norm[2]
        }
        suboptimal_msgs = []
        advice_lines = []

        for m in ['IC50', 'CV', 'slope', 'line_intensity']:
            threshold = 0.6
            if ((metric_targets[m]=="higher" and norm_metrics[m]<threshold) or
                (metric_targets[m]=="lower" and norm_metrics[m]<threshold)):
                suboptimal_msgs.append(m)

        # 2. For each suboptimal output, suggest which direction to move which input
        for metric in suboptimal_msgs:
            rf_single = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
            rf_single.fit(X, df[metric])
            imp_series = pd.Series(rf_single.feature_importances_, index=X.columns)
            top_feats = imp_series.sort_values(ascending=False).head(2).index.tolist()

            dir_advice = []
            for feat in top_feats:
                # Check if input feature is numeric (we can't suggest direction for categories)
                if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                    corr = df[feat].corr(df[metric])
                    if metric_targets[metric] == "lower":
                        # We want to lower the output
                        if corr > 0:
                            dir_advice.append(f"**decrease {feat.replace('_',' ')}**")
                        else:
                            dir_advice.append(f"**increase {feat.replace('_',' ')}**")
                    else:
                        # We want to raise the output
                        if corr > 0:
                            dir_advice.append(f"**increase {feat.replace('_',' ')}**")
                        else:
                            dir_advice.append(f"**decrease {feat.replace('_',' ')}**")
                else:
                    # For categorical or unknown, just say "adjust"
                    dir_advice.append(f"**adjust {feat.replace('_',' ')}**")
            advice_lines.append(f"To improve {metric}, {', and '.join(dir_advice)}.")

        if advice_lines:
            st.warning(f"⚠️ Suboptimal: {', '.join(suboptimal_msgs)} are dragging down the composite score.\n\n" +
                    " ".join(advice_lines))
        elif composite_sim > 0.8:
            st.success("🎉 **Excellent formulation!** This simulated experiment is predicted to have a high overall composite score.")
        elif composite_sim > 0.6:
            st.info("👍 **Good tradeoff:** This experiment is decent, but you could likely improve one or more metrics.")
        elif composite_sim > 0.4:
            st.warning("⚠️ **Suboptimal:** Some metrics are dragging down the composite score. Try adjusting parameters.")
        else:
            st.error("❌ **Poor result:** This simulated experiment is predicted to be low-performing. Try optimizing IC50 and CV.")



        # Radar/spider plot for outputs (optional, but recommended!)
        import plotly.graph_objects as go
        categories = ['IC50', 'Slope', 'Line Intensity', 'CV']
        radar_norm = [
            1 - norm[0],        # IC50 (lower is better)
            norm[1],            # Slope (higher is better)
            norm[2],            # Line Intensity (higher is better)
            1 - norm[3],        # CV (lower is better)
        ]
        fig_radar = go.Figure(data=[
            go.Scatterpolar(r=[*radar_norm, radar_norm[0]],
                            theta=[*categories, categories[0]],
                            fill='toself',
                            name='Prediction')
        ])
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Predicted Metric Profile (normalized)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# 5. Experiment Suggestions
elif page == "Suggest Experiments":
    st.title("🧬 Experiment Suggestions (Bayesian Optimization)")
    st.write("🚧 _Under construction!_ (Let me know if you want this fully built out right now)")

# 6. Download & Reports
elif page == "Download":
    st.title("⬇️ Download Results")
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.download_button("Download Current Data (CSV)", df.to_csv(index=False), "lfa_data.csv", "text/csv")
    else:
        st.warning("No data to download yet.")

# 7. Help & Info
elif page == "Help":
    st.title("ℹ️ Help & Info")
    st.markdown("""
    **What can this app do?**  
    - Simulate or upload LFA optimization data
    - Train Random Forest and analyze metrics
    - Explore features & interpret models (SHAP, feature importances)
    - Suggest new experiment settings via Bayesian Optimization
    - Download results

    **Composite Score** is calculated as:
    $$
    \\text{Composite} = \\frac{(1 - \\text{IC50}_\\text{norm}) + (1 - \\text{CV}_\\text{norm}) + \\text{slope}_\\text{norm} + \\text{intensity}_\\text{norm}}{4}
    $$
    """)

# -- END OF APP --
