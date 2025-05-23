import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap
from sklearn.tree import DecisionTreeRegressor, plot_tree
from scipy.stats import linregress
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from jinja2 import Template

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

def compute_composite_score(y_df):
    y_norm = (y_df - y_df.min()) / (y_df.max() - y_df.min() + 1e-9)
    score_ic50 = 1 - y_norm['IC50']
    score_cv = 1 - y_norm['CV']
    score_slope = y_norm['slope']
    score_intensity = y_norm['line_intensity']
    composite = (score_ic50 + score_cv + score_slope + score_intensity) / 4
    return composite

st.set_page_config(page_title="Eli Health LFA ML Prototype", layout="wide")
st.image("elihealth_logo-3.jpeg", width=200)
st.title("Eli Health LFA Optimization ML Prototype")

with st.expander("Generate Demo LFA Data"):
    if st.button("Generate random realistic LFA data (1000 rows)"):
        df = generate_lfa_data()
        df.to_csv("lfa_random_data.csv", index=False)
        st.success("Demo data saved as lfa_random_data.csv!")
        st.dataframe(df.head())

uploaded = st.file_uploader("Upload your LFA data CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head(20))
elif 'df' not in locals():
    st.warning("Upload your data or generate demo data.")

if 'df' in locals():
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    output_cols = ['IC50', 'slope', 'line_intensity', 'CV']
    input_cols = [c for c in df.columns if c not in output_cols]

    # OneHotEncoder: robust to sklearn version
    import sklearn
    from packaging import version
    if version.parse(sklearn.__version__) >= version.parse("1.2.0"):
        encoder = OneHotEncoder(sparse_output=False, drop='first')
    else:
        encoder = OneHotEncoder(sparse=False, drop='first')

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
    df['composite_score'] = compute_composite_score(y)
    y['composite_score'] = df['composite_score']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.subheader("Random Forest Metrics")
    metrics = []
    for i, col in enumerate(list(output_cols) + ['composite_score']):
        r2 = r2_score(y_test[col], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[col], y_pred[:, i]))
        mae = mean_absolute_error(y_test[col], y_pred[:, i])
        metrics.append({'Output': col, 'R2': r2, 'RMSE': rmse, 'MAE': mae})
    st.dataframe(pd.DataFrame(metrics))

    # Feature Importances
    importances = rf.feature_importances_
    fig, ax = plt.subplots(figsize=(8,4))
    feat_names = X.columns
    idx = np.argsort(importances)[::-1][:15]
    sns.barplot(x=importances[idx], y=feat_names[idx], ax=ax)
    ax.set_title("Top 15 Feature Importances (all outputs)")
    st.pyplot(fig)

    # SHAP, directionality, surrogate for each output (including composite)
    for out_col in list(output_cols) + ['composite_score']:
        st.markdown(f"---\n### SHAP Beeswarm & Surrogate Tree for: **{out_col}**")
        rf_single = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=3)
        rf_single.fit(X_train, y_train[out_col])
        explainer = shap.TreeExplainer(rf_single)
        shap_vals = explainer.shap_values(X_test)
        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_vals, X_test, show=False)
        st.pyplot(fig2)

        st.markdown("**Feature Directionality**")
        bullet_lines = []
        for j, feat in enumerate(X.columns):
            try:
                slope, intercept, r, p, std_err = linregress(X_test[feat], shap_vals[:, j])
                if abs(r) < 0.1:
                    bullet_lines.append(f"- {feat}: no clear effect")
                else:
                    emoji = "⬆️" if slope > 0 else "⬇️"
                    direction = "increases" if slope > 0 else "decreases"
                    bullet_lines.append(
                        f"- {emoji} Increasing **{feat}** {direction} **{out_col}** (corr: {r:+.2f})"
                    )
            except Exception as e:
                bullet_lines.append(f"- {feat}: (could not compute directionality)")
        st.markdown("\n".join(bullet_lines))

        st.markdown("**Surrogate Tree (Depth 3)**")
        surrogate = DecisionTreeRegressor(max_depth=3)
        y_rf_pred = rf_single.predict(X_train)
        surrogate.fit(X_train, y_rf_pred)
        fig3, ax3 = plt.subplots(figsize=(12,5))
        plot_tree(surrogate, feature_names=X.columns, filled=True, ax=ax3)
        st.pyplot(fig3)

    # Bayesian Optimization on composite_score
    st.subheader("Bayesian Optimization: Top 10 Experiment Suggestions (by composite_score)")
    def objective(x):
        pred = rf.predict([x])[0]
        mins = y_train[["IC50", "slope", "line_intensity", "CV"]].min().values
        maxs = y_train[["IC50", "slope", "line_intensity", "CV"]].max().values
        norm = (pred[:4] - mins) / (maxs - mins + 1e-9)
        score_ic50 = 1 - norm[0]
        score_cv = 1 - norm[3]
        score_slope = norm[1]
        score_intensity = norm[2]
        composite = (score_ic50 + score_cv + score_slope + score_intensity) / 4
        return -composite

    space = []
    for col in input_cols:
        if df[col].dtype == 'object':
            space.append(Real(0, df[col].nunique() - 1, name=col))
        else:
            space.append(Real(float(df[col].min()), float(df[col].max()), name=col))

    def decode_x(x):
        vals = []
        idx = 0
        for col in input_cols:
            if df[col].dtype == 'object':
                cats = df[col].unique()
                vals.append(cats[int(np.round(x[idx]))])
            else:
                vals.append(x[idx])
            idx += 1
        return vals

    def opt_objective(x):
        vals = decode_x(x)
        x_dict = {col: val for col, val in zip(input_cols, vals)}
        x_df = pd.DataFrame([x_dict])
        X_num_ = x_df.select_dtypes(include=[np.number])
        if categorical_cols:
            X_cat_ = pd.DataFrame(
                encoder.transform(x_df[categorical_cols]),
                columns=encoder.get_feature_names_out(categorical_cols)
            )
            X_eval = pd.concat([X_num_.reset_index(drop=True), X_cat_.reset_index(drop=True)], axis=1)
        else:
            X_eval = X_num_
        return objective(X_eval.values[0])

    res = gp_minimize(opt_objective, space, n_calls=30, random_state=42)
    all_params = [decode_x(x) for x in res.x_iters]
    all_outputs = rf.predict(
        pd.DataFrame([
            {col: val for col, val in zip(input_cols, p)} for p in all_params
        ]).pipe(lambda d: (
            pd.concat([
                d.select_dtypes(include=[np.number]).reset_index(drop=True),
                pd.DataFrame(
                    encoder.transform(d[categorical_cols]),
                    columns=encoder.get_feature_names_out(categorical_cols)
                ).reset_index(drop=True)
            ], axis=1)
            if categorical_cols else
            d.select_dtypes(include=[np.number])
        ))
    )
    bo_df = pd.DataFrame(all_params, columns=input_cols)
    for i, col in enumerate(list(output_cols) + ['composite_score']):
        bo_df[col] = all_outputs[:, i] if col != 'composite_score' else compute_composite_score(pd.DataFrame(all_outputs[:, :4], columns=output_cols))
    bo_df['composite_score'] = compute_composite_score(bo_df[output_cols])
    bo_top10 = bo_df.sort_values('composite_score', ascending=False).head(10)
    st.dataframe(bo_top10)

    # HTML report and download
    html_template = """
    <html>
    <head>
    <title>LFA Optimization Report</title>
    <style>
        body { font-family: Arial, Helvetica, sans-serif; margin: 30px;}
        h1 { background: #D8E6F2; padding: 20px; border-radius: 6px;}
        th, td { padding: 7px 15px; }
    </style>
    </head>
    <body>
    <h1>Eli Health LFA Optimization ML Report</h1>
    <h3>Random Forest Metrics</h3>
    {{ metrics_html }}
    <h3>Top 10 Bayesian Optimization Suggestions</h3>
    {{ bo_html }}
    </body>
    </html>
    """
    metrics_html = pd.DataFrame(metrics).to_html(index=False)
    bo_html = bo_top10.to_html(index=False)
    html_report = Template(html_template).render(metrics_html=metrics_html, bo_html=bo_html)
    html_path = "EliHealth_LFA_Report.html"
    with open(html_path, "w") as f:
        f.write(html_report)
    st.success(f"HTML report generated!")

    st.download_button(
        label="⬇️ Download HTML Report",
        data=html_report,
        file_name="EliHealth_LFA_Report.html",
        mime="text/html"
    )
