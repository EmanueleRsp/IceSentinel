import os
import glob
import logging

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


#################################################################
############################# UTILS #############################
#################################################################

# --------------------------
# Configuration
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

theme_colors = {
    'navy': '#001f3f',
    'ice_blue': '#7FDBFF',
    'snow_white': '#FFFFFF'
}

st.set_page_config(page_title="IceSentinel", layout="wide")


# --------------------------
# Cached Utilities
# --------------------------
@st.cache_resource
def load_model(model_name: str) -> Pipeline:
    path = os.path.join('models', f'{model_name}.pkl')
    logging.info(f"Loading model {path}")
    return joblib.load(path)

@st.cache_resource
def load_data(file) -> pd.DataFrame:
    logging.info("Loading data from uploaded CSV")
    return pd.read_csv(file)

@st.cache_resource
def _get_explainer(_clf, background_Xp, feature_names, output_names) -> shap.Explainer:
    logging.info("Creating SHAP Explainer")
    return shap.Explainer(
        _clf.predict_proba,       # ← wrap the probability function
        background_Xp,           # ← preprocessed background sample
        feature_names=feature_names,
        output_names=output_names
    )


# --------------------------
# SHAP Cache Handler
# --------------------------
def load_or_compute_shap(explainer, data, feature_names, class_names, dirpath='.', prefix='shap_values_class_', overwrite=False):
    # Find existing files
    pattern = os.path.join(dirpath, f"{prefix}*.csv")
    logging.info(f"Looking for SHAP values in {pattern}")
    csv_files = sorted(glob.glob(pattern))
    
    if csv_files and not overwrite:
        logging.info(f"Loading cached SHAP values from {csv_files}")
        loaded = np.stack([pd.read_csv(f).values for f in csv_files], axis=2)
        return loaded if loaded.ndim==3 else loaded[:, :, np.newaxis]
    
    # Compute SHAP values if not cached
    logging.info("Computing SHAP values...")
    os.makedirs(dirpath, exist_ok=True)
    shap_values = explainer.shap_values(data)
    # Save SHAP values to CSV for future use
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        df = pd.DataFrame(shap_values, columns=feature_names)
        fname = os.path.join(dirpath, prefix.rstrip('_') + '.csv')
        df.to_csv(fname, index=False)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        n_classes = shap_values.shape[2]
        for i in range(n_classes):
            df = pd.DataFrame(shap_values[:, :, i], columns=feature_names)
            fname = os.path.join(dirpath, f"{prefix}{class_names[i]}.csv")
            df.to_csv(fname, index=False)
    return shap_values


# --------------------------
# Prediction Utility
# --------------------------
def compute_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Apply the model to the DataFrame and return predictions and probabilities."""
    df = df.copy()
    preds = model.predict(df)
    preds_proba = model.predict_proba(df)
    mapping = {0: '🟩 1-Low', 1: '🟨 2-Moderate', 2: '🟧 3-Considerable', 3: '‼️ 4-High'}   # Map numeric labels to human-readable strings
    df['Predicted Level'] = [mapping.get(p, str(p)) for p in preds] # Add predicted level column
    classes = model.classes_    # Add a probability column for each class
    for idx, cls in enumerate(classes): 
        label = mapping.get(cls, str(cls)).split(' ', 1)[1] if ' ' in mapping.get(cls, str(cls)) else str(cls)
        col_name = f'Prob_{label}'
        df[col_name] = preds_proba[:, idx]
    return df


def initialize_session_state():
    """Initialize Streamlit session state variables for the app."""
    if 'models_dict' not in st.session_state:
        st.session_state.models_dict = {os.path.splitext(f)[0]: f for f in os.listdir('models') if f.endswith('.pkl')}
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = next(iter(st.session_state.models_dict.keys()))
    if 'uploaded_csv' not in st.session_state:
        st.session_state.uploaded_csv = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'pred_df' not in st.session_state:
        st.session_state.pred_df = None
    if 'model' not in st.session_state:
        # Load model and extract pipeline steps
        st.session_state.model = load_model(st.session_state.model_choice)
        # Remove any sampling steps (like RandomOverSampler) from the pipeline for preprocessing
        st.session_state.preprocess = Pipeline([step for step in st.session_state.model.steps[:-1] if step[0] != 'sampler'])
        st.session_state.classifier = st.session_state.model.steps[-1][1]
        st.session_state.feature_names = [feat.replace('selector__', '') for feat in st.session_state.preprocess.get_feature_names_out()]
        st.session_state.class_names = st.session_state.classifier.classes_
    if 'explainer' not in st.session_state and \
        st.session_state.classifier is not None and \
        st.session_state.preprocess is not None and \
        st.session_state.df is not None:
        st.session_state.explainer = _get_explainer(
            st.session_state.classifier, 
            st.session_state.preprocess.transform(st.session_state.df),
            feature_names=st.session_state.feature_names,
            output_names=st.session_state.class_names
        )
    else:
        st.session_state.explainer = None


#################################################################
########################### INTERFACE ###########################
#################################################################

def apply_css():
    css = f"""
    <style>
    .stApp {{ background-color: {theme_colors['navy']}; }}
    .stSidebar {{ background-color: #101414; }}
    .stButton>button {{ background-color: {theme_colors['ice_blue']}; color: {theme_colors['navy']}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def select_model(models_dict):
    choice = st.sidebar.radio("Select model", list(models_dict.keys()), index=0)
    if choice is None:
        return load_model(next(iter(models_dict.keys())))
    return load_model(choice)


def upload_and_show_predictions(model):
    uploaded = st.file_uploader("Upload CSV data", type='csv')
    if not uploaded:
        st.info("Please upload a CSV file to proceed. There's a sample file in the `interface\\test` folder (`X_test.csv`) that you can use to test the app.")
        return None, None   
    df = load_data(uploaded)
    pred_df = compute_predictions(df, model)
    st.write("#### Input data:")
    st.dataframe(df)
    st.write("#### Predictions:")
    # Expected danger level is computed as a weighted sum of probabilities
    prob_cols = [col for col in pred_df.columns if col.startswith('Prob_')]
    weights = np.arange(1, len(prob_cols) + 1)
    pred_df['Expected danger level'] = pred_df[prob_cols].values @ weights
    predictions_df = pred_df[['Predicted Level', 'Expected danger level'] + prob_cols]
    st.dataframe(predictions_df)
    return df, pred_df


def show_dashboard(pred_df):
    st.subheader("Dataset Overview")
    counts = pred_df['Predicted Level'].value_counts(normalize=True).mul(100).round(1)
    cols = st.columns(len(counts))
    for i, (lvl, pct) in enumerate(counts.items()):
        cols[i].metric(lvl, f"{pct}%")


def filters_section(pred_df):
    with st.expander("Filters"):
        levels = ['All'] + sorted(pred_df['Predicted Level'].unique())
        lvl = st.selectbox("Filter by level", levels)
        thr = st.slider("Min probability", 0.0, 1.0, 0.0)
        df = pred_df.copy()
        if lvl != 'All': 
            df = df[df['Predicted Level'] == lvl]
        prob_cols = [c for c in df if c.startswith('Prob_')]
        df = df[df[prob_cols].max(axis=1) >= thr]
        st.write("**Filtered Input Data**", df)
        st.write("**Filtered Predictions**", df[['Predicted Level'] + prob_cols])
        return df


def global_shap_section(df, preprocess, classifier, explainer, feature_names, class_names, models_dict):
    with st.expander("Global SHAP Analysis"):
        st.write("Insights about the model reasoning.")
        mode = st.selectbox("Mode", [
                "", 
                "Load test data", 
                # "Compute Now (with the current model)"
            ], 
            index=0
        )
        st.info("Test values were computed for the *2020 winter period* observations. ")
        if not mode:
            return
        # Load SHAP values
        if mode == "Load test data":
            choice = st.radio("Select model", list(models_dict.keys()), index=0, key="shap_model_choice")
            if choice is None:
                return
            model = load_model(choice)
            preprocess = Pipeline([step for step in model.steps[:-1] if hasattr(step[1], "transform")])
            classifier = model.steps[-1][1]
            feature_names = [feat.replace('feat_sel__', '').replace('selector__', '') for feat in preprocess.get_feature_names_out()]
            class_names = classifier.classes_
            explainer = _get_explainer(
                classifier, 
                background_Xp=preprocess.transform(df),
                feature_names=feature_names,
                output_names=class_names
            )
            data_proc = preprocess.transform(df)
            shp = load_or_compute_shap(
                explainer, data_proc, feature_names, class_names,
                dirpath=os.path.join('results', 'shap', 'csv', choice, 'winter_2020'),
                prefix='shap_values_class_',
                overwrite=False
            )
        # Compute SHAP values
        elif mode == "Compute Now":
            data_proc = preprocess.transform(df)
            shp = load_or_compute_shap(
                explainer, data_proc, feature_names, class_names,
                dirpath=(os.path.join('interface', 'test', 'shap_cache')), 
                prefix='shap_values_class_', 
                overwrite=True
            )
        logging.info(f"SHAP values shape: {shp.shape}")
        # Feature importance
        st.markdown("---")
        st.write("##### Feature Importance")
        fig = plt.figure()
        shap.summary_plot(
            shp, 
            data_proc,
            plot_type='bar', 
            class_names=class_names+1, 
            feature_names=feature_names, 
            max_display=30, 
            show=False
        )
        plt.title(f"SHAP Summary Plot - {choice}")
        st.pyplot(fig)

        # Summary dot grid
        st.markdown("---")
        st.write("##### SHAP Summary Plots")
        cols = st.columns(2)
        for i, cls in enumerate(class_names):
            with cols[i % 2]:
                st.write(f"**Class {cls+1}**")
                # Create a new figure for each plot
                fig = plt.figure()
                shap.summary_plot(
                    shp[:,:,i],
                    data_proc,
                    feature_names=feature_names,
                    plot_type='dot',
                    show=False
                )
                st.pyplot(fig)
                plt.clf()
        st.markdown(
            """
            **How to read these summary plots:**  
            - **Y-axis** orders features by overall importance.  
            - **Dots** represent samples; X-axis position shows each sample’s impact on predicting that class.  
            - **Color** reflects the original feature value (blue = low, red = high), helping spot how feature magnitude affects that class.
            """
        )

        # SHAP dependence plot for a selected feature
        st.markdown("---")
        st.write("#### SHAP Dependence Plot")
        feat_selected = st.selectbox(
            "Select feature:",
            options=feature_names,
            key="dep_feat_global"
        )
        cols = st.columns(2)
        for i, cls in enumerate(class_names):
            with cols[i % 2]:
                st.write(f"**Class {cls+1}**")
                shap.dependence_plot(
                    feature_names.index(feat_selected),
                    shp[:,:,i],
                    data_proc,
                    feature_names=feature_names,
                    show=False,
                )
                st.pyplot(plt.gcf())
                plt.clf()
        st.markdown(
            """
            **How to read these dependence plots:**  
            - **X-axis** shows the actual feature value for each sample.  
            - **Y-axis** shows the SHAP value (impact) for predicting that class.  
            - **Color** represents the value of a second feature (by default the most correlated one), highlighting interaction effects.  
            """
        )


def local_shap_section(df, preprocess, classifier, feature_names, class_names, model):
    st.subheader("Local SHAP Analysis")
    idx = st.number_input("Sample index", 0, len(df)-1, 0, 1)
    if not st.button("Compute Local SHAP"):
        return

    X = df.iloc[[idx]]
    Xp = preprocess.transform(X)
    
    explainer = _get_explainer(
        classifier, 
        background_Xp=preprocess.transform(df),
        feature_names=feature_names,
        output_names=list(class_names)
    )
    exp = explainer(Xp)
    pred = model.predict(X)[0]
    class_idx = list(class_names).index(pred)

    st.write("#### Details:")
    st.write(X)
    st.write("**Considered features:**")
    st.write(X[feature_names])
    st.caption(f"**Prediction: {pred+1}**")

    # Probability bar
    proba = model.predict_proba(X)[0]
    fig = go.Figure(go.Bar(
        x=proba, 
        y=['Low', 'Moderate', 'Considerable', 'High'], 
        orientation='h',
        text=[f'{p:.2%}' for p in proba],
        textposition='auto',
        marker_color=['green', 'yellow', 'orange', 'red']))
    fig.update_layout(
        title='Probability Distribution',
        xaxis_title='Probability',
        yaxis_title='Danger Level',
        showlegend=False,
        xaxis=dict(range=[0, 1]),
        plot_bgcolor='#001f3f', paper_bgcolor='#001f3f', font=dict(color='#001f3f'))
    st.plotly_chart(fig)

    # SHAP explanation for the sample
    X_exp = shap.Explanation(
        base_values=exp.base_values[0, class_idx],
        values=exp.values[0, :, class_idx],
        data=np.round(X[feature_names].to_numpy()[0], 3), 
        feature_names=feature_names
    )
    # Force plot of the predicted class
    st.write(f"##### SHAP Force")
    fig_fp = shap.plots.force(X_exp, matplotlib=True, show=False)
    st.pyplot(fig_fp)

    # Waterfall plots for each class
    st.write("##### SHAP Waterfall Plots")
    cols = st.columns(2)
    for i, cls in enumerate(class_names):
        with cols[i % 2]:
            st.write(f"**Class {cls+1}**")
            wf_fig, ax = plt.subplots()
            X_exp = shap.Explanation(
                base_values=exp.base_values[0, cls],
                values=exp.values[0, :, cls],
                data=np.round(X[feature_names].to_numpy()[0], 3), 
                feature_names=feature_names
            )
            shap.plots.waterfall(X_exp, max_display=10, show=False)
            st.pyplot(wf_fig)

    # Guidance on reading local plots
    st.markdown(
        """
        **How to interpret local SHAP plots:**  
        - **Waterfall**: Shows the cumulative impact of each feature on the individual prediction. Features pushing the output higher appear to the right, those lowering it to the left.  
        - **Force**: Visualizes the same information with arrows; red pushes towards the predicted risk level, blue towards the opposite, with the width proportional to the magnitude.  
        """
    )


def main():

    apply_css()
    initialize_session_state()

    st.title("IceSentinel")
    st.markdown("### An Avalanche Danger Level Classificator")
    
    st.session_state.model = select_model(st.session_state.models_dict)
    st.session_state.preprocess = Pipeline([step for step in st.session_state.model.steps[:-1] if hasattr(step[1], "transform")])
    st.session_state.classifier = st.session_state.model.steps[-1][1]
    st.session_state.feature_names = [feat.replace('feat_sel__', '').replace('selector__', '') for feat in st.session_state.preprocess.get_feature_names_out()]
    st.session_state.class_names = st.session_state.classifier.classes_
    
    if st.session_state.classifier is not None and \
        st.session_state.preprocess is not None and \
        st.session_state.df is not None:
        st.session_state.explainer = _get_explainer(
            st.session_state.classifier, 
            background_Xp=st.session_state.preprocess.transform(st.session_state.df),
            feature_names=st.session_state.feature_names,
            output_names=st.session_state.class_names
        )
    else:
        st.session_state.explainer = None

    # File uploader
    st.session_state.df, st.session_state.pred_df = upload_and_show_predictions(st.session_state.model)
    if st.session_state.df is None:
        return
    show_dashboard(st.session_state.pred_df)
    filters_section(st.session_state.pred_df)
    
    global_shap_section(
        st.session_state.df, 
        st.session_state.preprocess,
        st.session_state.classifier,
        st.session_state.explainer, 
        st.session_state.feature_names, 
        st.session_state.class_names,
        st.session_state.models_dict
    )
    
    st.markdown("---")
    
    local_shap_section(
        st.session_state.df, 
        st.session_state.preprocess, 
        st.session_state.classifier, 
        st.session_state.feature_names, 
        st.session_state.class_names,
        st.session_state.model
    )


if __name__ == '__main__':
    main()
