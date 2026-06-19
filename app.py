"""
Claim Amount Predictor — Streamlit App
----------------------------------------
Run locally with:
    streamlit run app.py

Requires the following files (produced by train_claim_amount_model.py)
to be in the same folder as this script:
    - claim_amount_model.pkl
    - target_encodings.pkl
    - global_mean.pkl
    - model_metadata.pkl
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Claim Amount Predictor",
    layout="centered",
)

# -----------------------------------------------------------------
# LOAD MODEL + METADATA (cached so it only loads once per session)
# -----------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    model = joblib.load(base / "claim_amount_model.pkl")
    target_encodings = joblib.load(base / "target_encodings.pkl")
    global_mean = joblib.load(base / "global_mean.pkl")
    metadata = joblib.load(base / "model_metadata.pkl")
    return model, target_encodings, global_mean, metadata

try:
    model, target_encodings, global_mean, metadata = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Model files not found. Make sure claim_amount_model.pkl, "
        "target_encodings.pkl, global_mean.pkl, and model_metadata.pkl "
        "are in the same folder as this app."
    )
    st.stop()

numeric_cols = metadata["numeric_cols"]
low_card_cols = metadata["low_card_cols"]
high_card_cols = metadata["high_card_cols"]
low_card_options = metadata["low_card_options"]
high_card_options = metadata["high_card_options"]
numeric_defaults = metadata["numeric_defaults"]


def _anon_label(index: int) -> str:
    """Turn 0,1,2,...,25,26,... into Option A, Option B, ..., Option Z, Option AA, ..."""
    letters = ""
    n = index
    while True:
        n, rem = divmod(n, 26)
        letters = chr(65 + rem) + letters
        if n == 0:
            break
        n -= 1
    return f"Option {letters}"

# -----------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------
st.title("Claim Amount Predictor")
st.caption(
    "Estimate the expected claim amount for a given claim record."
)

with st.expander("About this model"):
    st.markdown(
        f"""
        This app uses a **Linear Regression** model trained on historical
        claims data.

        | Metric | Value |
        |---|---|
        | MAE | {metadata['model_mae']:.2f} |
        | RMSE | {metadata['model_rmse']:.2f} |
        | R² | {metadata['model_r2']:.3f} |
        | Training rows | {metadata['n_train_rows']:,} |
        """
    )

st.divider()

# -----------------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------------
st.subheader("Claim Details")

with st.form("prediction_form"):
    user_inputs = {}

    # --- Numeric inputs ---
    if numeric_cols:
        st.markdown("**Numeric fields**")
        num_col_layout = st.columns(2)
        for i, col in enumerate(numeric_cols):
            with num_col_layout[i % 2]:
                default_val = numeric_defaults.get(col, 0.0)
                user_inputs[col] = st.number_input(
                    label=col.replace("_", " ").title(),
                    value=float(default_val),
                    format="%.2f",
                )

    # --- Low-cardinality categorical inputs (dropdowns) ---
    if low_card_cols:
        st.markdown("**Category fields**")
        cat_col_layout = st.columns(2)
        for i, col in enumerate(low_card_cols):
            options = low_card_options.get(col, [])
            with cat_col_layout[i % 2]:
                user_inputs[col] = st.selectbox(
                    label=col.replace("_", " ").title(),
                    options=options if options else ["(no data)"],
                )

    # --- High-cardinality categorical inputs (dropdowns, target-encoded) ---
    if high_card_cols:
        st.markdown("**Provider / Doctor / Treatment fields**")
        hc_col_layout = st.columns(2)
        for i, col in enumerate(high_card_cols):
            options = high_card_options.get(col, [])
            with hc_col_layout[i % 2]:
                if options:
                    # Show anonymized labels (e.g. "Option A", "Option B", ...)
                    # in the UI, but keep the real underlying value for the
                    # target-encoding lookup used in prediction.
                    display_labels = [_anon_label(idx) for idx in range(len(options))]
                    label_to_real = dict(zip(display_labels, options))
                    chosen_label = st.selectbox(
                        label=col.replace("_", " ").title(),
                        options=display_labels,
                    )
                    user_inputs[col] = label_to_real[chosen_label]
                else:
                    st.selectbox(
                        label=col.replace("_", " ").title(),
                        options=["(no data)"],
                    )
                    user_inputs[col] = None

    submitted = st.form_submit_button("Predict Claim Amount", use_container_width=True)

# -----------------------------------------------------------------
# PREDICTION LOGIC
# -----------------------------------------------------------------
if submitted:
    # Build a single-row dataframe matching training-time structure
    row = {}

    # numeric + low-card categorical columns go in as-is
    for col in numeric_cols:
        row[col] = user_inputs[col]
    for col in low_card_cols:
        row[col] = user_inputs[col]

    # high-cardinality columns -> apply the SAME target encoding used in training
    for col in high_card_cols:
        chosen_value = user_inputs[col]
        means = target_encodings.get(col, pd.Series(dtype=float))
        encoded_value = means.get(chosen_value, global_mean)
        row[col + "_te"] = encoded_value

    input_df = pd.DataFrame([row])

    # Ensure column order/presence matches what the pipeline's ColumnTransformer expects.
    # (The pipeline's ColumnTransformer selects columns by name, so order here doesn't
    # matter as long as all expected columns are present.)
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    st.subheader("Result")
    st.metric(label="Predicted Claim Amount", value=f"SRD {prediction:,.2f}")

    st.caption(
        "This is a model estimate based on historical patterns. "
        f"Typical error margin (MAE) is ± SRD {metadata['model_mae']:.2f}."
    )

# -----------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------
st.divider()
st.caption("Baseline Model Built by Akunna Anyamkpa")
