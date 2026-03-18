"""
miRNA Upregulation Predictor
=============================
Model : lgbm_mirna_model.pkl  (LightGBM + TargetEncoder + isotonic calibration)
Run   : streamlit run app.py

Prediction flow:
  1. Encode input with pipe (encoder step)
  2. Predict with calibrated_model (calibrated probabilities)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="miRNA Upregulation Predictor",
    page_icon="🧬",
    layout="wide"
)

# ── Load model bundle ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('lgbm_mirna_model_fixed.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_targetscan():
    """Load TargetScan miR_Family_Info.txt and build lookup."""
    try:
        ts = pd.read_csv('miR_Family_Info.txt', sep='\t', dtype=str)
        ts.columns = [c.strip() for c in ts.columns]
        broadly = ts[ts['Family Conservation?'] == '2'].copy()

        def normalize(name):
            name = str(name).strip().lower()
            name = re.sub(r'^[a-z]{3}-', '', name)
            name = re.sub(r'[-.](5p|3p|\*)$', '', name)
            return name

        broadly['norm'] = broadly['MiRBase ID'].apply(normalize)
        return broadly.set_index('norm')['miR family'].to_dict()
    except FileNotFoundError:
        return {}

def lookup_family(mirna_name: str, lookup: dict):
    def normalize(name):
        name = str(name).strip().lower()
        name = re.sub(r'^[a-z]{3}-', '', name)
        name = re.sub(r'[-.](5p|3p|\*)$', '', name)
        return name

    norm = normalize(mirna_name)
    if norm in lookup:
        return lookup[norm]
    norm2 = re.sub(r'-[12]$', '', norm)
    if norm2 in lookup:
        return lookup[norm2]
    return None


# ── Load resources ────────────────────────────────────────────
try:
    bundle           = load_model()
    pipe             = bundle['model']            # encoder + lgbm pipeline
    calibrated_model = bundle['calibrated_model'] # calibrated lgbm for probabilities
    encoder          = bundle['encoder']          # for encoding input before prediction
    metrics          = bundle['metrics']
    options          = bundle['options']
    lookup           = load_targetscan()
except FileNotFoundError:
    st.error(
        "**Missing file:** `lgbm_mirna_model.pkl` not found. "
        "Run the training script first and place the pkl here."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🧬 miRNA Upregulation Predictor")
st.markdown(
    "Predicts whether a miRNA is **upregulated** or **downregulated** "
    "during *Leishmania* infection based on experimental conditions."
)
st.divider()


# ══════════════════════════════════════════════════════════════
# LAYOUT — two columns
# ══════════════════════════════════════════════════════════════
col_input, col_result = st.columns([1, 1], gap="large")


# ── LEFT: Inputs ──────────────────────────────────────────────
with col_input:
    st.subheader("Experimental conditions")

    mirna_input = st.text_input(
        "miRNA name",
        placeholder="e.g. hsa-miR-155-5p, mmu-let-7f",
        help="Enter the miRBase ID. The seed family is looked up automatically."
    )

    parasite = st.selectbox(
        "Parasite species",
        options=options['parasite'],
        help="Leishmania species used in the experiment"
    )

    organism = st.selectbox(
        "Host organism",
        options=options['organism']
    )

    cell_type = st.selectbox(
        "Cell type",
        options=options['cell_type']
    )

    time = st.selectbox(
        "Time point (hours post-infection)",
        options=options['time'],
        format_func=lambda x: f"{x}h"
    )

    predict_btn = st.button("Predict", type="primary", use_container_width=True)


# ── RIGHT: Result ─────────────────────────────────────────────
with col_result:
    st.subheader("Prediction")

    if predict_btn:
        if not mirna_input.strip():
            st.warning("Please enter a miRNA name.")
        else:
            # ── Step 1: look up seed family ───────────────────
            family       = lookup_family(mirna_input.strip(), lookup)
            is_conserved = 1 if family else 0

            if family:
                st.info(f"Seed family found: **{family}**")
            else:
                st.warning(
                    f"**{mirna_input}** is not in the broadly conserved "
                    "TargetScan families. Prediction will rely on the other "
                    "features (parasite, organism, cell type, time)."
                )

            # ── Step 2: build input row ───────────────────────
            # parasite and cell type passed exactly as selectbox returns them
            # — no lowercasing — so TargetEncoder sees the same values as training
            parasite_celltype = f"{parasite}_{cell_type}"

            input_df = pd.DataFrame([{
                'parasite':          parasite,
                'organism':          organism,
                'cell type':         cell_type,
                'seed_family':       family if family else np.nan,
                'parasite_celltype': parasite_celltype,
                'time':              time,
                'is_conserved':      is_conserved,
            }])

            # ── Step 3: encode then predict with calibrated model ──
            try:
                # Encode using the trained encoder
                input_enc = encoder.transform(input_df)

                # Predict using calibrated model — gives realistic probabilities
                proba     = calibrated_model.predict_proba(input_enc)[0]
                pred      = calibrated_model.predict(input_enc)[0]
                prob_up   = proba[1]
                prob_down = proba[0]

                if pred == 1:
                    st.success("## ⬆ Upregulated")
                else:
                    st.error("## ⬇ Downregulated")

                st.markdown(f"**Confidence:** {max(prob_up, prob_down)*100:.1f}%")

                st.markdown("**Probability breakdown:**")
                prob_col1, prob_col2 = st.columns(2)
                prob_col1.metric("Upregulated",   f"{prob_up   * 100:.1f}%")
                prob_col2.metric("Downregulated", f"{prob_down * 100:.1f}%")

                st.progress(
                    float(prob_up),
                    text=f"↑ {prob_up*100:.1f}%  |  ↓ {prob_down*100:.1f}%"
                )

                with st.expander("Input summary"):
                    st.dataframe(input_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"**Prediction error:** {e}")

    else:
        st.markdown(
            "<div style='color: gray; margin-top: 2rem;'>"
            "Fill in the conditions on the left and click <b>Predict</b>."
            "</div>",
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Model performance")
st.caption(
    f"LightGBM + isotonic calibration · "
    f"{metrics['n_train']} training samples · "
    f"5-fold cross-validation · "
    f"206 rows total (68 non-conserved miRNAs use NaN for seed_family)"
)

m1, m2, m3 = st.columns(3)
m1.metric("ROC-AUC",  f"{metrics['auc_mean']:.3f}", f"± {metrics['auc_std']:.3f}")
m2.metric("Accuracy", f"{metrics['acc_mean']:.3f}", f"± {metrics['acc_std']:.3f}")
m3.metric("F1 Score", f"{metrics['f1_mean']:.3f}",  f"± {metrics['f1_std']:.3f}")

st.markdown("**AUC per fold:**")
fold_cols = st.columns(len(metrics['auc_folds']))
for col, (i, auc_val) in zip(fold_cols, enumerate(metrics['auc_folds'])):
    col.metric(f"Fold {i+1}", f"{auc_val:.3f}")

st.markdown("**Permutation feature importance** (how much AUC drops when each feature is shuffled):")
fi = pd.DataFrame(metrics['feature_importance'])
fi['importance'] = fi['importance'].round(4)
fi['std']        = fi['std'].round(4)
fi.columns       = ['Feature', 'Importance (AUC drop)', 'Std']
st.dataframe(
    fi.style.background_gradient(subset=['Importance (AUC drop)'], cmap='Greens'),
    use_container_width=True,
    hide_index=True
)

with st.expander("Best hyperparameters"):
    st.json(metrics['best_params'])
