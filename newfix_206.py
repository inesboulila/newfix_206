"""
miRNA Upregulation Predictor
=============================
Model : lgbm_mirna_model.pkl  (LightGBM + TargetEncoder pipeline)
Run   : streamlit run app.py

Features expected by the model (in order):
  CAT: parasite, organism, cell type, parasite_celltype
  NUM: time, is_conserved, seed_family_pct_up, seed_family_entropy, seed_family_n

Note: seed_family is looked up from TargetScan to compute the 3 variance
features, but is NOT passed directly to the model as a categorical.
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
    with open('lgbm_mirna_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_targetscan():
    """Load TargetScan miR_Family_Info.txt and build miRNA → family lookup."""
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
    """Try to find the seed family for a given miRNA name."""
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

def get_family_variance_features(family, family_stats):
    """
    Look up the 3 variance features for a seed family from the bundle.
    If family is unknown: pct_up=0.5 (no info), entropy=1.0 (max uncertainty), n=0.
    """
    if family and family in family_stats:
        stats = family_stats[family]
        return (
            stats['seed_family_pct_up'],
            stats['seed_family_entropy'],
            stats['seed_family_n'],
        )
    return (0.5, 1.0, 0)


# ── Load resources ────────────────────────────────────────────
try:
    bundle        = load_model()
    model         = bundle['model']
    metrics       = bundle['metrics']
    options       = bundle['options']
    family_stats  = bundle.get('family_stats', {})
    feature_names = bundle.get('feature_names', [])
    lookup        = load_targetscan()
except FileNotFoundError:
    st.error(
        "**Missing file:** `lgbm_mirna_model.pkl` not found. "
        "Run the training script first and place the pkl here."
    )
    st.stop()

# Detect model version from saved feature names
IS_V5 = 'seed_family_pct_up' in feature_names
IS_V4 = 'seed_family' in (bundle.get('cat_cols') or [])


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("🧬 miRNA Upregulation Predictor")
st.markdown(
    "Predicts whether a miRNA is **upregulated** or **downregulated** "
    "during *Leishmania* infection based on experimental conditions."
)
if IS_V5:
    st.caption(
        "Model v5 · seed family signal encoded via variance features "
        "(pct_up, entropy, n) · no redundancy with TargetEncoder"
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
        help="Enter the miRBase ID. The seed family is looked up automatically from TargetScan."
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
            # ── Step 1: look up seed family from TargetScan ───
            family       = lookup_family(mirna_input.strip(), lookup)
            is_conserved = 1 if family else 0

            if family:
                st.info(f"Seed family found: **{family}**")

                # Show variance stats so user understands prediction confidence
                if IS_V5 and family in family_stats:
                    pct_up  = family_stats[family]['seed_family_pct_up']
                    entropy = family_stats[family]['seed_family_entropy']
                    n       = int(family_stats[family]['seed_family_n'])
                    mix_label = (
                        "⚠️ highly mixed — direction varies across conditions"
                        if entropy > 0.8 else
                        "✅ consistent — usually same direction"
                    )
                    st.caption(
                        f"Family behaviour in training data: "
                        f"**{pct_up*100:.0f}% upregulated** across {n} observations · "
                        f"entropy {entropy:.2f} · {mix_label}"
                    )
            else:
                st.warning(
                    f"**{mirna_input}** seed family not found in TargetScan broadly "
                    "conserved families. Prediction relies on parasite, organism, "
                    "cell type, and time only."
                )

            # ── Step 2: build input row ───────────────────────
            para_clean        = parasite.lower().replace(' ', '')
            cell_clean        = cell_type.lower().strip()
            parasite_celltype = f"{para_clean}_{cell_clean}"

            # Get variance features (or neutral defaults if family unknown)
            pct_up, entropy, n = get_family_variance_features(family, family_stats)

            # v5 row — seed_family NOT included, only variance features represent family
            row = {
                'parasite':            para_clean,
                'organism':            organism,
                'cell type':           cell_clean,
                'parasite_celltype':   parasite_celltype,
                'time':                time,
                'is_conserved':        is_conserved,
                'seed_family_pct_up':  pct_up,
                'seed_family_entropy': entropy,
                'seed_family_n':       n,
            }

            # v4 fallback: old model expects seed_family as a raw categorical
            if IS_V4:
                row['seed_family'] = family if family else np.nan

            input_df = pd.DataFrame([row])

            # Reorder columns to match training order exactly
            input_df = input_df.reindex(columns=feature_names)

            # ── Step 3: predict ───────────────────────────────
            try:
                proba     = model.predict_proba(input_df)[0]
                pred      = model.predict(input_df)[0]
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
                    display_df = input_df.copy()
                    display_df.insert(0, 'miRNA', mirna_input.strip())
                    display_df.insert(1, 'seed_family_looked_up', family if family else 'not found')
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

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
    f"LightGBM · {metrics['n_train']} training samples · "
    f"5-fold stratified cross-validation"
)

m1, m2, m3 = st.columns(3)
m1.metric("ROC-AUC",  f"{metrics['auc_mean']:.3f}", f"± {metrics['auc_std']:.3f}")
m2.metric("Accuracy", f"{metrics['acc_mean']:.3f}", f"± {metrics['acc_std']:.3f}")
m3.metric("F1 Score", f"{metrics['f1_mean']:.3f}",  f"± {metrics['f1_std']:.3f}")

st.markdown("**AUC per fold:**")
fold_cols = st.columns(len(metrics['auc_folds']))
for col, (i, auc_val) in zip(fold_cols, enumerate(metrics['auc_folds'])):
    col.metric(f"Fold {i+1}", f"{auc_val:.3f}")

st.markdown("**Permutation feature importance** (avg AUC drop on held-out folds):")
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