"""
miRNA Upregulation Predictor
=============================
Model : lgbm_mirna_model.pkl  (LightGBM + TargetEncoder pipeline)
Run   : streamlit run app.py

Features expected by the model (in order):
  CAT: parasite, organism, cell type, seed_family, parasite_celltype
  NUM: time, is_conserved

At prediction time:
  1. If miRNA name is found in training data → show its exact observed behaviour
  2. If miRNA is unseen → fall back to seed family stats from TargetScan
  3. Model prediction runs normally either way — the lookup is informational context
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

def lookup_family_from_targetscan(mirna_name: str, ts_lookup: dict):
    """Normalize miRNA name and look up its seed family from TargetScan."""
    def normalize(name):
        name = str(name).strip().lower()
        name = re.sub(r'^[a-z]{3}-', '', name)
        name = re.sub(r'[-.](5p|3p|\*)$', '', name)
        return name

    norm = normalize(mirna_name)
    if norm in ts_lookup:
        return ts_lookup[norm]
    norm2 = re.sub(r'-[12]$', '', norm)
    if norm2 in ts_lookup:
        return ts_lookup[norm2]
    return None


# ── Load resources ────────────────────────────────────────────
try:
    bundle        = load_model()
    model         = bundle['model']
    metrics       = bundle['metrics']
    options       = bundle['options']
    mirna_lookup  = bundle.get('mirna_lookup', {})    # miRNA  → stats from training data
    family_lookup = bundle.get('family_lookup', {})   # family → stats from training data
    ts_lookup     = load_targetscan()
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
        placeholder="e.g. hsa-miR-146b, mmu-let-7f",
        help="Enter the miRBase ID. Known miRNAs are checked against training data first."
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
            mirna_clean = mirna_input.strip()

            # ── Step 1: check miRNA against training data ─────
            mirna_stats  = mirna_lookup.get(mirna_clean)
            family       = None
            family_stats = None
            is_conserved = 0

            if mirna_stats:
                # miRNA is known — use its exact observed behaviour
                family       = mirna_stats.get('seed_family')
                is_conserved = 1 if pd.notna(family) else 0

                n      = mirna_stats['n']
                pct_up = mirna_stats['pct_up']

                if mirna_stats['always_up']:
                    st.success(
                        f"✅ **{mirna_clean}** is in the training data — "
                        f"always **upregulated** across all {n} observation(s)."
                    )
                elif mirna_stats['always_down']:
                    st.error(
                        f"✅ **{mirna_clean}** is in the training data — "
                        f"always **downregulated** across all {n} observation(s)."
                    )
                else:
                    st.warning(
                        f"⚠️ **{mirna_clean}** is in the training data but shows "
                        f"**mixed behaviour**: upregulated {pct_up*100:.0f}% of the time "
                        f"across {n} observations. The model prediction depends on the "
                        f"specific conditions you selected."
                    )

            else:
                # miRNA is unseen — fall back to seed family
                st.info(
                    f"**{mirna_clean}** is not in the training data. "
                    f"Looking up its seed family for context..."
                )
                family = lookup_family_from_targetscan(mirna_clean, ts_lookup)
                is_conserved = 1 if family else 0

                if family:
                    family_stats = family_lookup.get(family)
                    if family_stats:
                        n      = family_stats['n']
                        pct_up = family_stats['pct_up']
                        if family_stats['always_up']:
                            st.info(
                                f"Seed family **{family}** found — "
                                f"always upregulated across {n} observations in training data."
                            )
                        elif family_stats['always_down']:
                            st.info(
                                f"Seed family **{family}** found — "
                                f"always downregulated across {n} observations in training data."
                            )
                        else:
                            st.warning(
                                f"Seed family **{family}** found but is **mixed**: "
                                f"upregulated {pct_up*100:.0f}% of the time across {n} observations. "
                                f"Prediction uncertainty is higher."
                            )
                    else:
                        st.info(f"Seed family **{family}** found via TargetScan.")
                else:
                    st.warning(
                        f"**{mirna_clean}** not found in training data or TargetScan. "
                        "Prediction relies on parasite, organism, cell type, and time only."
                    )

            # ── Step 2: build input row (unchanged from original) ──
            para_clean        = parasite.lower().replace(' ', '')
            cell_clean        = cell_type.lower().strip()
            parasite_celltype = f"{para_clean}_{cell_clean}"

            input_df = pd.DataFrame([{
                'parasite':          para_clean,
                'organism':          organism,
                'cell type':         cell_clean,
                'seed_family':       family if family else np.nan,
                'parasite_celltype': parasite_celltype,
                'time':              time,
                'is_conserved':      is_conserved,
            }])

            # ── Step 3: predict ───────────────────────────────
            try:
                proba     = model.predict_proba(input_df)[0]
                pred      = model.predict(input_df)[0]
                prob_up   = proba[1]
                prob_down = proba[0]

                st.divider()

                if pred == 1:
                    st.success("## ⬆ Upregulated")
                else:
                    st.error("## ⬇ Downregulated")

                # If miRNA behaviour in training data contradicts the model prediction, warn
                if mirna_stats:
                    if mirna_stats['always_up'] and pred == 0:
                        st.warning(
                            "⚠️ Note: this miRNA is always upregulated in training data, "
                            "but the model predicts downregulation for these specific conditions."
                        )
                    elif mirna_stats['always_down'] and pred == 1:
                        st.warning(
                            "⚠️ Note: this miRNA is always downregulated in training data, "
                            "but the model predicts upregulation for these specific conditions."
                        )

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
                    display_df.insert(0, 'miRNA', mirna_clean)
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
    f"LightGBM trained on {metrics['n_train']} samples · "
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
