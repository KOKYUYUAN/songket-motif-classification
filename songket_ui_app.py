"""Songket Motif Classifier — Multi-Model Comparison UI.

Run:  streamlit run songket_ui_app.py
Requires: streamlit, torch, torchvision, Pillow, pandas
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from songket_inference import (
    CLASS_NAMES,
    MODEL_DISPLAY_NAMES,
    classify_all,
    classify_image,
    confidence_guidance,
    default_model_path,
    load_all_models,
    load_model,
)

st.set_page_config(
    page_title="Songket Motif Classifier",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_COLORS = {
    "alexnet":   "#f093fb",
    "vgg19":     "#43e97b",
    "resnet50":  "#38bdf8",
    "googlenet": "#fb923c",
}

MODEL_BG = {
    "alexnet":   "rgba(240,147,251,0.12)",
    "vgg19":     "rgba(67,233,123,0.10)",
    "resnet50":  "rgba(56,189,248,0.10)",
    "googlenet": "rgba(251,146,60,0.10)",
}

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #fff !important; }
.stButton > button {
    background: linear-gradient(135deg, #f093fb, #f5576c) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 0.65rem 1.5rem !important;
}
[data-testid="stFileUploader"] {
    background: rgba(240,147,251,0.05) !important;
    border: 2px dashed rgba(240,147,251,0.35) !important;
    border-radius: 14px !important; padding: 1rem !important;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important; padding: 0.75rem 1rem !important;
}
[data-testid="stMetricLabel"] p { color: rgba(255,255,255,0.5) !important; font-size: 12px !important; }
[data-testid="stMetricValue"]   { color: #fff !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stRadio"] label   { color: rgba(255,255,255,0.8) !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

/* progress bar colour override per model */
.stProgress > div > div { border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
            border-radius:16px;padding:20px 28px;margin-bottom:16px;
            display:flex;align-items:center;gap:16px;">
    <div style="width:52px;height:52px;background:linear-gradient(135deg,#f093fb,#f5576c);
                border-radius:14px;display:flex;align-items:center;justify-content:center;
                font-size:26px;flex-shrink:0;">🧵</div>
    <div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:#fff;letter-spacing:-0.5px;">
            Songket Motif Classifier</div>
        <div style="font-size:13px;color:rgba(255,255,255,0.45);margin-top:3px;">
            Final Year Project &nbsp;·&nbsp; Deep Learning Model Comparison &nbsp;·&nbsp;
            PyTorch &nbsp;·&nbsp; AlexNet · VGG-19 · ResNet-50 · GoogLeNet</div>
    </div>
    <div style="margin-left:auto;background:rgba(67,233,123,0.15);border:1px solid rgba(67,233,123,0.3);
                color:#a0f0c4;border-radius:20px;padding:5px 14px;font-size:12px;font-weight:600;">
        3 Classes · 4 Models</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.divider()
    st.markdown("**Run mode**")
    run_mode = st.radio(
        "run_mode",
        ["🔬 Compare all 4 models", "🎯 Single model"],
        label_visibility="collapsed",
    )
    single_model_id = None
    if "Single" in run_mode:
        single_model_id = st.selectbox(
            "Choose model",
            options=["alexnet", "vgg19", "resnet50", "googlenet"],
            format_func=lambda x: MODEL_DISPLAY_NAMES[x],
        )
    st.divider()
    st.markdown("**Checkpoint paths**")
    st.caption("Leave blank to use default (same folder as script)")
    paths_override: dict[str, str | None] = {}
    for mid, dname in MODEL_DISPLAY_NAMES.items():
        val = st.text_input(dname, value="", placeholder=default_model_path(mid), key=f"path_{mid}")
        paths_override[mid] = val.strip() or None
    st.divider()
    st.markdown("**Classes**")
    for cls in CLASS_NAMES:
        st.markdown(f"- {cls.replace('_', ' ').title()}")

# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models — please wait…")
def get_all_models(path_key: str):
    return load_all_models(model_paths={k: v for k, v in paths_override.items() if v})


@st.cache_resource(show_spinner="Loading model…")
def get_single_model(model_id: str, path_override: str | None):
    return load_model(model_id, model_path=path_override or None)


# ---------------------------------------------------------------------------
# Native Streamlit result card  (no unsafe HTML)
# ---------------------------------------------------------------------------

def _render_result_card(result, is_best: bool = False) -> None:
    """Render one model result using only native Streamlit widgets."""
    color = MODEL_COLORS[result.model_id]

    # Card border via markdown — just a thin coloured top bar, safe HTML
    st.markdown(
        f'<div style="height:3px;border-radius:3px 3px 0 0;'
        f'background:{color};margin-bottom:0;"></div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # Model name row
        badge = " 🏆 Best" if is_best else ""
        st.markdown(
            f"<span style='color:{color};font-family:Syne,sans-serif;"
            f"font-size:13px;font-weight:700;'>{result.display_name}{badge}</span>",
            unsafe_allow_html=True,
        )

        if result.error:
            st.error(f"⚠ {result.error}")
            return

        # Prediction + confidence
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.markdown(f"**{result.predicted_label_pretty}**")
            st.caption(result.predicted_label)
        with col_b:
            st.metric(
                label="Confidence",
                value=f"{result.confidence * 100:.1f}%",
            )

        # Top-3 probability bars using st.progress
        st.caption("Top-3 probabilities")
        for item in result.top_k:
            pct = item.probability
            bar_col, pct_col = st.columns([5, 1])
            with bar_col:
                st.progress(pct, text=item.label.replace("_", " ").title())
            with pct_col:
                st.markdown(
                    f"<div style='padding-top:6px;font-size:12px;"
                    f"color:rgba(255,255,255,0.6);text-align:right;'>"
                    f"{pct*100:.1f}%</div>",
                    unsafe_allow_html=True,
                )

        st.caption(f"⏱ Inference: {result.inference_ms:.1f} ms")


# ---------------------------------------------------------------------------
# Upload section  (full width, then results below)
# ---------------------------------------------------------------------------

up_col, btn_col = st.columns([3, 1])
with up_col:
    uploaded = st.file_uploader(
        "Upload a Songket image (JPG · PNG · WebP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
with btn_col:
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run Classification", use_container_width=True)

if uploaded:
    img_col, _ = st.columns([1, 3])
    with img_col:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=f"{image.size[0]}×{image.size[1]} px", use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

st.markdown("### 📊 Results")

if not uploaded:
    st.info("Upload an image above and press **Run Classification** to begin.")

elif run_btn:
    image = Image.open(uploaded).convert("RGB")

    # ── Compare mode ──────────────────────────────────────────────────────
    if "Compare" in run_mode:
        path_key = str(sorted(paths_override.items()))
        try:
            with st.spinner("Loading all models…"):
                all_models = get_all_models(path_key)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Running inference on all 4 models…"):
            results = classify_all(image, all_models, top_k=3)

        valid   = [r for r in results if not r.error]
        best_id = max(valid, key=lambda r: r.confidence).model_id if valid else None

        # ── 4 cards in a 2×2 native grid ──────────────────────────────────
        col1, col2 = st.columns(2, gap="medium")
        pairs = list(zip(
            [r for i, r in enumerate(results) if i % 2 == 0],
            [r for i, r in enumerate(results) if i % 2 == 1],
        ))
        for left_res, right_res in pairs:
            with col1:
                _render_result_card(left_res,  is_best=(left_res.model_id  == best_id))
            with col2:
                _render_result_card(right_res, is_best=(right_res.model_id == best_id))

        st.divider()

        # ── Charts ────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2, gap="medium")
        with ch1:
            st.markdown("#### ⏱ Inference Time (ms)")
            st.bar_chart(
                {MODEL_DISPLAY_NAMES[r.model_id]: round(r.inference_ms, 2)
                 for r in results if not r.error},
                color="#f093fb",
            )
        with ch2:
            st.markdown("#### 🎯 Confidence (%)")
            st.bar_chart(
                {MODEL_DISPLAY_NAMES[r.model_id]: round(r.confidence * 100, 2)
                 for r in results if not r.error},
                color="#43e97b",
            )

        # ── Summary table ─────────────────────────────────────────────────
        st.markdown("#### 📋 Model Comparison Table")
        rows = []
        for r in results:
            rows.append({
                "Model":        r.display_name,
                "Prediction":   r.predicted_label_pretty if not r.error else "Error",
                "Confidence %": f"{r.confidence*100:.2f}" if not r.error else "—",
                "Inference ms": f"{r.inference_ms:.2f}"  if not r.error else "—",
                "Guidance":     confidence_guidance(r.confidence) if not r.error else r.error,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Single model mode ──────────────────────────────────────────────────
    else:
        mid = single_model_id
        try:
            with st.spinner(f"Loading {MODEL_DISPLAY_NAMES[mid]}…"):
                model, device = get_single_model(mid, paths_override.get(mid))
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Running inference…"):
            result = classify_image(image, model, device, model_id=mid, top_k=3)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Motif", result.predicted_label_pretty)
        m2.metric("Confidence",       f"{result.confidence*100:.2f}%")
        m3.metric("Inference Time",   f"{result.inference_ms:.2f} ms")

        _render_result_card(result, is_best=True)

        st.markdown("#### Top Class Probabilities")
        st.bar_chart(
            {item.label.replace("_", " ").title(): round(item.probability * 100, 2)
             for item in result.top_k},
            color=MODEL_COLORS[mid],
        )
        st.info(f"💡 {confidence_guidance(result.confidence)}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption("Songket Motif Classifier · FYP Deep Learning Project · AlexNet · VGG-19 · ResNet-50 · GoogLeNet · PyTorch + Streamlit")