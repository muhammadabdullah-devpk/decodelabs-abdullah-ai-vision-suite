import os
import time
import asyncio
from PIL import Image
import numpy as np
import cv2
import streamlit as st
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas

import utils

st.set_page_config(
    page_title="AI Vision Lab — Muhammad Abdullah",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🧠"
)

INJECTED_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #09090b !important; /* Premium high-contrast Zinc 950 black */
    color: #f4f4f5 !important; /* Zinc 100 - High contrast bright silver text */
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
}

/* === HIDE SIDEBAR COMPLETELY === */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

.block-container { padding: 2rem 2.5rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
h1,h2,h3 { color: #ffffff !important; font-weight: 800 !important; letter-spacing:-0.5px !important; }

/* === TABS CUSTOM STYLING === */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 2px solid #27272a;
    margin-bottom: 28px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #18181b;
    border: 1px solid #3f3f46;
    border-bottom: none;
    border-radius: 8px 8px 0px 0px;
    padding: 12px 28px;
    color: #d4d4d8;
    font-weight: 700;
    font-size: 14px;
    transition: all 0.2s ease-in-out;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff;
    background-color: #27272a;
    border-color: #52525b;
}
.stTabs [aria-selected="true"] {
    background-color: #0284c7 !important; /* Vibrant high-contrast blue */
    color: #ffffff !important;
    border-color: #0284c7 !important;
    box-shadow: 0 -4px 12px rgba(2, 132, 199, 0.2);
}

/* === CARDS === */
.card {
    background: #18181b; /* Zinc 900 */
    border: 1px solid #3f3f46; /* Zinc 700 - Light enough to see clearly */
    border-radius: 14px;
    padding: 24px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.card:hover { 
    border-color: #52525b; /* Zinc 600 */
    box-shadow: 0 6px 30px rgba(0,0,0,0.6); 
}
.card.blue  { border-top: 4px solid #38bdf8; } /* Cyan 400 */
.card.green { border-top: 4px solid #34d399; } /* Emerald 400 */
.card.purple{ border-top: 4px solid #c084fc; } /* Purple 400 */
.card.orange{ border-top: 4px solid #fbbf24; } /* Amber 400 */
.card.red   { border-top: 4px solid #f87171; } /* Red 400 */

/* === VISUAL VIEWPORT FRAME === */
.viewport-frame {
    background-color: #0d1117;
    border: 2px solid #3f3f46;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    margin-bottom: 16px;
}
.viewport-title {
    font-size: 11px;
    font-weight: 800;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    border-bottom: 1px solid #27272a;
    padding-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* === API / TERMINAL CONSOLE === */
.terminal-console {
    background-color: #050508 !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #34d399 !important; /* Glowing mint text */
    padding: 18px !important;
    font-size: 13.5px !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.9) !important;
    line-height: 1.6 !important;
    overflow-x: auto;
}

/* === PANEL HEADER === */
.panel-header {
    font-size: 15px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid #27272a;
    padding-bottom: 8px;
}

/* === PROFILE HERO / FOOTER CARD === */
.footer-hero {
    background: linear-gradient(135deg, #18181b 0%, #09090b 100%);
    border: 1px solid #3f3f46; border-radius: 16px;
    padding: 36px; margin-top: 40px; position: relative; overflow: hidden;
    box-shadow: 0 8px 30px rgba(0,0,0,0.6);
}
.footer-hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
    border-radius:50%;
}
.avatar-small {
    width: 60px; height: 60px; border-radius: 12px;
    background: linear-gradient(135deg, #38bdf8, #c084fc);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 900; color: #ffffff;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4);
    margin-bottom: 16px;
}
.hero-name { font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
.hero-role { font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin: 4px 0 16px; }
.hero-meta { font-size: 14px; color: #e4e4e7; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }

/* === VIBRANT BADGES === */
.badge { 
    display: inline-block; padding: 4px 12px; border-radius: 20px; 
    font-size: 11.5px; font-weight: 700; margin: 3px; border: 1px solid; 
}
.b-blue   { background: rgba(56, 189, 248, 0.12); border-color: rgba(56, 189, 248, 0.4); color: #38bdf8; }
.b-purple { background: rgba(192, 132, 252, 0.12); border-color: rgba(192, 132, 252, 0.4); color: #c084fc; }
.b-green  { background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.4); color: #34d399; }
.b-orange { background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.4); color: #fbbf24; }
.b-red    { background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.4); color: #f87171; }

/* === SOCIAL BUTTONS === */
.soc {
    display: inline-flex; align-items: center; gap: 8px;
    background: #27272a; border: 1px solid #4b5563; border-radius: 10px;
    padding: 10px 20px; color: #ffffff !important; text-decoration: none !important;
    font-size: 14px; font-weight: 700; margin-right: 12px; margin-bottom: 8px;
    transition: all 0.2s ease-in-out;
}
.soc:hover { 
    background: #0284c7; border-color: #0284c7; 
    color: #ffffff !important; box-shadow: 0 4px 16px rgba(2,132,199,0.4);
    transform: translateY(-1px);
}

/* === PIPELINE === */
.pipeline {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 4px; margin-top: 18px; border-radius: 12px; overflow: hidden;
}
.pstep {
    background: #18181b; border: 1px solid #3f3f46;
    padding: 22px 16px; position: relative;
    transition: all 0.2s; cursor: default;
}
.pstep:hover { background: #27272a; border-color: #38bdf8; z-index: 1; }
.pnum  { font-family:'JetBrains Mono',monospace; font-size: 11px; font-weight: 800; color: #38bdf8; letter-spacing: 1.5px; margin-bottom: 8px; }
.picon { font-size: 24px; margin-bottom: 8px; display: block; }
.ptitle{ font-size: 14px; font-weight: 800; color: #ffffff; margin-bottom: 6px; }
.pdesc { font-size: 12px; color: #d4d4d8; line-height: 1.6; } /* High contrast body descriptions */

/* === STATS ROW === */
.stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 12px; margin-bottom: 18px;
}
.stat {
    background: #18181b; border: 1px solid #3f3f46;
    border-radius: 12px; padding: 16px; text-align: center;
}
.stat-v { font-size: 28px; font-weight: 800; color: #0284c7; font-family:'JetBrains Mono',monospace; }
.stat-l { font-size: 11px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-top: 2px; }

/* === KV DATA TABLE === */
.kv { width: 100%; border-collapse: collapse; }
.kv tr { border-bottom: 1px solid #27272a; }
.kv tr:last-child { border-bottom: none; }
.kv td.k { padding: 12px 0; color: #a1a1aa; font-size: 13.5px; font-weight: 600; width: 42%; }
.kv td.v { padding: 12px 0; color: #ffffff; font-size: 13.5px; font-weight: 700; font-family:'JetBrains Mono',monospace; }

/* === SECTION LABEL === */
.lbl {
    font-size: 12px; font-weight: 800; color: #a1a1aa;
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #27272a;
}

/* === TOP BAR === */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0; margin-bottom: 24px; border-bottom: 2px solid #27272a;
}
.tb-badge {
    background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4);
    color: #38bdf8; border-radius: 6px; padding: 4px 12px;
    font-size: 13px; font-weight: 700;
}
.status {
    background: rgba(52, 211, 153, 0.15); border: 1px solid rgba(52, 211, 153, 0.4);
    color: #34d399; border-radius: 6px; padding: 5px 12px;
    font-size: 13px; font-weight: 700;
    display: flex; align-items: center; gap: 6px;
}
.dot { width: 8px; height: 8px; background: #34d399; border-radius: 50%; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity: 1;} 50%{opacity: 0.4;} }

/* === WIDGET OVERRIDES FOR HIGH CONTRAST === */
[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
}
[data-testid="stFileUploader"] {
    background-color: #18181b !important;
    border: 2px dashed #0284c7 !important;
    border-radius: 12px !important;
    padding: 24px !important;
    text-align: center !important;
    transition: all 0.2s ease-in-out !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
    background-color: #27272a !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.2);
}
[data-testid="stFileUploader"] section {
    background-color: transparent !important;
}
[data-testid="stFileUploader"] label {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 15px !important;
}
[data-testid="stFileUploader"] small {
    color: #a1a1aa !important;
    font-weight: 700 !important;
    font-size: 12.5px !important;
}
[data-testid="stFileUploader"] button {
    background-color: #27272a !important;
    border: 1px solid #3f3f46 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    padding: 6px 14px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #0284c7 !important;
    border-color: #0284c7 !important;
    color: #ffffff !important;
}
.stSelectbox > div > div { background:#18181b !important; border: 1px solid #52525b !important; border-radius: 8px !important; color:#ffffff !important; font-weight: 600 !important; }
.stTextInput > div > div > input { background:#18181b !important; border: 1px solid #52525b !important; border-radius: 8px !important; color:#ffffff !important; }
.stButton > button { background:#0284c7 !important; border: none !important; border-radius: 8px !important; color:#ffffff !important; font-weight: 700 !important; padding: 12px 24px !important; transition: all 0.2s !important; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important; }
.stButton > button:hover { background:#0369a1 !important; box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important; transform: translateY(-1px) !important; }
.stProgress > div > div > div { background:#0284c7 !important; border-radius: 4px !important; }
.stSuccess { background: rgba(52, 211, 153, 0.1) !important; border: 1px solid rgba(52, 211, 153, 0.3) !important; border-radius: 8px !important; color: #34d399 !important; }
.stInfo    { background: rgba(56, 189, 248, 0.1) !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; border-radius: 8px !important; color: #38bdf8 !important; }
.stWarning { background: rgba(251, 191, 36, 0.1) !important; border: 1px solid rgba(251, 191, 36, 0.3) !important; border-radius: 8px !important; color: #fbbf24 !important; }
.stError   { background: rgba(248, 113, 113, 0.1)  !important; border: 1px solid rgba(248, 113, 113, 0.3)  !important; border-radius: 8px !important; color: #f87171 !important; }
hr { border-color: #27272a !important; margin: 24px 0 !important; }

/* === COMPARISON GRID === */
.cmp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.cmp-label { font-size: 12px; color: #a1a1aa; text-align: center; margin-top: 6px; font-weight: 600; }

/* === DETECTION TABLE === */
.det-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.det-table th { text-align: left; padding: 10px 14px; background: #27272a; color: #ffffff; font-weight: 700; font-size: 11.5px; text-transform: uppercase; letter-spacing: 1px; border-bottom: 2px solid #3f3f46; }
.det-table td { padding: 12px 14px; border-bottom: 1px solid #27272a; color: #ffffff; font-family:'JetBrains Mono',monospace; font-size: 12.5px; }
.det-table tr:last-child td { border-bottom: none; }
.conf-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #38bdf8, #0284c7); }
</style>
"""
st.markdown(INJECTED_CSS, unsafe_allow_html=True)

# ─── TTS Widget ────────────────────────────────────────────────
def tts_widget(text):
    safe = text.replace('\\','\\\\').replace('"','\\"').replace('\n',' ')
    components.html(f"""
    <script>
    function spk(){{var u=new SpeechSynthesisUtterance("{safe}");speechSynthesis.cancel();speechSynthesis.speak(u);}}
    </script>
    <button onclick="spk()" style="background:#0284c7;color:#fff;border:none;padding:10px 20px;
    border-radius:8px;font-weight:700;font-size:13.5px;cursor:pointer;
    font-family:Inter,sans-serif;box-shadow:0 4px 12px rgba(2,132,199,.25);">
    🔊 &nbsp;Play Text-to-Speech</button>""", height=50)

# ─── Session Init ──────────────────────────────────────────────
if 'models_ready' not in st.session_state:
    st.session_state.models_ready = (
        os.path.exists("models/MobileNetSSD_deploy.caffemodel") and
        os.path.exists("models/yolov5n.onnx")
    )
if 'tesseract_path' not in st.session_state:
    st.session_state.tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if 'ocr_engine' not in st.session_state:
    st.session_state.ocr_engine = "Windows Native OCR (WinRT)"

# ─── TOP BAR ──────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div style="display:flex;align-items:center;gap:10px;">
        <span style="font-size:24px;font-weight:900;color:#ffffff;letter-spacing:-0.5px;">🧠 &nbsp;AI Vision Suite</span>
        <span class="tb-badge">Muhammad Abdullah</span>
    </div>
    <div class="status"><span class="dot"></span> System Online</div>
</div>""", unsafe_allow_html=True)

# ─── TABS NAVIGATION AT THE TOP ───────────────────────────────
tab_ov, tab_ocr, tab_cv, tab_det, tab_batch = st.tabs([
    "📋  Project Overview",
    "🔍  Document OCR Lab",
    "🧪  CV Filters Lab",
    "🎯  Object Detection",
    "📦  Batch Processor"
])

# ═══════════════════════════════════════════════════════════════
#  TAB 1 · PROJECT OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab_ov:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0284c7 0%, #18181b 100%); border: 1px solid #3f3f46; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <h3 style="margin-top:0; font-size:22px; font-weight:800; color:#ffffff;">WELCOME TO THE TEAM!</h3>
        <p style="font-size:14.5px; color:#ffffff; line-height:1.7; margin-top:8px; font-weight: 500;">
            Step into the role of an <strong>Artificial Intelligence Engineer</strong> at DecodeLabs. 
            Project 4 is your <strong>Optional Mastery Phase: Image or Text Recognition (Basic)</strong>. 
            By building this workflow, you demonstrate the ability to integrate pre-trained AI models, pre-process raw visual data, and construct functional end-to-end vision pipelines.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_ov_1, col_ov_2 = st.columns(2, gap="large")
    
    with col_ov_1:
        st.markdown('<div class="lbl">Path 1: Optical Character Recognition (OCR)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card blue" style="padding: 20px;">
            <p style="font-size: 13.5px; color: #ffffff; line-height: 1.6; font-weight: 600;">
                <strong>Objective:</strong> Extract machine-readable text strings from physical documents, scanned invoices, business cards, and IDs.
            </p>
            <ul style="font-size:13.5px; color:#e4e4e7; margin-left: 20px; line-height: 1.8;">
                <li><strong>Primary Engine:</strong> pytesseract (Google's OCR wrapper) / WinRT UWP APIs.</li>
                <li><strong>PSM Selection:</strong> Tuning Page Segmentation Modes (--psm 3, 6, 7, 11) for structured or sparse text layouts.</li>
                <li><strong>Systematic Image Pre-Processing:</strong> Grayscale conversion, Gaussian blurring, Otsu's adaptive binarization, deskewing, and CLAHE contrast enhancement.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_ov_2:
        st.markdown('<div class="lbl">Path 2: Deep Learning Object Detection</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card green" style="padding: 20px;">
            <p style="font-size: 13.5px; color: #ffffff; line-height: 1.6; font-weight: 600;">
                <strong>Objective:</strong> Identify, locate, and crop physical entities in raw visual matrices with normalized coordinates.
            </p>
            <ul style="font-size:13.5px; color:#e4e4e7; margin-left: 20px; line-height: 1.8;">
                <li><strong>Primary Models:</strong> YOLOv5-Nano (ONNX format) and MobileNet-SSD (Caffe framework).</li>
                <li><strong>Inference Input:</strong> Image scaling to 300x300 or 640x640, mean subtraction, and feed-forward blob construction.</li>
                <li><strong>Confidence Threshold Filtering:</strong> Drop false positives with a minimum validation standard (default: 80%).</li>
                <li><strong>Crop-to-OCR Integration:</strong> Crop bounding boxes of interest directly back into the character extraction pipeline.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('<div class="lbl">DecodeLabs Training Kit validation standards</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stats" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 20px;">
        <div class="card" style="padding: 16px; border-top: 4px solid #38bdf8;">
            <div style="font-size:14.5px; font-weight:800; color:#ffffff; margin-bottom:6px;">1. Library Integration</div>
            <div style="font-size:12.5px; color:#d4d4d8; line-height:1.6;">Seamless, error-free implementation of pytesseract, cv2.dnn, and async WinRT OCR.</div>
        </div>
        <div class="card" style="padding: 16px; border-top: 4px solid #34d399;">
            <div style="font-size:14.5px; font-weight:800; color:#ffffff; margin-bottom:6px;">2. Pre-Processing</div>
            <div style="font-size:12.5px; color:#d4d4d8; line-height:1.6;">Execution of grayscale, blur, deskew, and thresholding steps to isolate character contours.</div>
        </div>
        <div class="card" style="padding: 16px; border-top: 4px solid #fbbf24;">
            <div style="font-size:14.5px; font-weight:800; color:#ffffff; margin-bottom:6px;">3. Accuracy Gate</div>
            <div style="font-size:12.5px; color:#d4d4d8; line-height:1.6;">A validated confidence score cutoff filter to optimize predictions and reduce hallucinations.</div>
        </div>
        <div class="card" style="padding: 16px; border-top: 4px solid #f87171;">
            <div style="font-size:14.5px; font-weight:800; color:#ffffff; margin-bottom:6px;">4. Visual Confirmation</div>
            <div style="font-size:12.5px; color:#d4d4d8; line-height:1.6;">Pristine overlays highlighting keywords, bounding box coordinates, and entity tables.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PIPELINE (Enhanced) ──────────────────────────────────
    st.markdown('<div class="lbl">System Pipeline Architecture — End-to-End Data Flow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline" style="margin-bottom: 24px;">
        <div class="pstep">
            <div class="pnum">01 · INGEST</div>
            <span class="picon">📥</span>
            <div class="ptitle">Image Ingestion</div>
            <div class="pdesc">Raw RGB/BGR tensors (H×W×C) loaded from upload streams. Supports PNG, JPG, JPEG, RGBA.</div>
        </div>
        <div class="pstep">
            <div class="pnum">02 · PREPROCESS</div>
            <span class="picon">🔧</span>
            <div class="ptitle">CV Pre-processing</div>
            <div class="pdesc">Grayscale conversion → Gaussian blur denoising → Auto deskew warp correction → CLAHE contrast enhancement → Sharpening kernel.</div>
        </div>
        <div class="pstep">
            <div class="pnum">03 · BINARIZE</div>
            <span class="picon">⚡</span>
            <div class="ptitle">Adaptive Thresholding</div>
            <div class="pdesc">Otsu's algorithm auto-selects optimal threshold to separate foreground text from background noise.</div>
        </div>
        <div class="pstep">
            <div class="pnum">04 · INFER</div>
            <span class="picon">🧠</span>
            <div class="ptitle">Neural Inference</div>
            <div class="pdesc">YOLOv5n ONNX (NMS, 80 classes) or MobileNet-SSD Caffe. WinRT UWP / Tesseract PSM OCR for text regions.</div>
        </div>
        <div class="pstep">
            <div class="pnum">05 · EXTRACT</div>
            <span class="picon">📊</span>
            <div class="ptitle">NLP Parsing & Export</div>
            <div class="pdesc">Regex entity extraction: dates, amounts, emails, phones, IDs. ZIP export with CSV metadata.</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="lbl">⚡ Active Implemented System Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
        <div class="card" style="padding: 16px; border-top: 4px solid #38bdf8;">
            <div style="color: #38bdf8; font-weight: 800; font-size: 14.5px; margin-bottom: 8px;">🔍 Document OCR & NLP Features</div>
            <ul style="font-size: 12.5px; color: #d4d4d8; margin-left: 16px; line-height: 1.7;">
                <li><strong>Dual OCR Processing:</strong> Windows Native WinRT OCR & Google Tesseract engines.</li>
                <li><strong>Interactive Bounding Box (ROI):</strong> Draw custom canvas zones for targeted text extraction.</li>
                <li><strong>Google Lens Keyword Highlight:</strong> Real-time word matching and overlay drawing on image matrices.</li>
                <li><strong>Smart Entity Parsers:</strong> Automated regex parsing for Receipts/Invoices, Business Cards, and ID Cards.</li>
                <li><strong>Text-to-Speech (TTS):</strong> Instant audio readback of extracted document contents.</li>
            </ul>
        </div>
        <div class="card" style="padding: 16px; border-top: 4px solid #34d399;">
            <div style="color: #34d399; font-weight: 800; font-size: 14.5px; margin-bottom: 8px;">🧪 Computer Vision & Pre-processing Features</div>
            <ul style="font-size: 12.5px; color: #d4d4d8; margin-left: 16px; line-height: 1.7;">
                <li><strong>Image Denoising & Sharpening:</strong> Adjustable Gaussian Blur and Unsharp Mask sharpening.</li>
                <li><strong>Adaptive Binarization:</strong> Otsu's thresholding with manual slider controls.</li>
                <li><strong>Dynamic Deskewing:</strong> Automated text angle calculation and rotation snap.</li>
                <li><strong>Contrast Balancing:</strong> CLAHE (Contrast Limited Adaptive Histogram Equalization).</li>
                <li><strong>Dual Real-time Charts:</strong> Grayscale pixel histogram and RGB color channel distribution plots.</li>
            </ul>
        </div>
        <div class="card" style="padding: 16px; border-top: 4px solid #fbbf24;">
            <div style="color: #fbbf24; font-weight: 800; font-size: 14.5px; margin-bottom: 8px;">🎯 Deep Learning & Object Detection Features</div>
            <ul style="font-size: 12.5px; color: #d4d4d8; margin-left: 16px; line-height: 1.7;">
                <li><strong>Multi-Model Deployment:</strong> High-speed YOLOv5-Nano (ONNX) vs MobileNet-SSD (Caffe framework).</li>
                <li><strong>Tabular Detections Grid:</strong> Labeled object table showing ID, class, coordinates, and exact confidence percentages.</li>
                <li><strong>Class Distribution Chart:</strong> Horizontal bar graphs of detected visual class frequencies.</li>
                <li><strong>Crop-to-OCR Pipeline:</strong> Select any detected object bounding box and run character extraction on the cropped matrix.</li>
            </ul>
        </div>
        <div class="card" style="padding: 16px; border-top: 4px solid #f87171;">
            <div style="color: #f87171; font-weight: 800; font-size: 14.5px; margin-bottom: 8px;">📦 Enterprise Batch Pipeline Features</div>
            <ul style="font-size: 12.5px; color: #d4d4d8; margin-left: 16px; line-height: 1.7;">
                <li><strong>Bulk Operations Queue:</strong> Drop multiple images to run simultaneous OCR, DL detection, and NLP parsing.</li>
                <li><strong>Pipeline Progress Bar:</strong> Real-time process visual tracking.</li>
                <li><strong>Structured Exporter:</strong> Compile raw text, cropped images, and metadata into structured directories.</li>
                <li><strong>Dynamic ZIP Archiver:</strong> Create and download instant ZIP archives containing processing outputs and `metadata.csv`.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  TAB 2 · DOCUMENT OCR LAB
# ═══════════════════════════════════════════════════════════════
with tab_ocr:
    col_up, col_ctrl = st.columns([6, 4], gap="large")

    img_np = None
    cropped = None

    with col_up:
        st.markdown('<div class="panel-header">📥 &nbsp;Upload & Target Area Selection</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop an image (PNG / JPG / JPEG)", type=["png","jpg","jpeg"], key="ocr_up")
        use_roi = st.checkbox("✏️ Draw ROI — Crop region before OCR", value=False)

        if uploaded:
            pil = Image.open(uploaded)
            img_np = np.array(pil)
            if img_np.ndim == 2:       img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else:                      img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            oh, ow = img_np.shape[:2]

            st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
            st.markdown('<div class="viewport-title">🎬 Input Visual Stream</div>', unsafe_allow_html=True)
            if use_roi:
                sw = 680; sh = int(oh/ow*sw)
                res = st_canvas(fill_color="rgba(31,111,235,.15)", stroke_width=2,
                                stroke_color="#1f6feb", background_image=pil,
                                update_streamlit=True, height=sh, width=sw,
                                drawing_mode="rect", key="ocr_canvas")
                if res.json_data and res.json_data["objects"]:
                    o = res.json_data["objects"][-1]
                    x1 = max(0, int(o["left"]*ow/sw)); y1 = max(0, int(o["top"]*oh/sh))
                    x2 = min(ow, x1+int(o["width"]*ow/sw)); y2 = min(oh, y1+int(o["height"]*oh/sh))
                    if x2-x1>5 and y2-y1>5:
                        cropped = img_np[y1:y2, x1:x2]
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                        st.markdown('<div class="viewport-title">✂️ Cropped ROI Segment</div>', unsafe_allow_html=True)
                        st.image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.image(pil, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_ctrl:
        st.markdown('<div class="panel-header">⚙️ &nbsp;OCR Configuration Hub</div>', unsafe_allow_html=True)
        st.session_state.ocr_engine = st.selectbox("OCR Engine", [
            "Windows Native OCR (WinRT)", "Google Tesseract (pytesseract)"
        ], key="ocr_engine_select")
        if "Tesseract" in st.session_state.ocr_engine:
            st.session_state.tesseract_path = st.text_input(
                "Tesseract.exe path", value=st.session_state.tesseract_path, key="tess_path_input")
        template = st.selectbox("AI Parser Template", [
            "None — Raw Text Output", "🧾 Invoice / Receipt",
            "💼 Business Card", "🪪 ID Card Scanner"
        ], key="parser_template_select")
        search_kw = st.text_input("🔍 Keyword Highlight", placeholder="Search word on image...", key="ocr_search_input")
        show_analytics = st.checkbox("📊 Show NLP Text Analytics", value=True, key="ocr_show_analytics")
        run = st.button("⚡ Extract & Analyze", use_container_width=True, type="primary", key="ocr_run_btn")

        if img_np is not None and run:
            target = cropped if (use_roi and cropped is not None) else img_np
            raw_bytes = cv2.imencode('.png', target)[1].tobytes()
            t0 = time.time()
            with st.spinner("Running OCR pipeline..."):
                if "WinRT" in st.session_state.ocr_engine:
                    raw_text, words = asyncio.run(utils.run_uwp_ocr(raw_bytes))
                else:
                    raw_text, words = utils.run_tesseract_ocr(
                        target, tesseract_cmd=st.session_state.tesseract_path, psm=3)
            ms = (time.time()-t0)*1000

            # Stats row
            st.markdown('<div class="lbl">🔬 Extraction Diagnostics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stats">
                <div class="stat"><div class="stat-v">{ms:.0f}</div><div class="stat-l">ms latency</div></div>
                <div class="stat"><div class="stat-v">{len(raw_text)}</div><div class="stat-l">characters</div></div>
                <div class="stat"><div class="stat-v">{len(words)}</div><div class="stat-l">words</div></div>
            </div>""", unsafe_allow_html=True)

            if raw_text.strip():
                tts_widget(raw_text)

            # Keyword highlight overlay
            if search_kw and words:
                hi, cnt = utils.highlight_words(target.copy(), words, search_kw)
                st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                st.markdown(f'<div class="viewport-title">🔍 Overlay Highlights ({cnt} matches)</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(hi, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Download annotated image
            _, annotated_bytes = cv2.imencode('.png', target)
            st.download_button("📥 Download Source Image", data=annotated_bytes.tobytes(),
                               file_name="ocr_result.png", mime="image/png", key="ocr_download_img_btn", use_container_width=True)

            # NLP Analytics
            if show_analytics and raw_text.strip():
                st.markdown('<div class="lbl" style="margin-top:16px;">NLP Text Analytics</div>', unsafe_allow_html=True)
                stats, chart_buf = utils.get_text_analytics(raw_text)
                rows_html = "".join([f'<tr class="kv-row"><td class="kv-key">{k}</td><td class="kv-val">{v}</td></tr>'
                                     for k, v in stats.items()])
                st.markdown(f"""
                <div class="card" style="padding:16px; margin-bottom:12px;">
                    <table class="kv">{rows_html}</table>
                </div>""", unsafe_allow_html=True)
                if chart_buf:
                    st.image(chart_buf, caption="Word Frequency Chart", use_container_width=True)

            # Parsed output or raw text
            st.markdown('<div class="lbl" style="margin-top:16px;">Extracted Output Console</div>', unsafe_allow_html=True)
            if "None" in template:
                st.markdown(f'<div class="terminal-console">{raw_text}</div>', unsafe_allow_html=True)
            else:
                if "Invoice" in template:   parsed = utils.parse_invoice(raw_text)
                elif "Business" in template: parsed = utils.parse_business_card(raw_text)
                else:                        parsed = utils.parse_id_card(raw_text)
                rows = "".join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>' for k,v in parsed.items())
                st.markdown(f'<div class="card"><table class="kv">{rows}</table></div><br>', unsafe_allow_html=True)
                st.download_button("📥 Export JSON Data", data=str(parsed),
                                   file_name="parsed_output.json", mime="application/json", key="ocr_export_json_btn", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 3 · CV FILTERS LAB
# ═══════════════════════════════════════════════════════════════
with tab_cv:
    st.markdown('<div class="panel-header">🧪 &nbsp;Computer Vision Parameter Sandbox</div>', unsafe_allow_html=True)
    cv_up = st.file_uploader("Upload image for filter pipeline", type=["png","jpg","jpeg"], key="cv_up")

    if cv_up:
        pil_cv = Image.open(cv_up)
        img_cv = np.array(pil_cv)
        if img_cv.ndim == 2:       img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 4: img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
        else:                      img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        col_ctr, col_res = st.columns([3, 7], gap="large")

        with col_ctr:
            st.markdown('<div class="lbl">Algorithmic Controls</div>', unsafe_allow_html=True)

            st.markdown('<div class="card" style="padding:18px;">', unsafe_allow_html=True)
            st.markdown("<div style='font-size:13px; font-weight:700; color:#38bdf8; margin-bottom:6px;'>🔧 Noise Reduction</div>", unsafe_allow_html=True)
            blur_k = st.slider("Gaussian Blur Kernel", 1, 15, 3, step=2, key="cv_blur_slider")
            sharpening = st.checkbox("Unsharp Mask Sharpening", value=False, key="cv_sharpen_check")

            st.markdown("<div style='font-size:13px; font-weight:700; color:#34d399; margin-top:14px; margin-bottom:6px;'>⚡ Binarization Gateway</div>", unsafe_allow_html=True)
            use_otsu = st.checkbox("Otsu's Auto-Binarization", value=True, key="cv_otsu_check")
            manual_thresh = 127
            if not use_otsu:
                manual_thresh = st.slider("Manual Threshold", 0, 255, 127, key="cv_manual_thresh_slider")

            st.markdown("<div style='font-size:13px; font-weight:700; color:#fbbf24; margin-top:14px; margin-bottom:6px;'>🔆 Local Contrast</div>", unsafe_allow_html=True)
            use_clahe = st.checkbox("CLAHE Contrast Enhancement", value=False, key="cv_clahe_check")
            clahe_clip = 2.0
            if use_clahe:
                clahe_clip = st.slider("CLAHE Clip Limit", 0.5, 8.0, 2.0, 0.5, key="cv_clahe_clip_slider")

            st.markdown("<div style='font-size:13px; font-weight:700; color:#c084fc; margin-top:14px; margin-bottom:6px;'>📐 Matrix Alignment</div>", unsafe_allow_html=True)
            deskew = st.checkbox("Auto Deskew Correction", value=False, key="cv_deskew_check")

            st.markdown("<div style='font-size:13px; font-weight:700; color:#f87171; margin-top:14px; margin-bottom:6px;'>🎯 Canny Edge Detector</div>", unsafe_allow_html=True)
            canny = st.checkbox("Canny Edge Detector", value=False, key="cv_canny_check")
            c_lo, c_hi = 50, 150
            if canny:
                c_lo = st.slider("Canny Low", 0, 255, 50, key="cv_canny_lo_slider")
                c_hi = st.slider("Canny High", 0, 255, 150, key="cv_canny_hi_slider")

            st.markdown("<div style='font-size:13px; font-weight:700; color:#a1a1aa; margin-top:14px; margin-bottom:6px;'>🔬 Morphology Cleanup</div>", unsafe_allow_html=True)
            morph = st.selectbox("Morphological Op", ["None", "Dilation", "Erosion", "Opening", "Closing"], key="cv_morph_select")
            mk = st.slider("Morph Kernel Size", 1, 9, 3, step=2, key="cv_morph_kernel_slider")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_res:
            st.markdown('<div class="lbl">Visual Comparison Viewports</div>', unsafe_allow_html=True)

            gray   = utils.grayscale_image(img_cv)
            proc   = gray.copy()
            angle  = 0.0

            if use_clahe:
                proc = utils.apply_clahe(proc, clip_limit=clahe_clip)
            if deskew:
                proc, angle = utils.deskew_image(proc)
            proc = utils.blur_image(proc, blur_k)
            if sharpening:
                proc = utils.sharpen_image(proc)
            proc, thresh_v = utils.threshold_image(proc, manual_thresh, use_otsu)
            if canny:
                proc = cv2.Canny(proc, c_lo, c_hi)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
            if   morph == "Dilation": proc = cv2.dilate(proc, kernel)
            elif morph == "Erosion":  proc = cv2.erode(proc, kernel)
            elif morph == "Opening":  proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, kernel)
            elif morph == "Closing":  proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel)

            # Viewport Frames
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                st.markdown('<div class="viewport-title">📸 Raw Input Matrix</div>', unsafe_allow_html=True)
                st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                st.markdown(f'<div class="viewport-title">⚙️ Processed Signal Matrix (Rotation: {angle:.1f}°)</div>', unsafe_allow_html=True)
                st.image(proc, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Download processed
            _, enc = cv2.imencode('.png', proc)
            st.download_button("📥 Download Processed Matrix Output", data=enc.tobytes(),
                               file_name="processed.png", mime="image/png", key="cv_download_processed_btn", use_container_width=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # Two charts side-by-side in viewport frames
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                st.markdown('<div class="viewport-title">📊 Intensity Distribution Histogram</div>', unsafe_allow_html=True)
                hist_buf = utils.get_histogram_plot(gray, thresh_v)
                st.image(hist_buf, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with ch2:
                st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                st.markdown('<div class="viewport-title">🌈 RGB Frequency Distribution</div>', unsafe_allow_html=True)
                rgb_buf = utils.get_color_channels_plot(img_cv)
                st.image(rgb_buf, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TAB 4 · OBJECT DETECTION
# ═══════════════════════════════════════════════════════════════
with tab_det:
    st.markdown('<div class="panel-header">🎯 &nbsp;Deep Learning Bounding Box Inference Engine</div>', unsafe_allow_html=True)
    if not st.session_state.models_ready:
        col_dl, _ = st.columns([5, 5])
        with col_dl:
            st.markdown("""
            <div class="card blue" style="text-align:center;padding:40px;">
                <div style="font-size:40px;margin-bottom:14px;">🧠</div>
                <div style="font-size:18px;font-weight:700;color:#e6edf3;margin-bottom:8px;">Neural Weights Required</div>
                <div style="font-size:13px;color:#8b949e;margin-bottom:24px;">YOLOv5-Nano ONNX · MobileNet-SSD Caffe · ~27 MB total</div>
            </div>""", unsafe_allow_html=True)
            if st.button("📥 Download Neural Weights", use_container_width=True, key="det_download_weights_btn"):
                with st.spinner("Fetching from GitHub releases..."):
                    utils.download_models()
                    st.session_state.models_ready = True
                    st.success("✓ Weights downloaded successfully.")
                    st.rerun()
    else:
        det_up = st.file_uploader("Upload image for object detection", type=["png","jpg","jpeg"], key="det_up")

        if det_up:
            pil_det = Image.open(det_up)
            img_det = np.array(pil_det)
            if img_det.ndim == 2:       img_det = cv2.cvtColor(img_det, cv2.COLOR_GRAY2BGR)
            elif img_det.shape[2] == 4: img_det = cv2.cvtColor(img_det, cv2.COLOR_RGBA2BGR)
            else:                       img_det = cv2.cvtColor(img_det, cv2.COLOR_RGB2BGR)

            col_dc, col_dr = st.columns([3, 7], gap="large")

            with col_dc:
                st.markdown('<div class="lbl">Inference Configuration</div>', unsafe_allow_html=True)
                st.markdown('<div class="card" style="padding:18px;">', unsafe_allow_html=True)
                det_model = st.selectbox("Inference Model", ["YOLOv5-Nano (ONNX)", "MobileNet-SSD (Caffe)"], key="det_model_select")
                conf = st.slider("Confidence Gate Cutoff", 0.0, 1.0, 0.5, 0.05,
                                 help="Minimum confidence score to accept a detection", key="det_conf_slider")
                run_det = st.button("🔥 Deploy Model Pipeline", use_container_width=True, type="primary", key="det_run_btn")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_dr:
                if run_det:
                    t0 = time.time()
                    with st.spinner("Running forward pass through DNN..."):
                        if "YOLO" in det_model:
                            out_img, objs, err = utils.detect_objects_yolo(img_det, conf)
                        else:
                            out_img, objs, err = utils.detect_objects_ssd(img_det, conf)
                    ms = (time.time()-t0)*1000

                    if err:
                        st.error(err)
                    else:
                        st.markdown('<div class="lbl">Inference Diagnostics</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="stats">
                            <div class="stat"><div class="stat-v">{ms:.0f}</div><div class="stat-l">ms latency</div></div>
                            <div class="stat"><div class="stat-v">{len(objs)}</div><div class="stat-l">objects found</div></div>
                            <div class="stat"><div class="stat-v">{len(set(o['class'] for o in objs))}</div><div class="stat-l">unique classes</div></div>
                        </div>""", unsafe_allow_html=True)

                        # Annotated output in frame
                        st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                        st.markdown(f'<div class="viewport-title">👁️ Annotated Neural Bounding Boxes ({det_model.split()[0]})</div>', unsafe_allow_html=True)
                        st.image(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Download annotated image
                        _, ann_enc = cv2.imencode('.png', out_img)
                        st.download_button("📥 Download Annotated Image Output", data=ann_enc.tobytes(),
                                           file_name="detection_result.png", mime="image/png", key="det_download_annotated_btn", use_container_width=True)

                        if objs:
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown('<div class="lbl">Localization Coordinates Data</div>', unsafe_allow_html=True)
                            tab_det_t, tab_chart_t = st.tabs(["📋 Detection Coordinates Table", "📊 Class Frequency Histogram"])

                            with tab_det_t:
                                header = "<tr><th>#</th><th>Class Category</th><th>Confidence</th><th>Confidence Bar</th><th>Bounding Box (x,y,w,h)</th></tr>"
                                rows_det = ""
                                for i, o in enumerate(objs):
                                    pct = o['confidence']*100
                                    bar_w = int(pct)
                                    box = o['box']
                                    rows_det += f"""<tr>
                                        <td>#{i+1}</td>
                                        <td style="color:#38bdf8;font-weight:700;">{o['class']}</td>
                                        <td style="font-weight:600;color:#ffffff;">{pct:.1f}%</td>
                                        <td><div class="conf-bar" style="width:{bar_w}%;"></div></td>
                                        <td>({box[0]},{box[1]},{box[2]},{box[3]})</td>
                                    </tr>"""
                                st.markdown(f'<table class="det-table"><thead>{header}</thead><tbody>{rows_det}</tbody></table>', unsafe_allow_html=True)

                            with tab_chart_t:
                                chart = utils.get_class_distribution_chart(objs)
                                if chart:
                                    st.image(chart, use_container_width=True)

                            # Crop-to-OCR Integration
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.markdown('<div class="lbl">Target Crop-to-OCR Pipeline</div>', unsafe_allow_html=True)
                            opts = [f"#{i+1} · {o['class']} ({o['confidence']:.0%})" for i,o in enumerate(objs)]
                            sel = st.selectbox("Select detected bounding box to OCR", range(len(objs)), format_func=lambda x: opts[x], key="det_crop_ocr_select")
                            crop = objs[sel].get('crop')

                            if crop is not None and crop.size > 0:
                                cx, cy = st.columns([4, 6])
                                with cx:
                                    st.markdown('<div class="viewport-frame">', unsafe_allow_html=True)
                                    st.markdown('<div class="viewport-title">✂️ Crop ROI segment</div>', unsafe_allow_html=True)
                                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with cy:
                                    if st.button("⚡ Run OCR on Bounding Box", key="det_crop_ocr_run_btn", use_container_width=True):
                                        cb = cv2.imencode('.png', crop)[1].tobytes()
                                        with st.spinner("Extracting text from cropped bounding box..."):
                                            if "WinRT" in st.session_state.ocr_engine:
                                                ct, _ = asyncio.run(utils.run_uwp_ocr(cb))
                                            else:
                                                ct, _ = utils.run_tesseract_ocr(
                                                    crop, tesseract_cmd=st.session_state.tesseract_path)
                                        if ct.strip():
                                            st.markdown('<div class="lbl">OCR Extracted Console</div>', unsafe_allow_html=True)
                                            st.markdown(f'<div class="terminal-console">{ct}</div>', unsafe_allow_html=True)
                                            tts_widget(ct)
                                        else:
                                            st.info("No readable text found in this crop region.")


# ═══════════════════════════════════════════════════════════════
#  TAB 5 · BATCH PROCESSOR
# ═══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="panel-header">📦 &nbsp;Enterprise Batch Pipeline Queue</div>', unsafe_allow_html=True)
    if not st.session_state.models_ready:
        st.warning("⚠️ Neural weights are missing — go to **Object Detection** page to download them first.")
    else:
        batch_files = st.file_uploader("Upload multiple images to queue",
                                       type=["png","jpg","jpeg"],
                                       accept_multiple_files=True, key="batch_up")

        if batch_files:
            col_bs, col_br = st.columns([4, 6], gap="large")

            with col_bs:
                st.markdown('<div class="lbl">Configuration Queue</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="card blue" style="padding:18px;text-align:center;">
                    <div style="font-size:36px;font-weight:900;color:#0284c7;font-family:'JetBrains Mono',monospace;">{len(batch_files)}</div>
                    <div style="font-size:12px;color:#a1a1aa;font-weight:700;text-transform:uppercase;letter-spacing:1px;">images queued</div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="card" style="padding:16px;">', unsafe_allow_html=True)
                batch_model = st.selectbox("Detection Engine", [
                    "YOLOv5-Nano (ONNX)", "MobileNet-SSD (Caffe)"], key="bm")
                batch_parser = st.selectbox("Auto Parser Template", [
                    "🧾 Invoice / Receipt", "💼 Business Card", "🪪 ID Card Scanner"], key="bp")
                batch_conf = st.slider("Detection Confidence Gate", 0.0, 1.0, 0.5, 0.05, key="bc")
                st.markdown('</div>', unsafe_allow_html=True)
                run_batch = st.button("🔥 Launch Batch Pipeline Queue", type="primary", use_container_width=True, key="batch_run_btn")

            with col_br:
                if run_batch:
                    st.markdown('<div class="lbl">Active Processing Progress</div>', unsafe_allow_html=True)
                    prog    = st.progress(0)
                    status  = st.empty()
                    results = []

                    for i, f in enumerate(batch_files):
                        status.markdown(f'<div style="font-size:13.5px;color:#d4d4d8;">Processing {i+1}/{len(batch_files)}: <code style="background:#27272a;padding:2px 6px;border-radius:4px;color:#ffffff;font-family:\'JetBrains Mono\'">{f.name}</code></div>', unsafe_allow_html=True)
                        raw   = f.read()
                        arr   = np.frombuffer(raw, np.uint8)
                        img_b = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                        # OCR
                        if "WinRT" in st.session_state.ocr_engine:
                            txt, _ = asyncio.run(utils.run_uwp_ocr(raw))
                        else:
                            txt, _ = utils.run_tesseract_ocr(img_b, tesseract_cmd=st.session_state.tesseract_path)

                        # Detection
                        if "YOLO" in batch_model:
                            ann, dets, _ = utils.detect_objects_yolo(img_b, batch_conf)
                        else:
                            ann, dets, _ = utils.detect_objects_ssd(img_b, batch_conf)

                        # Parsing
                        if "Invoice" in batch_parser:   meta = utils.parse_invoice(txt)
                        elif "Business" in batch_parser: meta = utils.parse_business_card(txt)
                        else:                            meta = utils.parse_id_card(txt)

                        img_enc = cv2.imencode('.png', ann)[1].tobytes()
                        results.append({'filename':f.name,'img_bytes':img_enc,'text':txt,'metadata':meta,'detections':dets})
                        prog.progress((i+1)/len(batch_files))

                    status.markdown('<div class="stSuccess" style="padding:10px; margin-bottom:12px;">🎉 &nbsp;Batch pipeline queue execution complete!</div>', unsafe_allow_html=True)

                    # ZIP download
                    zip_data = utils.create_batch_zip(results)
                    st.download_button("📥 Download Compiled ZIP Archive", data=zip_data,
                                       file_name="ai_vision_batch.zip",
                                       mime="application/zip", use_container_width=True, key="batch_download_zip_btn")

                    # Summary table
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown('<div class="lbl">Pipeline Summary Records</div>', unsafe_allow_html=True)
                    hdr = "<tr><th>Filename</th><th>Objects Found</th><th>Text Snippet</th><th>Extracted Main Entity</th></tr>"
                    trows = ""
                    for r in results:
                        top_entity = list(r['metadata'].values())[0] if r['metadata'] else "—"
                        preview = r['text'][:60].replace('\n',' ')
                        if len(r['text']) > 60: preview += '...'
                        trows += f"""<tr>
                            <td style="color:#38bdf8;font-weight:700;">{r['filename']}</td>
                            <td style="font-weight:600;text-align:center;">{len(r['detections'])}</td>
                            <td style="color:#d4d4d8;">{preview}</td>
                            <td style="color:#ffffff;font-weight:600;">{top_entity}</td>
                        </tr>"""
                    st.markdown(f'<table class="det-table"><thead>{hdr}</thead><tbody>{trows}</tbody></table>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  UNIFIED FOOTER · DEVELOPER PROFILE (AT THE BOTTOM OF ALL PAGES)
# ═══════════════════════════════════════════════════════════════
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown('<div class="lbl">👤  Developed & Architected By</div>', unsafe_allow_html=True)

col_f_l, col_f_r = st.columns([5, 7], gap="large")

with col_f_l:
    st.markdown("""
    <div class="footer-hero">
        <div class="avatar-small">MA</div>
        <div class="hero-name">Muhammad Abdullah</div>
        <div class="hero-role">AI · ML · LLM · Full Stack AI Engineer</div>
        <div class="hero-meta">📍 Lahore, Pakistan</div>
        <div class="hero-meta">📧 meharabdullah4337@gmail.com</div>
        <div class="hero-meta">🎓 Computer Science Student @ Punjab University</div>
        <br>
        <a class="soc" href="https://github.com/muhammadabdullah-devpk" target="_blank">🐙 GitHub</a>
        <a class="soc" href="https://linkedin.com/in/muhammad-abdullah-devpk" target="_blank">💼 LinkedIn</a>
    </div>""", unsafe_allow_html=True)

with col_f_r:
    st.markdown("""
    <div style="background:#18181b; border:1px solid #3f3f46; border-radius:12px; padding:20px; height:100%;">
        <div style="font-size:12px; font-weight:800; color:#ffffff; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">Core Engineering Competencies</div>
        <div style="display: flex; flex-wrap: wrap;">
            <span class="badge b-blue">YOLOv5 / YOLOv8</span>
            <span class="badge b-blue">MobileNet-SSD</span>
            <span class="badge b-blue">ONNX Runtime</span>
            <span class="badge b-purple">Large Language Models</span>
            <span class="badge b-purple">Regex Parsing</span>
            <span class="badge b-green">OpenCV</span>
            <span class="badge b-green">WinRT UWP OCR</span>
            <span class="badge b-green">Tesseract OCR</span>
            <span class="badge b-orange">Streamlit Core</span>
            <span class="badge b-orange">Python Backend</span>
        </div>
        <p style="font-size:13px; color:#d4d4d8; line-height:1.6; margin-top:14px;">
            Specialized in integrating cutting-edge machine perception pipelines, optimizing deep neural network weights inference, and designing sleek, minimal developer-first interfaces.
        </p>
    </div>
    """, unsafe_allow_html=True)
