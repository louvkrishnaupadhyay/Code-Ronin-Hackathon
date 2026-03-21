import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
from detect import Detector
from embed import Embedder
from database import Database
from learn import Learner
from voice import VoiceAssist

# --- Configuration & Styling ---
st.set_page_config(page_title="CodeNova Adaptive Vision", layout="wide", page_icon="🛡️")

# Modern Premium UI Styles
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #0d1117, #010409);
        color: #c9d1d9;
    }
    
    /* Transparent Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 25, 30, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }

    /* Primary Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 160, 67, 0.4);
    }

    /* Label Styling */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-known { background: rgba(35, 134, 54, 0.2); color: #3fb950; border: 1px solid #3fb950; }
    .badge-unknown { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #f85149; }

    /* Titles */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    .main-title {
        background: linear-gradient(to right, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# --- Neural Engine Management (Safe Version-Aware Caching) ---
def get_caching_decorator():
    """Detects Streamlit version and returns the correct singleton decorator."""
    if hasattr(st, "cache_resource"):
        return st.cache_resource
    return st.experimental_singleton

# Apply the appropriate decorator based on system version (1.12.0 vs 1.18.0+)
@get_caching_decorator()
def load_neural_engines():
    """Load heavy models into RAM once and cache them (Prevents Meta Issue)."""
    with st.spinner("🚀 Initializing Neural Engines (YOLO & DINOv2)..."):
        # 1. FORCE SAFE YOLO LOADING (CRITICAL)
        det = Detector(model_path='yolov8n.pt')
        # 2. DINOv2 SAFETY
        emb = Embedder(model_name="facebook/dinov2-base")
        vce = VoiceAssist()
        return det, emb, vce

# Initialize through resource cache
detector, embedder, voice = load_neural_engines()
database = Database(embedding_dim=768) 
learner = Learner(detector, embedder, database)

# --- Sidebar Controls ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/shield.png", width=100)
    st.markdown("## System Controls")
    
    st.markdown("### 🛠️ Neural Settings")
    speech_on = st.checkbox("Vocal Notifications", value=True)
    confidence_threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.70)
    
    st.markdown("---")
    st.markdown("### 🚀 Database Management")
    st.write(f"Knowledge Base: **{len(database.labels)}** Classes")
    
    if st.button("🔄 Refresh Application"):
        st.experimental_rerun()

    if st.button("⚠️ Factory Reset DB"):
        if os.path.exists(database.vectors_path): os.remove(database.vectors_path)
        if os.path.exists(database.labels_path): os.remove(database.labels_path)
        st.experimental_rerun()

# --- Main App UI ---
st.markdown('<h1 class="main-title">🛡️ Adaptive Vision</h1>', unsafe_allow_html=True)
st.markdown("#### Real-time Zero-Shot Object Recognition System")

col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📸 Upload & Analyze")
    uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # 5. DEBUG CHECK: Confirm file received successfully
        st.caption(f"✅ File received: **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")

        # 7. FILE SIZE GUARD
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("File too large. Maximum allowed size is 200MB.")
            st.stop()

        # 2. CORRECT FILE UPLOAD HANDLING (PIL → RGB → NumPy)
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)   # This is RGB
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV/YOLO
        
        if 'detections' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_file.name:
            with st.spinner("🧠 Scanning for objects..."):
                detections = detector.detect_and_crop(img_cv)
                if not detections:
                    h, w, _ = img_cv.shape
                    detections = [{'bbox': [0, 0, w, h], 'crop': img_cv}]
                
                for det in detections:
                    embedding = embedder.get_embedding(det['crop'])
                    label, score = database.search_entry(embedding, threshold=confidence_threshold)
                    det['embedding'] = embedding
                    det['label'] = label
                    det['score'] = score
                
                st.session_state.detections = detections
                st.session_state.last_uploaded = uploaded_file.name
        else:
            detections = st.session_state.detections

        # Draw results
        for det in detections:
            color = (0, 255, 0) if det['label'] != "Unknown" else (0, 0, 255)
            img_cv = detector.draw_bbox(img_cv, det['bbox'], f"{det['label']} ({(det['score']*100):.0f}%)", color)
        
        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        if speech_on:
            voice.notify_detection([d['label'] for d in detections if d['label'] != "Unknown"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None and detections:
        st.markdown("### 🔍 Cognitive Results")
        
        for i, det in enumerate(detections):
            is_unknown = det['label'] == "Unknown"
            badge_class = "badge-unknown" if is_unknown else "badge-known"
            
            st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1em; font-weight: 600;"># {i+1} Detection</span>
                    <span class="status-badge {badge_class}">{det['label']}</span>
                </div>
                <p style="margin-top: 10px; color: #8b949e; font-size: 0.9em;">Match Probability: {det['score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if is_unknown:
                with st.form(key=f"form_{i}"):
                    new_label = st.text_input("Teach system a new label:")
                    if st.form_submit_button("🧠 Synchronize Label"):
                        if new_label.strip():
                            learner.learn_from_embedding(det['embedding'], new_label.strip())
                            st.success(f"Stored: '{new_label}'")
                            st.experimental_rerun()
    else:
        st.info("Upload an image in the left panel to begin object recognition.")
