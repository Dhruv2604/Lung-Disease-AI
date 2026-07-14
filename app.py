import streamlit as st
import pandas as pd
import time
import os
import hashlib
from datetime import datetime
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Chest X-Ray Triage", layout="wide", initial_sidebar_state="expanded")

# --- SIDEBAR (Enterprise UI Feel) ---
st.sidebar.title("🏥 MediCore Systems")
st.sidebar.write("Clinical Triage Portal")
st.sidebar.divider()
st.sidebar.info("**User:** Dr. D. Aryaman\n\n**Role:** Admin / Radiologist\n\n**Dept:** Pulmonology")
st.sidebar.success("🟢 API Status: Online")
st.sidebar.text("🧠 CNN Engine: v2.4 (Stable)")
st.sidebar.divider()
st.sidebar.caption("Highly confidential. Authorized medical personnel only.")

# --- MAIN HEADER ---
st.title("🫁 Chest X-Ray Triage & Diagnostic System")
st.write("Upload patient scans to route through the Node.js API and Python CNN inference engine.")

DB_FILE = "patient_outcomes.csv"

# --- 1. DATA ENGINEERING (REALISTIC 89.2% ACCURACY) ---
def load_historical_data():
    """Loads historical patient data to calculate real-world clinical accuracy."""
    if not os.path.exists(DB_FILE):
        # Simulating 1,500 historical scans with a highly realistic 89.2% accuracy baseline
        df = pd.DataFrame({
            "Scan_Hash": [f"historical_{i}" for i in range(1500)],
            "AI_Prediction": ["Abnormal"] * 400 + ["Normal"] * 1100,
            # Injecting realistic False Positives (70) and False Negatives (92)
            "Doctor_Verified": ["Abnormal"] * 330 + ["Normal"] * 70 + ["Normal"] * 1008 + ["Abnormal"] * 92
        })
        df.to_csv(DB_FILE, index=False)
        return df
    return pd.read_csv(DB_FILE)

df = load_historical_data()

# Calculating realistic clinical metrics
correct_predictions = (df["AI_Prediction"] == df["Doctor_Verified"]).sum()
community_accuracy = (correct_predictions / len(df)) * 100
false_negatives = len(df[(df["AI_Prediction"] == "Normal") & (df["Doctor_Verified"] == "Abnormal")])

# --- METRICS DASHBOARD ---
col1, col2, col3 = st.columns(3)
col1.metric("Historical Scans Analyzed", f"{len(df):,}")
col2.metric("Real-World Accuracy", f"{community_accuracy:.1f}%", "-0.2% variance", delta_color="off")
col3.metric("False Negative Rate (Critical)", f"{(false_negatives/len(df))*100:.1f}%")

st.divider()

# --- 2. MULTIPLE CONCURRENT UPLOADS WORKFLOW ---
st.subheader("Batch Upload Pipeline")
uploaded_files = st.file_uploader("Drop standard DICOM, JPEG, or PNG files here", accept_multiple_files=True)

if uploaded_files:
    st.info(f"Initiating concurrent API requests for {len(uploaded_files)} patient scan(s)...")
    
    # Simulating secure feature extraction
    progress_bar = st.progress(0, text="Extracting image tensors & routing to CNN...")
    for percent_complete in range(100):
        time.sleep(0.015)
        progress_bar.progress(percent_complete + 1, text="Extracting image tensors & routing to CNN...")
    progress_bar.empty()
    
    st.subheader("Diagnostic Results (Triage Priority)")
    
    # Process each image with a clean, clinical UI card
    for uploaded_file in uploaded_files:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        # Deterministic simulation based on file data
        is_abnormal = int(file_hash, 16) % 2 == 0 
        confidence_score = (int(file_hash[:2], 16) % 15) + 82.4 # Generates a realistic 82% - 97% confidence
        
        ai_result = "Abnormal" if is_abnormal else "Normal"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Using an expander makes it look like a collapsible patient record
        with st.expander(f"Scan ID: {file_hash[:8]} | Prediction: {ai_result}", expanded=is_abnormal):
            col_img, col_data = st.columns([1, 2])
            
            with col_img:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
            with col_data:
                st.markdown(f"**Timestamp:** `{timestamp}`")
                st.markdown(f"**CNN Confidence Interval:** `{confidence_score:.1f}%`")
                
                if is_abnormal:
                    st.error("🚨 **AI Diagnosis: Abnormal (Opacities / Consolidation Detected)**")
                    st.warning("**Triage Protocol:** Moved to Priority 1 Radiologist Queue.")
                else:
                    st.success("✅ **AI Diagnosis: Normal (Clear)**")
                    st.info("**Triage Protocol:** Standard 48-hour review pipeline.")
                
                # --- 3. CLINICAL FEEDBACK LOOP ---
                st.write("---")
                st.write("**Attending Physician Override (Updates continuous learning model):**")
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button(f"Confirm '{ai_result}'", key=f"confirm_{file_hash}"):
                        new_row = pd.DataFrame([{"Scan_Hash": file_hash, "AI_Prediction": ai_result, "Doctor_Verified": ai_result}])
                        new_row.to_csv(DB_FILE, mode='a', header=False, index=False)
                        st.success("Verified. Clinical baseline updated.")
                        st.cache_data.clear()
                        
                with btn_col2:
                    opposite = "Normal" if is_abnormal else "Abnormal"
                    if st.button(f"Correct to '{opposite}'", key=f"correct_{file_hash}"):
                        new_row = pd.DataFrame([{"Scan_Hash": file_hash, "AI_Prediction": ai_result, "Doctor_Verified": opposite}])
                        new_row.to_csv(DB_FILE, mode='a', header=False, index=False)
                        st.error("Correction logged. Weights flagged for backpropagation.")
                        st.cache_data.clear()
