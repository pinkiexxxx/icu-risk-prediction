import numpy as np
import streamlit as st
import shap
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

matplotlib.use('Agg')

st.set_page_config(
    page_title="ICU Mortality Risk Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    
    .main-header {
        background: linear-gradient(135deg, #005c97 0%, #363795 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin: 0;
    }
    
    .result-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 10px solid #ddd;
    }
    .result-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: #333;
        margin: 10px 0;
    }
    .result-label {
        font-weight: bold;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    
    .chart-section {
        margin-top: 30px;
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    .chart-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 20px;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 10px;
        border-left: 6px solid #005c97;
        padding-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "rf_model.pkl"
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    return classifier

classifier1 = load_model()

FEATURE_NAMES = [
    "Acute_kidney_injury", "Pneumonia", "Age", "Weight", "Heart_rate",
    "RDW", "WBC", "Anion_gap", "Chloride", "Creatinine", "Base_excess", "PH"
]

def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
        st.markdown("### Patient Data")
        st.markdown("Enter clinical parameters below:")
        with st.form("input_form"):
            st.markdown("#### 👤 Demographics")
            c1, c2 = st.columns(2)
            with c1: Age = st.number_input("Age", 18, 100, 60)
            with c2: Weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)

            st.markdown("#### ❤️ Vitals")
            Heart_rate = st.number_input("HR (bpm)", 20, 200, 80)

            st.markdown("#### 🧪 Labs")
            c5, c6 = st.columns(2)
            with c5: RDW = st.number_input("RDW (%)", 10.0, 30.0, 14.5)
            with c6: WBC = st.number_input("WBC (10⁹/L)", 0.1, 100.0, 8.0)
            c7, c8 = st.columns(2)
            with c7: Anion_gap = st.number_input("Anion gap (mEq/L)", 0.0, 50.0, 12.0)
            with c8: Chloride = st.number_input("Cl- (mEq/L)", 20.0, 200.0, 100.0)
            c9, c10 = st.columns(2)
            with c9: Creatinine = st.number_input("Creatinine (mg/dL)", 0.0, 20.0, 1.0)
            with c10: Base_excess = st.number_input("Base excess (mmol/L)", -30.0, 30.0, 0.0)
            PH = st.number_input("pH", 6.50, 8.00, 7.40, step=0.01, format="%.2f")

            st.markdown("#### 💊 Clinical Status")
            ak_map = {"No": 0, "Yes": 1}; Acute_kidney_injury = ak_map[st.selectbox("Acute Kidney Injury", list(ak_map.keys()))]
            pneumonia_map = {"No": 0, "Yes": 1}; Pneumonia = pneumonia_map[st.selectbox("Pneumonia", list(pneumonia_map.keys()))]
            
            st.markdown("---")
            predict_btn = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)

    st.markdown("""
    <div class="main-header">
        <h1>28-day Mortality Risk Prediction</h1>
        <p style="opacity: 0.9">ICU Patients with COPD combined with PH</p>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        data_values = np.array([[
            Acute_kidney_injury, Pneumonia, Age, Weight, Heart_rate,
            RDW, WBC, Anion_gap, Chloride, Creatinine, Base_excess, PH
        ]])
        df_input = pd.DataFrame(data_values, columns=FEATURE_NAMES)
        
        with st.spinner("Calculating risk score..."):
            pred_probs = classifier1.predict_proba(df_input)
            prob_pos = pred_probs[0][1]
            prob_percent = prob_pos * 100
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown(f"""
            <div class="result-box" style="border-left: 10px solid #005c97;">
                <div style="color:#666; font-size:1.1rem; font-weight:600;">ESTIMATED 28-DAY MORTALITY PROBABILITY</div>
                <div class="result-value">{prob_percent:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###")

        with st.spinner("Generating consistent interpretations..."):
            explainer = shap.TreeExplainer(classifier1)
            shap_values_full = explainer.shap_values(df_input)

            if isinstance(shap_values_full, list):
                shap_vals = shap_values_full[1][0]
            else:
                shap_vals = shap_values_full[0, :, 1] if len(shap_values_full.shape) == 3 else shap_values_full[0]
            shap_vals = np.array(shap_vals)

            current_base = explainer.expected_value
            if hasattr(current_base, '__len__'): current_base = current_base[1]
            final_base = float(current_base)
            final_values = shap_vals

            explanation = shap.Explanation(
                values=final_values,
                base_values=final_base,
                data=df_input.iloc[0, :].values,
                feature_names=FEATURE_NAMES
            )

            with st.container():
                st.markdown('<div class="chart-title">Force Plot</div>', unsafe_allow_html=True)
                
                plt.figure(figsize=(24, 5))
                shap.force_plot(
                    final_base, 
                    final_values, 
                    df_input.iloc[0, :], 
                    matplotlib=True, show=False, text_rotation=0
                )
                
                fig = plt.gcf()
                ax = plt.gca()
                for txt in ax.texts:
                    if "f(x)" in txt.get_text(): txt.set_visible(False)
                    
                    try:
                        float(txt.get_text())
                        txt.set_fontsize(22)
                        txt.set_fontweight('bold')
                        txt.set_color('#333')
                    except: pass      
                st.pyplot(fig, bbox_inches='tight')
                plt.clf()
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="chart-title">Waterfall Plot</div>', unsafe_allow_html=True)
                
                fig_waterfall, ax = plt.subplots(figsize=(10, 8))
                shap.plots.waterfall(explanation, max_display=12, show=False)
                st.pyplot(fig_waterfall, bbox_inches='tight')
                plt.clf()
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("<br><br><h3 style='text-align:center; color:#999;'>⬅️ Enter data to start analysis</h3>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
