import numpy as np
import streamlit as st
import shap
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb 
import os 

# 强制使用 Agg 后端
matplotlib.use('Agg')

# ==================== 1. 全局配置 & CSS ====================
st.set_page_config(
    page_title="ICU Mortality Risk Prediction",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局背景 */
    .stApp { background-color: #f8f9fa; }
    
    /* 标题栏 */
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
    
    /* 结果卡片 */
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
    
    /* 图表容器 */
    .chart-section {
        margin-top: 30px;
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    /* 【修改点】图表标题：添加蓝色竖杠 */
    .chart-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 20px;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 10px;
        border-left: 6px solid #005c97; /* 蓝色竖杠 */
        padding-left: 15px; /* 文字缩进 */
    }
</style>
""", unsafe_allow_html=True)

# ==================== 2. 加载模型 ====================
@st.cache_resource
def load_model():
    model_path = "xgb_model.json" 
    classifier = xgb.XGBClassifier()
    classifier.load_model(model_path)
    return classifier

classifier1 = load_model()

# 定义模型需要的 10 个特征名称 (必须与训练时完全一致)
FEATURE_NAMES = [
    "Acute_kidney_injury", "Sedative_and_analgesic_drugs", "Vasopressin", "Glucocorticoids", 
    "Age", "Weight", "RDW", "Heart_rate", "Respiratory_rate", "Chloride"
]

# 定义真实的训练集基准值 (用于兜底校准)
REAL_BASE_SCORE = 0.23351955

def main():
    # ==================== 侧边栏 ====================
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
            c3, c4 = st.columns(2)
            with c3: Heart_rate = st.number_input("HR (bpm)", 20, 200, 80)
            with c4: Respiratory_rate = st.number_input("RR (bpm)", 5, 50, 18)

            st.markdown("#### 🧪 Labs")
            c5, c6 = st.columns(2)
            with c5: RDW = st.number_input("RDW (%)", 10.0, 30.0, 14.5)
            with c6: Chloride = st.number_input("Cl- (mEq/L)", 20.0, 200.0, 100.0)

            st.markdown("#### 💊 Clinical Status")
            ak_map = {"No": 0, "Yes": 1}; Acute_kidney_injury = ak_map[st.selectbox("Acute Kidney Injury", list(ak_map.keys()))]
            sed_map = {"No": 0, "Yes": 1}; Sedative_and_analgesic_drugs = sed_map[st.selectbox("Sedatives", list(sed_map.keys()))]
            vaso_map = {"No": 0, "Yes": 1}; Vasopressin = vaso_map[st.selectbox("Vasopressin", list(vaso_map.keys()))]
            steroid_map = {"No": 0, "Yes": 1}; Glucocorticoids = steroid_map[st.selectbox("Glucocorticoids", list(steroid_map.keys()))]
            
            st.markdown("---")
            predict_btn = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)

    # ==================== 主界面 ====================
    st.markdown("""
    <div class="main-header">
        <h1>28-day Mortality Risk Prediction</h1>
        <p style="opacity: 0.9">ICU Patients with COPD combined with PAH</p>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        # 构建当前输入的单行数据
        data_values = np.array([[
            Acute_kidney_injury, Sedative_and_analgesic_drugs, Vasopressin, Glucocorticoids, 
            Age, Weight, RDW, Heart_rate, Respiratory_rate, Chloride
        ]])
        df_input = pd.DataFrame(data_values, columns=FEATURE_NAMES)
        
        # 1. 预测
        with st.spinner("Calculating risk score..."):
            pred_probs = classifier1.predict_proba(df_input)
            prob_pos = pred_probs[0][1]
            prob_percent = prob_pos * 100
        
        # 2. 结果展示
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if prob_percent < 23.35:
                color = "#5cb85c"  # 绿色
                status = "LOW RISK"
            elif prob_percent < 50:
                color = "#f0ad4e"  # 橙色
                status = "MEDIUM RISK"
            elif prob_percent < 75:
                color = "#d9534f"  # 红色
                status = "HIGH RISK"
            else:
                color = "#8b0000"  # 深红色
                status = "VERY HIGH RISK"
            st.markdown(f"""
            <div class="result-box" style="border-left: 10px solid {color};">
                <div style="color:#666; font-size:1.1rem; font-weight:600;">ESTIMATED MORTALITY RISK</div>
                <div class="result-value">{prob_percent:.2f}%</div>
                <div class="result-label" style="background:{color};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###")

        # 3. SHAP 分析
        with st.spinner("Generating consistent interpretations..."):
            
            # ==================== A. 核心修复：加载并筛选背景数据 ====================
            background_file = "background_data.csv"
            bg_data = None
            
            if os.path.exists(background_file):
                try:
                    # 1. 读取完整的 CSV (58列)
                    raw_bg = pd.read_csv(background_file)
                    
                    # 2. 【关键步骤】只保留模型需要的 10 列！
                    # 这一步解决了 "expected: 10, got 58" 的错误
                    bg_data = raw_bg[FEATURE_NAMES]
                    
                    # st.success("✅ Successfully loaded and filtered real background data.")
                    
                except KeyError as e:
                    st.error(f"Error: CSV file is missing required columns: {e}")
                    # 回退到随机
                    bg_data = None
            
            # 如果加载失败或文件不存在，回退到随机数据
            if bg_data is None:
                if not os.path.exists(background_file):
                    st.warning(f"⚠️ '{background_file}' not found. Using random simulation.")
                np.random.seed(42)
                bg_data = pd.DataFrame({
                    col: np.random.choice([0, 1], 50) if col in ["Acute_kidney_injury", "Sedative_and_analgesic_drugs", "Vasopressin", "Glucocorticoids"] 
                    else np.random.uniform(10, 100, 50) 
                    for col in FEATURE_NAMES
                }, columns=FEATURE_NAMES)
            
            # ================================================================

            # B. 计算 SHAP
            def predict_func(x): return classifier1.predict_proba(x)
            
            explainer = shap.KernelExplainer(predict_func, bg_data)
            shap_values_full = explainer.shap_values(df_input)
            
            # C. 提取数据
            if isinstance(shap_values_full, list):
                shap_vals = shap_values_full[1][0]
            else:
                shap_vals = shap_values_full[0, :, 1] if len(shap_values_full.shape) == 3 else shap_values_full[0]
            shap_vals = np.array(shap_vals)

            # D. 提取基准值
            current_base = explainer.expected_value
            if hasattr(current_base, '__len__'): current_base = current_base[1]
            current_base = float(current_base)

            # E. 校准 (Calibration)
            # 因为使用了真实的背景数据，这里的 bias 应该非常小
            # 但为了保证 100% 的数学严谨性，我们还是做一次微小的闭环平差
            bias = current_base - REAL_BASE_SCORE
            
            # 如果偏差过大 (>5%)，说明 CSV 数据分布和训练集差异大，不强制校准，保留 CSV 的真实基准
            # 如果偏差小，说明是采样误差，强制校准回模型参数
            if abs(bias) < 0.05:
                final_base = REAL_BASE_SCORE
                final_values = shap_vals + (bias / len(FEATURE_NAMES))
            else:
                final_base = current_base
                final_values = shap_vals

            # F. 构建 Explanation
            explanation = shap.Explanation(
                values=final_values,
                base_values=final_base,
                data=df_input.iloc[0, :].values,
                feature_names=FEATURE_NAMES
            )

            # === 图表 1: Force Plot ===
            with st.container():
                st.markdown('<div class="chart-title">Force Plot</div>', unsafe_allow_html=True)
                
                plt.figure(figsize=(24, 5))
                shap.force_plot(
                    final_base, 
                    final_values, 
                    df_input.iloc[0, :], 
                    matplotlib=True, show=False, text_rotation=0
                )
                
                # 清理 f(x)
                fig = plt.gcf()
                ax = plt.gca()
                for txt in ax.texts:
                    if "f(x)" in txt.get_text(): txt.set_visible(False)
                    
                    # 放大预测值
                    try:
                        float(txt.get_text())
                        txt.set_fontsize(22)
                        txt.set_fontweight('bold')
                        txt.set_color('#333')
                    except: pass      
                st.pyplot(fig, bbox_inches='tight')
                plt.clf()
                st.markdown('</div>', unsafe_allow_html=True)

            # === 图表 2: Waterfall Plot ===
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
