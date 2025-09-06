import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Health Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .safe-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .risk-prediction {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    .high-risk-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    }
    
    /* Use default Streamlit colors for better readability */
    .stSelectbox > div > div {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stNumberInput > div > div > input {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stSelectbox label {
        color: var(--text-color) !important;
    }
    
    .stNumberInput label {
        color: var(--text-color) !important;
    }
    
    /* Ensure dropdown text is visible */
    .stSelectbox [data-baseweb="select"] {
        color: var(--text-color) !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: var(--text-color) !important;
    }
    
    /* Remove sidebar gradient to use default colors */
    .sidebar .sidebar-content {
        background: var(--background-color);
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_data
def load_models():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        import joblib
        heart_model = joblib.load('models/heart_model_joblib.pkl')
        kidney_model = joblib.load('models/kidney_pipeline_joblib.pkl')
        #st.sidebar.success("✅ Models loaded successfully!")
        return heart_model, kidney_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.warning("""
        **Model Loading Issue Detected**
        
        Please run the retraining script to create compatible models:
        ```bash
        python retrain_models.py
        ```
        """)
        return None, None

# Load datasets for visualization
@st.cache_data
def load_datasets():
    try:
        heart_data = pd.read_csv('dataset/heart.csv')
        kidney_data = pd.read_csv('dataset/kidney_disease.csv')
        return heart_data, kidney_data
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None, None

# Main header
st.markdown('<h1 class="main-header">🏥 Health Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Predict Heart Disease and Kidney Disease with AI</p>', unsafe_allow_html=True)

# Load models and data
heart_model, kidney_model = load_models()
heart_data, kidney_data = load_datasets()

if heart_model is None or kidney_model is None:
    st.error("❌ Unable to load models. Please check if the model files exist in the models folder.")
    st.stop()

# Sidebar
st.sidebar.markdown("## 🎯 Prediction Options")
prediction_type = st.sidebar.selectbox(
    "Select Prediction Type",
    ["Heart Disease", "Kidney Disease"],
    help="Choose which disease prediction model to use"
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Dataset Overview")

if heart_data is not None and kidney_data is not None:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Heart Dataset", f"{len(heart_data)} samples")
    with col2:
        st.metric("Kidney Dataset", f"{len(kidney_data)} samples")

st.sidebar.markdown("---")
st.sidebar.markdown("## 💡 About This Tool")

st.sidebar.markdown("""
**🏥 Health Prediction Dashboard**

This AI-powered tool helps predict the risk of heart disease and kidney disease using advanced machine learning algorithms trained on medical datasets.

**✨ Key Features:**
- **Real-time Predictions** with confidence scores
- **Comprehensive Input Forms** for detailed analysis
- **Interactive Visualizations** for data exploration
- **Professional Medical Interface** designed for healthcare use

**🔬 Model Accuracy:**
- Heart Disease Model: ~82% accuracy
- Kidney Disease Model: Trained on clinical parameters
- Both models use validated medical datasets

**⚠️ Important Note:**
This tool is designed for educational and screening purposes only. Always consult healthcare professionals for medical decisions and proper diagnosis.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 How to Use")

st.sidebar.markdown("""
1. **Select Model Type** from the dropdown above
2. **Fill in Patient Data** using the input forms
3. **Click Predict** to get instant results
4. **Review Confidence Scores** and probabilities
5. **Consult Healthcare Provider** for medical advice

**📋 Data Required:**
- **Heart Disease**: 13 medical parameters
- **Kidney Disease**: 24 clinical features across 3 categories
""")

# Main content area
if prediction_type == "Heart Disease":
    st.markdown('<h2 class="sub-header">❤️ Heart Disease Prediction</h2>', unsafe_allow_html=True)
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient's age in years")
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", help="Patient's gender")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                         help="Type of chest pain experienced")
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=250, value=120, 
                                  help="Resting blood pressure in mm Hg")
        chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200, 
                              help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar", [0, 1], format_func=lambda x: "Normal" if x == 0 else "High", 
                          help="Fasting blood sugar > 120 mg/dl")
    
    with col2:
        st.markdown("### 🔬 Medical Test Results")
        restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                              format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x],
                              help="Resting electrocardiographic results")
        thalach = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150, 
                                 help="Maximum heart rate achieved")
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", 
                            help="Exercise induced angina")
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, 
                                 help="ST depression induced by exercise relative to rest")
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], 
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                            help="Slope of the peak exercise ST segment")
        ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3], 
                         help="Number of major vessels colored by flourosopy")
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x],
                           help="Thalassemia type")
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease", type="primary", use_container_width=True):
        # Prepare input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Make prediction
        prediction = heart_model.predict(input_data)[0]
        probability = heart_model.predict_proba(input_data)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if prediction == 0:
                st.markdown('<div class="prediction-card safe-prediction">', unsafe_allow_html=True)
                st.markdown("### ✅ No Heart Disease Risk")
                st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-card risk-prediction">', unsafe_allow_html=True)
                st.markdown("### ⚠️ Heart Disease Risk Detected")
                st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Show probability breakdown
        fig = go.Figure(data=[
            go.Bar(x=['No Disease', 'Heart Disease'], y=[probability[0]*100, probability[1]*100],
                   marker_color=['#4facfe', '#fa709a'])
        ])
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Outcome",
            yaxis_title="Probability (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif prediction_type == "Kidney Disease":
    st.markdown('<h2 class="sub-header">🫘 Kidney Disease Prediction</h2>', unsafe_allow_html=True)
    
    # Create tabs for different input categories
    tab1, tab2, tab3 = st.tabs(["📋 Basic Info", "🔬 Lab Results", "🏥 Medical History"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50, help="Patient's age in years")
            bp = st.number_input("Blood Pressure", min_value=50, max_value=200, value=80, help="Blood pressure in mm Hg")
            sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.1, value=1.02, step=0.01, 
                                help="Specific gravity of urine")
            al = st.number_input("Albumin", min_value=0, max_value=5, value=1, help="Albumin level (0-5)")
            su = st.number_input("Sugar", min_value=0, max_value=5, value=0, help="Sugar level (0-5)")
        
        with col2:
            rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"], help="Red blood cell count")
            pc = st.selectbox("Pus Cells", ["normal", "abnormal"], help="Pus cell presence")
            pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"], help="Pus cell clumps")
            ba = st.selectbox("Bacteria", ["notpresent", "present"], help="Bacterial presence")
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            bgr = st.number_input("Blood Glucose Random", min_value=50, max_value=500, value=121, 
                                 help="Random blood glucose level")
            bu = st.number_input("Blood Urea", min_value=10, max_value=200, value=36, help="Blood urea level")
            sc = st.number_input("Serum Creatinine", min_value=0.5, max_value=20.0, value=1.2, step=0.1, 
                                help="Serum creatinine level")
            sod = st.number_input("Sodium", min_value=100, max_value=200, value=142, help="Sodium level")
        
        with col2:
            pot = st.number_input("Potassium", min_value=2.0, max_value=8.0, value=4.0, step=0.1, 
                                 help="Potassium level")
            hemo = st.number_input("Hemoglobin", min_value=5.0, max_value=20.0, value=15.4, step=0.1, 
                                  help="Hemoglobin level")
            pcv = st.number_input("Packed Cell Volume", min_value=20, max_value=60, value=44, 
                                 help="Packed cell volume")
            wc = st.number_input("White Blood Cell Count", min_value=2000, max_value=20000, value=7800, 
                                help="White blood cell count")
            rc = st.number_input("Red Blood Cell Count", min_value=2.0, max_value=8.0, value=5.2, step=0.1, 
                                help="Red blood cell count")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            htn = st.selectbox("Hypertension", ["no", "yes"], help="History of hypertension")
            dm = st.selectbox("Diabetes Mellitus", ["no", "yes"], help="History of diabetes")
            cad = st.selectbox("Coronary Artery Disease", ["no", "yes"], help="History of CAD")
            appet = st.selectbox("Appetite", ["good", "poor"], help="Appetite status")
        
        with col2:
            pe = st.selectbox("Pedal Edema", ["no", "yes"], help="Pedal edema presence")
            ane = st.selectbox("Anemia", ["no", "yes"], help="Anemia presence")
    
    # Prediction button
    if st.button("🔮 Predict Kidney Disease", type="primary", use_container_width=True):
        # Prepare input data (simplified for demonstration)
        # Note: You may need to adjust this based on your actual model's preprocessing
        try:
            # Create a sample input - you'll need to adjust this based on your model's requirements
            input_data = pd.DataFrame({
                'age': [age],
                'bp': [bp],
                'sg': [sg],
                'al': [al],
                'su': [su],
                'rbc': [rbc],
                'pc': [pc],
                'pcc': [pcc],
                'ba': [ba],
                'bgr': [bgr],
                'bu': [bu],
                'sc': [sc],
                'sod': [sod],
                'pot': [pot],
                'hemo': [hemo],
                'pcv': [pcv],
                'wc': [wc],
                'rc': [rc],
                'htn': [htn],
                'dm': [dm],
                'cad': [cad],
                'appet': [appet],
                'pe': [pe],
                'ane': [ane]
            })
            
            # Make prediction
            prediction = kidney_model.predict(input_data)[0]
            probability = kidney_model.predict_proba(input_data)[0]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col2:
                if prediction == 0:  # Assuming 0 = no disease, 1 = disease
                    st.markdown('<div class="prediction-card safe-prediction">', unsafe_allow_html=True)
                    st.markdown("### ✅ No Kidney Disease")
                    st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-card risk-prediction">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ Kidney Disease Detected")
                    st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Show probability breakdown
            fig = go.Figure(data=[
                go.Bar(x=['No Disease', 'Kidney Disease'], y=[probability[0]*100, probability[1]*100],
                       marker_color=['#4facfe', '#fa709a'])
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Outcome",
                yaxis_title="Probability (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.info("Please check if all required fields are filled correctly.")

# Data visualization section
if heart_data is not None and kidney_data is not None:
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📊 Dataset Visualizations</h2>', unsafe_allow_html=True)
    
    viz_tab1, viz_tab2 = st.tabs(["Heart Disease Data", "Kidney Disease Data"])
    
    with viz_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(heart_data, x='age', nbins=20, title="Age Distribution")
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Target distribution
            target_counts = heart_data['target'].value_counts()
            fig_target = px.pie(values=target_counts.values, names=['No Disease', 'Heart Disease'], 
                               title="Disease Distribution")
            fig_target.update_layout(height=400)
            st.plotly_chart(fig_target, use_container_width=True)
    
    with viz_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(kidney_data, x='age', nbins=20, title="Age Distribution")
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Classification distribution
            class_counts = kidney_data['classification'].value_counts()
            fig_class = px.pie(values=class_counts.values, names=class_counts.index, 
                              title="Disease Classification")
            fig_class.update_layout(height=400)
            st.plotly_chart(fig_class, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>🏥 Health Prediction Dashboard | Built with Streamlit & Machine Learning</p>
    <p><small>⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.</small></p>
</div>
""", unsafe_allow_html=True)
