# 🏥 Health Prediction Dashboard

A beautiful Streamlit web application for predicting Heart Disease and Kidney Disease using machine learning models.

## ✨ Features

- **Heart Disease Prediction**: Predict heart disease risk based on patient data
- **Kidney Disease Prediction**: Predict kidney disease using comprehensive lab results
- **Beautiful UI**: Modern, responsive design with gradient cards and animations
- **Interactive Visualizations**: Data distribution charts and prediction probabilities
- **Real-time Predictions**: Instant results with confidence scores
- **Comprehensive Input Forms**: Organized input fields for different data categories

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── models/                         # Trained ML models
│   ├── heart_model.pkl            # Heart disease prediction model
│   └── kidney_pipeline.pkl        # Kidney disease prediction model
├── dataset/                        # Training datasets
│   ├── heart.csv                  # Heart disease dataset
│   └── kidney_disease.csv         # Kidney disease dataset
└── model training/                 # Jupyter notebooks for model training
    ├── Heart_disease_prediction.ipynb
    └── Kidney_Disease_Prediction.ipynb
```

## 🎯 How to Use

### Heart Disease Prediction
1. Select "Heart Disease" from the sidebar
2. Fill in patient information:
   - Basic demographics (age, sex)
   - Medical history (chest pain type, blood pressure, cholesterol)
   - Test results (ECG, heart rate, exercise tests)
3. Click "Predict Heart Disease" to get instant results

### Kidney Disease Prediction
1. Select "Kidney Disease" from the sidebar
2. Fill in information across three tabs:
   - **Basic Info**: Demographics and basic vitals
   - **Lab Results**: Comprehensive lab test results
   - **Medical History**: Existing conditions and symptoms
3. Click "Predict Kidney Disease" to get instant results

## 🔧 Model Information

### Heart Disease Model Features
- **Input Features**: 13 features including age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Output**: Binary classification (0 = No Disease, 1 = Heart Disease Risk)
- **Confidence**: Probability scores for both outcomes

### Kidney Disease Model Features
- **Input Features**: 24 features including lab results, medical history, and demographics
- **Output**: Binary classification (0 = No Disease, 1 = Kidney Disease)
- **Confidence**: Probability scores for both outcomes

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Gradient Cards**: Beautiful color-coded prediction results
- **Interactive Charts**: Plotly visualizations for data exploration
- **Sidebar Navigation**: Easy switching between prediction types
- **Tabbed Interface**: Organized input forms for complex data
- **Real-time Feedback**: Instant predictions with confidence scores

## ⚠️ Important Notes

- **Educational Purpose**: This tool is designed for educational and demonstration purposes
- **Medical Disclaimer**: Always consult healthcare professionals for actual medical decisions
- **Model Accuracy**: Model performance depends on training data quality and preprocessing
- **Data Privacy**: No patient data is stored or transmitted

## 🛠️ Customization

### Adding New Models
1. Train your model and save as `.pkl` file in the `models/` folder
2. Update the `load_models()` function in `app.py`
3. Add corresponding input fields and prediction logic

### Styling Changes
- Modify the CSS in the `st.markdown()` section at the top of `app.py`
- Update colors, fonts, and layout as needed

### Adding New Visualizations
- Use Plotly for interactive charts
- Add new tabs or sections for additional data views

## 📊 Dataset Information

### Heart Disease Dataset
- **Source**: UCI Machine Learning Repository
- **Size**: 303 samples
- **Features**: 13 medical attributes
- **Target**: Binary classification (presence/absence of heart disease)

### Kidney Disease Dataset
- **Source**: UCI Machine Learning Repository  
- **Size**: 400 samples
- **Features**: 24 clinical attributes
- **Target**: Binary classification (chronic kidney disease)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check that all model files exist in the `models/` folder
2. Verify dataset files are in the `dataset/` folder
3. Ensure all dependencies are installed correctly
4. Check the console for error messages

---

**Built with ❤️ using Streamlit and Machine Learning**
