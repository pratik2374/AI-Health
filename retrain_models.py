"""
Script to retrain models with current scikit-learn version
This helps resolve compatibility issues with pickle files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

def retrain_heart_model():
    """Retrain heart disease prediction model"""
    print("🫀 Retraining Heart Disease Model...")
    
    # Load data
    df = pd.read_csv('dataset/heart.csv')
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost model (same as original)
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Heart Model Accuracy: {accuracy:.3f}")
    
    # Save with joblib (more reliable than pickle)
    joblib.dump(model, 'models/heart_model_joblib.pkl')
    print("💾 Heart model saved as heart_model_joblib.pkl")
    
    return model

def retrain_kidney_model():
    """Retrain kidney disease prediction model"""
    print("🫘 Retraining Kidney Disease Model...")
    
    # Load data
    df = pd.read_csv('dataset/kidney_disease.csv')
    
    # Remove id column
    df = df.drop('id', axis=1)
    
    # Handle missing values and categorical variables
    # This is a simplified version - you may need to adjust based on your original preprocessing
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from categorical columns
    if 'classification' in categorical_cols:
        categorical_cols.remove('classification')
    
    # Fill missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    
    # Convert categorical variables to string to avoid mixed types
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    # Prepare features and target
    X = df.drop('classification', axis=1)
    y = df['classification'].map({'ckd': 1, 'notckd': 0})
    
    # Remove rows with NaN target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Create preprocessing steps
    numerical_transformer = StandardScaler()
    categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Create full pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
    ])
    
    # Train pipeline
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Kidney Model Accuracy: {accuracy:.3f}")
    
    # Save with joblib
    joblib.dump(pipeline, 'models/kidney_pipeline_joblib.pkl')
    print("💾 Kidney model saved as kidney_pipeline_joblib.pkl")
    
    return pipeline

if __name__ == "__main__":
    print("🚀 Starting Model Retraining Process...")
    print("=" * 50)
    
    try:
        # Retrain heart model
        heart_model = retrain_heart_model()
        print()
        
        # Retrain kidney model
        kidney_model = retrain_kidney_model()
        print()
        
        print("🎉 All models retrained successfully!")
        print("\n📝 Next steps:")
        print("1. Update app.py to use the new joblib models")
        print("2. Or install scikit-learn==1.6.1 to use original models")
        
    except Exception as e:
        print(f"❌ Error during retraining: {e}")
        print("\n💡 Alternative solution:")
        print("Run: pip install scikit-learn==1.6.1")
