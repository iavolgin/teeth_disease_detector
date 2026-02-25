import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent / 'weights' / 'classificator'

def load_classifier():
    """Загрузка модели и препроцессоров."""
    model = joblib.load(MODEL_DIR / 'model.pkl')
    scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
    ohe = joblib.load(MODEL_DIR / 'ohe.pkl')
    label_encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    
    return model, scaler, ohe, label_encoder


def preprocess_metadata(metadata_list: list, scaler, ohe) -> np.ndarray:
    """
    Предобработка метаданных для классификации.
    
    Args:
        metadata_list: Список словарей от compute_metadata
        scaler: StandardScaler
        ohe: OneHotEncoder
        
    Returns:
        Массив признаков shape (N, 8)
    """
    if not metadata_list:
        return np.array([])
    
    numeric_features = ['x_norm', 'y_norm', 'aspect_ratio', 'neighbours_right', 'neighbours_top']
    
    X_numeric = []
    X_quadrant = []
    
    for meta in metadata_list:
        numeric_row = [meta[feat] for feat in numeric_features]
        X_numeric.append(numeric_row)
        X_quadrant.append([meta['quadrant']])
    
    X_numeric = np.array(X_numeric)
    X_quadrant = np.array(X_quadrant)
    
    X_numeric_scaled = scaler.transform(X_numeric)
    X_quadrant_encoded = ohe.transform(X_quadrant)
    
    X_final = np.hstack([X_numeric_scaled, X_quadrant_encoded])
    
    return X_final


def predict_teeth(metadata_list: list, model, scaler, ohe, label_encoder) -> list:
    """
    Предсказание классов зубов по метаданным.
    
    Args:
        metadata_list: Список словарей от compute_metadata
        model: Обученная модель
        scaler: StandardScaler
        ohe: OneHotEncoder
        label_encoder: LabelEncoder
        
    Returns:
        Список предсказанных меток зубов (например, [11, 12, 13, ...])
    """
    if not metadata_list:
        return []
    
    X = preprocess_metadata(metadata_list, scaler, ohe)
    y_pred_encoded = model.predict(X)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    return y_pred.tolist()