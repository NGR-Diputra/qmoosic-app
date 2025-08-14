import sys
import os
import numpy as np
import joblib

# Tambahkan path ke root folder agar import 'features' bisa dilakukan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.feature_extraction import feature_extraction
from misc.model import OneVsRestSVM_QP, SVM_QP

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, '..', 'model')

mi_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))
pca_transform = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
best_svm_model = joblib.load(os.path.join(MODEL_DIR, "best_svm_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

def predict(file_path):
    spectral_features = feature_extraction(file_path)
    spectral_features = np.array(spectral_features).reshape(1, -1)

    scaled_features = scaler.transform(spectral_features)
    mi_spectral_features = scaled_features[:, mi_features]
    pca_transformed_features = pca_transform.transform(mi_spectral_features)

    prediction = best_svm_model.predict(pca_transformed_features)
    return prediction[0]
