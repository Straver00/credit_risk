import json
import numpy as np

class ScalerLigero:
    """Scaler ligero que reemplaza joblib"""
    def __init__(self, archivo_params):
        with open(archivo_params, 'r') as f:
            params = json.load(f)
        self.min_ = np.array(params["min_"], dtype=np.float32)
        self.scale_ = np.array(params["scale_"], dtype=np.float32)
    
    def transform(self, X):
        return (X - self.min_) * self.scale_
