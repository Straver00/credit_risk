import numpy as np
import onnxruntime as ort
from .encoder import codificar_formulario_modificado
from utils.visuals import user_hist_to_file, probability_to_score
import pandas as pd
import joblib

THRESHOLD = 0.4046

def crear_session_optimizada():
    providers = ['CPUExecutionProvider']
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena = False
    sess_options.enable_mem_pattern = False
    return ort.InferenceSession("modelo/model.onnx", sess_options=sess_options, providers=providers)

def cargar_scaler():
    return joblib.load("modelo/scalerMinMax.pkl")

def realizar_prediccion(form_data, session, scale, input_name, output_name, valores_df):
    datos_codificados = codificar_formulario_modificado(form_data)

    if not isinstance(datos_codificados, (list, np.ndarray)):
        raise ValueError("codificar_formulario_modificado debe devolver una lista o array")

    entrada = np.array([datos_codificados], dtype=np.float32)

    print("entrada shape:", entrada.shape)
    print("entrada[:, :2] shape:", entrada[:, :2].shape)
    print("scaler expects:", scale.n_features_in_)

    entrada_escalada = entrada.copy()
    entrada_escalada[:, :2] = scale.transform(entrada[:, :2])

    print("entrada escalada:", entrada_escalada)

    prediccion = session.run([output_name], {input_name: entrada_escalada})[0][0][0]
    aprobado = prediccion <= THRESHOLD
    credit_score = float(probability_to_score(prediccion))
    valores = valores_df.values.reshape(-1)
    histograma_path = user_hist_to_file(prediccion, valores)

    return {
        'score': float(prediccion),
        'approved': aprobado,
        'threshold': THRESHOLD,
        'credit_score': round(credit_score, 2),
        'histograma_path': histograma_path
    }
