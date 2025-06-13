import numpy as np
import onnxruntime as ort
from .encoder import codificar_formulario_modificado
from .scaler_ligero import ScalerLigero
from utils.visuals import user_hist_to_file, probability_to_score
import pandas as pd

THRESHOLD = 0.4046

def crear_session_optimizada():
  providers = ['CPUExecutionProvider']
  sess_options = ort.SessionOptions()
  sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
  sess_options.enable_cpu_mem_arena = False
  sess_options.enable_mem_pattern = False

  return ort.InferenceSession("modelo/model.onnx", sess_options=sess_options, providers=providers)

def cargar_scaler():
    return ScalerLigero("modelo/scaler_params.json")

def realizar_prediccion(form_data, session, scale, input_name, output_name, valores_df):
    datos_codificados = codificar_formulario_modificado(form_data)
    entrada = np.array([datos_codificados], dtype=np.float32)
    entrada_escalada = entrada.copy()
    entrada_escalada[:, :2] = scale.transform(entrada[:, :2])

    prediccion = session.run([output_name], {input_name: entrada_escalada})[0][0][0]
    aprobado = prediccion <= THRESHOLD

    # Calcula el score crediticio
    credit_score = float(probability_to_score(prediccion))

    # Usar el DataFrame directamente (valores_df)
    valores = valores_df.values.reshape(-1)

    # Genera el histograma y devuelve la ruta
    histograma_path = user_hist_to_file(prediccion, valores)

    return {
        'score': float(prediccion),
        'approved': aprobado,
        'threshold': THRESHOLD,
        'credit_score': round(credit_score, 2),
        'histograma_path': histograma_path
    }

