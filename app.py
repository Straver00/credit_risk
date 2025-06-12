# disable oneDNN optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
import json

app = Flask(__name__)

class ScalerLigero:
    """Scaler ligero que reemplaza joblib"""
    def __init__(self, archivo_params):
        with open(archivo_params, 'r') as f:
            params = json.load(f)
        self.min_ = np.array(params["min_"], dtype=np.float32)
        self.scale_ = np.array(params["scale_"], dtype=np.float32)
    
    def transform(self, X):
        return (X - self.min_) * self.scale_

def codificar_formulario_modificado(data):
    """Codifica los datos del formulario manteniendo la misma lógica"""
    resultado = []

    # Numéricos directos
    resultado.append(float(data["monto"]))
    resultado.append(float(data["ingresos"]))

    # Binario: ingresos verificables
    resultado.append(1.0 if data["ingresos_verificables"] else -1.0)

    # Tiempo de pago
    resultado.append(1.0 if str(data["tiempo_pago"]) == "36" else -1.0)

    # Empleo (orden estricto del trabajo)
    orden_empleo = ["1", "10+", "2", "3", "4", "5", "6", "7", "8", "9", "<1"]
    if data["tiempo_trabajo"] not in orden_empleo:
        resultado.extend([-1.0] * len(orden_empleo))
    else:
        for cat in orden_empleo:
            resultado.append(1.0 if data["tiempo_trabajo"] == cat else -1.0)

    # Propiedad de la vivienda
    tipos_propiedad = ["hipoteca", "propia", "alquiler"]
    if data["tipo_propiedad"] not in tipos_propiedad:
        resultado.extend([-1.0] * len(tipos_propiedad))
    else:
        for tipo in tipos_propiedad:
            resultado.append(1.0 if data["tipo_propiedad"] == tipo else -1.0)

    # Propósito del préstamo
    propositos = [
        "carro", "tarjeta_credito", "consolidar_debito", "educativo",
        "mejorar_casa", "comprar_casa", "compra_importante", "salud",
        "mudanza", "energia_renovable", "microempresa", "vacaciones", "boda"
    ]
    if data["proposito_prestamo"] not in propositos:
        resultado.extend([-1.0] * len(propositos))
    else:
        for p in propositos:
            resultado.append(1.0 if data["proposito_prestamo"] == p else -1.0)

    return resultado

def realizar_prediccion(form_data):
    """Función centralizada para realizar predicciones"""
    try:
        # Preprocesamiento
        datos_codificados = codificar_formulario_modificado(form_data)
        entrada = np.array([datos_codificados], dtype=np.float32)
        
        # Escalar solo las primeras 2 columnas
        entrada_escalada = entrada.copy()
        entrada_escalada[:, :2] = scale.transform(entrada[:, :2])

        # Predicción con ONNX
        prediccion = session.run(
            [output_name], 
            {input_name: entrada_escalada}
        )[0][0][0]

        # Aplicar umbral
        resultado = True if prediccion <= 0.4046 else False
        
        print(f"Predicción: {prediccion:.4f} -> {'Aprobado' if resultado else 'Rechazado'}")
        
        return {
            'score': float(prediccion),
            'approved': resultado,
            'threshold': 0.4046
        }
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        import traceback
        traceback.print_exc()
        return None

# Configuración optimizada de ONNX Runtime
def crear_session_optimizada():
    """Crea una sesión ONNX optimizada"""
    providers = ['CPUExecutionProvider']
    
    # Opciones de optimización
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena = False  # Reduce uso de memoria
    sess_options.enable_mem_pattern = False
    
    try:
        session = ort.InferenceSession(
            "modelo/model.onnx", 
            sess_options=sess_options,
            providers=providers
        )
        return session
    except Exception as e:
        print(f"Error cargando modelo ONNX: {e}")
        return None

# Cargar modelo y scaler una sola vez al iniciar
session = crear_session_optimizada()

# Verificar que el modelo se cargó correctamente
if session is None:
    raise Exception("No se pudo cargar el modelo ONNX")

# Obtener información del modelo
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape

print(f"✅ Modelo cargado:")
print(f"   Input: {input_name} {input_shape}")
print(f"   Output: {output_name}")

# Cargar scaler optimizado
try:
    scale = ScalerLigero("modelo/scaler_params.json")
    print("✅ Scaler cargado")
except Exception as e:
    print(f"Error cargando scaler: {e}")
    # Fallback a joblib si existe
    try:
        import joblib
        scale = joblib.load("modelo/scalerMinMax.pkl")
        print("⚠️  Usando joblib como fallback")
    except:
        raise Exception("No se pudo cargar ningún scaler")

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    prediccion_score = None
    
    if request.method == "POST":
        # Procesar datos del formulario
        form_data = request.form.to_dict()
        form_data["ingresos_verificables"] = "ingresos_verificables" in request.form

        # Realizar predicción
        prediction_result = realizar_prediccion(form_data)
        
        if prediction_result:
            resultado = prediction_result['approved']
            prediccion_score = prediction_result['score']

    return render_template("index.html", 
                         resultado=resultado, 
                         prediccion_score=prediccion_score)

if __name__ == "__main__":

    app.run(debug=True)