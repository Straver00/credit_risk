# disable oneDNN optimizations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
from utils.predictor import crear_session_optimizada, cargar_scaler, realizar_prediccion
from utils.visuals import user_hist_to_file
import pandas as pd

app = Flask(__name__)

# Inicialización de modelo y scaler
session = crear_session_optimizada()
if session is None:
    raise Exception("No se pudo cargar el modelo ONNX")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
scale = cargar_scaler()



@app.route("/", methods=["GET", "POST"])
def index():
    valores_df = pd.read_csv("data/probabilities.csv")
    resultado = None
    prediccion_score = None
    credit_score = None
    histograma = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        form_data["ingresos_verificables"] = "ingresos_verificables" in request.form
        prediction_result = realizar_prediccion(form_data, session, scale, input_name, output_name, valores_df)


        if prediction_result:
            resultado = prediction_result['approved']
            prediccion_score = prediction_result['score']
            credit_score = prediction_result['credit_score']
            histograma = prediction_result['histograma_path']
    return render_template("index.html",
                           resultado=resultado,
                           prediccion_score=prediccion_score,
                           credit_score=credit_score,
                           histograma=histograma)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
