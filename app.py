from flask import Flask, render_template, request
from keras.models import load_model
import joblib
import numpy as np


app = Flask(__name__)

def codificar_formulario_modificado(data):
    resultado = []

    # Numéricos directos
    resultado.append(data["monto"])
    resultado.append(data["ingresos"])

    # Binario: ingresos verificables
    resultado.append(1 if data["ingresos_verificables"] else -1)

    # Tiempo de pago
    resultado.append(1 if data["tiempo_pago"] == 36 else -1)

    # Empleo (orden estricto del trabajo)
    orden_empleo = ["1", "10+", "2", "3", "4", "5", "6", "7", "8", "9", "<1"]
    if data["tiempo_trabajo"] not in orden_empleo:
        resultado.extend([-1] * len(orden_empleo))
    else:
        for cat in orden_empleo:
            resultado.append(1 if data["tiempo_trabajo"] == cat else -1)

    # Propiedad de la vivienda
    tipos_propiedad = ["hipoteca", "propia", "alquiler"]
    if data["tipo_propiedad"] not in tipos_propiedad:
        resultado.extend([-1] * len(tipos_propiedad))
    else:
        for tipo in tipos_propiedad:
            resultado.append(1 if data["tipo_propiedad"] == tipo else -1)

    # Propósito del préstamo
    propositos = [
        "carro", "tarjeta_credito", "consolidar_debito", "educativo",
        "mejorar_casa", "comprar_casa", "compra_importante", "salud",
        "mudanza", "energia_renovable", "microempresa", "vacaciones", "boda"
    ]
    if data["proposito_prestamo"] not in propositos:
        resultado.extend([-1] * len(propositos))
    else:
        for p in propositos:
            resultado.append(1 if data["proposito_prestamo"] == p else -1)

    return resultado

modelo = load_model("modelo/bestModel.keras")
scale = joblib.load("modelo/scalerMinMax.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    prediccion = None
    entrada = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        form_data["ingresos_verificables"] = "ingresos_verificables" in request.form

        # Preprocesamiento
        datos_codificados = codificar_formulario_modificado(form_data)
        entrada = np.array([datos_codificados], dtype='float32')
        entrada[:, :2] = scale.transform(entrada[:, :2])

        # Predicción binaria
        prediccion = modelo.predict(entrada)[0][0]
        # si es 0 es aprobado, si es 1 es rechazado
        resultado =  True if prediccion <= 0.4046 else False

    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
      app.run(debug=True)
