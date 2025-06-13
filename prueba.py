import numpy as np
import joblib
from keras.models import load_model

# Cargar modelo previamente entrenado
modelo = load_model("modelo/bestModel.keras")
scale = joblib.load("modelo/scalerMinMax.pkl")
print(modelo.summary())


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

entrada = codificar_formulario_modificado({
    "monto": 30000,  # monto
    "ingresos": 100000,  # ingreso
    "ingresos_verificables": True,  # verificación
    "tiempo_pago": 36,  # term (36 meses)
    "tiempo_trabajo": "3",  # empleo (3 años)
    "tipo_propiedad": "alquiler",  # propiedad (rentado)
    "proposito_prestamo": "educativo"  # propósito (educativo)
})


# Solo escalar las 2 primeras características numéricas
entrada = np.array([entrada], dtype='float32')
entrada[:, :2] = scale.transform(entrada[:, :2])

# Realizar la predicción
prediccion = modelo.predict(entrada)

# Mostrar resultado
print("Resultado:", prediccion)

from utils.visuals import generate_base_hist_svg
