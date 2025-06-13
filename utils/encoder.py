def codificar_formulario_modificado(data):
    resultado = []

    resultado.append(float(data["monto"]))
    resultado.append(float(data["ingresos"]))
    resultado.append(1.0 if data["ingresos_verificables"] else -1.0)
    resultado.append(1.0 if str(data["tiempo_pago"]) == "36" else -1.0)

    orden_empleo = ["1", "10+", "2", "3", "4", "5", "6", "7", "8", "9", "<1"]
    for cat in orden_empleo:
        resultado.append(1.0 if data["tiempo_trabajo"] == cat else -1.0)

    tipos_propiedad = ["hipoteca", "propia", "alquiler"]
    for tipo in tipos_propiedad:
        resultado.append(1.0 if data["tipo_propiedad"] == tipo else -1.0)

    propositos = [
        "carro", "tarjeta_credito", "consolidar_debito", "educativo",
        "mejorar_casa", "comprar_casa", "compra_importante", "salud",
        "mudanza", "energia_renovable", "microempresa", "vacaciones", "boda"
    ]
    for p in propositos:
        resultado.append(1.0 if data["proposito_prestamo"] == p else -1.0)

    return resultado
