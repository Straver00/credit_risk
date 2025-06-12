# Script para convertir (ejecutar una vez)
import tensorflow as tf
import tf2onnx

model = tf.keras.models.load_model("modelo/bestModel.keras")
onnx_model, _ = tf2onnx.convert.from_keras(model)
with open("modelo/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())