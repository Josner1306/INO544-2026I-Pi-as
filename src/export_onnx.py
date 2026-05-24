import tensorflow as tf
import tf2onnx
import os

# Cargar el modelo Keras
print("Cargando modelo Keras...")
model = tf.keras.models.load_model('model/modelo_pinas.h5')
print("Modelo cargado correctamente.")

# Definir la función concreta con la forma de entrada exacta
@tf.function
def model_function(x):
    return model(x)

# Especificar el tensor de entrada: [1, 224, 224, 3] (batch=1, 224x224, RGB)
input_signature = tf.TensorSpec(shape=[1, 224, 224, 3], dtype=tf.float32, name='input')
concrete_func = model_function.get_concrete_function(input_signature)

# Convertir a ONNX
print("Convirtiendo a ONNX...")
onnx_model, _ = tf2onnx.convert.from_function(
    concrete_func,
    opset=13,
    output_path=None
)

# Guardar el archivo ONNX
os.makedirs('model', exist_ok=True)
output_path = 'model/pinas_equipo.onnx'
with open(output_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

print(f"✅ Exportación exitosa. Archivo guardado en: {output_path}")
print(f"   Tamaño: {os.path.getsize(output_path) / 1024:.2f} KB")