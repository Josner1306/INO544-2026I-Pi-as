import tensorflow as tf

print("Cargando modelo Keras...")
model = tf.keras.models.load_model('model/modelo_pinas.h5')
print("Modelo cargado. Guardando como SavedModel...")
tf.saved_model.save(model, 'model/saved_model')
print("✅ SavedModel guardado en model/saved_model")