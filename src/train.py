import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# ================= CARGAR DATOS =================
print("Cargando datos desde dataset/...")
X_train = np.load("dataset/X_train.npy")
y_train = np.load("dataset/y_train.npy")
X_val = np.load("dataset/X_val.npy")
y_val = np.load("dataset/y_val.npy")
X_test = np.load("dataset/X_test.npy")
y_test = np.load("dataset/y_test.npy")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ================= NORMALIZAR =================
# Convertir píxeles de [0,255] a [0,1] para mejor entrenamiento
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# ================= DATA AUGMENTATION =================
# Se aplica SOLO al conjunto de entrenamiento, en tiempo real
datagen = ImageDataGenerator(
    rotation_range=20,          # rotación aleatoria hasta ±20°
    zoom_range=0.15,            # zoom aleatorio entre 0.85 y 1.15
    brightness_range=[0.8, 1.2], # brillo entre 80% y 120%
    horizontal_flip=True,       # espejeado horizontal
    width_shift_range=0.1,      # desplazamiento horizontal ±10%
    height_shift_range=0.1      # desplazamiento vertical ±10%
)

# Ajustar el generador a los datos de entrenamiento
datagen.fit(X_train)

# ================= CONSTRUIR EL MODELO CNN =================
model = models.Sequential([
    # Primera capa convolucional + pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    
    # Segunda capa convolucional + pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Tercera capa convolucional + pooling
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Clasificador
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),          # Reduce overfitting
    layers.Dense(1, activation='sigmoid')   # Salida binaria (piña o no piña)
])

# ================= HIPERPARÁMETROS =================
LEARNING_RATE = 0.001   # Tasa de aprendizaje elegida (justificar en README)
EPOCHS = 20
BATCH_SIZE = 32

# Compilar el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Resumen del modelo (opcional, para ver las capas)
model.summary()

# ================= ENTRENAMIENTO =================
print("Iniciando entrenamiento...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    verbose=1
)

# ================= EVALUACIÓN EN TEST =================
print("\nEvaluando en conjunto de prueba...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Precisión en test: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"✅ Pérdida en test:   {test_loss:.4f}")

# ================= GUARDAR GRÁFICAS =================
plt.figure(figsize=(12, 4))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_accuracy'], label='Validación', marker='o')
plt.title('Precisión del modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento', marker='o')
plt.plot(history.history['val_loss'], label='Validación', marker='o')
plt.title('Pérdida del modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# Guardar la figura
os.makedirs('src', exist_ok=True)
plt.savefig('src/grafica_rendimiento.png', dpi=150, bbox_inches='tight')
print("📊 Gráfica guardada en src/grafica_rendimiento.png")

# Mostrar la gráfica (opcional, puede no mostrarse en CMD)
plt.show()

# ================= GUARDAR MODELO (KERAS) =================
os.makedirs('model', exist_ok=True)
model.save('model/modelo_pinas.h5')
print("💾 Modelo guardado en model/modelo_pinas.h5")