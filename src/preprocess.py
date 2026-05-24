import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ================= CONFIGURACIÓN =================
INPUT_DIR = "data/"          # carpeta donde están tus 400 imágenes de piñas
OUTPUT_DIR = "dataset/"      # donde se guardarán los archivos .npy
IMG_SIZE = (224, 224)        # tamaño estándar para la CNN
TEST_RATIO = 0.2             # 20% para prueba
VAL_RATIO = 0.1              # 10% del total para validación (sobre el entrenamiento restante)

# ================= CARGAR IMÁGENES =================
def load_images(folder):
    images = []
    print(f"Buscando imágenes en: {folder}")
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            if img is None:
                print(f"  ⚠️ No se pudo leer: {file}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convertir a RGB
            img = cv2.resize(img, IMG_SIZE)             # redimensionar
            images.append(img)
    print(f"✅ Cargadas {len(images)} imágenes correctamente.")
    return np.array(images)

# Cargar todas las imágenes (asumimos que todas son piñas -> clase positiva = 1)
X = load_images(INPUT_DIR)
y = np.ones(len(X))   # vector de etiquetas: 1 = piña

if len(X) == 0:
    print("❌ No se encontraron imágenes en 'data/'. Verifica la ruta.")
    exit(1)

# ================= DIVIDIR EN TRAIN, VAL, TEST =================
# Primero separamos train+val del test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=42
)

# Luego del train+val, separamos train y val (90% y 10% de train_val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VAL_RATIO, random_state=42
)

print(f"📊 División final:")
print(f"   Entrenamiento: {X_train.shape[0]} imágenes")
print(f"   Validación:   {X_val.shape[0]} imágenes")
print(f"   Prueba:       {X_test.shape[0]} imágenes")

# ================= GUARDAR EN FORMATO .NPY =================
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print(f"✅ Datos guardados en: {OUTPUT_DIR}")