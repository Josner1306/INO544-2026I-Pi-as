# DO YOU TENSORFLOW? — Ves lo que veo 👁️🤖
## IUJO — Feria de Haceres Período I-2026
### Unidad Curricular: INO-544 (Investigación de Operaciones)

---

## 👥 Integrantes y Roles
* **Integrante 1:** [Nombre Completo] - [Cédula] - *Rol: Ingeniero de Datos (Dataset y Preprocesamiento)*
* **Integrante 2:** [Nombre Completo] - [Cédula] - *Rol: Arquitecto de IA (Modelado y Entrenamiento)*
* **Integrante 3:** [Nombre Completo] - [Cédula] - *Rol: Ingeniero de Despliegue (Exportación ONNX y Pruebas)*

---

## 🎯 1. Clase/Tema Seleccionado
* **Tema asignado:** Piñas
* **Descripción del Objeto:** Fruta tropical de cáscara rugosa con escamas hexagonales, corona de hojas espinosas y color que va del verde al amarillo anaranjado cuando madura. Forma ovalada u oblonga.

---

## 📊 2. Gestión del Dataset (Ingeniería de Datos)
* **Cantidad de imágenes originales recopiladas:** 400
* **Estrategia de Data Augmentation aplicada:**
    * *Rotación:* ±20 grados
    * *Zoom:* 0.85 a 1.15
    * *Cambios de Brillo:* 0.8 a 1.2
    * *Otras transformaciones:* Desplazamiento horizontal/vertical (10%), espejeado horizontal.
* **Total de imágenes generadas para el entrenamiento:** Aumentación en tiempo real, cada época ve transformaciones distintas. Equivalente a conjunto virtual infinito.
* **Resolución y formato estandarizado:** 224x224 píxeles, JPG, canales RGB (Formato Tensor: `[1, 224, 224, 3]`).

---

## 🧠 3. Arquitectura del Modelo y Entrenamiento
* **Framework utilizado:** TensorFlow / Keras
* **Descripción de la Red (CNN):** 3 capas convolucionales (32, 64 y 128 filtros) con activación ReLU, cada una seguida de MaxPooling de 2x2. Luego una capa Flatten, una capa densa de 128 neuronas con ReLU, Dropout 0.5 y capa de salida densa con 1 neurona y activación sigmoide.
* **Hiperparámetros óptimos seleccionados:**
    * *Función de pérdida (Loss):* Binary Crossentropy
    * *Optimizador:* Adam
    * *Tasa de Aprendizaje (Learning Rate):* 0.001
    * *Épocas (Epochs):* 20
    * *Tamaño de lote (Batch Size):* 32

### 💡 Justificación Crítica (Control de Autoría)
*Explicación detallada de por qué eligieron esa tasa de aprendizaje y el impacto en las gráficas de pérdida:*

> [Aquí escribirán su justificación REAL después de probar. Ejemplo: "Probamos learning rates de 0.1, 0.01, 0.001 y 0.0001. Con 0.1 la pérdida divergió rápidamente. Con 0.01 hubo oscilaciones. Con 0.001 la pérdida disminuyó de manera estable y la validación siguió de cerca al entrenamiento sin sobreajuste. Por eso seleccionamos 0.001."]

---

## 📈 4. Métricas de Rendimiento (Testing - 20%)
* **Precisión final (Accuracy) en la data de test:** [Ej. 92.4%] (lo pondrás después de entrenar)
* **Pérdida final (Loss) en la data de test:** [Ej. 0.15] (lo pondrás después de entrenar)

![Gráfica de Entrenamiento](src/grafica_rendimiento.png)

---

## ⚙️ 5. Especificación de Exportación ONNX
* **Nombre del archivo:** `model/pinas_equipo.onnx`
* **Tensor de Entrada (Input Shape):** `[1, 224, 224, 3]` (Tipo: `float32`)
* **Tensor de Salida (Output Shape):** `[1, 1]` (Tipo: `float32`)
* **Función de activación final:** Sigmoide (Rango de salida de 0.0 a 1.0 para conversión a porcentaje).

---

## 🚀 6. Instrucciones de Ejecución Local
Para replicar el preprocesamiento y el entrenamiento del modelo:

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/INO544-2026I-Piñas.git
   cd INO544-2026I-Piñas 
