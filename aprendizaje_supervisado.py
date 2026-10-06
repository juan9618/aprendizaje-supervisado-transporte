import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import export_text


# ============================================================
# MODELO DE APRENDIZAJE SUPERVISADO
# SISTEMA DE TRANSPORTE MASIVO
# ============================================================


# ============================================================
# 1. CARGAR EL DATASET
# ============================================================

print("=" * 60)
print("      MODELO DE APRENDIZAJE SUPERVISADO")
print("      SISTEMA DE TRANSPORTE MASIVO")
print("=" * 60)

print("\nCargando dataset...")

datos = pd.read_csv("rutas_transporte.csv")

print("Dataset cargado correctamente.")

print("\nPrimeros registros:")
print(datos.head())


# ============================================================
# 2. INFORMACIÓN DEL DATASET
# ============================================================

print("\n" + "=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)

print("\nCantidad de registros:", len(datos))

print("\nColumnas disponibles:")

for columna in datos.columns:
    print("-", columna)


# ============================================================
# 3. SEPARAR VARIABLES
# ============================================================

# Variables utilizadas para realizar la predicción.

X = datos[
    [
        "distancia_km",
        "tiempo_min",
        "transbordos",
        "congestion",
        "demanda",
        "hora_pico",
        "costo"
    ]
]


# Variable objetivo.

y = datos["ruta_recomendada"]


print("\n" + "=" * 60)
print("VARIABLES DEL MODELO")
print("=" * 60)

print("\nVariables de entrada:")

print(
    "distancia_km, tiempo_min, transbordos,"
)

print(
    "congestion, demanda, hora_pico, costo"
)

print("\nVariable objetivo: ruta_recomendada")


# ============================================================
# 4. DIVIDIR LOS DATOS
# ============================================================

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("DIVISIÓN DEL DATASET")
print("=" * 60)

print(
    "\nDatos utilizados para entrenamiento:",
    len(X_entrenamiento)
)

print(
    "Datos utilizados para prueba:",
    len(X_prueba)
)


# ============================================================
# 5. CREAR EL ÁRBOL DE DECISIÓN
# ============================================================

modelo = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)


print("\n" + "=" * 60)
print("ENTRENAMIENTO DEL MODELO")
print("=" * 60)

print("\nEntrenando árbol de decisión...")

modelo.fit(
    X_entrenamiento,
    y_entrenamiento
)

print("Modelo entrenado correctamente.")


# ============================================================
# 6. REALIZAR PREDICCIONES
# ============================================================

print("\nRealizando predicciones...")

predicciones = modelo.predict(
    X_prueba
)

print("Predicciones realizadas correctamente.")


# ============================================================
# 7. EVALUAR EL MODELO
# ============================================================

precision = accuracy_score(
    y_prueba,
    predicciones
)

matriz = confusion_matrix(
    y_prueba,
    predicciones
)


print("\n" + "=" * 60)
print("RESULTADOS DEL MODELO")
print("=" * 60)

print(
    f"\nPrecisión del modelo: {precision * 100:.2f}%"
)

print("\nMatriz de confusión:")

print(matriz)

print("\nReporte de clasificación:")

print(
    classification_report(
        y_prueba,
        predicciones,
        zero_division=0
    )
)


# ============================================================
# 8. MOSTRAR REGLAS APRENDIDAS
# ============================================================

print("\n" + "=" * 60)
print("REGLAS APRENDIDAS POR EL ÁRBOL")
print("=" * 60)

nombres_variables = [
    "distancia_km",
    "tiempo_min",
    "transbordos",
    "congestion",
    "demanda",
    "hora_pico",
    "costo"
]

reglas = export_text(
    modelo,
    feature_names=nombres_variables
)

print(reglas)


# ============================================================
# 9. REALIZAR UNA PREDICCIÓN NUEVA
# ============================================================

print("\n" + "=" * 60)
print("PREDICCIÓN DE UNA NUEVA RUTA")
print("=" * 60)


nueva_ruta = pd.DataFrame(
    [
        {
            "distancia_km": 3.0,
            "tiempo_min": 16,
            "transbordos": 0,
            "congestion": 1,
            "demanda": 2,
            "hora_pico": 0,
            "costo": 3000
        }
    ]
)


prediccion_nueva = modelo.predict(
    nueva_ruta
)


print("\nCaracterísticas de la nueva ruta:")

print(nueva_ruta)


if prediccion_nueva[0] == 1:

    print(
        "\nResultado: RUTA RECOMENDADA"
    )

else:

    print(
        "\nResultado: RUTA NO RECOMENDADA"
    )


# ============================================================
# 10. FINALIZACIÓN
# ============================================================

print("\n" + "=" * 60)

print(
    "Proceso de aprendizaje supervisado finalizado."
)

print("=" * 60)
