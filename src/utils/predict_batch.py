"""
predict_batch.py

Genera predicciones de riesgo de abandono (Dropout) para un conjunto de estudiantes
usando el modelo final ya entrenado, y exporta los resultados a un CSV ordenado por
riesgo (de mayor a menor probabilidad de Dropout).

Uso por defecto: predice sobre el conjunto de test guardado en 04_modeling.joblib
(útil para validar/inspeccionar resultados del propio proyecto).

Para usarlo con estudiantes nuevos reales, sustituye la carga de X_nuevo por tu propio
DataFrame (debe tener las mismas columnas que las usadas en entrenamiento).
"""

import joblib
import pandas as pd

# --- Rutas ---
# este script está pensado para ejecutarse desde la RAÍZ del proyecto
# (ML_rendimiento/), igual que: python src/utils/predict_batch.py
MODEL_PATH = "src/notebooks/src/models/modelo_dropout_final.joblib"
MODELING_ARTIFACT_PATH = "src/notebooks/src/artifacts/04_modeling.joblib"  # de aquí sacamos X_test de ejemplo
OUTPUT_CSV = "src/notebooks/src/artifacts/predicciones_dropout.csv"


def cargar_modelo(model_path: str = MODEL_PATH):
    """Carga el pipeline completo (preprocesado + modelo) ya entrenado."""
    return joblib.load(model_path)


def cargar_estudiantes_ejemplo(artifact_path: str = MODELING_ARTIFACT_PATH) -> pd.DataFrame:
    """
    Carga el conjunto de test guardado en el pipeline del proyecto, a modo de ejemplo.
    Sustituye esta función por la carga de tus propios datos nuevos si tienes un CSV
    o base de datos con estudiantes recién matriculados.
    """
    art = joblib.load(artifact_path)
    return art["X_test"]


def predecir_riesgo(modelo, X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el modelo a X y devuelve un DataFrame con:
    - la predicción de clase (Dropout / Enrolled / Graduate)
    - la probabilidad de cada clase
    - una columna 'riesgo_dropout_pct' lista para ordenar/filtrar
    """
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    clases = modelo.named_steps["clf"].classes_

    resultado = X.copy()
    resultado["prediccion"] = predicciones

    for i, clase in enumerate(clases):
        resultado[f"prob_{clase}"] = (probabilidades[:, i] * 100).round(1)

    resultado["riesgo_dropout_pct"] = resultado["prob_Dropout"]
    return resultado


def main():
    print("Cargando modelo...")
    modelo = cargar_modelo()

    print("Cargando estudiantes...")
    X = cargar_estudiantes_ejemplo()

    print(f"Generando predicciones para {len(X)} estudiantes...")
    resultado = predecir_riesgo(modelo, X)

    # Ordenamos de mayor a menor riesgo de abandono: los primeros de la lista
    # son los estudiantes a los que el equipo de orientación debería contactar antes.
    resultado_ordenado = resultado.sort_values("riesgo_dropout_pct", ascending=False)

    resultado_ordenado.to_csv(OUTPUT_CSV, index=False)
    print(f"Predicciones guardadas en: {OUTPUT_CSV}")

    print("\nTop 5 estudiantes con mayor riesgo de abandono:")
    columnas_resumen = ["prediccion", "prob_Dropout", "prob_Enrolled", "prob_Graduate"]
    print(resultado_ordenado[columnas_resumen].head(5).to_string())


if __name__ == "__main__":
    main()