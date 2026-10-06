# Sistema de Aprendizaje Supervisado para Rutas de Transporte

## Descripción

Este proyecto desarrolla un modelo de aprendizaje supervisado aplicado a un sistema de transporte masivo.

El objetivo es utilizar un árbol de decisión para clasificar rutas de transporte como recomendadas o no recomendadas a partir de diferentes características de la ruta.

## Tecnologías utilizadas

- Python 3
- Pandas
- Scikit-learn
- Árbol de decisión
- Dataset en formato CSV

## Dataset

El conjunto de datos utilizado es un dataset simulado con 20 registros.

Las variables utilizadas son:

- distancia_km
- tiempo_min
- transbordos
- congestion
- demanda
- hora_pico
- costo

La variable objetivo es:

- ruta_recomendada

Donde:

- 1 = Ruta recomendada
- 0 = Ruta no recomendada

## Modelo

Se utilizó un modelo de clasificación mediante un árbol de decisión utilizando la biblioteca Scikit-learn.

Los datos fueron divididos en:

- 14 registros para entrenamiento.
- 6 registros para prueba.

## Resultados

El modelo obtuvo una precisión del 100 % sobre el conjunto de prueba.

La matriz de confusión obtenida fue:

    [[3 0]
     [0 3]]

El árbol de decisión identificó la distancia como el principal criterio de clasificación de las rutas.

## Archivos del proyecto

- `rutas_transporte.csv`: conjunto de datos utilizado para entrenar y probar el modelo.
- `aprendizaje_supervisado.py`: código fuente del modelo de aprendizaje supervisado.

## Ejecución

Para ejecutar el proyecto es necesario tener Python instalado junto con las bibliotecas Pandas y Scikit-learn.

El programa carga el dataset, entrena el árbol de decisión, realiza las predicciones y muestra los resultados de evaluación.

## Nota

El dataset utilizado es de carácter académico y simulado. Los resultados obtenidos sirven para demostrar el funcionamiento del modelo de aprendizaje supervisado y no representan datos oficiales ni información de tráfico en tiempo real.
