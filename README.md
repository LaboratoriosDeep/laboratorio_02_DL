# Lab02-DL-2026-01

Proyecto base para el Laboratorio 02 de Deep Learning.

La idea del laboratorio, siguiendo la presentacion, es tratar el problema como
seis experimentos independientes:

- `GDS`
- `GDS_R1`
- `GDS_R2`
- `GDS_R3`
- `GDS_R4`
- `GDS_R5`

Cada experimento toma una sola columna objetivo y la transforma a formato
one-hot. Esta version deja una base minima para que los estudiantes puedan
ejecutar un experimento completo antes de ampliar el laboratorio.

## Que esta implementado

- Carga basica de datos desde `csv` o `sav`
- Seleccion de columnas de entrada
- Codificacion one-hot del target activo
- `Dataset` de PyTorch
- Red neuronal poco profunda
- Entrenamiento base con `BCEWithLogitsLoss`
- Validacion anidada minima con folds internos y externos
- `hamming_loss` para evaluar predicciones multilabel

## Que queda como TODO

- Busqueda de hiperparametros dentro de la validacion interna
- Metricas adicionales
  En esta version solo queda activa `hamming_loss`. Las otras metricas se dejan
  comentadas en `src/evaluation.py` para que los alumnos las implementen luego.
- Algoritmo de incertidumbre con Monte Carlo Dropout
- Comparacion entre los seis experimentos

## Estructura

```text
Lab02-DL-2026-01/
|-- dataset/
|   `-- README.md
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- models.py
|   |-- preprocessing.py
|   `-- uncertainty.py
|-- main.py
|-- .gitignore
|-- environment.yml
`-- README.md
```

## Uso esperado

1. Copiar el dataset dentro de `dataset/`.
2. Activar el entorno del proyecto.
3. Ejecutar un experimento base.

```bash
python main.py --data-path dataset/archivo.csv --target-name GDS_R2
```

Ese comando:

- carga el dataset,
- prepara `X` y `Y`,
- entrena una red neuronal poco profunda,
- aplica validacion externa con folds estratificados,
- aplica validacion interna dentro de cada fold externo,
- y reporta `Hamming Loss` como metrica minima.

## Flujo implementado en esta version minima

La implementacion sigue la idea general mostrada en la presentacion:

1. Elegir una columna objetivo, por ejemplo `GDS_R2`.
2. Transformarla a formato one-hot.
3. Crear `5` folds externos estratificados para evaluacion final.
4. Crear `3` folds internos dentro de cada entrenamiento externo.
5. Entrenar la misma configuracion base en cada fold.
6. Reportar `Hamming Loss` en validacion interna y prueba externa.

La busqueda de hiperparametros todavia no se implementa. Se deja asi a
proposito para que los alumnos puedan entender primero el flujo minimo y luego
extenderlo.

## Argumentos utiles

- `--data-path`: ruta al archivo `csv` o `sav`.
- `--target-name`: experimento a ejecutar. Por defecto `GDS_R2`.
- `--hidden-dim`: neuronas de la capa oculta.
- `--dropout`: dropout de la red.
- `--learning-rate`: learning rate de Adam.
- `--weight-decay`: weight decay de Adam.
- `--batch-size`: batch size de entrenamiento.
- `--epochs`: epocas por fold.
- `--threshold`: umbral para convertir probabilidades en etiquetas.
- `--outer-folds`: folds externos. Por defecto `5`.
- `--inner-folds`: folds internos. Por defecto `3`.
- `--device`: `cpu`, `cuda` o `auto`.

Para una primera prueba rapida conviene bajar las epocas:

```bash
python main.py --data-path dataset/archivo.csv --target-name GDS_R2 --epochs 5
```

## Plan sugerido para alumnos

1. Comprender el problema y revisar las columnas del dataset.
2. Elegir un experimento objetivo, por ejemplo `GDS_R2`.
3. Verificar la codificacion one-hot del target y la validacion base.
4. Agregar nuevas metricas ademas de `Hamming Loss`.
5. Incorporar busqueda de hiperparametros en la validacion interna.
6. Comparar resultados entre experimentos.
7. Implementar incertidumbre al final del laboratorio.
