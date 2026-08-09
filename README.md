# SignBridge: reconocimiento de alfabeto ASL
Proyecto Python para explorar el conjunto ASL Alphabet, preparar particiones reproducibles y definir los experimentos de clasificación de imágenes de manos.

## Estructura

```text
sign-language-recognition/
├── analisis_exploratorio.py   # Inventario, gráficos y particiones
├── entrenar_modelos.py        # CNN, MLP, SVM y transformaciones
├── evaluar_fotos.py           # Pruebas oficiales y por integrante
├── src/                       # Datos, preprocesamiento y modelos
├── tests/                     # Pruebas automatizadas
├── data/external/             # Fotografías aportadas por el equipo
├── artifacts/                 # Métricas, figuras y pesos generados
└── requirements.txt
```

`data/raw`, `data/processed` y `artifacts` se excluyen de Git por su tamaño.

## Instalación

```powershell
py -m pip install -r requirements.txt
```

## 1. Análisis y particiones

Con los datos en `data/raw`:

```powershell
py analisis_exploratorio.py
```

También puede indicarse `--data-dir "C:\ruta\al\dataset"` o solicitar la descarga con `--download`.

## 2. Entrenamiento y comparación

```powershell
py entrenar_modelos.py
```

Se entrenan dos CNN con variaciones de hiperparámetros, dos MLP, SVM-HOG con dos valores de C y versiones con transformaciones. Los resultados existentes se reutilizan; `--force` obliga a repetirlos.

Resultados generados:

- `data/manifests/`: entrenamiento, validación y prueba.
- `artifacts/models/`: estados de PyTorch y SVM.
- `artifacts/metrics/`: historiales, predicciones y métricas.
- `artifacts/figures/`: exploración, comparación y matrices de confusión.

## 3. Fotografías externas

Coloque al menos cinco letras distintas por persona en `data/external/<integrante>/`, con nombres como `A_01.jpg`, y ejecute:

```powershell
py evaluar_fotos.py
```

La etiqueta debe aparecer al inicio del nombre. Se aceptan A–Z, `del`, `nothing` y `space`.

## Pruebas

```powershell
py -m unittest discover -s tests -v
```

Las pruebas crean imágenes sintéticas, comprueban cabeceras y particiones, validan HOG y verifican las dimensiones de las tres arquitecturas neuronales.

## Configuración reproducible

- Semilla global: `42`.
- Submuestra: `600` imágenes por clase, 17 400 en total.
- Particiones: 70 % entrenamiento, 15 % validación y 15 % prueba.
- Entrada: RGB de `64 × 64`, tipo `float32`, valores en `[0, 1]`.
- Selección: macro F1 de validación; la prueba se consulta después.
- Mejor resultado actual: SVM-HOG con C=1, macro F1 de prueba `0.9443`.

Fuente: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
