# SignBridge: reconocimiento de alfabeto ASL

Proyecto Python para explorar el conjunto **ASL Alphabet**, preparar particiones reproducibles y definir los experimentos de clasificación de imágenes de manos.

## Estructura

```text
sign-language-recognition/
├── analisis_exploratorio.py   # Programa principal
├── src/
│   ├── config.py              # Parámetros reproducibles
│   ├── data.py                # Descarga, inventario y particiones
│   ├── eda.py                 # Gráficos y estadísticas visuales
│   ├── preprocessing.py       # Carga y normalización con Pillow/NumPy
│   └── model_catalog.py       # Modelos y transformaciones seleccionados
├── tests/
│   └── test_data_pipeline.py  # Prueba con imágenes sintéticas
├── docs/
│   └── informe.md             # Informe técnico
└── requirements.txt
```

Las carpetas `data/` y `artifacts/` se crean durante la ejecución y no se versionan.

## Instalación

Con el intérprete usado para este proyecto:

```powershell
-m pip install -r requirements.txt
```

No se requieren OpenCV, TensorFlow ni dependencias de interfaces interactivas para ejecutar el análisis actual.

## Ejecución

Si ASL Alphabet ya está descargado:

```powershell
py analisis_exploratorio.py --data-dir "C:\ruta\al\dataset"
```

Para descargar la copia pública de Kaggle, que puede superar 1 GB:

```powershell
py analisis_exploratorio.py --download
```

Sin argumentos, el programa busca datos en `data/raw` y en la caché local de Kaggle. Si todavía no existen, muestra las dos opciones anteriores y termina limpiamente.

Los resultados se guardan en:

- `data/manifests/`: asignación de entrenamiento, validación y prueba.
- `artifacts/figures/`: distribución, ejemplos y comparaciones visuales.
- `artifacts/tables/`: resúmenes, controles y catálogo de modelos.

## Pruebas

```powershell
py -m unittest discover -s tests -v
```

La prueba crea temporalmente 290 imágenes sintéticas, comprueba sus cabeceras, genera las tres particiones y verifica que no compartan archivos.

## Decisiones actuales

- Semilla global: `42`.
- Submuestra: hasta `600` imágenes por clase.
- Particiones estratificadas: 70 % entrenamiento, 15 % validación y 15 % prueba.
- Entrada: RGB de `64 × 64`, tipo `float32`, valores en `[0, 1]`.
- El pequeño directorio oficial de prueba queda fuera de la evaluación cuantitativa.

Fuente: [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
