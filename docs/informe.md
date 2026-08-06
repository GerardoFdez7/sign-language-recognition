# SignBridge: análisis y diseño del clasificador de alfabeto ASL

## 1. Contexto y objetivo

SignBridge busca reconocer, letra por letra, las señas realizadas frente a una cámara. En esta etapa se estudia el conjunto **ASL Alphabet**, se fija una estrategia de datos reproducible y se decide qué familias de modelos se compararán. La variable objetivo es la carpeta de cada imagen: 26 letras y las acciones `del`, `nothing` y `space`.

La unidad de observación es una fotografía RGB de una mano. Por tanto, no hay variables tabulares originales más allá de la etiqueta y los metadatos del archivo; los cruces relevantes se construyen entre clase, resolución, formato, tamaño en bytes, brillo, contraste, saturación y densidad de bordes.

## 2. Preguntas de exploración

1. ¿Cuántas clases e imágenes hay y la distribución está balanceada?
2. ¿Todas las imágenes comparten formato, resolución y modo de color?
3. ¿Existe variación de iluminación, contraste y fondo dentro de una misma clase?
4. ¿Qué letras poseen configuraciones manuales parecidas?
5. ¿Cómo crear entrenamiento, validación y prueba sin depender del pequeño directorio oficial de prueba?
6. ¿Qué reducción de resolución conserva la forma de la mano con un costo de cómputo razonable?

El programa responde estas preguntas mediante conteos, inspección de cabeceras, galerías y estadísticas visuales. Las tablas se calculan desde los archivos y no dependen de valores escritos manualmente.

## 3. Fuente y estructura

La ficha del conjunto [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) informa:

- 87 000 imágenes de entrenamiento en 29 clases.
- 26 letras de A a Z, más `del`, `nothing` y `space`.
- fotografías de 200 × 200 píxeles.
- 29 archivos en el directorio oficial de prueba, uno por clase.
- licencia GPL-2.0.

Si se verifican 3 000 archivos por clase, el conjunto está perfectamente balanceado: cada clase representa `1/29 = 3.45 %` del total. El programa no asume que la copia local está intacta: vuelve a contar los archivos, calcula la razón entre la clase más pequeña y la más grande, e inspecciona una muestra de cabeceras con Pillow. Esto permite descubrir descargas incompletas o archivos dañados.

El directorio oficial de prueba no se usa para estimar desempeño porque una observación por clase produciría una métrica muy inestable. Se reserva únicamente como demostración cualitativa posterior.

También existe una diferencia entre el objetivo simplificado y el lenguaje real. ASL se expresa mediante movimientos de manos y rostro, no solo con poses aisladas; en el deletreo, J y Z trazan su forma en el aire. Una fotografía solo captura un instante de esa trayectoria. Por ello, el clasificador planteado reconoce categorías del conjunto, pero todavía no es un traductor de ASL ni modela secuencias. Esta distinción evita atribuir al prototipo capacidades que sus datos no permiten.

## 4. Variabilidad observada y comparaciones visuales

La galería generada incluye A, B, M, U y V, con tres observaciones por letra. Así se comparan simultáneamente diferencias entre clases y cambios dentro de la misma clase. Además se calculan cuatro descriptores en una muestra reproducible de cada categoría:

- **Brillo:** promedio de intensidad en escala de grises.
- **Contraste:** desviación estándar de la intensidad.
- **Saturación:** promedio del canal S en HSV.
- **Densidad de bordes:** proporción de gradientes horizontales y verticales que superan un umbral fijo.

Estos descriptores no sustituyen la imagen ni son la entrada principal de las CNN. Su función es comprobar si el modelo podría aprender atajos relacionados con iluminación o fondo en vez de la configuración de los dedos.

### Letras con diferencias sutiles

- **M, N y S:** comparten un puño cerrado. La diferencia principal está en la posición del pulgar y en cuántos dedos lo cubren; son detalles pequeños al reducir la imagen.
- **U, V y R:** mantienen dos dedos extendidos. U los presenta juntos, V separados y R cruzados. Un ángulo de cámara oblicuo puede ocultar la separación o el cruce.

El programa presenta estos grupos en paneles separados. Esta observación justifica conservar tres canales y una resolución de 64 × 64 en lugar de convertir tempranamente a escala de grises o reducir a un tamaño extremo.

Hay una limitación importante: las carpetas no incluyen identificadores explícitos de persona o sesión. Un particionado aleatorio puede colocar fotogramas muy parecidos en conjuntos distintos y producir una estimación optimista. El manifiesto garantiza que ningún archivo exacto se repita, pero no puede garantizar independencia por persona. Para un producto real se necesitaría recopilar identidad de participante y usar una división por grupos.

## 5. Muestreo y particiones

Se seleccionan como máximo 600 archivos de cada clase con semilla 42. Con las 29 clases completas se obtienen 17 400 imágenes, una reducción del 80 % frente a los 87 000 archivos, suficiente para comparar prototipos sin agotar la memoria de un entorno académico.

La separación se hace en dos pasos estratificados:

| Partición | Proporción | Archivos por clase | Total esperado |
|---|---:|---:|---:|
| Entrenamiento | 70 % | 420 | 12 180 |
| Validación | 15 % | 90 | 2 610 |
| Prueba | 15 % | 90 | 2 610 |

La validación servirá para seleccionar hiperparámetros y aplicar parada temprana. La prueba permanecerá aislada hasta elegir la configuración final. Las transformaciones aleatorias se aplicarán solo durante el entrenamiento.

Los CSV de `data/manifests/` hacen auditable la asignación. Cada imagen aparece una sola vez y todas las clases deben estar presentes en cada partición; el programa falla de forma explícita si no se cumplen esas condiciones.

## 6. Preprocesamiento

El canal definido en `src/preprocessing.py` realiza:

1. lectura bajo demanda desde la ruta registrada;
2. decodificación JPEG forzada a RGB;
3. cambio de 200 × 200 a 64 × 64 con interpolación bilineal y antialiasing;
4. conversión a `float32`;
5. normalización de `[0, 255]` a `[0, 1]`;
6. lotes de 32 cargados bajo demanda para no ocupar toda la memoria.

No se recorta la mano ni se elimina el fondo en esta versión. Una segmentación defectuosa podría borrar dedos, y mantener la imagen completa establece una línea base clara. Tampoco se aplican filtros de suavizado: los bordes finos entre dedos son discriminativos.

La reducción a 64 × 64 cambia cada muestra de 120 000 a 12 288 valores, aproximadamente 89.8 % menos. Esto reduce memoria y tiempo, a la vez que conserva más detalle que 32 × 32. El control `inspect_preprocessing` verifica forma, tipo y rango antes de cualquier entrenamiento.

## 7. Transformaciones previstas

Se evaluarán rotaciones pequeñas, traslaciones, zoom moderado y cambios suaves de contraste. Representan variaciones plausibles de cámara y luz. No se usará reflejo horizontal de forma automática: una reflexión altera la lateralidad y puede producir una configuración que no representa la misma observación que la etiqueta original. Tampoco se proponen rotaciones grandes, recortes agresivos ni deformaciones elásticas, porque pueden cambiar u ocultar la posición relativa de los dedos.

Cada transformación será visualizada antes de entrenar. Después se comparará un experimento sin transformaciones contra otro con el conjunto seguro, manteniendo las mismas particiones y semilla.

## 8. Modelos seleccionados

| Modelo | Papel en la comparación | Hiperparámetros principales |
|---|---|---|
| CNN base | Línea base que explota estructura espacial | filtros 32-64-128, `dropout`, tasa de aprendizaje |
| CNN con BatchNorm | CNN más profunda y regularizada | 3/4 bloques, L2, `dropout` |
| MLP | Control sin convoluciones | 1/2 capas, 256/512 unidades |
| SVM con HOG | Método clásico sensible a contornos | kernel, C y gamma |

Se selecciona **SVM con HOG** como método no neuronal porque HOG resume orientaciones de bordes, relevantes para siluetas de manos, y una SVM funciona bien en espacios de muchas dimensiones. La CNN base medirá el beneficio mínimo de las convoluciones; la segunda CNN permitirá estudiar profundidad y regularización; el MLP mostrará el costo de perder el sesgo espacial.

El criterio primario será `macro F1`, acompañado de exactitud, matriz de confusión, `precision` y `recall` por clase. Aunque las clases están balanceadas, `macro F1` evita ocultar fallos sistemáticos en letras difíciles. También se registrarán número de parámetros, tiempo por época y latencia por imagen.

### Protocolo experimental

1. Mantener fijas la submuestra, las particiones y la semilla.
2. Entrenar cada configuración con Adam, entropía cruzada y parada temprana sobre `val_macro_f1` o, si no está disponible, `val_loss`.
3. Buscar pocos valores definidos de antemano; no ajustar decisiones mirando la prueba.
4. Elegir una configuración por familia usando validación.
5. Comparar las familias bajo el mismo preprocesamiento.
6. Evaluar una única vez en prueba y analizar especialmente M/N/S y U/V/R.
7. Repetir con las transformaciones seguras para medir robustez.

## 9. Riesgos y próximos pasos

- Confirmar con la copia descargada los conteos, formatos y estadísticas visuales.
- Investigar duplicados exactos y cercanos, especialmente entre particiones.
- Entrenar las configuraciones seleccionadas y completar la tabla de métricas.
- Incorporar fotografías externas de distintas personas, fondos e iluminaciones.
- Para una evaluación responsable, recopilar datos con diversidad de tonos de piel, lateralidad, tamaño de mano, dispositivos y ángulos, con consentimiento y separación por participante.

## 10. Reproducibilidad

Todo número derivado se genera mediante `analisis_exploratorio.py`. Las imágenes originales son inmutables, los CSV registran la procedencia de cada observación y la semilla se centraliza en `src/config.py`. El entorno se declara en `requirements.txt`.

### Referencias

- Akash Nagaraj. [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), Kaggle, 2018.
- NIDCD. [American Sign Language](https://www.nidcd.nih.gov/health/american-sign-language).
- scikit-learn. [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
- Lifeprint. [Fingerspelling](https://www.lifeprint.com/asl101/pages-layout/fingerspelling.htm).
