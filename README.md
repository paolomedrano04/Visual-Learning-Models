# AXON: cómo aprende una red neuronal

DS5343 Visualización de Datos, UTEC 2026-2. Flask y D3 v7, sin otras librerías de visualización.

AXON muestra qué "ve" una red neuronal por dentro mientras aprende. Cada punto es una de 1.000 imágenes de validación. Su posición resume las activaciones de una capa, proyectadas a 2D con t-SNE, y su color es la clase real. La app recorre dos ejes: las épocas de entrenamiento (T1) y las capas de la red (T2). Cubre tres datasets: CIFAR-10, MNIST y Fashion-MNIST.

## Ejecutar

```
pip install -r requirements.txt
python app.py
```

Abre http://127.0.0.1:5000. Los datos procesados, D3 y las fuentes van incluidos, así que la app funciona sin internet.

## Las cuatro preguntas

| | Pregunta | Vistas que la responden | Hallazgo (CIFAR-10) |
|---|---|---|---|
| P1 (T1) | ¿En qué época aparece la estructura de clases? | Proyección animada por épocas, curva de validación y pureza, medidores | El mayor salto ocurre en la primera época: la pureza pasa de 17,7 % a 35,7 %. Desde la época 24 solo hay ajuste fino. |
| P2 (T2) | ¿En qué capa se separan las clases? | Proyección por capas con haces de trayectorias, barras de pureza y sonda lineal, mapa de neuronas | El bloque 3 aporta el mayor salto de pureza (+23,6 puntos). |
| P3 | ¿Qué clases confunde la red y qué imágenes hacen de puente? | Matriz de confusión clicable, red de clases por vecinos compartidos, fotos puente | Gato y Perro comparten el 19,1 % de sus vecinos. Gato es la clase más dispersa. |
| P4 | ¿Qué imágenes le cuestan y cómo cambia de opinión? | Ranking de dificultad, trayectoria individual por épocas, historial de decisiones | 188 imágenes nunca se estabilizan en la respuesta correcta. |

En MNIST y Fashion-MNIST aparece un hallazgo adicional. **Sin entrenar, cada capa pierde información**: en MNIST la pureza cae de 76,8 % en píxeles a 25,7 % en la salida. Entrenar invierte esa tendencia. En la vista P2 se ve con las marcas rojas.

## Cómo se aplica el mantra de visualización

- **Vista general:** la matriz época × capa (botón "Vista general") y la línea de tiempo con la curva de pureza y la vista previa de cada etapa.
- **Zoom y filtro:** zoom y arrastre sobre el lienzo; ocultar o aislar clases (clic y doble clic en la leyenda); "Solo errores"; resaltados desde la matriz de confusión, la red de clases y el mapa de neuronas. Al acercar más de 3×, los puntos se convierten en las imágenes reales.
- **Detalle a demanda:** pasar el cursor muestra la imagen y su predicción. Al hacer clic se fija la imagen y se ven:
  - su trayectoria por épocas o capas;
  - sus 10 vecinos en el espacio de activaciones;
  - las probabilidades del modelo;
  - su historial de decisiones.

La animación entre etapas interpola posiciones y deja una estela corta en cada punto, para que se pueda seguir el movimiento. El recorrido guiado (5 pasos) responde las cuatro preguntas en orden y sirve como guion de la exposición.

## Método

**Modelos**

- **CIFAR-10.** Red convolucional de 3 bloques (32, 64 y 96 canales), entrenada con PyTorch:
  - 20.000 fotos de entrenamiento y 4.000 de validación.
  - Mejor época: 30. Exactitud de prueba: 81,55 % sobre las 10.000 fotos oficiales.
  - Por época se usó el tercer bloque. Para las capas, `pipeline/cifar.py` reimplementa la red en NumPy con los pesos guardados y reproduce el 100 % de las predicciones originales. Así obtiene cinco capas sin reentrenar: entrada, bloques 1 a 3 y logits.
- **MNIST y Fashion-MNIST.** Perceptrón multicapa 784, 256, 128, 64, 32, 10 (ReLU, Adam), entrenado en NumPy durante 20 épocas:
  - 20.000 imágenes de entrenamiento y 5.000 de validación.
  - Las 1.000 imágenes visualizadas salen de la partición oficial de prueba (100 por clase).
  - Se guardan todas las capas en 11 épocas de control, así que la matriz época × capa está completa (66 proyecciones).
  - Exactitud de prueba: 97,53 % en MNIST y 87,81 % en Fashion-MNIST.

**Proyección**

- Se usa t-SNE con perplejidad 35.
- El ancla es la última capa oculta en la mejor época.
- Las demás etapas se inicializan con el ancla a escala mínima (1e-4), para conservar la orientación sin imponer grupos falsos.
- Después se alinean con Procrustes (rotación y reflexión).
- Esta idea sigue a Rauber et al., *Visualizing the Hidden Activity of Artificial Neural Networks* (IEEE TVCG, 2017).

**Haces de trayectorias**

- Se calculan en el navegador con KDE edge bundling (Hurter et al., 2012), con `d3.blur2`.
- Cada trayectoria se remuestrea. En cada iteración, sus puntos intermedios se desplazan hacia el gradiente de densidad de todas las trayectorias y luego se suavizan.
- Las posiciones de cada etapa quedan fijas.
- El deslizador "Agrupación" controla la intensidad.

**Métricas**

- *Vecinos de su clase (pureza):* fracción de los 10 vecinos más cercanos, en el espacio de activaciones original, que comparten la clase.
- *Sonda lineal:* exactitud de una regresión logística entrenada solo con las activaciones de esa capa (validación cruzada de 3 particiones). Mide qué tan separables son las clases en esa capa.
- *Fidelidad 2D:* cuántos de esos 10 vecinos siguen siendo vecinos en el mapa 2D. Con el color "Fidelidad" se ve punto por punto dónde el mapa engaña.
- *Red de vecinos (`edges.json`):* grafo kNN de la etapa final.
  - Las fotos puente son las de mayor intermediación (betweenness), ponderada por la proporción de vecinos de otra clase.
  - La red de clases agrega los vecinos compartidos entre cada par de clases.
- *Dificultad:* se combinan cuatro señales:
  - baja confianza final;
  - error final;
  - vecinos de otra clase;
  - número de cambios de predicción entre épocas.
- *Mapa de neuronas:* cada neurona (o canal) se proyecta según su patrón de activación sobre las 1.000 imágenes. El color indica la clase que más la activa y el tamaño, su selectividad.

## Decisiones de diseño

- **El holograma es el protagonista.** Ocupa casi toda la pantalla. El título, las lecturas de métricas, la leyenda y los controles flotan sobre él sin cajas, y solo ganan un fondo translúcido al acercar el zoom.
- **Rótulos directos.** El nombre de cada clase aparece sobre su grupo solo cuando el grupo es compacto. Así, la aparición de los rótulos es en sí misma una lectura de cuándo la red separa las clases. Los gráficos del inspector también rotulan sus líneas al final, sin leyendas aparte.
- **Línea de tiempo con datos.** Las épocas se ubican en escala logarítmica, porque el aprendizaje se concentra al inicio. Sobre la línea se dibuja la pureza de cada etapa, y al pasar el cursor se ve una vista previa de la proyección.
- **Dos colores de interfaz con significado fijo.**
  - Cian: estado normal y aciertos.
  - Ámbar: selección, errores y lo que hay que mirar.
  - Los 10 colores de clase están reservados para los datos y evitan esos dos tonos.
- **Tipografía.**
  - Saira (variable, con ejes de peso y ancho) para la interfaz. El título usa un trazo ligero y ancho, que da el aire técnico del holograma sin caer en una fuente de videojuego.
  - IBM Plex Mono solo para cifras, con dígitos tabulares para que los números no bailen al animarse.
- **Fondo oscuro con brillo aditivo.** Las zonas densas se iluminan solas, así que la densidad se lee sin un mapa de calor aparte.
- **Encuadre por etapa.** Cada etapa se centra y se ajusta al lienzo, porque t-SNE no tiene escala absoluta. Las distancias entre grupos no se leen como magnitudes, y la métrica de fidelidad 2D lo hace explícito.
- **Hallazgos calculados desde los datos.** Se generan con `pipeline/insights.py`, de modo que los números del texto siempre coinciden con los de la vista.

## Archivos de datos (por dataset en `data/<dataset>/`)

| Archivo | Contenido |
|---|---|
| `nodes.json` | Una entrada por imagen: clase, probabilidades finales, confianza, cambios de predicción, época en que se aprendió, puntaje de dificultad |
| `edges.json` | Grafo kNN (aristas `source`, `target`, mutua, entre clases), fotos puente, red de clases y cohesión por clase |
| `frames.json` | Una celda por época y capa: coordenadas t-SNE, predicciones, sonda lineal, vecinos y métricas |
| `meta.json` | Capas, épocas, curva de validación, exactitud de prueba y hallazgos |
| `neurons.json` | Mapa de neuronas de cada capa oculta |
| `atlas.png` | Las 1.000 imágenes en una sola lámina |

## Reproducir el pipeline

```
pip install -r requirements-pipeline.txt
python -m pipeline.cifar              # usa pipeline/original (entrenamiento con PyTorch)
python -m pipeline.descargar_datos    # MNIST y Fashion-MNIST
python -m pipeline.mlp mnist
python -m pipeline.mlp fashion
python -m pipeline.insights
```

El entrenamiento original de CIFAR-10 está en `pipeline/original/generate.py` y requiere PyTorch y el archivo oficial `cifar-10-python.tar.gz`.

## Límites

- En CIFAR-10 la matriz época × capa tiene forma de cruz. El entrenamiento original guardó el bloque 3 en cada época y todas las capas solo en la época seleccionada. Completarla requiere reentrenar guardando todas las capas por época.
- t-SNE conserva vecindarios locales, no distancias globales. Por eso la app reporta la fidelidad 2D junto a cada proyección.

## Guion de exposición (unos 10 minutos)

1. **Apertura (1 min).** Pregunta inicial: ¿qué pasa dentro de una red mientras aprende? Mostrar CIFAR-10 en la época 0: una nube sin forma.
2. **P1 (2 min).** Reproducir las épocas. Señalar el salto de la época 0 a la 1 en la curva. Abrir la vista general para mostrar la estructura completa.
3. **P2 (2,5 min).** Cambiar a MNIST, pestaña Capas, y activar los haces. Reproducir de la entrada a la salida y comparar con la figura del paper. Señalar las marcas rojas: sin entrenar, cada capa pierde información. Pasar el cursor por el mapa de neuronas.
4. **P3 (2 min).** Volver a CIFAR-10. Clic en la arista entre Gato y Perro de la red de clases y acercar hasta ver las fotos reales. Mostrar una foto puente.
5. **P4 (1,5 min).** Abrir la imagen más difícil y seguir su trayectoria y su historial de decisiones.
6. **Cierre (1 min).** Mostrar Fashion-MNIST (Camiseta y Camisa) para cerrar la idea: la red falla donde las clases se parecen de verdad.

## Créditos

Pipeline original de entrenamiento de CIFAR-10 y primera versión de la visualización: [nombre de la compañera]. Ampliación, MNIST, Fashion-MNIST e interfaz AXON: Paolo Medrano Terán.

Interfaz holográfica original; no usa marcas ni personajes de terceros. Fuentes Saira e IBM Plex Mono bajo licencia SIL Open Font License (incluida en `static/fonts/`).
