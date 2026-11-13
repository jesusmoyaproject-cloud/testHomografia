# Homografía 2D Estabilizada para Videos de Fútbol

Módulo de visión por computadora y deep learning para proyectar secuencias de partidos de fútbol grabados a un plano táctico 2D (minimapa) con homografía estabilizada, libre de parpadeos y vibraciones.

## Características Principales

1. **Inferencia del Frame (YOLO11 Pose)**:
   - Detección de los 30 keypoints clave de la cancha de fútbol utilizando modelos YOLO11-pose (PyTorch `.pt`, OpenVINO e inferencia optimizada con TensorRT `.engine` FP16).
2. **Filtro de Confianza con Histéresis (Schmitt Trigger)**:
   - Umbral de activación alto ($c \ge 0.50$) para evitar falsos positivos y umbral de sostenimiento ($c \ge 0.25$) con exclusión de bordes de cuadro.
3. **Buffer Histórico y Filtro de Mediana Espacial**:
   - Ventana deslizante temporal ($n=5$) con cálculo de mediana espacial $(x, y)$ para eliminar picos de ruido y micro-vibraciones.
4. **Validación Geométrica Robusta**:
   - Verificación de mínimo matemático ($\ge 4$ keypoints activos).
5. **Estimación Robusta con MAGSAC++**:
   - `cv2.USAC_MAGSAC` para descarte de outliers geométricos y cálculo de homografía $H_t$ frame a frame hacia el plano métrico real.
6. **Suavizado de la Matriz de Homografía (EMA)**:
   - Filtro de media móvil exponencial directamente sobre la matriz de homografía ($H_{final} = \alpha H_t + (1 - \alpha) H_{t-1}$).
7. **Proyección Inversa Rígida y Renderizado Side-by-Side**:
   - Proyección de la plantilla FIFA 2D sobre la vista de cámara ($H_{final}^{-1}$) y generación de video comparativo en tiempo real.
8. **Threaded I/O Asíncrono**:
   - Captura y escritura de video multihilo (`ThreadedVideoCapture` y `ThreadedVideoWriter`) con cola acotada y cero pérdida de frames en video offline.
9. **Exportación de Métricas CSV**:
   - Registro frame a frame de keypoints detectados vs reales (en píxeles y centímetros) y cálculo de distancias de error.

---

## Estructura del Proyecto

```text
├── config.yaml               # Configuración para modelo PyTorch (.pt)
├── config_openvino.yaml      # Configuración para modelo OpenVINO
├── config_engine.yaml        # Configuración para motor TensorRT (.engine)
├── inference.py              # Pipeline principal de inferencia y generación
├── pitch_config.py           # Diccionario de keypoints y lienzo 2D FIFA
├── stabilizer.py             # Filtros temporales, MAGSAC++, EMA y proyección
├── requirements.txt          # Dependencias de Python
├── models/
│   └── engine/               # Carpeta para almacenar modelos .engine
└── output/                   # Carpeta de salida para videos y CSVs
```

---

## Instalación

```bash
git clone https://github.com/jesusmoyaproject-cloud/testHomografia.git
cd testHomografia
pip install -r requirements.txt
```

---

## Uso

### Con modelo PyTorch (.pt)
```bash
python inference.py --config config.yaml
```

### Con modelo OpenVINO
```bash
python inference.py --config config_openvino.yaml
```

### Con modelo TensorRT (.engine)
```bash
python inference.py --config config_engine.yaml
```
