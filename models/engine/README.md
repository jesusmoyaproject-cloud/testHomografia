# Carpeta para Modelos TensorRT (.engine)

Coloca aquí tus modelos optimizados en formato TensorRT `.engine`, por ejemplo:
- `best.engine`
- `yolov11m-homography.engine`

### ¿Cómo compilar un archivo .pt a .engine (FP16, imgsz=1280)?
Si cuentas con una GPU NVIDIA con TensorRT instalado, puedes compilarlo desde la terminal con:
```bash
yolo export model="ruta/a/best.pt" format=engine half=True imgsz=1280 device=0
```
Y luego copiar el archivo `.engine` generado a esta carpeta `models/engine/best.engine`.

