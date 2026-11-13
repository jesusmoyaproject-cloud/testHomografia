"""
Pipeline principal de Inferencia y Homografía 2D Estabilizada para Videos de Fútbol.
Ejecuta los 7 pasos de la arquitectura:
1. Inferencia del Frame (YOLO11m-pose, optimizado en FP16 / TensorRT)
2. Filtro de Confianza Pre-Buffer (c >= conf_threshold)
3. Buffer Histórico (ventana fija n) y Filtro de Mediana Espacial
4. Validación de Puntos Mínimos (>= 4 keypoints activos)
5. Estimación Robusta de Homografía con MAGSAC++ (cv2.USAC_MAGSAC)
6. Suavizado de la Matriz de Homografía con EMA (H_final = 0.8 * H_t + 0.2 * H_{t-1})
7. Proyección al plano táctico 2D, generación de video Side-by-Side y exportación CSV.
"""

import argparse
import csv
import logging
from pathlib import Path
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

from pitch_config import CENTER_KEYPOINTS, KP_MAP, Pitch2DCanvas
from stabilizer import HomographyStabilizer, KeypointFrameRecord, KeypointTemporalFilter


class _SuppressHalfDeprecatedFilter(logging.Filter):
    """Oculta la advertencia de Ultralytics sobre la deprecación de 'half' en inferencia."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not ("'half' is deprecated" in msg or "Use 'quantize' instead" in msg)


def _install_ultralytics_log_filter() -> None:
    """Registra el filtro en el logger de Ultralytics sin duplicarlo."""
    for logger_name in ("ultralytics", "ultralytics.utils"):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(f, _SuppressHalfDeprecatedFilter) for f in logger.filters):
            logger.addFilter(_SuppressHalfDeprecatedFilter())


PITCH_EDGES: Tuple[Tuple[str, str], ...] = (
    # Perímetro exterior
    ("01", "06"),
    ("06", "32"),
    ("32", "27"),
    ("27", "01"),
    # Línea central
    ("15", "18"),
    # Área grande izquierda y área chica izquierda
    ("02", "10"),
    ("10", "13"),
    ("13", "05"),
    ("03", "07"),
    ("07", "08"),
    ("08", "04"),
    # Área grande derecha y área chica derecha
    ("28", "20"),
    ("20", "23"),
    ("23", "31"),
    ("29", "25"),
    ("25", "26"),
    ("26", "30"),
)


class ThreadedVideoCapture:
    """Lectura asíncrona de video mediante cola acotada (sin pérdida de frames en archivos offline)."""

    def __init__(self, video_path: str, max_frames: Optional[int] = None, queue_size: int = 64):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video de entrada: {video_path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = 30.0 if (fps <= 0 or np.isnan(fps)) else fps
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_frames = int(max_frames) if (max_frames is not None and max_frames > 0) else None

        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        read_count = 0
        try:
            while not self._stop_event.is_set():
                if self.max_frames is not None and read_count >= self.max_frames:
                    break
                ret, frame = self.cap.read()
                if not ret:
                    break
                read_count += 1
                while not self._stop_event.is_set():
                    try:
                        self._queue.put((True, frame), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            while not self._stop_event.is_set():
                try:
                    self._queue.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
            self.cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retorna el siguiente frame en orden estricto o (False, None) al finalizar."""
        item = self._queue.get()
        if item is None:
            return False, None
        return item

    def release(self) -> None:
        """Detiene el hilo lector y libera el recurso de captura de forma segura."""
        self._stop_event.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


class ThreadedVideoWriter:
    """Escritura asíncrona de video mediante cola acotada para no bloquear el bucle de inferencia."""

    def __init__(self, output_path: str, fourcc: int, fps: float, frame_size: Tuple[int, int], queue_size: int = 64):
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"No se pudo abrir el escritor de video de salida: {output_path}")

        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def _writer_loop(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self.writer.write(frame)
        finally:
            self.writer.release()

    def write(self, frame: np.ndarray) -> None:
        """Encola un frame renderizado para su escritura en disco."""
        self._queue.put(frame)

    def release(self) -> None:
        """Vacía la cola pendiente, cierra el hilo escritor y libera el archivo de salida."""
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)


def load_config(config_path: str) -> Dict[str, Any]:
    """Carga el archivo de configuración YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device_and_half(requested_device: str, requested_half: bool) -> Tuple[str, bool]:
    """Verifica disponibilidad de GPU CUDA para habilitar FP16 o hace fallback seguro a CPU."""
    try:
        import torch
        if "cuda" in requested_device.lower() and not torch.cuda.is_available():
            print("[INFO] CUDA no detectado en el entorno actual. Usando CPU (FP32).")
            return "cpu", False
        return requested_device, (requested_half and "cuda" in requested_device.lower())
    except ImportError:
        return "cpu", False


def load_yolo_model(
    model_path: str,
    use_trt: bool,
    device: str,
    half: bool,
    imgsz: int,
    task: str = "pose",
):
    """
    Carga el modelo YOLO11m-pose (o detector) desde .pt, .engine (TensorRT FP16)
    o directorio OpenVINO (*_openvino_model).
    """
    from ultralytics import YOLO

    path_obj = Path(model_path)
    if not path_obj.exists():
        if path_obj.suffix.lower() == ".engine":
            raise FileNotFoundError(
                f"No se encontró el motor TensorRT (.engine) en: {path_obj.resolve()}\n"
                f"Asegúrate de colocar tu archivo .engine en esa ruta (ej. 'models/engine/{path_obj.name}') "
                f"o compilarlo desde tu modelo .pt con: yolo export model=... format=engine half=True imgsz={imgsz} device={device}"
            )
        raise FileNotFoundError(f"No se encontró el modelo en: {path_obj.resolve()}")

    if path_obj.is_dir() or "openvino" in path_obj.name.lower():
        print(f"[INFO] Cargando modelo OpenVINO ({task}): {path_obj}")
        return YOLO(str(path_obj), task=task)

    if path_obj.suffix.lower() == ".engine":
        print(f"[INFO] Cargando motor TensorRT ({task}): {path_obj}")
        return YOLO(str(path_obj), task=task)

    if use_trt and path_obj.suffix.lower() == ".pt" and "cuda" in device.lower():
        engine_path = path_obj.with_suffix(".engine")
        if engine_path.exists():
            print(f"[INFO] Cargando motor TensorRT FP16 existente: {engine_path}")
            return YOLO(str(engine_path), task=task)
        try:
            print(f"[INFO] Exportando {path_obj.name} a TensorRT FP16 (.engine)...")
            base_model = YOLO(str(path_obj), task=task)
            exported = base_model.export(format="engine", half=half, imgsz=imgsz, device=device)
            return YOLO(str(exported), task=task)
        except Exception as exc:
            print(f"[WARN] No se pudo exportar a TensorRT ({exc}). Usando modelo PyTorch (.pt).")

    print(f"[INFO] Cargando modelo YOLO: {path_obj} (device={device}, half={half})")
    return YOLO(str(path_obj), task=task)


def extract_best_pitch_keypoints(results) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae los 30 keypoints (x, y) y sus confianzas (c) de la detección de cancha con mayor score.
    Si no hay detecciones en el frame, retorna arreglos de ceros de forma (30, 2) y (30,).
    """
    xy_zeros = np.zeros((30, 2), dtype=np.float32)
    conf_zeros = np.zeros((30,), dtype=np.float32)

    if not results or len(results) == 0:
        return xy_zeros, conf_zeros

    res = results[0]
    if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
        return xy_zeros, conf_zeros

    kps_xy = res.keypoints.xy.cpu().numpy()  # (N_dets, K, 2)
    kps_conf = (
        res.keypoints.conf.cpu().numpy()
        if res.keypoints.conf is not None
        else np.ones((kps_xy.shape[0], kps_xy.shape[1]), dtype=np.float32)
    )

    # Seleccionar la instancia de cancha con mayor confianza general
    best_idx = 0
    if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) > 0:
        best_idx = int(res.boxes.conf.argmax().item())

    num_kps = min(30, kps_xy.shape[1])
    xy_zeros[:num_kps] = kps_xy[best_idx, :num_kps]
    conf_zeros[:num_kps] = kps_conf[best_idx, :num_kps]
    return xy_zeros, conf_zeros


def _is_point_drawable(pt: Tuple[float, float], width: int, height: int) -> bool:
    """Evita desbordamientos en cv2.line/circle cuando una proyección perspectiva cae muy lejos."""
    x, y = pt
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return (-width <= x <= 2 * width) and (-height <= y <= 2 * height)


def draw_frame_annotations(
    frame: np.ndarray,
    records: List[KeypointFrameRecord],
    frame_idx: int,
    homography_status: str,
    tracked_objects: List[Tuple[ Tuple[int, int, int, int], float, int ]],
    ball_class_id: int,
    projected_pitch_pts: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Dibuja en el frame izquierdo del partido los puntos detectados, su ID, confianza y estado."""
    vis = frame.copy()
    frame_h, frame_w = vis.shape[:2]

    # Dibujar plantilla rígida proyectada con H_final^-1 (referencia visual estable estilo Roboflow)
    if projected_pitch_pts:
        for kp_a, kp_b in PITCH_EDGES:
            pt_a = projected_pitch_pts.get(kp_a)
            pt_b = projected_pitch_pts.get(kp_b)
            if pt_a is None or pt_b is None:
                continue
            if _is_point_drawable(pt_a, frame_w, frame_h) and _is_point_drawable(pt_b, frame_w, frame_h):
                cv2.line(
                    vis,
                    (int(round(pt_a[0])), int(round(pt_a[1]))),
                    (int(round(pt_b[0])), int(round(pt_b[1]))),
                    (255, 190, 90),
                    1,
                    cv2.LINE_AA,
                )
        for pt in projected_pitch_pts.values():
            if _is_point_drawable(pt, frame_w, frame_h):
                px, py = int(round(pt[0])), int(round(pt[1]))
                if 0 <= px < frame_w and 0 <= py < frame_h:
                    cv2.circle(vis, (px, py), 3, (255, 220, 120), -1, lineType=cv2.LINE_AA)

    # Dibujar jugadores y balón si hay modelo de tracking activo
    for (x1, y1, x2, y2), conf, cls_id in tracked_objects:
        is_ball = cls_id == ball_class_id
        color = (0, 215, 255) if is_ball else (255, 180, 0)
        label = f"Ball {conf:.2f}" if is_ball else f"Player {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
        )

    # Dibujar keypoints detectados y estabilizados
    for rec in records:
        if not rec.is_active or rec.median_x is None or rec.median_y is None:
            continue

        pt_med = (int(round(rec.median_x)), int(round(rec.median_y)))
        pt_raw = (int(round(rec.raw_x)), int(round(rec.raw_y)))

        # Color según si es parte del círculo/línea central o resto de la cancha y si fue medido o retenido
        if rec.status == "ACTIVE_MEASURED":
            color = (0, 255, 255) if rec.kp_id in CENTER_KEYPOINTS else (0, 255, 0)
        else:
            color = (0, 165, 255)  # Naranja cuando se mantiene última coordenada válida por oclusión

        # Punto crudo pequeño en magenta para visualizar la corrección de la mediana
        if rec.passed_conf_filter:
            cv2.circle(vis, pt_raw, 3, (255, 0, 255), -1, lineType=cv2.LINE_AA)

        # Punto estabilizado por mediana
        radius = 6 if rec.is_inlier_magsac else 4
        cv2.circle(vis, pt_med, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt_med, radius + 2, (20, 20, 20), 1, lineType=cv2.LINE_AA)

        label = f"KP{rec.kp_id} ({rec.conf:.2f})"
        cv2.putText(
            vis,
            label,
            (pt_med[0] + 7, pt_med[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (pt_med[0] + 7, pt_med[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # Banner informativo superior en la vista del partido
    active_count = sum(1 for r in records if r.is_active)
    header = f"Frame: {frame_idx} | KPs Activos: {active_count}/30 | {homography_status}"
    cv2.rectangle(vis, (10, 10), (min(vis.shape[1] - 10, 720), 46), (20, 20, 20), -1)
    cv2.putText(
        vis, header, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA
    )

    return vis


def _compute_camera_fov_polygon_px(
    h_final_cm: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
    pitch_canvas: Pitch2DCanvas,
) -> Optional[np.ndarray]:
    """
    Proyecta el rectángulo visible del frame de cámara hacia el plano 2D con recorte homogéneo
    (w >= eps) para evitar inversiones cuando el borde superior de la imagen supera el horizonte.
    """
    if h_final_cm is None:
        return None

    corners = np.array(
        [
            [0.0, 0.0],
            [float(frame_w - 1), 0.0],
            [float(frame_w - 1), float(frame_h - 1)],
            [0.0, float(frame_h - 1)],
        ],
        dtype=np.float64,
    )
    homog_in = np.column_stack([corners, np.ones(4, dtype=np.float64)])
    homog_out = (h_final_cm @ homog_in.T).T  # (4, 3): [X, Y, w]

    # Recorte Sutherland-Hodgman contra el plano w >= w_min para puntos válidos delante de cámara
    w_min = 0.02
    clipped_pts: List[np.ndarray] = []
    n = len(homog_out)
    for i in range(n):
        p1 = homog_out[i]
        p2 = homog_out[(i + 1) % n]
        in1 = p1[2] >= w_min
        in2 = p2[2] >= w_min
        if in1:
            clipped_pts.append(p1)
        if in1 != in2:
            denom = p2[2] - p1[2]
            if abs(denom) > 1e-9:
                t = (w_min - p1[2]) / denom
                clipped_pts.append(p1 + t * (p2 - p1))

    if len(clipped_pts) < 3:
        return None

    poly_px: List[List[int]] = []
    for p in clipped_pts:
        cm_x = float(p[0] / p[2])
        cm_y = float(p[1] / p[2])
        px_x, px_y = pitch_canvas.cm_to_px(cm_x, cm_y)
        if not (np.isfinite(px_x) and np.isfinite(px_y)):
            return None
        poly_px.append(
            [
                int(np.clip(round(px_x), -2000, pitch_canvas.width_px + 2000)),
                int(np.clip(round(px_y), -2000, pitch_canvas.height_px + 2000)),
            ]
        )

    return np.array(poly_px, dtype=np.int32).reshape(-1, 1, 2)


def draw_2d_minimap_annotations(
    pitch_canvas: Pitch2DCanvas,
    records: List[KeypointFrameRecord],
    projected_kps_px: Dict[int, Tuple[float, float]],
    projected_objects_px: List[Tuple[Tuple[float, float], float, int]],
    ball_class_id: int,
    projected_kps_cm: Optional[Dict[int, Tuple[float, float]]] = None,
    fov_polygon_px: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Dibuja en el plano 2D (lado izquierdo) la zona detectada por la cámara, el punto real
    de cada keypoint, el punto detectado proyectado, su confianza y la distancia en cm.
    """
    canvas = pitch_canvas.get_clean_canvas()

    # Pintar la zona de cancha que detecta la cámara (FOV) restringida al rectángulo de la cancha
    if fov_polygon_px is not None and len(fov_polygon_px) >= 3:
        m = pitch_canvas.margin_px
        pitch_x1, pitch_y1 = m, m
        pitch_x2 = pitch_canvas.width_px - m
        pitch_y2 = pitch_canvas.height_px - m

        fov_mask = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.uint8)
        cv2.fillPoly(fov_mask, [fov_polygon_px], 255)
        pitch_mask = np.zeros_like(fov_mask)
        pitch_mask[pitch_y1:pitch_y2, pitch_x1:pitch_x2] = 255
        clipped_mask = cv2.bitwise_and(fov_mask, pitch_mask)

        if np.any(clipped_mask):
            overlay = canvas.copy()
            overlay[clipped_mask > 0] = (170, 245, 170)
            cv2.addWeighted(overlay, 0.24, canvas, 0.76, 0.0, dst=canvas)

            contours, _ = cv2.findContours(clipped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, (225, 255, 225), 1, lineType=cv2.LINE_AA)

    errors_cm: List[float] = []

    # Dibujar punto real, punto detectado, línea de error y distancia en cm
    for rec in records:
        if rec.kp_index not in projected_kps_px:
            continue

        det_px_x, det_px_y = projected_kps_px[rec.kp_index]
        real_px_x, real_px_y = pitch_canvas.coords_px[rec.kp_id]
        real_cm_x, real_cm_y = pitch_canvas.coords_cm[rec.kp_id]

        if projected_kps_cm is not None and rec.kp_index in projected_kps_cm:
            det_cm_x, det_cm_y = projected_kps_cm[rec.kp_index]
        else:
            det_cm_x = (det_px_x - pitch_canvas.margin_px) * (100.0 / pitch_canvas.ppm)
            det_cm_y = (det_px_y - pitch_canvas.margin_px) * (100.0 / pitch_canvas.ppm)
        dist_cm = float(np.hypot(det_cm_x - real_cm_x, det_cm_y - real_cm_y))
        errors_cm.append(dist_cm)

        pt_det = (
            int(np.clip(round(det_px_x), 0, canvas.shape[1] - 1)),
            int(np.clip(round(det_px_y), 0, canvas.shape[0] - 1)),
        )
        pt_real = (int(round(real_px_x)), int(round(real_px_y)))

        # Línea entre el keypoint real de la cancha y el keypoint detectado proyectado
        cv2.line(canvas, pt_real, pt_det, (180, 220, 255), 1, lineType=cv2.LINE_AA)

        # Punto real del keypoint en la plantilla 2D (blanco con borde verde oscuro)
        cv2.circle(canvas, pt_real, 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_real, 5, (0, 140, 0), 1, lineType=cv2.LINE_AA)

        # Círculo del punto detectado en el plano 2D
        color = (0, 255, 255) if rec.is_inlier_magsac else (0, 140, 255)
        cv2.circle(canvas, pt_det, 5, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_det, 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Mostrar ID, confianza y distancia en cm respecto al punto real
        text_2d = f"{rec.kp_id}({rec.conf:.2f}) {dist_cm:.1f}cm"
        cv2.putText(
            canvas,
            text_2d,
            (pt_det[0] + 6, pt_det[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text_2d,
            (pt_det[0] + 6, pt_det[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    # Dibujar jugadores y balón proyectados en el minimapa 2D (Paso 7)
    for (px_x, px_y), conf, cls_id in projected_objects_px:
        pt = (
            int(np.clip(round(px_x), 0, canvas.shape[1] - 1)),
            int(np.clip(round(px_y), 0, canvas.shape[0] - 1)),
        )
        if cls_id == ball_class_id:
            cv2.circle(canvas, pt, 6, (0, 215, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pt, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 7, (255, 120, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pt, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Título y leyenda en el lienzo 2D
    mean_err_str = f" | Err medio: {np.mean(errors_cm):.1f}cm" if errors_cm else ""
    cv2.putText(
        canvas,
        f"PLANO 2D (MAGSAC++ & EMA){mean_err_str} | Blanco:Real Cian:Det",
        (pitch_canvas.margin_px, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def combine_side_by_side(left_canvas_2d: np.ndarray, right_camera_frame: np.ndarray) -> np.ndarray:
    """Combina el plano 2D (izquierda) y el video del partido (derecha) en una sola fila."""
    target_h = right_camera_frame.shape[0]
    h_l, w_l = left_canvas_2d.shape[:2]
    scale = target_h / float(h_l)
    new_w_l = int(round(w_l * scale))
    resized_left = cv2.resize(left_canvas_2d, (new_w_l, target_h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([resized_left, right_camera_frame])


def run_pipeline(
    config_path: str = "config.yaml",
    cli_video: Optional[str] = None,
    cli_output_video: Optional[str] = None,
    cli_max_frames: Optional[int] = None,
) -> None:
    """Ejecuta el pipeline completo de homografía estabilizada."""
    _install_ultralytics_log_filter()
    cfg = load_config(config_path)

    paths_cfg = cfg.get("paths", {})
    inf_cfg = cfg.get("inference", {})
    filt_cfg = cfg.get("filtering_and_homography", {})
    pitch_cfg = cfg.get("pitch", {})
    track_cfg = cfg.get("tracking", {})

    pose_model_path = paths_cfg.get("pose_model_path", "models/yolo11m-pose.pt")
    detector_model_path = paths_cfg.get("detector_model_path", "")

    # 1. Video de entrada (CLI prevalece sobre YAML)
    video_path = cli_video if cli_video else paths_cfg.get("video_path", "input/partido.mp4")
    output_dir = Path(paths_cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Nombre de video resultante (CLI prevalece sobre YAML)
    output_video_name = cli_output_video if cli_output_video else paths_cfg.get("output_video_name", "homografia_2d_output.mp4")
    output_video_path = output_dir / output_video_name

    # Archivo CSV de reporte asociado
    output_csv_name = paths_cfg.get("output_csv_name")
    if cli_output_video or not output_csv_name:
        output_csv_name = f"detecciones_{Path(output_video_name).stem}.csv"
    output_csv_path = output_dir / output_csv_name

    # 3. Cantidad de frames a procesar (CLI prevalece sobre YAML; si no se especifica o es <=0, procesa todos)
    raw_max_frames = cli_max_frames if cli_max_frames is not None else inf_cfg.get("max_frames", None)
    max_frames: Optional[int] = None
    if raw_max_frames is not None:
        try:
            val = int(raw_max_frames)
            if val > 0:
                max_frames = val
        except (ValueError, TypeError):
            max_frames = None

    is_openvino_pose = Path(pose_model_path).is_dir() or "openvino" in Path(pose_model_path).name.lower()
    if is_openvino_pose and "cuda" in str(inf_cfg.get("device", "")).lower():
        device, half = "cpu", False
    else:
        device, half = resolve_device_and_half(
            str(inf_cfg.get("device", "cuda:0")),
            bool(inf_cfg.get("half_precision", True)),
        )
    imgsz = int(inf_cfg.get("imgsz", 1280))
    # Si es un directorio OpenVINO con metadata.yaml, sincronizar imgsz con el tamaño fijo del modelo
    metadata_file = Path(pose_model_path) / "metadata.yaml"
    if metadata_file.exists():
        try:
            meta = load_config(str(metadata_file))
            meta_imgsz = meta.get("imgsz")
            if isinstance(meta_imgsz, list) and len(meta_imgsz) > 0:
                imgsz = int(meta_imgsz[0])
            elif isinstance(meta_imgsz, int):
                imgsz = meta_imgsz
        except Exception:
            pass
    use_trt = bool(inf_cfg.get("use_tensorrt_if_available", False))

    # Inicializar plantilla 2D de la cancha
    pitch_canvas = Pitch2DCanvas(
        largo_cancha=float(pitch_cfg.get("largo_cancha", 105.0)),
        ancho_cancha=float(pitch_cfg.get("ancho_cancha", 68.0)),
        pixels_per_meter=float(pitch_cfg.get("pixels_per_meter", 10.0)),
        margin_px=int(pitch_cfg.get("margin_px", 50)),
    )

    # Inicializar Filtro Temporal de Mediana (Pasos 2 y 3) y Estabilizador de Homografía (Pasos 4, 5 y 6)
    kp_filter = KeypointTemporalFilter(
        num_keypoints=len(KP_MAP),
        conf_threshold=float(filt_cfg.get("conf_threshold", 0.30)),
        conf_enter_threshold=float(filt_cfg.get("conf_enter_threshold", 0.50)),
        conf_exit_threshold=float(filt_cfg.get("conf_exit_threshold", 0.25)),
        min_confirm_frames=int(filt_cfg.get("min_confirm_frames", 2)),
        window_size=int(filt_cfg.get("median_window_size", 5)),
        hold_last_valid_frames=int(filt_cfg.get("hold_last_valid_frames", 8)),
        border_margin_px=float(filt_cfg.get("border_margin_px", 12.0)),
        max_jump_px=float(filt_cfg.get("max_jump_px", 100.0)),
    )
    homography_stabilizer = HomographyStabilizer(
        pitch_canvas=pitch_canvas,
        min_keypoints=int(filt_cfg.get("min_keypoints", 4)),
        magsac_reproj_thresh_cm=float(filt_cfg.get("magsac_reproj_thresh_cm", 120.0)),
        magsac_max_iters=int(filt_cfg.get("magsac_max_iters", 10000)),
        magsac_confidence=float(filt_cfg.get("magsac_confidence", 0.999)),
        ema_alpha=float(filt_cfg.get("ema_alpha", 0.55)),
        max_coast_frames=int(filt_cfg.get("max_coast_frames", 15)),
        regularize_keypoints_with_homography=bool(
            filt_cfg.get("regularize_keypoints_with_homography", True)
        ),
        keypoint_regularization_weight=float(
            filt_cfg.get("keypoint_regularization_weight", 0.75)
        ),
    )

    # Cargar modelo YOLO11m-pose (Paso 1)
    pose_model = load_yolo_model(pose_model_path, use_trt, device, half, imgsz, task="pose")

    # Cargar modelo detector de jugadores/balón opcional (Paso 7)
    detector_model = None
    if detector_model_path and Path(detector_model_path).exists():
        detector_model = load_yolo_model(detector_model_path, use_trt, device, half, imgsz, task="detect")

    cap = ThreadedVideoCapture(str(video_path), max_frames=max_frames, queue_size=64)
    fps = cap.fps
    total_video_frames = cap.total_video_frames

    if max_frames is not None and max_frames > 0:
        frames_to_process = min(max_frames, total_video_frames) if total_video_frames > 0 else max_frames
        info_frames_str = f"{frames_to_process} frames"
    else:
        frames_to_process = total_video_frames if total_video_frames > 0 else -1
        info_frames_str = f"{total_video_frames} frames (todos)" if total_video_frames > 0 else "todos los frames"

    writer: Optional[ThreadedVideoWriter] = None

    # Precalcular vértices de la plantilla 2D en cm para proyección inversa rígida (H_final^-1)
    pitch_kp_ids = list(pitch_canvas.coords_cm.keys())
    pitch_pts_cm = np.array([pitch_canvas.coords_cm[k] for k in pitch_kp_ids], dtype=np.float32)

    csv_columns = [
        "frame",
        "kp_index",
        "keypoint_detectado",
        "keypoint_real",
        "conf",
        "paso_filtro_conf",
        "estado_kp",
        "inlier_magsac",
        "frame_raw_px_x",
        "frame_raw_px_y",
        "buffer_px_x",
        "buffer_px_y",
        "frame_mediana_px_x",
        "frame_mediana_px_y",
        "pos_detectado_px_x",
        "pos_detectado_px_y",
        "pos_detectado_cm_x",
        "pos_detectado_cm_y",
        "pos_real_px_x",
        "pos_real_px_y",
        "pos_real_cm_x",
        "pos_real_cm_y",
        "distancia_detectado_vs_real_px",
        "distancia_detectado_vs_real_cm",
    ]

    print(f"[INFO] Procesando {info_frames_str} de '{video_path}' (Threaded I/O)...")

    frame_idx = 0
    try:
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(csv_columns)

            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # 1. Inferencia del Frame (YOLO Pose)
                pose_predict_kwargs: Dict[str, Any] = {
                    "source": frame,
                    "device": device,
                    "imgsz": imgsz,
                    "verbose": False,
                }
                if half and not is_openvino_pose and Path(pose_model_path).suffix.lower() != ".engine":
                    pose_predict_kwargs["half"] = True
                pose_results = pose_model.predict(**pose_predict_kwargs)
                raw_xy, raw_conf = extract_best_pitch_keypoints(pose_results)

                # 2 y 3. Filtro de Confianza Pre-Buffer + Buffer Histórico y Filtro de Mediana
                kp_records = kp_filter.update(raw_xy, raw_conf, frame_shape=frame.shape[:2])

                # 4, 5 y 6. Validación (>= 4 pts) + Estimación MAGSAC++ + Suavizado EMA
                _, h_status = homography_stabilizer.estimate_and_smooth(
                    kp_records, kp_filter=kp_filter, frame_shape=frame.shape[:2]
                )

                # 7. Proyección de los keypoints activos al plano 2D para visualización y cálculo de error
                projected_kps_cm: Dict[int, Tuple[float, float]] = {}
                projected_kps_px: Dict[int, Tuple[float, float]] = {}

                active_indices = [
                    r.kp_index
                    for r in kp_records
                    if r.is_active and r.median_x is not None and r.median_y is not None
                ]
                if active_indices and homography_stabilizer.h_final_cm is not None:
                    pts_med = np.array(
                        [[kp_records[i].median_x, kp_records[i].median_y] for i in active_indices],
                        dtype=np.float32,
                    )
                    proj_res = homography_stabilizer.project_points_to_2d(pts_med)
                    if proj_res is not None:
                        coords_cm, coords_px = proj_res
                        for idx_pos, kp_i in enumerate(active_indices):
                            projected_kps_cm[kp_i] = (
                                float(coords_cm[idx_pos, 0]),
                                float(coords_cm[idx_pos, 1]),
                            )
                            projected_kps_px[kp_i] = (
                                float(coords_px[idx_pos, 0]),
                                float(coords_px[idx_pos, 1]),
                            )

                # 7b. Proyección opcional de jugadores y balón (si detector_model está configurado)
                tracked_objects: List[Tuple[Tuple[int, int, int, int], float, int]] = []
                projected_objects_px: List[Tuple[Tuple[float, float], float, int]] = []
                ball_class_id = int(track_cfg.get("ball_class_id", 0))

                if detector_model is not None:
                    det_predict_kwargs: Dict[str, Any] = {
                        "source": frame,
                        "device": device,
                        "imgsz": imgsz,
                        "conf": float(track_cfg.get("conf_threshold", 0.35)),
                        "verbose": False,
                    }
                    if half and Path(detector_model_path).suffix.lower() == ".pt":
                        det_predict_kwargs["half"] = True
                    det_results = detector_model.predict(**det_predict_kwargs)
                    if det_results and len(det_results) > 0 and det_results[0].boxes is not None:
                        boxes = det_results[0].boxes
                        xyxy = boxes.xyxy.cpu().numpy()
                        confs = boxes.conf.cpu().numpy()
                        clss = boxes.cls.cpu().numpy().astype(int)
                        foot_points = []
                        for (x1, y1, x2, y2), c_box, cls_id in zip(xyxy, confs, clss):
                            tracked_objects.append(((int(x1), int(y1), int(x2), int(y2)), float(c_box), int(cls_id)))
                            # Punto base de los pies (centro inferior del Bounding Box)
                            foot_points.append([(x1 + x2) * 0.5, y2])

                        if foot_points and homography_stabilizer.h_final_cm is not None:
                            proj_objs = homography_stabilizer.project_points_to_2d(
                                np.array(foot_points, dtype=np.float32)
                            )
                            if proj_objs is not None:
                                _, objs_px = proj_objs
                                for i_obj, (_, c_box, cls_id) in enumerate(tracked_objects):
                                    projected_objects_px.append(
                                        ((float(objs_px[i_obj, 0]), float(objs_px[i_obj, 1])), c_box, cls_id)
                                    )

                # Guardar fila por cada keypoint en el CSV
                for rec in kp_records:
                    real_cm_x, real_cm_y = pitch_canvas.coords_cm[rec.kp_id]
                    real_px_x, real_px_y = pitch_canvas.coords_px[rec.kp_id]

                    if rec.kp_index in projected_kps_cm:
                        det_cm_x, det_cm_y = projected_kps_cm[rec.kp_index]
                        det_px_x, det_px_y = projected_kps_px[rec.kp_index]
                        dist_cm = float(np.hypot(det_cm_x - real_cm_x, det_cm_y - real_cm_y))
                        dist_px = float(np.hypot(det_px_x - real_px_x, det_px_y - real_px_y))
                        det_px_x_str = f"{det_px_x:.2f}"
                        det_px_y_str = f"{det_px_y:.2f}"
                        det_cm_x_str = f"{det_cm_x:.2f}"
                        det_cm_y_str = f"{det_cm_y:.2f}"
                        dist_px_str = f"{dist_px:.2f}"
                        dist_cm_str = f"{dist_cm:.2f}"
                    else:
                        det_px_x_str = ""
                        det_px_y_str = ""
                        det_cm_x_str = ""
                        det_cm_y_str = ""
                        dist_px_str = ""
                        dist_cm_str = ""

                    med_x_str = f"{rec.median_x:.2f}" if rec.median_x is not None else ""
                    med_y_str = f"{rec.median_y:.2f}" if rec.median_y is not None else ""
                    buf_x_str = "[" + ", ".join(f"{v:.2f}" for v in rec.buffer_x) + "]"
                    buf_y_str = "[" + ", ".join(f"{v:.2f}" for v in rec.buffer_y) + "]"

                    csv_writer.writerow(
                        [
                            frame_idx,
                            rec.kp_index,
                            rec.kp_id if rec.is_active else "",
                            rec.kp_id,
                            f"{rec.conf:.4f}",
                            rec.passed_conf_filter,
                            rec.status,
                            rec.is_inlier_magsac,
                            f"{rec.raw_x:.2f}",
                            f"{rec.raw_y:.2f}",
                            buf_x_str,
                            buf_y_str,
                            med_x_str,
                            med_y_str,
                            det_px_x_str,
                            det_px_y_str,
                            det_cm_x_str,
                            det_cm_y_str,
                            f"{real_px_x:.2f}",
                            f"{real_px_y:.2f}",
                            f"{real_cm_x:.2f}",
                            f"{real_cm_y:.2f}",
                            dist_px_str,
                            dist_cm_str,
                        ]
                    )

                # Proyección inversa rígida de la plantilla 2D al frame usando H_final^-1
                projected_pitch_pts: Optional[Dict[str, Tuple[float, float]]] = None
                if homography_stabilizer.h_final_cm is not None:
                    proj_frame_pts = homography_stabilizer.project_pitch_to_frame(pitch_pts_cm)
                    if proj_frame_pts is not None:
                        projected_pitch_pts = {
                            kp_id: (float(proj_frame_pts[i, 0]), float(proj_frame_pts[i, 1]))
                            for i, kp_id in enumerate(pitch_kp_ids)
                        }

                # Calcular zona de cancha visible por la cámara (FOV) en el plano 2D
                fov_polygon_px = _compute_camera_fov_polygon_px(
                    homography_stabilizer.h_final_cm,
                    frame.shape[1],
                    frame.shape[0],
                    pitch_canvas,
                )

                # Renderizar vista izquierda (plano 2D) y vista derecha (partido) en una sola fila
                minimap_view = draw_2d_minimap_annotations(
                    pitch_canvas=pitch_canvas,
                    records=kp_records,
                    projected_kps_px=projected_kps_px,
                    projected_objects_px=projected_objects_px,
                    ball_class_id=ball_class_id,
                    projected_kps_cm=projected_kps_cm,
                    fov_polygon_px=fov_polygon_px,
                )
                camera_view = draw_frame_annotations(
                    frame=frame,
                    records=kp_records,
                    frame_idx=frame_idx,
                    homography_status=h_status,
                    tracked_objects=tracked_objects,
                    ball_class_id=ball_class_id,
                    projected_pitch_pts=projected_pitch_pts,
                )
                combined_frame = combine_side_by_side(minimap_view, camera_view)

                if writer is None:
                    out_h, out_w = combined_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = ThreadedVideoWriter(
                        str(output_video_path), fourcc, fps, (out_w, out_h), queue_size=64
                    )

                writer.write(combined_frame)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    if isinstance(frames_to_process, int) and frames_to_process > 0:
                        print(f"[INFO] Frame {frame_idx}/{frames_to_process} procesado...")
                    else:
                        print(f"[INFO] Frame {frame_idx} procesado...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"[OK] Procesamiento finalizado ({frame_idx} frames).")
    print(f"[OK] Video guardado en: {output_video_path.resolve()}")
    print(f"[OK] CSV de detecciones y mediana guardado en: {output_csv_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia YOLO11m-pose y Homografía 2D Estabilizada (Mediana + MAGSAC++ + EMA)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración YAML (por defecto: config.yaml)",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Ruta al video de entrada (anula video_path del archivo .yaml)",
    )
    parser.add_argument(
        "--output-video",
        "-o",
        type=str,
        default=None,
        help="Nombre del video resultante (anula output_video_name del archivo .yaml)",
    )
    parser.add_argument(
        "--max-frames",
        "-n",
        type=int,
        default=None,
        help="Cantidad de frames a procesar. Si no se especifica o es <= 0, procesa todos los frames.",
    )
    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        cli_video=args.video,
        cli_output_video=args.output_video,
        cli_max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

