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
from pathlib import Path
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import yaml

from pitch_config import CENTER_KEYPOINTS, KP_MAP, Pitch2DCanvas
from stabilizer import HomographyStabilizer, KeypointFrameRecord, KeypointTemporalFilter


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

    def __init__(self, video_path: str, max_frames: int = 5000, queue_size: int = 64):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video de entrada: {video_path}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = 30.0 if (fps <= 0 or np.isnan(fps)) else fps
        self.total_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_frames = max(0, int(max_frames))

        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        read_count = 0
        try:
            while not self._stop_event.is_set() and read_count < self.max_frames:
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


def draw_2d_minimap_annotations(
    pitch_canvas: Pitch2DCanvas,
    records: List[KeypointFrameRecord],
    projected_kps_px: Dict[int, Tuple[float, float]],
    projected_objects_px: List[Tuple[Tuple[float, float], float, int]],
    ball_class_id: int,
) -> np.ndarray:
    """
    Dibuja en el plano 2D (lado derecho) los puntos detectados proyectados con su confianza (conf),
    conectándolos con su ubicación real en la plantilla 2D, además de jugadores/balón proyectados.
    """
    canvas = pitch_canvas.get_clean_canvas()

    # Dibujar keypoints detectados proyectados en el plano 2D con su conf
    for rec in records:
        if rec.kp_index not in projected_kps_px:
            continue

        det_px_x, det_px_y = projected_kps_px[rec.kp_index]
        real_px_x, real_px_y = pitch_canvas.coords_px[rec.kp_id]

        pt_det = (
            int(np.clip(round(det_px_x), 0, canvas.shape[1] - 1)),
            int(np.clip(round(det_px_y), 0, canvas.shape[0] - 1)),
        )
        pt_real = (int(round(real_px_x)), int(round(real_px_y)))

        # Línea entre el keypoint real de la cancha y el keypoint detectado proyectado
        cv2.line(canvas, pt_real, pt_det, (180, 220, 255), 1, lineType=cv2.LINE_AA)

        # Círculo del punto detectado en el plano 2D
        color = (0, 255, 255) if rec.is_inlier_magsac else (0, 140, 255)
        cv2.circle(canvas, pt_det, 6, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt_det, 7, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Mostrar ID del keypoint y su nivel de confianza (conf) en el plano 2D
        text_2d = f"{rec.kp_id}:{rec.conf:.2f}"
        cv2.putText(
            canvas,
            text_2d,
            (pt_det[0] + 6, pt_det[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text_2d,
            (pt_det[0] + 6, pt_det[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
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

    # Título en el lienzo 2D
    cv2.putText(
        canvas,
        "PLANO 2D ESTABILIZADO (MAGSAC++ & EMA) - KEYPOINTS + CONF",
        (pitch_canvas.margin_px, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def combine_side_by_side(left_frame: np.ndarray, right_canvas: np.ndarray) -> np.ndarray:
    """Combina el video del partido (izquierda) y el plano 2D (derecha) ajustando a la misma altura."""
    target_h = left_frame.shape[0]
    h_r, w_r = right_canvas.shape[:2]
    scale = target_h / float(h_r)
    new_w_r = int(round(w_r * scale))
    resized_right = cv2.resize(right_canvas, (new_w_r, target_h), interpolation=cv2.INTER_LINEAR)
    return np.hstack([left_frame, resized_right])


def run_pipeline(config_path: str = "config.yaml") -> None:
    """Ejecuta el pipeline completo de homografía estabilizada."""
    cfg = load_config(config_path)

    paths_cfg = cfg.get("paths", {})
    inf_cfg = cfg.get("inference", {})
    filt_cfg = cfg.get("filtering_and_homography", {})
    pitch_cfg = cfg.get("pitch", {})
    track_cfg = cfg.get("tracking", {})

    pose_model_path = paths_cfg.get("pose_model_path", "models/yolo11m-pose.pt")
    detector_model_path = paths_cfg.get("detector_model_path", "")
    video_path = paths_cfg.get("video_path", "input/partido.mp4")
    output_dir = Path(paths_cfg.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video_path = output_dir / paths_cfg.get("output_video_name", "homografia_2d_output.mp4")
    output_csv_path = output_dir / paths_cfg.get("output_csv_name", "detecciones_keypoints_homografia.csv")

    # Límite de frames configurable desde el YAML (por defecto 3000 si no se especifica)
    configured_max_frames = int(inf_cfg.get("max_frames", 3000))
    max_frames = configured_max_frames if configured_max_frames > 0 else 3000

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
        conf_enter_threshold=float(filt_cfg.get("conf_enter_threshold", 0.42)),
        conf_exit_threshold=float(filt_cfg.get("conf_exit_threshold", 0.22)),
        min_confirm_frames=int(filt_cfg.get("min_confirm_frames", 2)),
        window_size=int(filt_cfg.get("median_window_size", 5)),
        hold_last_valid_frames=int(filt_cfg.get("hold_last_valid_frames", 8)),
        border_margin_px=float(filt_cfg.get("border_margin_px", 12.0)),
        max_jump_px=float(filt_cfg.get("max_jump_px", 65.0)),
    )
    homography_stabilizer = HomographyStabilizer(
        pitch_canvas=pitch_canvas,
        min_keypoints=int(filt_cfg.get("min_keypoints", 4)),
        magsac_reproj_thresh_cm=float(filt_cfg.get("magsac_reproj_thresh_cm", 120.0)),
        magsac_max_iters=int(filt_cfg.get("magsac_max_iters", 10000)),
        magsac_confidence=float(filt_cfg.get("magsac_confidence", 0.999)),
        ema_alpha=float(filt_cfg.get("ema_alpha", 0.50)),
        max_coast_frames=int(filt_cfg.get("max_coast_frames", 15)),
        regularize_keypoints_with_homography=bool(
            filt_cfg.get("regularize_keypoints_with_homography", True)
        ),
        keypoint_regularization_weight=float(
            filt_cfg.get("keypoint_regularization_weight", 0.65)
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
    frames_to_process = min(max_frames, total_video_frames) if total_video_frames > 0 else max_frames

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

    print(f"[INFO] Procesando hasta {frames_to_process} frames de '{video_path}' (Threaded I/O)...")

    frame_idx = 0
    try:
        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(csv_columns)

            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # 1. Inferencia del Frame (YOLO Pose)
                pose_results = pose_model.predict(
                    source=frame,
                    device=device,
                    half=half,
                    imgsz=imgsz,
                    verbose=False,
                )
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
                    det_results = detector_model.predict(
                        source=frame,
                        device=device,
                        half=half,
                        imgsz=imgsz,
                        conf=float(track_cfg.get("conf_threshold", 0.35)),
                        verbose=False,
                    )
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

                # Renderizar vista izquierda (partido) y vista derecha (plano 2D)
                left_view = draw_frame_annotations(
                    frame=frame,
                    records=kp_records,
                    frame_idx=frame_idx,
                    homography_status=h_status,
                    tracked_objects=tracked_objects,
                    ball_class_id=ball_class_id,
                    projected_pitch_pts=projected_pitch_pts,
                )
                right_view = draw_2d_minimap_annotations(
                    pitch_canvas=pitch_canvas,
                    records=kp_records,
                    projected_kps_px=projected_kps_px,
                    projected_objects_px=projected_objects_px,
                    ball_class_id=ball_class_id,
                )
                combined_frame = combine_side_by_side(left_view, right_view)

                if writer is None:
                    out_h, out_w = combined_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = ThreadedVideoWriter(
                        str(output_video_path), fourcc, fps, (out_w, out_h), queue_size=64
                    )

                writer.write(combined_frame)
                frame_idx += 1

                if frame_idx % 100 == 0:
                    print(f"[INFO] Frame {frame_idx}/{frames_to_process} procesado...")
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
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración YAML (por defecto: config.yaml)",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()

