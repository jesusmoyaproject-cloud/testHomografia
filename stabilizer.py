"""
Módulo de estabilización temporal y espacial definitivo para Homografía 2D en Fútbol:
- Paso 2: Filtro de Confianza Pre-Buffer con Exclusión de Bordes (anti-clamping) e Histéresis (Schmitt Trigger)
- Paso 3: Buffer Histórico (n frames) con Compensación de Movimiento Global de Cámara, Mediana Espacial y EMA Adaptativo
- Paso 4: Validación de Puntos Mínimos (>= 4 keypoints activos) y No-Colinealidad Geométrica
- Paso 5: Estimación Robusta de Homografía con MAGSAC++ (cv2.USAC_MAGSAC) y Verificación Proyectiva
- Paso 6: Suavizado Adaptativo de la Matriz de Homografía con EMA (lento en estático, rápido en paneo)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

from pitch_config import KP_MAP, Pitch2DCanvas


@dataclass
class KeypointFrameRecord:
    """Registro completo del estado de un keypoint en un frame para auditoría y exportación CSV."""
    kp_index: int
    kp_id: str
    raw_x: float
    raw_y: float
    conf: float
    passed_conf_filter: bool
    is_active: bool
    status: str  # "ACTIVE_MEASURED", "ACTIVE_HELD", "INACTIVE"
    buffer_x: List[float] = field(default_factory=list)
    buffer_y: List[float] = field(default_factory=list)
    median_x: Optional[float] = None
    median_y: Optional[float] = None
    is_inlier_magsac: bool = False


class KeypointTemporalFilter:
    """
    Implementa los Pasos 2 y 3 con estabilización desacoplada contra parpadeos, vibraciones y paneos:
    1. Exclusión de margen de borde (evita puntos pegados a x=0 o x=W cuando salen de cámara).
    2. Histéresis de confianza (conf_enter / conf_exit) + confirmación temporal (min_confirm_frames).
    3. Compensación del movimiento global de cámara (pan/tilt) en el buffer histórico antes de la mediana,
       eliminando el retraso de fase (phase lag) durante paneos rápidos.
    4. Suavizado adaptativo (EMA sobre la mediana compensada) para eliminar micro-vibraciones en estático
       sin introducir retraso cuando la cámara se mueve.
    """

    def __init__(
        self,
        num_keypoints: int = 30,
        conf_threshold: float = 0.30,
        conf_enter_threshold: Optional[float] = None,
        conf_exit_threshold: Optional[float] = None,
        min_confirm_frames: int = 2,
        window_size: int = 5,
        hold_last_valid_frames: int = 8,
        border_margin_px: float = 12.0,
        max_jump_px: float = 100.0,
        smooth_alpha_slow: float = 1.0,
        smooth_alpha_fast: float = 1.0,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.conf_threshold = float(conf_threshold)
        self.conf_enter = (
            float(conf_enter_threshold)
            if conf_enter_threshold is not None
            else max(self.conf_threshold, 0.50)
        )
        self.conf_exit = (
            float(conf_exit_threshold)
            if conf_exit_threshold is not None
            else max(0.22, self.conf_threshold - 0.05)
        )
        self.min_confirm_frames = max(1, int(min_confirm_frames))
        self.window_size = int(window_size)
        self.hold_last_valid_frames = int(hold_last_valid_frames)
        self.border_margin_px = float(border_margin_px)
        self.max_jump_px = float(max_jump_px)
        self.smooth_alpha_slow = float(smooth_alpha_slow)
        self.smooth_alpha_fast = float(smooth_alpha_fast)

        self.buffers_x: Dict[int, Deque[float]] = {
            i: deque(maxlen=self.window_size) for i in range(num_keypoints)
        }
        self.buffers_y: Dict[int, Deque[float]] = {
            i: deque(maxlen=self.window_size) for i in range(num_keypoints)
        }

        self.last_valid_xy: Dict[int, Optional[Tuple[float, float]]] = {
            i: None for i in range(num_keypoints)
        }
        self.smoothed_xy: Dict[int, Optional[Tuple[float, float]]] = {
            i: None for i in range(num_keypoints)
        }
        self.prev_raw_valid_xy: Dict[int, Optional[Tuple[float, float]]] = {
            i: None for i in range(num_keypoints)
        }
        self.is_currently_active: Dict[int, bool] = {
            i: False for i in range(num_keypoints)
        }
        self.confirm_streak: Dict[int, int] = {
            i: 0 for i in range(num_keypoints)
        }
        self.inlier_streak: Dict[int, int] = {
            i: 0 for i in range(num_keypoints)
        }
        self.missed_frames: Dict[int, int] = {
            i: 0 for i in range(num_keypoints)
        }
        self.last_camera_delta: Tuple[float, float] = (0.0, 0.0)

    def reset_all(self) -> None:
        """Limpia todos los buffers históricos tras un corte de cámara prolongado."""
        for i in range(self.num_keypoints):
            self.buffers_x[i].clear()
            self.buffers_y[i].clear()
            self.last_valid_xy[i] = None
            self.smoothed_xy[i] = None
            self.prev_raw_valid_xy[i] = None
            self.is_currently_active[i] = False
            self.confirm_streak[i] = 0
            self.inlier_streak[i] = 0
            self.missed_frames[i] = 0
        self.last_camera_delta = (0.0, 0.0)

    def _is_within_safe_frame(
        self, x: float, y: float, frame_w: Optional[int], frame_h: Optional[int]
    ) -> bool:
        """Descarta coordenadas pegadas al borde donde YOLO aplica clamping cuando el punto sale de cámara."""
        margin = self.border_margin_px
        if x <= margin or y <= margin:
            return False
        if frame_w is not None and x >= (frame_w - margin):
            return False
        if frame_h is not None and y >= (frame_h - margin):
            return False
        return True

    def _estimate_camera_motion(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        frame_w: Optional[int],
        frame_h: Optional[int],
    ) -> Tuple[float, float]:
        """
        Estima el desplazamiento global (dx, dy) de la cámara entre el frame t-1 y t
        usando la mediana de las diferencias de los keypoints confiables comunes.
        """
        dxs: List[float] = []
        dys: List[float] = []

        for idx in range(self.num_keypoints):
            if idx >= len(keypoints_xy) or idx >= len(keypoints_conf):
                continue
            rx, ry = float(keypoints_xy[idx, 0]), float(keypoints_xy[idx, 1])
            conf = float(keypoints_conf[idx])
            prev_xy = self.prev_raw_valid_xy[idx]

            if (
                conf >= self.conf_exit
                and self._is_within_safe_frame(rx, ry, frame_w, frame_h)
                and prev_xy is not None
            ):
                dx = rx - prev_xy[0]
                dy = ry - prev_xy[1]
                if abs(dx) < 110.0 and abs(dy) < 80.0:
                    dxs.append(dx)
                    dys.append(dy)

        if len(dxs) >= 2:
            return (float(np.median(dxs)), float(np.median(dys)))
        return (0.0, 0.0)

    def _shift_buffers_by_camera_motion(self, dx: float, dy: float) -> None:
        """
        Alinea las coordenadas históricas del buffer y el estado suavizado al frame actual
        compensando el paneo (dx, dy) de la cámara.
        """
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            return

        for idx in range(self.num_keypoints):
            if len(self.buffers_x[idx]) > 0:
                self.buffers_x[idx] = deque(
                    (val + dx for val in self.buffers_x[idx]),
                    maxlen=self.window_size,
                )
                self.buffers_y[idx] = deque(
                    (val + dy for val in self.buffers_y[idx]),
                    maxlen=self.window_size,
                )
            if self.last_valid_xy[idx] is not None:
                lx, ly = self.last_valid_xy[idx]
                self.last_valid_xy[idx] = (lx + dx, ly + dy)
            if self.smoothed_xy[idx] is not None:
                sx, sy = self.smoothed_xy[idx]
                self.smoothed_xy[idx] = (sx + dx, sy + dy)

    def deactivate_point(self, idx: int, record: KeypointFrameRecord) -> None:
        """Desactiva completamente un keypoint y limpia su buffer."""
        self.is_currently_active[idx] = False
        self.confirm_streak[idx] = 0
        self.inlier_streak[idx] = 0
        self.buffers_x[idx].clear()
        self.buffers_y[idx].clear()
        self.last_valid_xy[idx] = None
        self.smoothed_xy[idx] = None
        record.is_active = False
        record.status = "INACTIVE"
        record.median_x = None
        record.median_y = None
        record.buffer_x = []
        record.buffer_y = []
        record.is_inlier_magsac = False

    def _apply_adaptive_smooth(self, idx: int, med_x: float, med_y: float) -> Tuple[float, float]:
        """
        Suavizado adaptativo sobre la mediana compensada:
        - Cuando la cámara/punto está casi quieto, aplica alpha bajo (elimina vibración residual).
        - Cuando la cámara hace paneo rápido, aplica alpha alto (sin retraso).
        """
        prev_s = self.smoothed_xy[idx]
        if prev_s is None:
            self.smoothed_xy[idx] = (med_x, med_y)
            return (med_x, med_y)

        cam_speed = float(np.hypot(self.last_camera_delta[0], self.last_camera_delta[1]))
        # Interpolar alpha según velocidad de cámara (0 px -> slow, >= 6 px/frame -> fast)
        t = float(np.clip(cam_speed / 6.0, 0.0, 1.0))
        alpha = self.smooth_alpha_slow + t * (self.smooth_alpha_fast - self.smooth_alpha_slow)

        sx = alpha * med_x + (1.0 - alpha) * prev_s[0]
        sy = alpha * med_y + (1.0 - alpha) * prev_s[1]
        self.smoothed_xy[idx] = (float(sx), float(sy))
        return self.smoothed_xy[idx]

    def update(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: np.ndarray,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[KeypointFrameRecord]:
        """
        Procesa los 30 keypoints predichos en el frame actual aplicando filtro de borde,
        histéresis de confianza, compensación de movimiento de cámara, mediana espacial y EMA adaptativo.
        """
        frame_h = frame_shape[0] if frame_shape is not None else 720
        frame_w = frame_shape[1] if frame_shape is not None else 1280

        cam_dx, cam_dy = self._estimate_camera_motion(keypoints_xy, keypoints_conf, frame_w, frame_h)
        self.last_camera_delta = (cam_dx, cam_dy)
        self._shift_buffers_by_camera_motion(cam_dx, cam_dy)

        records: List[KeypointFrameRecord] = []

        for idx in range(self.num_keypoints):
            kp_id = KP_MAP.get(idx, f"{idx:02d}")
            raw_x = float(keypoints_xy[idx, 0]) if idx < len(keypoints_xy) else 0.0
            raw_y = float(keypoints_xy[idx, 1]) if idx < len(keypoints_xy) else 0.0
            conf = float(keypoints_conf[idx]) if idx < len(keypoints_conf) else 0.0

            in_safe_frame = self._is_within_safe_frame(raw_x, raw_y, frame_w, frame_h)
            was_active = self.is_currently_active[idx]

            # Histéresis de confianza (Schmitt Trigger)
            effective_thresh = self.conf_exit if was_active else self.conf_enter
            passed_conf = bool(in_safe_frame and (conf >= effective_thresh))

            # Verificar salto brusco anormal respecto a la posición compensada pura
            if passed_conf and was_active and self.last_valid_xy[idx] is not None:
                exp_x, exp_y = self.last_valid_xy[idx]
                jump_dist = float(np.hypot(raw_x - exp_x, raw_y - exp_y))
                if jump_dist > self.max_jump_px:
                    passed_conf = False

            if passed_conf:
                self.prev_raw_valid_xy[idx] = (raw_x, raw_y)
                self.confirm_streak[idx] += 1

                if not was_active and self.confirm_streak[idx] < self.min_confirm_frames:
                    self.buffers_x[idx].append(raw_x)
                    self.buffers_y[idx].append(raw_y)
                    records.append(
                        KeypointFrameRecord(
                            kp_index=idx,
                            kp_id=kp_id,
                            raw_x=raw_x,
                            raw_y=raw_y,
                            conf=conf,
                            passed_conf_filter=False,
                            is_active=False,
                            status="INACTIVE",
                            buffer_x=list(self.buffers_x[idx]),
                            buffer_y=list(self.buffers_y[idx]),
                            median_x=None,
                            median_y=None,
                        )
                    )
                    continue

                self.is_currently_active[idx] = True
                self.missed_frames[idx] = 0
                self.buffers_x[idx].append(raw_x)
                self.buffers_y[idx].append(raw_y)

                med_x = float(np.median(self.buffers_x[idx]))
                med_y = float(np.median(self.buffers_y[idx]))
                self.last_valid_xy[idx] = (med_x, med_y)

                # Suavizado adaptativo sobre la mediana para eliminar micro-vibraciones en estático
                out_x, out_y = self._apply_adaptive_smooth(idx, med_x, med_y)

                records.append(
                    KeypointFrameRecord(
                        kp_index=idx,
                        kp_id=kp_id,
                        raw_x=raw_x,
                        raw_y=raw_y,
                        conf=conf,
                        passed_conf_filter=True,
                        is_active=True,
                        status="ACTIVE_MEASURED",
                        buffer_x=list(self.buffers_x[idx]),
                        buffer_y=list(self.buffers_y[idx]),
                        median_x=out_x,
                        median_y=out_y,
                    )
                )
            else:
                self.prev_raw_valid_xy[idx] = None
                self.confirm_streak[idx] = 0
                self.missed_frames[idx] += 1

                prev_s = self.smoothed_xy[idx] if self.smoothed_xy[idx] is not None else self.last_valid_xy[idx]
                held_in_bounds = (
                    prev_s is not None
                    and self._is_within_safe_frame(prev_s[0], prev_s[1], frame_w, frame_h)
                )

                if (
                    was_active
                    and held_in_bounds
                    and self.missed_frames[idx] <= self.hold_last_valid_frames
                ):
                    records.append(
                        KeypointFrameRecord(
                            kp_index=idx,
                            kp_id=kp_id,
                            raw_x=raw_x,
                            raw_y=raw_y,
                            conf=conf,
                            passed_conf_filter=False,
                            is_active=True,
                            status="ACTIVE_HELD",
                            buffer_x=list(self.buffers_x[idx]),
                            buffer_y=list(self.buffers_y[idx]),
                            median_x=float(prev_s[0]),
                            median_y=float(prev_s[1]),
                        )
                    )
                else:
                    self.is_currently_active[idx] = False
                    self.inlier_streak[idx] = 0
                    self.buffers_x[idx].clear()
                    self.buffers_y[idx].clear()
                    self.last_valid_xy[idx] = None
                    self.smoothed_xy[idx] = None

                    records.append(
                        KeypointFrameRecord(
                            kp_index=idx,
                            kp_id=kp_id,
                            raw_x=raw_x,
                            raw_y=raw_y,
                            conf=conf,
                            passed_conf_filter=False,
                            is_active=False,
                            status="INACTIVE",
                            buffer_x=[],
                            buffer_y=[],
                            median_x=None,
                            median_y=None,
                        )
                    )

        return records


class HomographyStabilizer:
    """
    Implementa los Pasos 4, 5, 6 y 7 de forma limpia y desacoplada:
    - Paso 4: Validación de >= 4 keypoints activos, no-colinealidad y coherencia espacial izquierda-derecha.
    - Paso 5: Estimación robusta de homografía H_t con MAGSAC++ (cv2.USAC_MAGSAC) y verificación proyectiva.
    - Paso 6: Suavizado EMA adaptativo sobre H_t (bajo en cámara fija, alto en paneo) + reset en cortes.
    - Paso 7: Proyección de puntos al plano táctico 2D sin retroalimentar H_final^-1 al filtro de entrada.
    """

    def __init__(
        self,
        pitch_canvas: Pitch2DCanvas,
        min_keypoints: int = 4,
        magsac_reproj_thresh_cm: float = 120.0,
        magsac_max_iters: int = 10000,
        magsac_confidence: float = 0.999,
        ema_alpha: float = 0.55,
        ema_alpha_slow: float = 0.45,
        ema_alpha_fast: float = 0.80,
        max_coast_frames: int = 15,
        regularize_keypoints_with_homography: bool = True,
        keypoint_regularization_weight: float = 0.75,
    ) -> None:
        self.pitch_canvas = pitch_canvas
        self.min_keypoints = int(min_keypoints)
        self.magsac_reproj_thresh_cm = float(magsac_reproj_thresh_cm)
        self.magsac_max_iters = int(magsac_max_iters)
        self.magsac_confidence = float(magsac_confidence)
        self.ema_alpha = float(ema_alpha)
        self.ema_alpha_slow = float(ema_alpha_slow)
        self.ema_alpha_fast = float(ema_alpha_fast)
        self.max_coast_frames = int(max_coast_frames)
        self.regularize_keypoints_with_homography = bool(regularize_keypoints_with_homography)
        self.keypoint_regularization_weight = float(keypoint_regularization_weight)

        self.h_final_cm: Optional[np.ndarray] = None
        self.last_status: str = "UNINITIALIZED"
        self.last_inliers_count: int = 0
        self.coast_frames: int = 0

    @staticmethod
    def _normalize_homography(h_mat: np.ndarray) -> np.ndarray:
        """Normaliza la matriz 3x3 dividiendo por H[2, 2] para mantener consistencia de escala antes de EMA."""
        if abs(h_mat[2, 2]) > 1e-8:
            return h_mat / h_mat[2, 2]
        norm = np.linalg.norm(h_mat)
        return h_mat / norm if norm > 1e-8 else h_mat

    def _has_valid_geometry_spread(self, active_records: List[KeypointFrameRecord]) -> bool:
        """
        Verifica que los puntos activos no sean colineales ni degenerados en cancha ni en imagen.
        """
        if len(active_records) < self.min_keypoints:
            return False

        pts_cm = np.array(
            [self.pitch_canvas.coords_cm[r.kp_id] for r in active_records], dtype=np.float64
        )
        pts_px = np.array(
            [[r.median_x, r.median_y] for r in active_records], dtype=np.float64
        )

        unique_x_m = np.unique(np.round(pts_cm[:, 0] / 100.0, 1))
        unique_y_m = np.unique(np.round(pts_cm[:, 1] / 100.0, 1))
        if len(unique_x_m) < 2 or len(unique_y_m) < 2:
            return False

        # Verificar que los puntos no sean colineales en imagen (área/dispersión 2D mínima)
        centered_px = pts_px - np.mean(pts_px, axis=0)
        cov_px = np.cov(centered_px.T)
        eigvals = np.linalg.eigvalsh(cov_px)
        if float(np.min(eigvals)) < 25.0:
            return False

        return True

    def _is_homography_well_conditioned(
        self,
        h_mat: np.ndarray,
        inlier_src: np.ndarray,
        inlier_dst_cm: np.ndarray,
    ) -> bool:
        """
        Verifica que H_t no tenga inversión de perspectiva en la región visible del video
        ni error de reproyección excesivo en los inliers.
        """
        if h_mat is None or not np.all(np.isfinite(h_mat)):
            return False

        h_norm = self._normalize_homography(h_mat.astype(np.float64))
        det = float(np.linalg.det(h_norm))
        if abs(det) < 1e-9 or abs(det) > 1e6:
            return False

        w_vals = h_norm[2, 0] * inlier_src[:, 0] + h_norm[2, 1] * inlier_src[:, 1] + h_norm[2, 2]
        if np.any(w_vals <= 0.05):
            return False

        proj = cv2.perspectiveTransform(inlier_src.reshape(-1, 1, 2), h_norm).reshape(-1, 2)
        errs = np.linalg.norm(proj - inlier_dst_cm, axis=1)
        if float(np.max(errs)) > self.magsac_reproj_thresh_cm * 1.35:
            return False

        # Verificar que el jacobiano local en el centroide de los inliers conserve orientación positiva
        cx, cy = float(np.mean(inlier_src[:, 0])), float(np.mean(inlier_src[:, 1]))
        w_c = h_norm[2, 0] * cx + h_norm[2, 1] * cy + h_norm[2, 2]
        if abs(w_c) < 1e-6:
            return False
        px_c = (h_norm[0, 0] * cx + h_norm[0, 1] * cy + h_norm[0, 2]) / w_c
        py_c = (h_norm[1, 0] * cx + h_norm[1, 1] * cy + h_norm[1, 2]) / w_c
        j11 = (h_norm[0, 0] - h_norm[2, 0] * px_c) / w_c
        j12 = (h_norm[0, 1] - h_norm[2, 1] * px_c) / w_c
        j21 = (h_norm[1, 0] - h_norm[2, 0] * py_c) / w_c
        j22 = (h_norm[1, 1] - h_norm[2, 1] * py_c) / w_c
        if (j11 * j22 - j12 * j21) <= 0.0:
            return False

        return True

    def _enforce_consistency_with_h_final(
        self,
        records: List[KeypointFrameRecord],
        kp_filter: Optional[KeypointTemporalFilter],
        strict_mask: Optional[np.ndarray] = None,
        active_records: Optional[List[KeypointFrameRecord]] = None,
        h_was_updated: bool = True,
        frame_w: int = 1280,
        frame_h: int = 720,
    ) -> None:
        """
        Valida consistencia de los puntos contra H_final y aplica regularización proyectiva suave
        sin contaminar el estado interno de kp_filter (last_valid_xy / smoothed_xy).
        """
        if self.h_final_cm is None:
            return

        # Nunca desactivar keypoints cuando H_final no se actualizó (p. ej. coasting/paneo),
        # ya que una matriz rezagada provocaría falsos descartes en cadena.
        if not h_was_updated:
            return

        try:
            h_inv = np.linalg.inv(self.h_final_cm)
        except np.linalg.LinAlgError:
            h_inv = None

        magsac_map: Dict[int, bool] = {}
        if strict_mask is not None and active_records is not None:
            for r, m in zip(active_records, strict_mask):
                magsac_map[r.kp_index] = bool(m)

        # Umbral amplio para tolerar retardo transitorio de EMA durante zoom/paneo rápido
        outlier_deactivate_thresh_cm = self.magsac_reproj_thresh_cm * 2.5

        for rec in records:
            if not rec.is_active or rec.median_x is None or rec.median_y is None:
                continue

            idx = rec.kp_index
            pt_arr = np.array([[[rec.median_x, rec.median_y]]], dtype=np.float32)
            proj_cm = cv2.perspectiveTransform(pt_arr, self.h_final_cm).reshape(2)
            real_cm = self.pitch_canvas.coords_cm[rec.kp_id]
            err_cm = float(np.hypot(proj_cm[0] - real_cm[0], proj_cm[1] - real_cm[1]))

            is_inlier = magsac_map.get(idx, err_cm <= self.magsac_reproj_thresh_cm)
            if is_inlier and err_cm <= outlier_deactivate_thresh_cm:
                rec.is_inlier_magsac = True
                if kp_filter is not None:
                    kp_filter.inlier_streak[idx] += 1
            else:
                rec.is_inlier_magsac = False
                # Solo desactivar si MAGSAC++ lo rechazó Y además diverge fuertemente (> 2.5x umbral)
                if kp_filter is not None and err_cm > outlier_deactivate_thresh_cm:
                    if kp_filter.inlier_streak[idx] < 2 or h_inv is None:
                        kp_filter.deactivate_point(idx, rec)
                        continue
                    if rec.passed_conf_filter and len(kp_filter.buffers_x[idx]) > 1:
                        kp_filter.buffers_x[idx].pop()
                        kp_filter.buffers_y[idx].pop()

                    real_cm_arr = np.array([[[real_cm[0], real_cm[1]]]], dtype=np.float64)
                    ideal_px = cv2.perspectiveTransform(real_cm_arr, h_inv).reshape(2)
                    if (
                        np.all(np.isfinite(ideal_px))
                        and kp_filter._is_within_safe_frame(
                            float(ideal_px[0]), float(ideal_px[1]), frame_w, frame_h
                        )
                        and float(np.hypot(ideal_px[0] - rec.median_x, ideal_px[1] - rec.median_y)) <= 45.0
                    ):
                        rec.status = "ACTIVE_HELD"
                        rec.is_inlier_magsac = True
                        rec.median_x = float(ideal_px[0])
                        rec.median_y = float(ideal_px[1])
                    else:
                        kp_filter.deactivate_point(idx, rec)
                        continue

            # Regularización geométrica suave para la visualización,
            # SOLO si ideal_px es coherente (<= 25 px) y SIN sobrescribir kp_filter
            if (
                self.regularize_keypoints_with_homography
                and h_inv is not None
                and rec.is_active
                and rec.median_x is not None
                and rec.median_y is not None
            ):
                real_cm_arr = np.array([[[real_cm[0], real_cm[1]]]], dtype=np.float64)
                ideal_px = cv2.perspectiveTransform(real_cm_arr, h_inv).reshape(2)
                if np.all(np.isfinite(ideal_px)):
                    dist_px = float(np.hypot(ideal_px[0] - rec.median_x, ideal_px[1] - rec.median_y))
                    if dist_px <= 25.0:
                        w_reg = 0.85 if rec.status == "ACTIVE_HELD" else self.keypoint_regularization_weight
                        w_reg = float(np.clip(w_reg, 0.0, 1.0))
                        rec.median_x = float((1.0 - w_reg) * rec.median_x + w_reg * ideal_px[0])
                        rec.median_y = float((1.0 - w_reg) * rec.median_y + w_reg * ideal_px[1])

    def _handle_failed_frame(
        self,
        reason: str,
        records: List[KeypointFrameRecord],
        kp_filter: Optional[KeypointTemporalFilter] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        self.coast_frames += 1
        self.last_inliers_count = 0
        if self.h_final_cm is not None and self.coast_frames > self.max_coast_frames:
            self.h_final_cm = None
            if kp_filter is not None:
                kp_filter.reset_all()
            self.last_status = f"CAMERA_CUT_RESET ({reason})"
        elif self.h_final_cm is None:
            self.last_status = f"WAITING_VALID_H ({reason})"
        else:
            self.last_status = f"REUSED_PREV_H ({reason})"

        self._enforce_consistency_with_h_final(records, kp_filter, h_was_updated=False)
        return self.h_final_cm, self.last_status

    def estimate_and_smooth(
        self,
        records: List[KeypointFrameRecord],
        kp_filter: Optional[KeypointTemporalFilter] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Ejecuta la validación de puntos mínimos, MAGSAC++ y suavizado EMA adaptativo sobre H_t.
        """
        frame_h = frame_shape[0] if frame_shape is not None else 720
        frame_w = frame_shape[1] if frame_shape is not None else 1280
        if self.h_final_cm is not None and self.coast_frames <= 3 and kp_filter is not None:
            gate_cm = self.magsac_reproj_thresh_cm * 2.5
            for r in records:
                if r.is_active and r.median_x is not None and r.median_y is not None:
                    if kp_filter.inlier_streak[r.kp_index] < 2:
                        pt_arr = np.array([[[r.median_x, r.median_y]]], dtype=np.float32)
                        proj_cm = cv2.perspectiveTransform(pt_arr, self.h_final_cm).reshape(2)
                        real_cm = self.pitch_canvas.coords_cm[r.kp_id]
                        err_cm = float(np.hypot(proj_cm[0] - real_cm[0], proj_cm[1] - real_cm[1]))
                        if err_cm > gate_cm:
                            kp_filter.deactivate_point(r.kp_index, r)

        active_records = [
            r for r in records
            if r.is_active and r.median_x is not None and r.median_y is not None
        ]

        if len(active_records) < self.min_keypoints:
            return self._handle_failed_frame(
                f"active={len(active_records)} < {self.min_keypoints}",
                records,
                kp_filter,
            )

        if not self._has_valid_geometry_spread(active_records):
            return self._handle_failed_frame(
                "DEGENERATE_OR_INVERTED_GEOMETRY",
                records,
                kp_filter,
            )

        src_pts = np.array(
            [[r.median_x, r.median_y] for r in active_records], dtype=np.float32
        )
        dst_pts_cm = np.array(
            [self.pitch_canvas.coords_cm[r.kp_id] for r in active_records],
            dtype=np.float32,
        )

        # Paso 5: Estimación Robusta de Homografía con MAGSAC++
        try:
            h_t, inlier_mask = cv2.findHomography(
                srcPoints=src_pts,
                dstPoints=dst_pts_cm,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=self.magsac_reproj_thresh_cm,
                maxIters=self.magsac_max_iters,
                confidence=self.magsac_confidence,
            )
        except cv2.error:
            h_t, inlier_mask = None, None

        if h_t is None or not np.all(np.isfinite(h_t)):
            return self._handle_failed_frame("MAGSAC_FAILED", records, kp_filter)

        mask_flat = (
            inlier_mask.ravel().astype(bool)
            if inlier_mask is not None
            else np.ones(len(active_records), dtype=bool)
        )

        proj_cm = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), h_t).reshape(-1, 2)
        reproj_errors_cm = np.linalg.norm(proj_cm - dst_pts_cm, axis=1)
        strict_mask = mask_flat & (reproj_errors_cm <= self.magsac_reproj_thresh_cm)
        self.last_inliers_count = int(np.sum(strict_mask))

        if self.last_inliers_count < self.min_keypoints:
            return self._handle_failed_frame(
                f"inliers={self.last_inliers_count} < {self.min_keypoints}",
                records,
                kp_filter,
            )

        inlier_records = [r for r, m in zip(active_records, strict_mask) if m]
        if not self._has_valid_geometry_spread(inlier_records):
            return self._handle_failed_frame("INLIERS_COLLINEAR", records, kp_filter)

        inlier_src = src_pts[strict_mask]
        inlier_dst = dst_pts_cm[strict_mask]

        if not self._is_homography_well_conditioned(h_t, inlier_src, inlier_dst):
            return self._handle_failed_frame("ILL_CONDITIONED_H", records, kp_filter)

        h_t_norm = self._normalize_homography(h_t.astype(np.float64))

        # Paso 6: Suavizado EMA adaptativo de H_t (considerando paneo + cambio de escala/zoom)
        cam_speed = 0.0
        if kp_filter is not None:
            cam_speed = float(np.hypot(kp_filter.last_camera_delta[0], kp_filter.last_camera_delta[1]))
        t_cam = float(np.clip(cam_speed / 6.0, 0.0, 1.0))

        self.coast_frames = 0
        if self.h_final_cm is None:
            self.h_final_cm = h_t_norm
            self.last_status = f"INITIALIZED_MAGSAC (inliers={self.last_inliers_count}/{len(active_records)})"
        else:
            # Detectar si hay zoom/cambio de perspectiva midiendo la diferencia entre H_final previa y H_t actual
            prev_proj = cv2.perspectiveTransform(inlier_src.reshape(-1, 1, 2), self.h_final_cm).reshape(-1, 2)
            curr_proj = cv2.perspectiveTransform(inlier_src.reshape(-1, 1, 2), h_t_norm).reshape(-1, 2)
            h_shift_cm = float(np.percentile(np.linalg.norm(curr_proj - prev_proj, axis=1), 75))
            t_zoom = float(np.clip(h_shift_cm / 45.0, 0.0, 1.0))
            adaptive_alpha = self.ema_alpha_slow + max(t_cam, t_zoom) * (self.ema_alpha_fast - self.ema_alpha_slow)

            blended = self._normalize_homography(
                adaptive_alpha * h_t_norm + (1.0 - adaptive_alpha) * self.h_final_cm
            )
            curr_max_err = float(np.max(np.linalg.norm(curr_proj - inlier_dst, axis=1)))
            blend_proj = cv2.perspectiveTransform(inlier_src.reshape(-1, 1, 2), blended).reshape(-1, 2)
            blend_max_err = float(np.max(np.linalg.norm(blend_proj - inlier_dst, axis=1)))
            blend_not_distorted = blend_max_err <= max(self.magsac_reproj_thresh_cm, curr_max_err * 1.5)

            if blend_not_distorted and self._is_homography_well_conditioned(blended, inlier_src, inlier_dst):
                self.h_final_cm = blended
                self.last_status = f"UPDATED_MAGSAC_EMA (inliers={self.last_inliers_count}/{len(active_records)})"
            elif self.last_inliers_count >= 5:
                self.h_final_cm = h_t_norm
                self.last_status = f"RESET_TO_MAGSAC (inliers={self.last_inliers_count}/{len(active_records)})"
            else:
                return self._handle_failed_frame("INCOMPATIBLE_4PT_H", records, kp_filter)

        self._enforce_consistency_with_h_final(
            records,
            kp_filter,
            strict_mask,
            active_records,
            h_was_updated=True,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        return self.h_final_cm, self.last_status

    def project_points_to_2d(
        self,
        points_xy_px: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Paso 7: Proyecta un conjunto de coordenadas (N, 2) desde la perspectiva de la cámara
        hacia el plano táctico 2D usando H_final.
        """
        if self.h_final_cm is None or len(points_xy_px) == 0:
            return None

        pts = np.asarray(points_xy_px, dtype=np.float32).reshape(-1, 1, 2)
        projected_cm = cv2.perspectiveTransform(pts, self.h_final_cm).reshape(-1, 2)

        projected_px = np.zeros_like(projected_cm, dtype=np.float64)
        for i, (cm_x, cm_y) in enumerate(projected_cm):
            px_x, px_y = self.pitch_canvas.cm_to_px(float(cm_x), float(cm_y))
            projected_px[i, 0] = px_x
            projected_px[i, 1] = px_y

        return projected_cm.astype(np.float64), projected_px

    def project_pitch_to_frame(
        self,
        points_cm: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Proyecta coordenadas reales de la cancha (en cm) hacia el espacio de píxeles del frame
        usando la inversa de H_final (H_final^-1), al estilo de Roboflow ViewTransformer inverso.
        """
        if self.h_final_cm is None or len(points_cm) == 0:
            return None
        try:
            h_inv = np.linalg.inv(self.h_final_cm)
        except np.linalg.LinAlgError:
            return None

        pts = np.asarray(points_cm, dtype=np.float64).reshape(-1, 1, 2)
        projected_px = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
        if not np.all(np.isfinite(projected_px)):
            return None
        return projected_px
