"""
Definición geométrica de los 30 keypoints de la cancha de fútbol,
conversiones de unidades (metros, centímetros, píxeles) y renderizado del plano táctico 2D.
"""

from typing import Dict, Tuple
import cv2
import numpy as np

# Dimensiones por defecto (en metros)
LARGO_CANCHA: float = 105.0
ANCHO_CANCHA: float = 68.0

# Mapa de índice de salida del modelo YOLO Pose (0..29) al identificador de keypoint de la cancha
KP_MAP: Dict[int, str] = {
    0: '01',  1: '02',  2: '03',  3: '04',  4: '05',  5: '06',
    6: '07',  7: '08',  8: '10',  9: '11', 10: '12', 11: '13',
    12: '15', 13: '16', 14: '17', 15: '18',
    16: '20', 17: '21', 18: '22', 19: '23',
    20: '25', 21: '26', 22: '27', 23: '28',
    24: '29', 25: '30', 26: '31', 27: '32',
    28: '14', 29: '19'
}

CENTER_KEYPOINTS = {"14", "15", "16", "17", "18", "19"}

MX, MY = LARGO_CANCHA / 2.0, ANCHO_CANCHA / 2.0

PITCH_COORDS_METERS: Dict[str, Tuple[float, float]] = {
    '01': (0.0, 0.0),                        '02': (0.0, MY - 20.16),
    '03': (0.0, MY - 9.16),                  '04': (0.0, MY + 9.16),
    '05': (0.0, MY + 20.16),                 '06': (0.0, ANCHO_CANCHA),
    '07': (5.5, MY - 9.16),                  '08': (5.5, MY + 9.16),
    '10': (16.5, MY - 20.16),                '11': (16.5, MY - 7.31),
    '12': (16.5, MY + 7.31),                 '13': (16.5, MY + 20.16),
    '14': (MX - 9.15, MY),                   '15': (MX, 0.0),
    '16': (MX, MY - 9.15),                   '17': (MX, MY + 9.15),
    '18': (MX, ANCHO_CANCHA),                '19': (MX + 9.15, MY),
    '20': (LARGO_CANCHA - 16.5, MY - 20.16), '21': (LARGO_CANCHA - 16.5, MY - 7.31),
    '22': (LARGO_CANCHA - 16.5, MY + 7.31),  '23': (LARGO_CANCHA - 16.5, MY + 20.16),
    '25': (LARGO_CANCHA - 5.5, MY - 9.16),   '26': (LARGO_CANCHA - 5.5, MY + 9.16),
    '27': (LARGO_CANCHA, 0.0),               '28': (LARGO_CANCHA, MY - 20.16),
    '29': (LARGO_CANCHA, MY - 9.16),         '30': (LARGO_CANCHA, MY + 9.16),
    '31': (LARGO_CANCHA, MY + 20.16),        '32': (LARGO_CANCHA, ANCHO_CANCHA),
}


def build_pitch_coords_meters(
    largo_cancha: float = LARGO_CANCHA,
    ancho_cancha: float = ANCHO_CANCHA
) -> Dict[str, Tuple[float, float]]:
    """Construye el diccionario de coordenadas reales en metros para dimensiones personalizadas."""
    mx, my = largo_cancha / 2.0, ancho_cancha / 2.0
    return {
        '01': (0.0, 0.0),                        '02': (0.0, my - 20.16),
        '03': (0.0, my - 9.16),                  '04': (0.0, my + 9.16),
        '05': (0.0, my + 20.16),                 '06': (0.0, ancho_cancha),
        '07': (5.5, my - 9.16),                  '08': (5.5, my + 9.16),
        '10': (16.5, my - 20.16),                '11': (16.5, my - 7.31),
        '12': (16.5, my + 7.31),                 '13': (16.5, my + 20.16),
        '14': (mx - 9.15, my),                   '15': (mx, 0.0),
        '16': (mx, my - 9.15),                   '17': (mx, my + 9.15),
        '18': (mx, ancho_cancha),                '19': (mx + 9.15, my),
        '20': (largo_cancha - 16.5, my - 20.16), '21': (largo_cancha - 16.5, my - 7.31),
        '22': (largo_cancha - 16.5, my + 7.31),  '23': (largo_cancha - 16.5, my + 20.16),
        '25': (largo_cancha - 5.5, my - 9.16),   '26': (largo_cancha - 5.5, my + 9.16),
        '27': (largo_cancha, 0.0),               '28': (largo_cancha, my - 20.16),
        '29': (largo_cancha, my - 9.16),         '30': (largo_cancha, my + 9.16),
        '31': (largo_cancha, my + 20.16),        '32': (largo_cancha, ancho_cancha),
    }


class Pitch2DCanvas:
    """
    Gestiona la plantilla del plano 2D de la cancha y las conversiones entre
    metros (m), centímetros (cm) y píxeles (px) del minimapa.
    """

    def __init__(
        self,
        largo_cancha: float = LARGO_CANCHA,
        ancho_cancha: float = ANCHO_CANCHA,
        pixels_per_meter: float = 10.0,
        margin_px: int = 50,
    ) -> None:
        self.largo_m = float(largo_cancha)
        self.ancho_m = float(ancho_cancha)
        self.ppm = float(pixels_per_meter)
        self.margin_px = int(margin_px)

        self.width_px = int(round(self.largo_m * self.ppm)) + 2 * self.margin_px
        self.height_px = int(round(self.ancho_m * self.ppm)) + 2 * self.margin_px

        self.coords_meters: Dict[str, Tuple[float, float]] = build_pitch_coords_meters(
            self.largo_m, self.ancho_m
        )
        self.coords_cm: Dict[str, Tuple[float, float]] = {
            k: (v[0] * 100.0, v[1] * 100.0) for k, v in self.coords_meters.items()
        }
        self.coords_px: Dict[str, Tuple[float, float]] = {
            k: self.meters_to_px(v[0], v[1]) for k, v in self.coords_meters.items()
        }

        self._base_canvas = self._render_base_pitch()

    def meters_to_px(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Convierte coordenadas (x, y) en metros a píxeles del lienzo 2D."""
        px_x = self.margin_px + x_m * self.ppm
        px_y = self.margin_px + y_m * self.ppm
        return (float(px_x), float(px_y))

    def cm_to_px(self, x_cm: float, y_cm: float) -> Tuple[float, float]:
        """Convierte coordenadas (x, y) en centímetros a píxeles del lienzo 2D."""
        return self.meters_to_px(x_cm / 100.0, y_cm / 100.0)

    def px_to_cm(self, px_x: float, px_y: float) -> Tuple[float, float]:
        """Convierte coordenadas (x, y) en píxeles del lienzo 2D a centímetros reales."""
        x_m = (px_x - self.margin_px) / self.ppm
        y_m = (px_y - self.margin_px) / self.ppm
        return (float(x_m * 100.0), float(y_m * 100.0))

    def _m2p_int(self, x_m: float, y_m: float) -> Tuple[int, int]:
        px_x, px_y = self.meters_to_px(x_m, y_m)
        return (int(round(px_x)), int(round(px_y)))

    def _render_base_pitch(self) -> np.ndarray:
        """Dibuja la cancha 2D reglamentaria con sus líneas y keypoints reales de referencia."""
        canvas = np.full((self.height_px, self.width_px, 3), (34, 110, 43), dtype=np.uint8)

        # Franjas de césped estéticas
        num_stripes = 14
        stripe_w = self.largo_m / num_stripes
        for i in range(num_stripes):
            if i % 2 == 0:
                p1 = self._m2p_int(i * stripe_w, 0.0)
                p2 = self._m2p_int((i + 1) * stripe_w, self.ancho_m)
                cv2.rectangle(canvas, p1, p2, (40, 122, 49), -1)

        line_color = (235, 235, 235)
        lw = 2
        mx, my = self.largo_m / 2.0, self.ancho_m / 2.0

        # Contorno exterior (01 - 06 - 32 - 27)
        cv2.rectangle(
            canvas,
            self._m2p_int(0.0, 0.0),
            self._m2p_int(self.largo_m, self.ancho_m),
            line_color,
            lw,
        )

        # Línea de medio campo (15 - 18)
        cv2.line(canvas, self._m2p_int(mx, 0.0), self._m2p_int(mx, self.ancho_m), line_color, lw)

        # Círculo central (radio 9.15 m) y punto central
        center_pt = self._m2p_int(mx, my)
        radius_px = int(round(9.15 * self.ppm))
        cv2.circle(canvas, center_pt, radius_px, line_color, lw, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center_pt, 3, line_color, -1, lineType=cv2.LINE_AA)

        # Área grande izquierda (02 - 10 - 13 - 05)
        cv2.rectangle(
            canvas,
            self._m2p_int(0.0, my - 20.16),
            self._m2p_int(16.5, my + 20.16),
            line_color,
            lw,
        )
        # Área chica izquierda (03 - 07 - 08 - 04)
        cv2.rectangle(
            canvas,
            self._m2p_int(0.0, my - 9.16),
            self._m2p_int(5.5, my + 9.16),
            line_color,
            lw,
        )
        # Punto penal izquierdo (11 m) y semicírculo (11 - 12)
        pen_left = self._m2p_int(11.0, my)
        cv2.circle(canvas, pen_left, 3, line_color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(
            canvas,
            pen_left,
            (radius_px, radius_px),
            0,
            -53,
            53,
            line_color,
            lw,
            lineType=cv2.LINE_AA,
        )

        # Área grande derecha (20 - 28 - 31 - 23)
        cv2.rectangle(
            canvas,
            self._m2p_int(self.largo_m - 16.5, my - 20.16),
            self._m2p_int(self.largo_m, my + 20.16),
            line_color,
            lw,
        )
        # Área chica derecha (25 - 29 - 30 - 26)
        cv2.rectangle(
            canvas,
            self._m2p_int(self.largo_m - 5.5, my - 9.16),
            self._m2p_int(self.largo_m, my + 9.16),
            line_color,
            lw,
        )
        # Punto penal derecho y semicírculo (21 - 22)
        pen_right = self._m2p_int(self.largo_m - 11.0, my)
        cv2.circle(canvas, pen_right, 3, line_color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(
            canvas,
            pen_right,
            (radius_px, radius_px),
            0,
            127,
            233,
            line_color,
            lw,
            lineType=cv2.LINE_AA,
        )

        # Dibujar los 30 keypoints reales de la plantilla en gris tenue como referencia
        for kp_id, (px_x, px_y) in self.coords_px.items():
            pt = (int(round(px_x)), int(round(px_y)))
            cv2.circle(canvas, pt, 4, (190, 190, 190), 1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                kp_id,
                (pt[0] - 8, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (210, 210, 210),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def get_clean_canvas(self) -> np.ndarray:
        """Retorna una copia fresca de la plantilla 2D de la cancha."""
        return self._base_canvas.copy()

