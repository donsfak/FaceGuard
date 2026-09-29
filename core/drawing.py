"""Affichage OpenCV des résultats (scripts CLI). Le nom est écrit AU-DESSUS du
visage, comme demandé dans le cahier des charges."""

import cv2

from core import config

# Couleurs BGR alignées sur l'interface web
COLOR_KNOWN = (191, 212, 45)     # #2DD4BF  identité reconnue
COLOR_UNKNOWN = (11, 158, 245)   # #F59E0B  inconnu
COLOR_SPOOF = (68, 68, 239)      # #EF4444  fraude
COLOR_PENDING = (250, 165, 96)   # #60A5FA  vivacité en cours d'analyse
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for(result) -> tuple:
    if result["identity"] == config.SPOOF_LABEL:
        return COLOR_SPOOF
    if result["identity"] == config.UNKNOWN_LABEL:
        return COLOR_UNKNOWN
    if not result.get("is_real", True):
        return COLOR_PENDING
    return COLOR_KNOWN


def draw_result(frame, result, show_kps=True, show_liveness=True):
    x1, y1, x2, y2 = result["box"]
    color = color_for(result)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{result['identity']}  {result['similarity']:.2f}"
    (tw, th), base = cv2.getTextSize(label, FONT, 0.6, 2)
    top = y1 - th - base - 8
    if top < 0:                      # visage collé en haut de l'image
        top = y2
    lx = max(0, min(x1, frame.shape[1] - tw - 10))   # l'étiquette ne sort pas de l'image
    cv2.rectangle(frame, (lx, top), (lx + tw + 10, top + th + base + 8), color, cv2.FILLED)
    cv2.putText(frame, label, (lx + 5, top + th + 4), FONT, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    if show_liveness and "liveness" in result:
        (sw, _), _ = cv2.getTextSize(result["liveness"], FONT, 0.45, 1)
        cv2.putText(frame, result["liveness"], (max(0, min(x1, frame.shape[1] - sw)), min(y2 + 18, frame.shape[0] - 4)),
                    FONT, 0.45, color, 1, cv2.LINE_AA)
    if show_kps and result.get("kps"):
        for kx, ky in result["kps"]:
            cv2.circle(frame, (int(kx), int(ky)), 2, (255, 255, 0), -1)


def draw_hud(frame, fps: float, n_faces: int, n_known: int):
    text = f"FPS {fps:4.1f} | visages {n_faces} | reconnus {n_known}"
    cv2.rectangle(frame, (0, 0), (len(text) * 10 + 16, 30), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, text, (8, 21), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
