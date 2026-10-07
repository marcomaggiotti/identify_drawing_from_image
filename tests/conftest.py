import cv2
import numpy as np
import pytest

from drawing_identifier.config import load_config


@pytest.fixture
def cv_config():
    return load_config(overrides={"default_vlm": "none"})


def draw_text(img, s, org, scale=1.0, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)


@pytest.fixture
def simple_drawing():
    """Clean diagram: nested ellipses, a rectangle, triangle, diamond, two overlapping circles, a heavy line."""
    img = np.full((760, 1100, 3), 255, np.uint8)
    cv2.ellipse(img, (230, 200), (170, 110), 0, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(img, (230, 200), (80, 50), 0, 0, 360, (0, 0, 0), 3)
    draw_text(img, "A", (215, 212))
    draw_text(img, "loves", (90, 290), 0.8)
    cv2.rectangle(img, (520, 90), (780, 310), (0, 0, 0), 3)
    draw_text(img, "B", (640, 210))
    cv2.polylines(img, [np.array([[950, 80], [1060, 300], [840, 300]])], True, (0, 0, 0), 3)
    cv2.polylines(img, [np.array([[230, 430], [350, 540], [230, 650], [110, 540]])], True, (0, 0, 0), 3)
    cv2.circle(img, (600, 560), 110, (0, 0, 0), 3)
    cv2.circle(img, (740, 560), 110, (0, 0, 0), 3)
    # heavy line from the rectangle to the right edge of the ellipse
    cv2.line(img, (520, 200), (400, 200), (0, 0, 0), 7)
    return img
