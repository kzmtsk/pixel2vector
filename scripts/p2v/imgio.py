"""Image loading and colour helpers."""
import numpy as np
import cv2
from PIL import Image


def load_rgba(path):
    """Return float32 RGBA array in [0,1], shape (H,W,4), and a flag for real transparency."""
    im = Image.open(path)
    im.load()
    if im.mode in ("P", "LA", "L", "1", "I;16", "I", "F", "CMYK", "RGB", "YCbCr"):
        im = im.convert("RGBA")
    elif im.mode != "RGBA":
        im = im.convert("RGBA")
    a = np.asarray(im).astype(np.float32) / 255.0
    has_alpha = bool((a[..., 3] < 0.99).any())
    return a, has_alpha


def composite(rgba, bg=(1.0, 1.0, 1.0)):
    """Composite RGBA over a solid background -> RGB float32."""
    rgb = rgba[..., :3]
    al = rgba[..., 3:4]
    bg = np.asarray(bg, np.float32).reshape(1, 1, 3)
    return rgb * al + bg * (1 - al)


def rgb_to_lab(rgb):
    """float32 RGB [0,1] (...,3) -> Lab (L 0..100)."""
    shp = rgb.shape
    x = np.ascontiguousarray(rgb.reshape(-1, 1, 3).astype(np.float32))
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)
    return lab.reshape(shp)


def lab_to_rgb(lab):
    shp = lab.shape
    x = np.ascontiguousarray(lab.reshape(-1, 1, 3).astype(np.float32))
    rgb = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
    return np.clip(rgb.reshape(shp), 0, 1)


def to_hex(rgb):
    r, g, b = [int(round(float(v) * 255)) for v in rgb]
    return "#%02X%02X%02X" % (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
