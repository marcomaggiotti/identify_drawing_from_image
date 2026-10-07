"""Handwriting-like text rendering for synthetic pages.

Uses OpenCV's Hershey *script* fonts with per-glyph jitter, slant and elastic
distortion. Point ``fonts_dir`` at a folder of handwriting TTF/OTF fonts
(e.g. Google Fonts "Homemade Apple", "Caveat", "Dawning of a New Day") to get
much more realistic text.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Words and phrases taken from the example manuscripts (Peirce's logic notebooks)
LABEL_WORDS = [
    "loves", "rejects", "benefactress", "benefactress of", "contains", "contained", "hears", "arrests",
    "makes C", "R robs", "calls", "policeman", "flatters", "is wise", "obeys", "gives", "is r", "is q",
    "is sold", "Impass", "is mortal", "is a man", "lover of", "servant of", "good", "rich", "M", "P",
    "R", "K", "is r to", "fills", "place", "u", "v", "w",
]
LETTERS = ["A", "B", "C", "D", "E", "I", "O", "p", "q", "r", "s", "a", "b", "x", "y", "M", "A'", "A''", "r1", "q2", "Tc"]
SENTENCES = [
    "Some A is r every B",
    "Every B is r'd by some A",
    "There is a man and on any occasion",
    "either he is not robbed by anybody",
    "or he calls upon whatever policeman may hear him",
    "to arrest the robber",
    "Every collection of A's is r to an A",
    "A possible collection of A's is that one which",
    "contains no A that is contained in the collection",
    "Let any B be distinguished from any other",
    "by the fact that one is q to an A",
    "Let any A be an A' if and only if",
    "There is a B that is q to whatever A' there may be",
    "Given two classes A and B and two relations p and q",
    "the two objects may be taken as two qualities",
    "There is a benefactress of everybody",
    "Everybody loves some benefactress of him",
    "Somebody rejects everybody unless",
    "Anybody either has been rejected by somebody",
    "or loves some benefactress of himself",
    "This since we can by Rules I and III insert",
    "what we please within odd enclosures, gives",
    "Hence by deiteration we get",
    "This, on substituting what a and b represent, becomes",
    "By Rule II we can now put two ovals round",
    "Either u contains u or there is some place",
    "If v contains whatever u contains",
    "Undelined means assumed",
    "There is but one way of coloring these",
    "Same ten regions on another projection",
    "you are a good girl",
    "you obey mamma",
    "Logic universal",
    "For example, let the problem be",
    "How many ways can cut four given rays",
]


@dataclass
class TextPatch:
    mask: np.ndarray  # uint8 ink alpha 0..255
    text: str


_FONT_CACHE: dict[str, list[str]] = {}


def list_fonts(fonts_dir: str | None) -> list[str]:
    if not fonts_dir:
        return []
    if fonts_dir not in _FONT_CACHE:
        files = []
        for ext in ("ttf", "otf", "TTF", "OTF"):
            files += glob.glob(os.path.join(fonts_dir, "**", f"*.{ext}"), recursive=True)
        _FONT_CACHE[fonts_dir] = sorted(files)
    return _FONT_CACHE[fonts_dir]


def _hershey(text: str, height: float, rng: np.random.Generator) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX if rng.random() < 0.6 else cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    if rng.random() < 0.3:
        font |= cv2.FONT_ITALIC
    thick = int(rng.integers(1, 3)) if height < 28 else int(rng.integers(2, 4))
    (_, h1), _ = cv2.getTextSize("Ag", font, 1.0, thick)
    scale = height / max(h1, 1)
    # render glyph by glyph with baseline/size jitter, subscripts for trailing digits
    glyphs = []
    for i, ch in enumerate(text):
        sub = ch.isdigit() and i > 0 and text[i - 1].isalpha()
        s = scale * (0.6 if sub else rng.uniform(0.9, 1.1))
        (gw, gh), base = cv2.getTextSize(ch, font, s, thick)
        dy = (0.35 * height if sub else rng.normal(0, 0.05 * height))
        glyphs.append((ch, s, gw, gh, base, dy))
    total_w = int(sum(g[2] for g in glyphs) + 0.15 * height * len(glyphs) + 2 * height)
    H = int(3 * height)
    img = np.zeros((H, max(8, total_w)), np.uint8)
    x = int(0.5 * height)
    base_y = int(2.0 * height)
    for ch, s, gw, gh, base, dy in glyphs:
        if ch == " ":
            x += int(0.45 * height)
            continue
        cv2.putText(img, ch, (x, int(base_y + dy)), font, s, 255, thick, cv2.LINE_AA)
        x += gw + int(rng.normal(0.02, 0.04) * height)
    return img


def _ttf(text: str, height: float, font_path: str) -> np.ndarray:
    size = max(8, int(height * 1.4))
    font = ImageFont.truetype(font_path, size)
    l, t, r, b = font.getbbox(text)
    W, H = int(r - l + 2 * height), int(b - t + 2 * height)
    im = Image.new("L", (max(8, W), max(8, H)), 0)
    ImageDraw.Draw(im).text((height - l, height - t), text, fill=255, font=font)
    return np.asarray(im)


def _distort(mask: np.ndarray, height: float, rng: np.random.Generator) -> np.ndarray:
    h, w = mask.shape
    # slant + small rotation
    shear = rng.normal(0.25, 0.15)
    rot = np.deg2rad(rng.normal(0, 2.5))
    M = np.array([[np.cos(rot), -np.sin(rot) + shear, 0], [np.sin(rot), np.cos(rot), 0]], np.float32)
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    M[:, 2] -= mn
    out = cv2.warpAffine(mask, M, (int(mx[0] - mn[0]) + 2, int(mx[1] - mn[1]) + 2), flags=cv2.INTER_LINEAR)
    # elastic wobble
    hh, ww = out.shape
    sigma = max(2.0, height * 0.35)
    alpha = height * rng.uniform(0.04, 0.12)
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (hh, ww)).astype(np.float32), (0, 0), sigma) * alpha * sigma
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (hh, ww)).astype(np.float32), (0, 0), sigma) * alpha * sigma
    xs, ys = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
    return cv2.remap(out, xs + dx, ys + dy, cv2.INTER_LINEAR, borderValue=0)


def _crop(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask > 12)
    if len(xs) == 0:
        return mask[:1, :1]
    return mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _restroke(mask: np.ndarray, height: float, rng: np.random.Generator, pen: float | None = None) -> np.ndarray:
    """Redraw glyphs as a monoline pen trace (skeleton + round pen): looks written, not typeset."""
    from skimage.morphology import skeletonize

    sk = skeletonize(mask > 60)
    p = pen if pen else height * rng.uniform(0.07, 0.13)
    p = max(1, int(round(p * rng.uniform(0.8, 1.1))))
    out = cv2.dilate(sk.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p, p)))
    return cv2.GaussianBlur(out, (0, 0), 0.6)


def render_text(text: str, height: float, rng: np.random.Generator, fonts: list[str] | None = None, pen: float | None = None) -> TextPatch:
    """Ink-alpha patch of hand-looking ``text``; ``height`` is roughly the cap height in px."""
    used_ttf = False
    if fonts and rng.random() < 0.85:
        try:
            m = _ttf(text, height * 2, fonts[int(rng.integers(0, len(fonts)))])
            used_ttf = True
        except Exception:
            m = _hershey(text, height * 2, rng)
    else:
        m = _hershey(text, height * 2, rng)
    m = _crop(m)
    # normalise the overall size: rendered at 2x, scale so the line is ~1.3-1.7 x height tall
    target_h = height * rng.uniform(1.25, 1.7) * (1.0 if any(c in text for c in "gjpqyfhkldtb") else 0.8)
    s = target_h / max(1, m.shape[0])
    m = cv2.resize(m, (max(1, int(m.shape[1] * s)), max(1, int(m.shape[0] * s))), interpolation=cv2.INTER_AREA)
    if not used_ttf or rng.random() < 0.5:
        m = _restroke(m, height, rng, pen)
    m = np.pad(m, int(height * 0.6))
    m = _distort(m, height, rng)
    # thin glyph strokes lose contrast when scaled down: bring the ink back to full strength
    m = np.clip(m.astype(np.float32) * (255.0 / max(1.0, float(np.percentile(m[m > 12], 90)) if (m > 12).any() else 255.0)), 0, 255)
    return TextPatch(_crop(m.astype(np.uint8)), text)


def strike_through(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cross a word out with a (wavy) line or a zig-zag."""
    out = mask.copy()
    h, w = out.shape
    y = h * rng.uniform(0.4, 0.65)
    xs = np.linspace(0, w - 1, max(4, w // 6))
    ys = y + np.sin(xs / max(4.0, h * 0.5) + rng.uniform(0, 6)) * h * 0.08
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, 255, max(1, h // 9), cv2.LINE_AA)
    if rng.random() < 0.3:
        cv2.polylines(out, [(pts + [0, int(h * 0.15)]).reshape(-1, 1, 2)], False, 255, max(1, h // 10), cv2.LINE_AA)
    return out


def random_label(rng: np.random.Generator) -> str:
    return LETTERS[int(rng.integers(0, len(LETTERS)))] if rng.random() < 0.5 else LABEL_WORDS[int(rng.integers(0, len(LABEL_WORDS)))]


def random_sentence(rng: np.random.Generator) -> str:
    if rng.random() < 0.7:
        return SENTENCES[int(rng.integers(0, len(SENTENCES)))]
    words = [w for s in SENTENCES for w in s.split()]
    return " ".join(words[int(i)] for i in rng.integers(0, len(words), int(rng.integers(3, 8))))
