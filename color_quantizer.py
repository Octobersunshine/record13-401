import os
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
from PIL import Image


class QuantizeMethod(str, Enum):
    MEDIAN_CUT = "median_cut"
    MAX_COVERAGE = "max_coverage"
    KMEANS = "kmeans"


class DitherMethod(str, Enum):
    NONE = "none"
    FLOYD_STEINBERG = "floyd_steinberg"
    ATKINSON = "atkinson"
    BURKES = "burkes"
    SIERRA = "sierra"
    SIERRA_LITE = "sierra_lite"
    BAYER2 = "bayer2"
    BAYER4 = "bayer4"
    BAYER8 = "bayer8"
    ORDERED = "ordered"


_DIFFUSION_MATRICES = {
    DitherMethod.FLOYD_STEINBERG: {
        "offsets": [(0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)],
        "divisor": 16,
    },
    DitherMethod.ATKINSON: {
        "offsets": [(0, 1, 1), (0, 2, 1), (1, -1, 1), (1, 0, 1), (1, 1, 1), (2, 0, 1)],
        "divisor": 8,
    },
    DitherMethod.BURKES: {
        "offsets": [
            (0, 1, 8), (0, 2, 4),
            (1, -2, 2), (1, -1, 4), (1, 0, 8), (1, 1, 4), (1, 2, 2),
        ],
        "divisor": 32,
    },
    DitherMethod.SIERRA: {
        "offsets": [
            (0, 1, 5), (0, 2, 3),
            (1, -2, 2), (1, -1, 4), (1, 0, 5), (1, 1, 4), (1, 2, 2),
            (2, -1, 2), (2, 0, 3), (2, 1, 2),
        ],
        "divisor": 32,
    },
    DitherMethod.SIERRA_LITE: {
        "offsets": [(0, 1, 2), (1, -1, 1), (1, 0, 1)],
        "divisor": 4,
    },
}


_BAYER_MATRICES = {
    2: np.array([
        [0, 2],
        [3, 1],
    ], dtype=np.float64) / 4.0 - 0.5,
    4: np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ], dtype=np.float64) / 16.0 - 0.5,
    8: np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ], dtype=np.float64) / 64.0 - 0.5,
}


class ColorQuantizer:
    def __init__(
        self,
        colors: int = 256,
        method: QuantizeMethod = QuantizeMethod.MEDIAN_CUT,
        dither: DitherMethod = DitherMethod.NONE,
        palette: Optional[List[Tuple[int, int, int]]] = None,
        dither_strength: float = 1.0,
    ):
        if colors < 2 or colors > 256:
            raise ValueError("colors must be between 2 and 256")
        if not 0.0 <= dither_strength <= 2.0:
            raise ValueError("dither_strength must be between 0.0 and 2.0")
        self.colors = colors
        self.method = method
        self.dither = dither
        self.palette = palette
        self.dither_strength = dither_strength

    def _get_pil_dither(self) -> int:
        if self.dither == DitherMethod.FLOYD_STEINBERG:
            return Image.Dither.FLOYDSTEINBERG
        elif self.dither in (DitherMethod.ORDERED, DitherMethod.BAYER4, DitherMethod.BAYER8):
            return Image.Dither.ORDERED
        return Image.Dither.NONE

    def _build_palette_array(self, img: Image.Image) -> np.ndarray:
        if self.palette:
            return np.array(self.palette, dtype=np.float64)

        if self.method == QuantizeMethod.KMEANS:
            return self._compute_kmeans_palette(img)

        pil_method = Image.Quantize.MEDIANCUT
        if self.method == QuantizeMethod.MAX_COVERAGE:
            pil_method = Image.Quantize.MAXCOVERAGE

        quantized = img.quantize(colors=self.colors, method=pil_method)
        palette_data = quantized.getpalette()
        colors_list = []
        if palette_data:
            for i in range(0, min(len(palette_data), self.colors * 3), 3):
                colors_list.append((palette_data[i], palette_data[i + 1], palette_data[i + 2]))
        return np.array(colors_list, dtype=np.float64)

    def _compute_kmeans_palette(self, img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img, dtype=np.float64)
        pixels = arr.reshape(-1, 3)

        n_samples = len(pixels)
        if n_samples <= self.colors:
            unique_pixels = np.unique(pixels, axis=0)
            return unique_pixels.astype(np.float64)

        if n_samples > 20000:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(n_samples, 20000, replace=False)
            pixels_sample = pixels[sample_idx]
        else:
            pixels_sample = pixels

        rng = np.random.default_rng(42)
        centroids = pixels_sample[rng.choice(len(pixels_sample), self.colors, replace=False)].copy()

        for _ in range(30):
            distances = np.linalg.norm(pixels_sample[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for k in range(self.colors):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = pixels_sample[mask].mean(axis=0)

            if np.allclose(centroids, new_centroids, atol=0.5):
                break
            centroids = new_centroids

        return centroids

    def _quantize_to_palette(self, img: Image.Image, palette: np.ndarray) -> np.ndarray:
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3)

        distances = np.linalg.norm(pixels[:, None, :] - palette[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        quantized = palette[labels].astype(np.uint8)

        return quantized.reshape(h, w, 3)

    def _apply_diffusion_dither(
        self,
        img: Image.Image,
        palette: np.ndarray,
        dither_method: DitherMethod,
    ) -> Image.Image:
        if dither_method not in _DIFFUSION_MATRICES:
            raise ValueError(f"Unsupported diffusion dither: {dither_method}")

        matrix = _DIFFUSION_MATRICES[dither_method]
        offsets = matrix["offsets"]
        divisor = matrix["divisor"]
        strength = self.dither_strength

        arr = np.array(img.convert("RGB"), dtype=np.float64)
        h, w, _ = arr.shape
        palette_arr = palette.astype(np.float64)

        for y in range(h):
            for x in range(w):
                old_pixel = arr[y, x].copy()
                distances = np.linalg.norm(palette_arr - old_pixel, axis=1)
                new_pixel = palette_arr[np.argmin(distances)]
                arr[y, x] = new_pixel
                quant_error = (old_pixel - new_pixel) * strength

                for dy, dx, weight in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        arr[ny, nx] += quant_error * weight / divisor

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _apply_bayer_dither(
        self,
        img: Image.Image,
        palette: np.ndarray,
        matrix_size: int,
    ) -> Image.Image:
        if matrix_size not in _BAYER_MATRICES:
            raise ValueError(f"Unsupported Bayer matrix size: {matrix_size}")

        bayer = _BAYER_MATRICES[matrix_size]
        strength = self.dither_strength * 64.0

        arr = np.array(img.convert("RGB"), dtype=np.float64)
        h, w, _ = arr.shape

        tile_h = (h + matrix_size - 1) // matrix_size
        tile_w = (w + matrix_size - 1) // matrix_size
        threshold = np.tile(bayer, (tile_h, tile_w))[:h, :w]
        threshold_3d = np.stack([threshold, threshold, threshold], axis=2)

        dithered = arr + threshold_3d * strength
        dithered = np.clip(dithered, 0, 255)

        pixels = dithered.reshape(-1, 3)
        palette_arr = palette.astype(np.float64)
        distances = np.linalg.norm(pixels[:, None, :] - palette_arr[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        result = palette_arr[labels].astype(np.uint8).reshape(h, w, 3)

        return Image.fromarray(result, mode="RGB")

    def _apply_dither(
        self,
        img: Image.Image,
        palette: np.ndarray,
    ) -> Image.Image:
        if self.dither == DitherMethod.NONE:
            arr = self._quantize_to_palette(img, palette)
            return Image.fromarray(arr, mode="RGB")

        if self.dither in _DIFFUSION_MATRICES:
            return self._apply_diffusion_dither(img, palette, self.dither)

        if self.dither == DitherMethod.BAYER2:
            return self._apply_bayer_dither(img, palette, 2)
        if self.dither in (DitherMethod.BAYER4, DitherMethod.ORDERED):
            return self._apply_bayer_dither(img, palette, 4)
        if self.dither == DitherMethod.BAYER8:
            return self._apply_bayer_dither(img, palette, 8)

        arr = self._quantize_to_palette(img, palette)
        return Image.fromarray(arr, mode="RGB")

    def _quantize_kmeans(self, img: Image.Image) -> Image.Image:
        palette = self._compute_kmeans_palette(img)
        return self._apply_dither(img, palette)

    def _quantize_pillow(self, img: Image.Image) -> Image.Image:
        if self.dither == DitherMethod.NONE:
            if img.mode != "RGB":
                img = img.convert("RGB")

            pil_method = Image.Quantize.MEDIANCUT
            if self.method == QuantizeMethod.MAX_COVERAGE:
                pil_method = Image.Quantize.MAXCOVERAGE

            kwargs = {
                "colors": self.colors,
                "method": pil_method,
                "dither": Image.Dither.NONE,
            }

            if self.palette:
                palette_img = Image.new("P", (1, 1))
                flat_palette = []
                for rgb in self.palette:
                    flat_palette.extend(rgb)
                while len(flat_palette) < 768:
                    flat_palette.extend([0, 0, 0])
                palette_img.putpalette(flat_palette)
                kwargs["palette"] = palette_img

            return img.quantize(**kwargs).convert("RGB")

        palette = self._build_palette_array(img)
        return self._apply_dither(img, palette)

    def quantize_image(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> Tuple[Image.Image, dict]:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        img = Image.open(input_path)

        if self.method == QuantizeMethod.KMEANS:
            result = self._quantize_kmeans(img)
        else:
            result = self._quantize_pillow(img)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() in (".png",):
            result.save(output_path, format="PNG", optimize=True)
        elif output_path.suffix.lower() in (".jpg", ".jpeg"):
            result.save(output_path, format="JPEG", quality=85, optimize=True)
        else:
            result.save(output_path)

        original_size = input_path.stat().st_size
        compressed_size = output_path.stat().st_size
        unique_colors = len(set(result.getdata()))

        info = {
            "original_size": original_size,
            "compressed_size": compressed_size,
            "size_reduction_pct": round((1 - compressed_size / original_size) * 100, 2) if original_size > 0 else 0,
            "unique_colors": unique_colors,
            "target_colors": self.colors,
            "method": self.method.value,
            "dither": self.dither.value,
            "dither_strength": self.dither_strength,
            "original_mode": img.mode,
            "original_size_px": img.size,
        }

        return result, info

    def quantize_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        extensions: Optional[List[str]] = None,
    ) -> List[dict]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")

        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]

        results = []
        for ext in extensions:
            for img_path in input_dir.glob(f"*{ext}"):
                out_path = output_dir / img_path.name
                try:
                    _, info = self.quantize_image(img_path, out_path)
                    info["filename"] = img_path.name
                    results.append(info)
                except Exception as e:
                    results.append({
                        "filename": img_path.name,
                        "error": str(e),
                    })

        return results

    def extract_palette(self, img: Image.Image) -> List[Tuple[int, int, int]]:
        palette_arr = self._build_palette_array(img)
        return [(int(r), int(g), int(b)) for r, g, b in palette_arr]

    def create_palette_image(
        self,
        palette: List[Tuple[int, int, int]],
        swatch_size: int = 50,
        columns: int = 8,
    ) -> Image.Image:
        rows = (len(palette) + columns - 1) // columns
        width = columns * swatch_size
        height = rows * swatch_size

        palette_img = Image.new("RGB", (width, height), (255, 255, 255))

        for idx, color in enumerate(palette):
            row = idx // columns
            col = idx % columns
            x = col * swatch_size
            y = row * swatch_size
            for dy in range(swatch_size):
                for dx in range(swatch_size):
                    palette_img.putpixel((x + dx, y + dy), color)

        return palette_img
