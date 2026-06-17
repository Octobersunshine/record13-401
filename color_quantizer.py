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
    ORDERED = "ordered"


class ColorQuantizer:
    def __init__(
        self,
        colors: int = 256,
        method: QuantizeMethod = QuantizeMethod.MEDIAN_CUT,
        dither: DitherMethod = DitherMethod.NONE,
        palette: Optional[List[Tuple[int, int, int]]] = None,
    ):
        if colors < 2 or colors > 256:
            raise ValueError("colors must be between 2 and 256")
        self.colors = colors
        self.method = method
        self.dither = dither
        self.palette = palette

    def _get_pil_dither(self) -> int:
        if self.dither == DitherMethod.FLOYD_STEINBERG:
            return Image.Dither.FLOYDSTEINBERG
        elif self.dither == DitherMethod.ORDERED:
            return Image.Dither.ORDERED
        return Image.Dither.NONE

    def _quantize_pillow(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        pil_method = Image.Quantize.MEDIANCUT
        if self.method == QuantizeMethod.MAX_COVERAGE:
            pil_method = Image.Quantize.MAXCOVERAGE

        kwargs = {
            "colors": self.colors,
            "method": pil_method,
            "dither": self._get_pil_dither(),
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

    def _quantize_kmeans(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img, dtype=np.float64)
        h, w, _ = arr.shape
        pixels = arr.reshape(-1, 3)

        n_samples = len(pixels)
        if n_samples <= self.colors:
            return img

        rng = np.random.default_rng(42)
        centroids = pixels[rng.choice(n_samples, self.colors, replace=False)].copy()

        for _ in range(20):
            distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = centroids.copy()
            for k in range(self.colors):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = pixels[mask].mean(axis=0)

            if np.allclose(centroids, new_centroids, atol=1.0):
                break
            centroids = new_centroids

        distances = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        quantized_pixels = centroids[labels].astype(np.uint8)
        quantized_arr = quantized_pixels.reshape(h, w, 3)

        result = Image.fromarray(quantized_arr, mode="RGB")

        if self.dither == DitherMethod.FLOYD_STEINBERG:
            result = self._apply_floyd_steinberg(img, centroids)
        elif self.dither == DitherMethod.ORDERED:
            result = result.convert("P", palette=Image.Palette.ADAPTIVE, colors=self.colors)
            result = result.convert("RGB")

        return result

    def _apply_floyd_steinberg(self, img: Image.Image, palette: np.ndarray) -> Image.Image:
        arr = np.array(img, dtype=np.float64)
        h, w, _ = arr.shape

        for y in range(h):
            for x in range(w):
                old_pixel = arr[y, x].copy()
                distances = np.linalg.norm(palette - old_pixel, axis=1)
                new_pixel = palette[np.argmin(distances)]
                arr[y, x] = new_pixel
                quant_error = old_pixel - new_pixel

                if x + 1 < w:
                    arr[y, x + 1] += quant_error * 7 / 16
                if x - 1 >= 0 and y + 1 < h:
                    arr[y + 1, x - 1] += quant_error * 3 / 16
                if y + 1 < h:
                    arr[y + 1, x] += quant_error * 5 / 16
                if x + 1 < w and y + 1 < h:
                    arr[y + 1, x + 1] += quant_error * 1 / 16

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

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
        if img.mode != "RGB":
            img = img.convert("RGB")

        quantized = img.quantize(colors=self.colors, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette()

        colors_list = []
        if palette:
            for i in range(0, min(len(palette), self.colors * 3), 3):
                colors_list.append((palette[i], palette[i + 1], palette[i + 2]))

        return colors_list

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
