from pathlib import Path
from color_quantizer import ColorQuantizer, QuantizeMethod, DitherMethod
from PIL import Image
import numpy as np


def create_sample_image(output_path: Path, size: int = 300):
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    for y in range(size):
        for x in range(size):
            r = int(255 * x / size)
            g = int(255 * y / size)
            b = int(255 * (1 - abs(x - y) / size))
            pixels[x, y] = (r, g, b)
    img.save(output_path)
    return img


def create_smooth_gradient(output_path: Path, size: int = 400):
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    for y in range(size):
        t = y / size
        for x in range(size):
            s = x / size
            r = int(128 + 100 * np.sin(np.pi * s) * np.cos(np.pi * t * 0.5))
            g = int(128 + 80 * np.sin(np.pi * t * 1.5))
            b = int(128 + 90 * np.cos(np.pi * s * t * 2))
            pixels[x, y] = (np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255))
    img.save(output_path)
    return img


def example_basic_usage():
    print("=" * 60)
    print("示例 1: 基础使用 - 256 色量化")
    print("=" * 60)

    sample_dir = Path("./samples")
    sample_dir.mkdir(exist_ok=True)

    input_path = sample_dir / "sample_gradient.png"
    output_path = sample_dir / "sample_256color.png"

    create_sample_image(input_path)
    print(f"已生成测试图片: {input_path}")

    quantizer = ColorQuantizer(colors=256)
    result, info = quantizer.quantize_image(input_path, output_path)

    print(f"输出图片: {output_path}")
    print(f"原始大小: {info['original_size'] / 1024:.2f} KB")
    print(f"压缩后大小: {info['compressed_size'] / 1024:.2f} KB")
    print(f"压缩率: {info['size_reduction_pct']:.2f}%")
    print(f"实际颜色数: {info['unique_colors']}")
    print()


def example_dither_comparison():
    print("=" * 60)
    print("示例 2: 抖动算法对比（修复色块问题）")
    print("=" * 60)

    sample_dir = Path("./samples")
    sample_dir.mkdir(exist_ok=True)

    input_path = sample_dir / "smooth_gradient.png"
    create_smooth_gradient(input_path, size=300)
    print(f"已生成平滑渐变测试图: {input_path}")
    print()

    dither_methods = [
        (DitherMethod.NONE, "无抖动（可见色块）"),
        (DitherMethod.FLOYD_STEINBERG, "Floyd-Steinberg 抖动"),
        (DitherMethod.ATKINSON, "Atkinson 抖动"),
        (DitherMethod.BURKES, "Burkes 抖动"),
        (DitherMethod.SIERRA, "Sierra 抖动"),
        (DitherMethod.SIERRA_LITE, "Sierra Lite 抖动"),
        (DitherMethod.BAYER4, "Bayer 4x4 有序抖动"),
        (DitherMethod.BAYER8, "Bayer 8x8 有序抖动"),
    ]

    for dither, desc in dither_methods:
        output_path = sample_dir / f"dither_{dither.value}.png"
        quantizer = ColorQuantizer(
            colors=16,
            method=QuantizeMethod.MEDIAN_CUT,
            dither=dither,
        )
        _, info = quantizer.quantize_image(input_path, output_path)
        print(f"  {desc:<30} -> {output_path.name} ({info['unique_colors']} 色)")

    print()
    print("提示：对比 dither_none.png（有色块）与其他抖动算法的输出，")
    print("      可明显看到抖动算法如何消除颜色量化产生的色块。")
    print()


def example_dither_strength():
    print("=" * 60)
    print("示例 3: 抖动强度调节")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "smooth_gradient.png"

    strengths = [0.3, 0.7, 1.0, 1.5]
    for strength in strengths:
        output_path = sample_dir / f"strength_{strength}.png"
        quantizer = ColorQuantizer(
            colors=16,
            dither=DitherMethod.FLOYD_STEINBERG,
            dither_strength=strength,
        )
        _, info = quantizer.quantize_image(input_path, output_path)
        print(f"  抖动强度 {strength:<4} -> {output_path.name}")

    print()
    print("提示：强度越低，色块越明显；强度越高，颗粒感越强。")
    print("      默认 1.0 通常是最佳平衡点。")
    print()


def example_custom_palette():
    print("=" * 60)
    print("示例 4: 自定义调色板 + Bayer 有序抖动（复古风格）")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "sample_gradient.png"
    output_path = sample_dir / "sample_retro_bayer.png"

    retro_palette = [
        (0, 0, 0),
        (255, 255, 255),
        (136, 0, 0),
        (170, 255, 238),
        (204, 68, 204),
        (0, 204, 85),
        (0, 0, 170),
        (238, 238, 119),
        (221, 136, 85),
        (102, 68, 0),
        (255, 119, 119),
        (51, 51, 51),
        (119, 119, 119),
        (170, 255, 102),
        (0, 136, 255),
        (187, 187, 187),
    ]

    quantizer = ColorQuantizer(
        colors=len(retro_palette),
        palette=retro_palette,
        dither=DitherMethod.BAYER4,
        dither_strength=1.2,
    )
    result, info = quantizer.quantize_image(input_path, output_path)

    print(f"输出图片: {output_path}")
    print(f"实际颜色数: {info['unique_colors']}")
    print(f"抖动方式: {info['dither']}")
    print()


def example_palette_extraction():
    print("=" * 60)
    print("示例 5: 提取调色板并生成调色板预览图")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "sample_gradient.png"
    palette_output = sample_dir / "extracted_palette.png"

    img = Image.open(input_path)

    quantizer = ColorQuantizer(colors=32)
    palette = quantizer.extract_palette(img)

    print(f"提取了 {len(palette)} 种颜色:")
    for i, color in enumerate(palette[:10]):
        print(f"  颜色 {i + 1}: RGB{color}")
    if len(palette) > 10:
        print(f"  ... 还有 {len(palette) - 10} 种颜色")

    palette_img = quantizer.create_palette_image(palette, swatch_size=40, columns=8)
    palette_img.save(palette_output)
    print(f"\n调色板预览图已保存: {palette_output}")
    print()


def example_all_methods_kmeans():
    print("=" * 60)
    print("示例 6: K-Means + 多种抖动对比")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "smooth_gradient.png"

    dither_options = [DitherMethod.NONE, DitherMethod.FLOYD_STEINBERG, DitherMethod.BAYER8]
    for dither in dither_options:
        output_path = sample_dir / f"kmeans_{dither.value}.png"
        quantizer = ColorQuantizer(
            colors=32,
            method=QuantizeMethod.KMEANS,
            dither=dither,
        )
        _, info = quantizer.quantize_image(input_path, output_path)
        print(f"  K-Means + {dither.value:<18} -> {output_path.name}")
    print()


def example_batch_processing():
    print("=" * 60)
    print("示例 7: 批量处理（带抖动）")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_dir = sample_dir / "batch_input"
    output_dir = sample_dir / "batch_output_dithered"

    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    for i in range(3):
        test_img = Image.new("RGB", (200, 200))
        pixels = test_img.load()
        for y in range(200):
            for x in range(200):
                r = (x * (i + 1)) % 256
                g = (y * (i + 2)) % 256
                b = ((x + y) * (i + 3)) % 256
                pixels[x, y] = (r, g, b)
        test_img.save(input_dir / f"test_{i}.png")

    print(f"已在 {input_dir} 中生成 3 张测试图片")

    quantizer = ColorQuantizer(
        colors=64,
        method=QuantizeMethod.MAX_COVERAGE,
        dither=DitherMethod.FLOYD_STEINBERG,
    )
    results = quantizer.quantize_batch(input_dir, output_dir)

    print(f"\n批量处理完成，输出目录: {output_dir}")
    for r in results:
        if "error" in r:
            print(f"  {r['filename']}: 错误 - {r['error']}")
        else:
            print(f"  {r['filename']}: {r['size_reduction_pct']:.1f}% 压缩率, {r['unique_colors']} 色, 抖动={r['dither']}")
    print()


def main():
    print("\n图片颜色量化服务 - 抖动修复版示例演示\n")

    try:
        example_basic_usage()
        example_dither_comparison()
        example_dither_strength()
        example_custom_palette()
        example_palette_extraction()
        example_all_methods_kmeans()
        example_batch_processing()

        print("=" * 60)
        print("所有示例运行成功！查看 ./samples 目录查看结果。")
        print()
        print("抖动算法说明：")
        print("  误差扩散类：floyd_steinberg, atkinson, burkes, sierra, sierra_lite")
        print("    - 优点：过渡自然，色块消除效果好")
        print("    - 缺点：可能有噪点感，速度较慢")
        print("  有序抖动类：bayer2, bayer4, bayer8, ordered")
        print("    - 优点：规则纹理，速度快，复古风格")
        print("    - 缺点：可能有明显的网格纹理")
        print()
        print("使用建议：")
        print("  - 追求最佳质量：sierra 或 floyd_steinberg")
        print("  - 复古游戏风格：bayer4 或 bayer8")
        print("  - 平衡速度与质量：atkinson 或 sierra_lite")
        print("=" * 60)
    except ImportError as e:
        print(f"依赖缺失: {e}")
        print("请先运行: pip install -r requirements.txt")
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
