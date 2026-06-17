from pathlib import Path
from color_quantizer import ColorQuantizer, QuantizeMethod, DitherMethod, PaletteType
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


def example_custom_color_count():
    print("=" * 60)
    print("示例 1: 自定义颜色数量（2-256）")
    print("=" * 60)

    sample_dir = Path("./samples")
    sample_dir.mkdir(exist_ok=True)

    input_path = sample_dir / "sample_gradient.png"
    create_sample_image(input_path)
    print(f"已生成测试图片: {input_path}")

    color_counts = [2, 4, 8, 16, 32, 64, 128, 256]
    for c in color_counts:
        output_path = sample_dir / f"colors_{c:03d}.png"
        quantizer = ColorQuantizer(colors=c, dither=DitherMethod.FLOYD_STEINBERG)
        _, info = quantizer.quantize_image(input_path, output_path)
        print(f"  {c:3d} 色 -> {output_path.name} (实际 {info['unique_colors']} 色)")

    print()


def example_palette_types():
    print("=" * 60)
    print("示例 2: 调色板类型对比")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "smooth_gradient.png"
    create_smooth_gradient(input_path, size=256)
    print(f"已生成平滑渐变测试图: {input_path}")
    print()

    palette_tests = [
        (PaletteType.GRAYSCALE, "灰度调色板", 16, DitherMethod.FLOYD_STEINBERG),
        (PaletteType.SEPIA, "棕褐色怀旧", 16, DitherMethod.SIERRA),
        (PaletteType.WEB_SAFE, "Web 安全色", 216, DitherMethod.NONE),
        (PaletteType.VGA_16, "VGA 16 色", 16, DitherMethod.BAYER4),
        (PaletteType.EGA, "EGA 16 色", 16, DitherMethod.BAYER4),
        (PaletteType.CGA, "CGA 4 色", 4, DitherMethod.NONE),
        (PaletteType.NES, "NES 任天堂", 40, DitherMethod.FLOYD_STEINBERG),
        (PaletteType.GAMEBOY, "Game Boy 4 色", 4, DitherMethod.ORDERED),
        (PaletteType.RETRO_GAME, "复古游戏 16 色", 16, DitherMethod.BAYER4),
        (PaletteType.VGA_256, "VGA 256 色", 256, DitherMethod.NONE),
        (PaletteType.WARM, "暖色调", 16, DitherMethod.ATKINSON),
        (PaletteType.COOL, "冷色调", 16, DitherMethod.ATKINSON),
        (PaletteType.PASTEL, "柔和色调", 16, DitherMethod.SIERRA_LITE),
        (PaletteType.NEON, "霓虹色调", 16, DitherMethod.BAYER8),
        (PaletteType.MONOCHROME_AMBER, "琥珀色单色", 10, DitherMethod.FLOYD_STEINBERG),
        (PaletteType.MONOCHROME_GREEN, "绿色单色", 10, DitherMethod.FLOYD_STEINBERG),
        (PaletteType.AUTO, "自动提取 (K-Means)", 16, DitherMethod.FLOYD_STEINBERG),
    ]

    for pt, desc, colors, dither in palette_tests:
        output_path = sample_dir / f"palette_{pt.value}.png"
        quantizer = ColorQuantizer(
            colors=colors,
            palette_type=pt,
            dither=dither,
            dither_strength=1.1,
        )
        _, info = quantizer.quantize_image(input_path, output_path)
        print(f"  {desc:<20} -> {output_path.name} ({info['unique_colors']} 色)")

    print()


def example_custom_palette():
    print("=" * 60)
    print("示例 3: 自定义调色板")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "sample_gradient.png"

    sunset_palette = [
        (0, 0, 64),
        (64, 0, 128),
        (128, 32, 160),
        (192, 64, 128),
        (255, 96, 96),
        (255, 160, 64),
        (255, 224, 96),
        (255, 255, 192),
    ]

    output_path = sample_dir / "custom_sunset.png"
    quantizer = ColorQuantizer(
        colors=len(sunset_palette),
        palette=sunset_palette,
        dither=DitherMethod.FLOYD_STEINBERG,
        dither_strength=1.2,
    )
    _, info = quantizer.quantize_image(input_path, output_path)
    print(f"日落主题自定义调色板: {output_path.name} ({info['unique_colors']} 色)")
    print(f"调色板类型: {info['palette_type']}")

    output_palette = sample_dir / "custom_sunset_palette.png"
    palette_img = quantizer.create_palette_image(sunset_palette, swatch_size=40, columns=4)
    palette_img.save(output_palette)
    print(f"调色板预览图: {output_palette}")
    print()


def example_dither_comparison():
    print("=" * 60)
    print("示例 4: 抖动算法对比（修复色块）")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "smooth_gradient.png"

    dither_methods = [
        (DitherMethod.NONE, "无抖动（可见色块）"),
        (DitherMethod.FLOYD_STEINBERG, "Floyd-Steinberg"),
        (DitherMethod.ATKINSON, "Atkinson"),
        (DitherMethod.BURKES, "Burkes"),
        (DitherMethod.SIERRA, "Sierra"),
        (DitherMethod.SIERRA_LITE, "Sierra Lite"),
        (DitherMethod.BAYER4, "Bayer 4x4"),
        (DitherMethod.BAYER8, "Bayer 8x8"),
    ]

    for dither, desc in dither_methods:
        output_path = sample_dir / f"dither_{dither.value}.png"
        quantizer = ColorQuantizer(
            colors=16,
            palette_type=PaletteType.AUTO,
            dither=dither,
        )
        _, info = quantizer.quantize_image(input_path, output_path)
        print(f"  {desc:<18} -> {output_path.name}")

    print()
    print("提示：对比 dither_none.png 与其他抖动算法的输出，")
    print("      可明显看到抖动如何消除颜色量化产生的色块。")
    print()


def example_color_count_with_palette():
    print("=" * 60)
    print("示例 5: 调色板 + 颜色数量组合（灰度/棕褐色）")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "sample_gradient.png"

    for pt in [PaletteType.GRAYSCALE, PaletteType.SEPIA]:
        for c in [2, 4, 8, 16, 32, 64]:
            output_path = sample_dir / f"{pt.value}_{c:03d}.png"
            quantizer = ColorQuantizer(
                colors=c,
                palette_type=pt,
                dither=DitherMethod.FLOYD_STEINBERG,
            )
            _, info = quantizer.quantize_image(input_path, output_path)
            print(f"  {pt.value:12s} {c:3d} 色 -> {output_path.name}")

    print()


def example_cli_usage_demo():
    print("=" * 60)
    print("示例 6: 作为库使用 - API 调用示例")
    print("=" * 60)

    sample_dir = Path("./samples")
    input_path = sample_dir / "sample_gradient.png"

    print()
    print("代码示例 1: 使用 Game Boy 调色板")
    print("  quantizer = ColorQuantizer(")
    print("      palette_type=PaletteType.GAMEBOY,")
    print("      dither=DitherMethod.BAYER4,")
    print("  )")

    quantizer_gb = ColorQuantizer(
        palette_type=PaletteType.GAMEBOY,
        dither=DitherMethod.BAYER4,
    )
    output_gb = sample_dir / "api_gameboy.png"
    _, info = quantizer_gb.quantize_image(input_path, output_gb)
    print(f"  结果: {output_gb.name}, 调色板={info['palette_type']}")

    print()
    print("代码示例 2: 使用 NES 调色板 + Sierra 抖动 + 自定义强度")
    print("  quantizer = ColorQuantizer(")
    print("      palette_type=PaletteType.NES,")
    print("      dither=DitherMethod.SIERRA,")
    print("      dither_strength=1.3,")
    print("  )")

    quantizer_nes = ColorQuantizer(
        palette_type=PaletteType.NES,
        dither=DitherMethod.SIERRA,
        dither_strength=1.3,
    )
    output_nes = sample_dir / "api_nes.png"
    _, info = quantizer_nes.quantize_image(input_path, output_nes)
    print(f"  结果: {output_nes.name}, 调色板={info['palette_type']}")

    print()


def main():
    print("\n图片颜色量化服务 - 完整版示例演示\n")

    try:
        example_custom_color_count()
        example_palette_types()
        example_custom_palette()
        example_dither_comparison()
        example_color_count_with_palette()
        example_cli_usage_demo()

        print("=" * 60)
        print("所有示例运行成功！查看 ./samples 目录查看结果。")
        print()
        print("颜色数量支持: 2-256 色（通过 colors 参数指定）")
        print()
        print("调色板类型:")
        print("  自动提取: auto（从图片自适应提取）")
        print("  标准色板: web_safe(216), vga_16(16), vga_256(256)")
        print("  复古硬件: cga(4), ega(16), nes(56), gameboy(4)")
        print("  风格化: grayscale, sepia, warm, cool, pastel, neon")
        print("  单色: monochrome_amber, monochrome_green")
        print("  自定义: 传入 RGB 元组列表")
        print()
        print("CLI 快速命令:")
        print("  python cli.py input.jpg -o output.png -c 32 --palette-type nes -d sierra")
        print("  python cli.py input.jpg -o output.png -c 16 --palette-type sepia")
        print("  python cli.py input.jpg -o output.png -c 64 --palette-type grayscale")
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
