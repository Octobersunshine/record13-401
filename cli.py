import argparse
import json
import sys
from pathlib import Path

from color_quantizer import ColorQuantizer, QuantizeMethod, DitherMethod, PaletteType


def print_info_table(info_list):
    if not info_list:
        return

    headers = ["文件名", "原始大小", "压缩后大小", "压缩率%", "颜色数", "调色板", "算法", "抖动"]
    rows = []
    for info in info_list:
        if "error" in info:
            rows.append([info.get("filename", "?"), "-", "-", "-", "-", "-", "-", f"错误: {info['error']}"])
        else:
            rows.append([
                info.get("filename", "-"),
                f"{info['original_size'] / 1024:.1f}KB",
                f"{info['compressed_size'] / 1024:.1f}KB",
                f"{info['size_reduction_pct']:.1f}%",
                str(info['unique_colors']),
                info.get('palette_type', 'auto'),
                info['method'],
                info['dither'],
            ])

    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers))))


def main():
    parser = argparse.ArgumentParser(
        description="图片颜色量化工具 - 减少图片颜色数量，用于压缩或艺术效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单张图片量化为 256 色（自动调色板）
  python cli.py input.jpg -o output.png -c 256

  # 使用 NES 复古游戏调色板 + Bayer 抖动
  python cli.py input.jpg -o output.png --palette-type nes -d bayer4

  # 灰度调色板 + 64 色级别
  python cli.py input.jpg -o output.png -c 64 --palette-type grayscale

  # 棕褐色怀旧风 + Sierra 抖动
  python cli.py input.jpg -o output.png --palette-type sepia -d sierra

  # 使用 K-Means 算法 + Floyd-Steinberg 抖动
  python cli.py input.jpg -o output.png -c 64 -m kmeans -d floyd_steinberg

  # 批量处理目录中的所有图片，使用 GameBoy 调色板
  python cli.py -i ./input_dir -o ./output_dir --palette-type gameboy

  # 导出调色板
  python cli.py input.jpg --export-palette palette.png

支持的调色板类型:
  auto, web_safe, grayscale, sepia, vga_16, vga_256, cga, ega, nes,
  gameboy, warm, cool, pastel, neon, monochrome_amber, monochrome_green,
  retro_game, custom
        """,
    )

    parser.add_argument("input", nargs="?", help="输入图片路径（单文件模式）")
    parser.add_argument("-i", "--input-dir", help="输入目录（批量模式）")
    parser.add_argument("-o", "--output", required=True, help="输出图片路径或输出目录")
    parser.add_argument(
        "-c", "--colors", type=int, default=256, help="目标颜色数量 (2-256)，默认 256"
    )
    parser.add_argument(
        "-m", "--method",
        choices=[m.value for m in QuantizeMethod],
        default=QuantizeMethod.MEDIAN_CUT.value,
        help="量化算法: median_cut(默认), max_coverage, kmeans",
    )
    parser.add_argument(
        "-d", "--dither",
        choices=[d.value for d in DitherMethod],
        default=DitherMethod.NONE.value,
        help="抖动方式: none(默认), floyd_steinberg, atkinson, burkes, sierra, sierra_lite, bayer2, bayer4, bayer8, ordered",
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        default=1.0,
        help="抖动强度 (0.0-2.0)，默认 1.0。值越高抖动越明显，0 为无抖动。",
    )
    parser.add_argument(
        "--palette-type",
        choices=[p.value for p in PaletteType if p != PaletteType.CUSTOM],
        default=PaletteType.AUTO.value,
        help="调色板类型: auto(默认, 从图片提取), web_safe, grayscale, sepia, vga_16, vga_256, cga, ega, nes, gameboy, warm, cool, pastel, neon, monochrome_amber, monochrome_green, retro_game",
    )
    parser.add_argument(
        "--custom-palette",
        nargs="+",
        type=str,
        help="自定义调色板，格式为 R,G,B 三元组，空格分隔。如: \"255,0,0 0,255,0 0,0,255\"",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"],
        help="批量处理时的文件扩展名，默认: .jpg .jpeg .png .bmp .gif .tiff .webp",
    )
    parser.add_argument("--export-palette", help="导出调色板图片的路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")

    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("请指定输入图片路径 (input) 或输入目录 (--input-dir)")

    custom_palette = None
    if args.custom_palette:
        try:
            custom_palette = []
            for arg in args.custom_palette:
                for color_str in arg.split():
                    r, g, b = [int(x) for x in color_str.split(",")]
                    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                        raise ValueError(f"颜色值必须在 0-255 之间: {color_str}")
                    custom_palette.append((r, g, b))
        except ValueError as e:
            print(f"自定义调色板解析错误: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        quantizer = ColorQuantizer(
            colors=args.colors,
            method=QuantizeMethod(args.method),
            dither=DitherMethod(args.dither),
            dither_strength=args.dither_strength,
            palette_type=PaletteType(args.palette_type),
            palette=custom_palette,
        )
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)

    all_results = []

    if args.input_dir:
        try:
            results = quantizer.quantize_batch(
                input_dir=args.input_dir,
                output_dir=args.output,
                extensions=args.extensions,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"批量处理失败: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            _, info = quantizer.quantize_image(args.input, args.output)
            info["filename"] = Path(args.input).name
            all_results.append(info)

            if args.export_palette:
                from PIL import Image
                img = Image.open(args.input)
                palette = quantizer.extract_palette(img)
                palette_img = quantizer.create_palette_image(palette)
                palette_img.save(args.export_palette)
                print(f"调色板已导出: {args.export_palette}")
        except Exception as e:
            print(f"处理失败: {e}", file=sys.stderr)
            sys.exit(1)

    if args.json:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
    else:
        print_info_table(all_results)


if __name__ == "__main__":
    main()
