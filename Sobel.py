import argparse
import sys
import os
import glob
import time
import numpy as np
import cv2
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

def save_to_excel(results: list[dict], xlsx_path: str, scale: float, swap: bool = False) -> None:
    """Write per-image Sobel results to a scale-specific sheet in an Excel file."""
    if os.path.exists(xlsx_path):
        wb = openpyxl.load_workbook(xlsx_path)
    else:
        wb = openpyxl.Workbook()
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    sheet_name = f"Sobel_{int(scale * 100)}pct" + ("_swap" if swap else "")
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    headers = ["Image", "Width (px)", "Height (px)", "Scale",
               "Road Pixels", "Total Pixels", "Road %",
               "Close Kernel (px, 0 = skipped)", "Processing Time (s)"]
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill(fill_type="solid", fgColor="375623")

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row, r in enumerate(results, 2):
        ws.cell(row=row, column=1, value=r["stem"]).font = Font(name="Arial")
        ws.cell(row=row, column=2, value=r["width"]).font = Font(name="Arial")
        ws.cell(row=row, column=3, value=r["height"]).font = Font(name="Arial")
        ws.cell(row=row, column=4, value=r["scale"]).font = Font(name="Arial")
        ws.cell(row=row, column=5, value=r["road_px"]).font = Font(name="Arial")
        ws.cell(row=row, column=6, value=r["total_px"]).font = Font(name="Arial")
        pct_cell = ws.cell(row=row, column=7, value=round(r["road_px"] / r["total_px"], 4))
        pct_cell.font = Font(name="Arial")
        pct_cell.number_format = "0.0%"
        ws.cell(row=row, column=8, value=r["close_k"]).font = Font(name="Arial")
        ws.cell(row=row, column=9, value=round(r["elapsed"], 3)).font = Font(name="Arial")

    avg_row = len(results) + 2
    ws.cell(row=avg_row, column=1, value="AVERAGE").font = Font(name="Arial", bold=True)
    for col, key in [(2, "width"), (3, "height"), (5, "road_px"), (9, "elapsed")]:
        vals = [r[key] for r in results]
        ws.cell(row=avg_row, column=col, value=round(sum(vals) / len(vals), 3)).font = Font(name="Arial", bold=True)

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 22

    wb.save(xlsx_path)
    print(f"\n  Saved Sobel results ({int(scale*100)}%) : {xlsx_path}")

def preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

    return image, smooth, gray

def apply_sobel(smooth: np.ndarray) -> np.ndarray:
    Gx = cv2.Sobel(smooth, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
    Gy = cv2.Sobel(smooth, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3)

    abs_Gx = cv2.convertScaleAbs(Gx)
    abs_Gy = cv2.convertScaleAbs(Gy)
    mag = cv2.addWeighted(abs_Gx, 0.5, abs_Gy, 0.5, 0)

    return mag

def threshold(mag: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.astype(bool)

def adaptive_close(mask: np.ndarray, close_kernel: int = 0) -> tuple[np.ndarray, int]:
    """
    If close_kernel > 0, use that fixed size. Otherwise measure the largest
    hole strictly inside the mask (excluding background which touches the border)
    and pick a kernel just large enough to fill it.
    Returns the closed mask and the kernel size used (0 = no closing applied).
    """
    mask_uint = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)

    if close_kernel > 0:
        k = close_kernel if close_kernel % 2 == 1 else close_kernel + 1
    else:
        inverted = cv2.bitwise_not(mask_uint)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

        hole_areas = []
        for label in range(1, num_labels):
            component = (labels == label)
            touches_border = (
                component[0, :].any() or component[-1, :].any() or
                component[:, 0].any() or component[:, -1].any()
            )
            if not touches_border:
                hole_areas.append(stats[label, cv2.CC_STAT_AREA])

        if not hole_areas:
            return mask, 0

        largest = max(hole_areas)
        k = int(np.ceil(np.sqrt(largest)))
        if k % 2 == 0:
            k += 1
        k = max(3, k)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask_uint, cv2.MORPH_CLOSE, kernel)

    if mask.max() <= 1:
        return (closed > 0).astype(mask.dtype), k
    return closed.astype(mask.dtype), k

def find_drivable_region(edge_mask: np.ndarray) -> np.ndarray:
    H, W = edge_mask.shape
    road_zone = edge_mask[H//2:, :]
    inverted = (~road_zone).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    if num_labels < 2:
        return np.zeros((H, W), dtype=np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    drivable_zone = (labels == largest_label).astype(np.uint8) * 255

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[H//2:, :] = drivable_zone

    return full_mask

def overlay_and_save(
        image: np.ndarray,
        full_mask: np.ndarray,
        output_path: str,
        alpha: float = 0.4,
) -> np.ndarray:
    green_layer = np.zeros_like(image)
    green_layer[:, :, 1] = 255

    output = image.copy().astype(np.float32)
    mask_bool = full_mask > 0
    output[mask_bool] = (
        (1.0 - alpha) * image[mask_bool].astype(np.float32)
        + alpha * green_layer[mask_bool].astype(np.float32)
    )
    output = np.clip(output, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, output)
    print(f"  Saved overlay     : {output_path}")
    return output

def find_drivable_region_floodfill(edge_mask: np.ndarray) -> np.ndarray:
    """FF region method applied to the Sobel edge mask"""
    H, W = edge_mask.shape
    edge_map = edge_mask.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    walls = cv2.dilate(edge_map, kernel, iterations=2)

    cv2.rectangle(walls, (0, 0), (W - 1, H - 1), 255, thickness=1)

    roi_top = H // 2
    walls[:roi_top, :] = 255

    fillable = cv2.bitwise_not(walls)

    seed_col = W // 2
    seed_row = H - 2
    while seed_row > roi_top and fillable[seed_row, seed_col] == 0:
        seed_row -= 1

    if fillable[seed_row, seed_col] == 0:
        return np.zeros((H, W), dtype=np.uint8)

    flood_mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    cv2.floodFill(fillable, flood_mask, (seed_col, seed_row), 128)

    return (fillable == 128).astype(np.uint8) * 255

def apply_green_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """FF uint8 blend, returned as an array (the caller saves it)."""
    GREEN = np.array([0, 255, 0], dtype=np.uint8)

    output = image.copy()
    green_layer = np.full_like(image, GREEN)

    region = mask > 0
    output[region] = (
        alpha * green_layer[region] + (1 - alpha) * image[region]
    ).astype(np.uint8)

    return output

def collect_image_paths(inputs: list[str]) -> list[str]:
    """Expand a mix of file paths and directory paths into a flat list of image paths."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fname in sorted(os.listdir(inp)):
                if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                    paths.append(os.path.join(inp, fname))
        elif os.path.isfile(inp):
            paths.append(inp)
        else:
            matched = sorted(glob.glob(inp))
            if matched:
                paths.extend(matched)
            else:
                print(f"Warning: '{inp}' not found, skipping.")
    return paths

def process_image(image_path: str, args) -> dict | None:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = args.outdir if args.outdir else os.path.dirname(image_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{stem}] Loading image...")
    try:
        image, smooth, gray = preprocess(image_path)
    except FileNotFoundError as e:
        print(f"  Error: {e}, skipping.")
        return None

    if args.scale != 1.0:
        h, w = image.shape[:2]
        new_w = max(1, int(w * args.scale))
        new_h = max(1, int(h * args.scale))
        image  = cv2.resize(image,  (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray   = cv2.resize(gray,   (new_w, new_h), interpolation=cv2.INTER_AREA)
        smooth = cv2.resize(smooth, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_tag = f"_scale{int(args.scale * 100)}pct"
        print(f"  Resized to {new_w}×{new_h} px ({args.scale:.0%})")
    else:
        scale_tag = ""

    if args.swap:
        scale_tag += "_swap"
        print("  Region method     : Seeded Flood-fill ")
    else:
        print("  Region method     : Maximal Blob Extraction ")

    print(f"  Image size : {image.shape[1]}×{image.shape[0]} px")

    print("  Applying Sobel operator...")
    t_start = time.perf_counter()
    mag = apply_sobel(smooth)

    print("  Thresholding with Otsu's method...")
    edge_mask = threshold(mag)

    print("  Finding drivable region...")
    if args.swap:
        full_mask = find_drivable_region_floodfill(edge_mask)
    else:
        full_mask = find_drivable_region(edge_mask)
    full_mask, close_k = adaptive_close(full_mask, args.close_kernel)
    t_elapsed = time.perf_counter() - t_start

    h, w = image.shape[:2]
    if close_k > 0:
        print(f"  Adaptive close    : kernel={close_k}px")
    else:
        print(f"  Adaptive close    : no holes found, skipped")
    print(f"  Processing time   : {t_elapsed:.3f}s")
    road_px = int(np.sum(full_mask > 0))
    total_px = full_mask.size
    print(f"  Road pixels: {road_px:,} / {total_px:,} ({100 * road_px / total_px:.1f}%)")

    gray_path = os.path.join(out_dir, f"{stem}{scale_tag}_sobel_gray.png")
    cv2.imwrite(gray_path, gray)
    print(f"  Saved grayscale   : {gray_path}")

    blurred_path = os.path.join(out_dir, f"{stem}{scale_tag}_sobel_blurred.png")
    cv2.imwrite(blurred_path, smooth)
    print(f"  Saved blurred     : {blurred_path}")

    edges_path = os.path.join(out_dir, f"{stem}{scale_tag}_sobel_edges.png")
    cv2.imwrite(edges_path, mag)
    print(f"  Saved gradient mag: {edges_path}")

    edges_binary_path = os.path.join(out_dir, f"{stem}{scale_tag}_sobel_edges_binary.png")
    cv2.imwrite(edges_binary_path, edge_mask.astype(np.uint8) * 255)
    print(f"  Saved binary edges: {edges_binary_path}")

    mask_path = os.path.join(out_dir, f"{stem}{scale_tag}_sobel_mask.png")
    cv2.imwrite(mask_path, full_mask)
    print(f"  Saved binary mask : {mask_path}")

    overlay_path = os.path.join(out_dir, f"{stem}{scale_tag}_sobel_result.png")
    if args.swap:
        output = apply_green_overlay(image, full_mask, alpha=args.alpha)
        cv2.imwrite(overlay_path, output)
        print(f"  Saved overlay     : {overlay_path}")
    else:
        output = overlay_and_save(image, full_mask, overlay_path, alpha=args.alpha)

    if not args.no_display:
        try:
            cv2.imshow(f"[{stem}] Drivable Region Detection", output)
            print("  Press any key to advance to next image.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("  No display available - result saved to disk only.")

    return {"stem": stem, "width": w, "height": h, "scale": args.scale,
            "road_px": road_px, "total_px": total_px,
            "close_k": close_k, "elapsed": t_elapsed}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sobel-based drivable region detector — batch mode"
    )
    parser.add_argument(
        "inputs", nargs="*", default=["images"],
        help="One or more image files, directories, or glob patterns. "
             "Defaults to the 'images' folder in the current directory "
             "if nothing is given — every image inside it is processed."
    )
    parser.add_argument(
        "--outdir", default=None, metavar="PATH",
        help="Directory to write output files (default: same dir as each input)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4, metavar="A",
        help="Green overlay opacity in [0, 1] (default: 0.4)"
    )
    parser.add_argument(
        "--close-kernel", type=int, default=0, metavar="K",
        help="Morphological close kernel size (odd int, e.g. 15). "
             "0 = adaptive per image (default: 0)"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, metavar="S",
        help="Resize factor before processing, e.g. 0.5 for 50%% (default: 1.0)"
    )
    parser.add_argument(
        "--swap", action="store_true",
        help="Use Canny's region method (flood fill + uint8 overlay) instead of "
             "Sobel's largest connected component. Outputs get a _swap tag."
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Skip interactive display; save to disk only"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        if args.inputs == ["images"]:
            sys.exit(
                "Error: no images found in './images' (default folder). "
                "Create it and add your image files, or pass a folder/file path instead, "
                "e.g. python Sobel.py path/to/your/folder"
            )
        sys.exit("Error: no valid image files found.")

    print(f"Found {len(image_paths)} image(s) to process.")

    results = []
    for path in image_paths:
        r = process_image(path, args)
        if r:
            results.append(r)

    if results:
        total_time = sum(r["elapsed"] for r in results)
        print(f"\nTotal processing time : {total_time:.3f}s")
        print(f"Average per image     : {total_time / len(results):.3f}s")
        out_dir = args.outdir if args.outdir else "."
        os.makedirs(out_dir, exist_ok=True)
        save_to_excel(results, os.path.join(out_dir, "results.xlsx"), args.scale, args.swap)

    print("\nDone.")

if __name__ == "__main__":
    main()
