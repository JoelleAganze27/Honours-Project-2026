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
    """Write per-image Canny results to a scale-specific sheet in an Excel file."""
    if os.path.exists(xlsx_path):
        wb = openpyxl.load_workbook(xlsx_path)
    else:
        wb = openpyxl.Workbook()
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    sheet_name = f"Canny_{int(scale * 100)}pct" + ("_swap" if swap else "")
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    headers = ["Image", "Width (px)", "Height (px)", "Scale",
               "Close Kernel (px, 0 = skipped)", "Processing Time (s)"]
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill(fill_type="solid", fgColor="2E75B6")

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
        ws.cell(row=row, column=5, value=r["close_k"]).font = Font(name="Arial")
        ws.cell(row=row, column=6, value=round(r["elapsed"], 3)).font = Font(name="Arial")

    avg_row = len(results) + 2
    ws.cell(row=avg_row, column=1, value="AVERAGE").font = Font(name="Arial", bold=True)
    for col, key in [(2, "width"), (3, "height"), (6, "elapsed")]:
        vals = [r[key] for r in results]
        ws.cell(row=avg_row, column=col, value=round(sum(vals) / len(vals), 3)).font = Font(name="Arial", bold=True)

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 22

    wb.save(xlsx_path)
    print(f"\n  Saved Canny results ({int(scale*100)}%) : {xlsx_path}")

def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def gaussian_blur(gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

def compute_gradients(blurred: np.ndarray):
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return magnitude, direction

def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    H, W = magnitude.shape
    nms = np.zeros_like(magnitude)

    angle = np.degrees(direction) % 180

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            a = angle[i, j]
            m = magnitude[i, j]

            if (0 <= a < 22.5) or (157.5 <= a < 180):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if m >= n1 and m >= n2:
                nms[i, j] = m

    return nms

def double_threshold_and_hysteresis(
    nms: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> np.ndarray:

    STRONG = 255
    WEAK = 128

    edge = np.zeros_like(nms, dtype=np.uint8)
    edge[nms > high_thresh] = STRONG
    edge[(nms >= low_thresh) & (nms <= high_thresh)] = WEAK

    H, W = edge.shape
    visited = np.zeros((H, W), dtype=bool)
    queue = list(zip(*np.where(edge == STRONG)))

    while queue:
        i, j = queue.pop()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and not visited[ni, nj]:
                    if edge[ni, nj] == WEAK:
                        edge[ni, nj] = STRONG
                        visited[ni, nj] = True
                        queue.append((ni, nj))

    edge[edge != STRONG] = 0
    return edge

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

def extract_drivable_region(edge_map: np.ndarray) -> np.ndarray:
    H, W = edge_map.shape

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

    drivable_mask = (fillable == 128).astype(np.uint8)
    return drivable_mask

def apply_green_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    GREEN = np.array([0, 255, 0], dtype=np.uint8)

    output = image.copy()
    green_layer = np.full_like(image, GREEN)

    region = mask.astype(bool)
    output[region] = (
        alpha * green_layer[region] + (1 - alpha) * image[region]
    ).astype(np.uint8)

    return output

def extract_drivable_region_components(edge_map: np.ndarray) -> np.ndarray:
    """BE region method applied to the Canny edge map"""
    H, W = edge_map.shape
    edge_mask = edge_map > 0
    road_zone = edge_mask[H // 2:, :]
    inverted = (~road_zone).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )
    if num_labels < 2:
        return np.zeros((H, W), dtype=np.uint8)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    drivable_zone = (labels == largest_label).astype(np.uint8)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[H // 2:, :] = drivable_zone
    return full_mask

def apply_green_overlay_float32(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """BE's float32 blend, returned as an array (no file write)."""
    green_layer = np.zeros_like(image)
    green_layer[:, :, 1] = 255

    output = image.copy().astype(np.float32)
    mask_bool = mask > 0
    output[mask_bool] = (
        (1.0 - alpha) * image[mask_bool].astype(np.float32)
        + alpha * green_layer[mask_bool].astype(np.float32)
    )
    return np.clip(output, 0, 255).astype(np.uint8)

def canny_drivable_overlay(
    image: np.ndarray,
    low_thresh:  float = 80.0,
    high_thresh: float = 200.0,
    alpha:       float = 0.45,
    kernel_size: int = 5,
    sigma:       float = 1.4,
    close_kernel: int = 0,
    swap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, kernel_size, sigma)
    mag, dire = compute_gradients(blurred)
    nms = non_maximum_suppression(mag, dire)
    edge_map = double_threshold_and_hysteresis(nms, low_thresh, high_thresh)
    if swap:
        drivable = extract_drivable_region_components(edge_map)   # BE method
    else:
        drivable = extract_drivable_region(edge_map)              # FF method
    drivable, close_k = adaptive_close(drivable, close_kernel)
    if swap:
        result = apply_green_overlay_float32(image, drivable, alpha)
    else:
        result = apply_green_overlay(image, drivable, alpha)
    return result, edge_map, drivable, blurred, gray, close_k

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
            # Try glob
            matched = sorted(glob.glob(inp))
            if matched:
                paths.extend(matched)
            else:
                print(f"Warning: '{inp}' not found, skipping.")
    return paths

def process_image(image_path: str, args) -> dict | None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image at '{image_path}', skipping.")
        return None

    stem = os.path.splitext(os.path.basename(image_path))[0]

    if args.scale != 1.0:
        h, w = image.shape[:2]
        new_w = max(1, int(w * args.scale))
        new_h = max(1, int(h * args.scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_tag = f"_scale{int(args.scale * 100)}pct"
        print(f"\n[{stem}] Resized to {new_w}×{new_h} px ({args.scale:.0%}) — "
              f"Canny (low={args.low}, high={args.high}) alpha={args.alpha}")
    else:
        scale_tag = ""
        print(f"\n[{stem}] {image.shape[1]}×{image.shape[0]} px — "
              f"Canny (low={args.low}, high={args.high}) alpha={args.alpha}")

    if args.swap:
        scale_tag += "_swap"
        print("  Region method     : Maximal Blob Extraction")
    else:
        print("  Region method     : Seeded Flood-fill")

    t_start = time.perf_counter()
    result, edge_map, drivable, blurred, gray, close_k = canny_drivable_overlay(
        image,
        low_thresh=args.low,
        high_thresh=args.high,
        alpha=args.alpha,
        close_kernel=args.close_kernel,
        swap=args.swap,
    )
    t_elapsed = time.perf_counter() - t_start

    h, w = image.shape[:2]
    if close_k > 0:
        print(f"  Adaptive close    : kernel={close_k}px")
    else:
        print(f"  Adaptive close    : no holes found, skipped")
    print(f"  Processing time   : {t_elapsed:.3f}s")

    # Output paths
    out_dir = args.outdir if args.outdir else os.path.dirname(image_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    def out(suffix):
        return os.path.join(out_dir, f"{stem}{scale_tag}_{suffix}")

    gray_path = out("canny_gray.png")
    cv2.imwrite(gray_path, gray)
    print(f"  Saved grayscale   : {gray_path}")

    blurred_path = out("canny_blurred.png")
    cv2.imwrite(blurred_path, blurred)
    print(f"  Saved blurred     : {blurred_path}")

    edges_path = out("canny_edges.png")
    cv2.imwrite(edges_path, edge_map)
    print(f"  Saved edge map    : {edges_path}")

    mask_path = out("canny_mask.png")
    cv2.imwrite(mask_path, drivable * 255)
    print(f"  Saved binary mask : {mask_path}")

    if args.save:
        overlay_path = out(args.save)
        cv2.imwrite(overlay_path, result)
        print(f"  Saved overlay     : {overlay_path}")
    else:
        cv2.imshow(f"[{stem}] Original", image)
        if not args.no_intermediates:
            cv2.imshow(f"[{stem}] Canny edges", edge_map)
            cv2.imshow(f"[{stem}] Drivable mask", drivable * 255)
        cv2.imshow(f"[{stem}] Drivable region overlay", result)
        print("  Press any key to advance to next image.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {"stem": stem, "width": w, "height": h,
            "scale": args.scale, "close_k": close_k, "elapsed": t_elapsed}

def parse_args():
    p = argparse.ArgumentParser(
        description="Canny edge detection with drivable region green overlay — batch mode"
    )
    p.add_argument(
        "inputs", nargs="*", default=["images"],
        help="One or more image files, directories, or glob patterns. "
             "Defaults to the 'images' folder in the current directory "
             "if nothing is given — every image inside it is processed."
    )
    p.add_argument("--low",    type=float, default=50.0,  help="Low threshold (default: 50)")
    p.add_argument("--high",   type=float, default=100.0, help="High threshold (default: 100)")
    p.add_argument("--alpha",  type=float, default=0.45,  help="Overlay opacity 0-1 (default: 0.45)")
    p.add_argument("--close-kernel", type=int, default=0, metavar="K",
                   help="Morphological close kernel size (odd int, e.g. 15). "
                        "0 = adaptive per image (default: 0)")
    p.add_argument("--scale",  type=float, default=1.0,
                   help="Resize factor before processing, e.g. 0.5 for 50%% (default: 1.0)")
    p.add_argument("--swap", action="store_true",
                   help="Use Sobel's region method (largest connected component + float32 "
                        "overlay) instead of Canny's flood fill. Outputs get a _swap tag.")
    p.add_argument("--save",   type=str,   default=None,
                   help="Suffix for saved overlay filename (e.g. 'overlay.png' → img1_overlay.png). "
                        "Omit to display interactively.")
    p.add_argument("--outdir", type=str,   default=None,
                   help="Directory to write output files (default: same dir as each input)")
    p.add_argument("--no-intermediates", action="store_true",
                   help="Skip showing intermediate edge/mask windows")
    return p.parse_args()

def main():
    args = parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        if args.inputs == ["images"]:
            sys.exit(
                "Error: no images found in './images' (default folder). "
                "Create it and add your image files, or pass a folder/file path instead, "
                "e.g. python Canny.py path/to/your/folder"
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
