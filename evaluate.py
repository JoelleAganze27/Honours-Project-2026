import numpy as np
import cv2
import json
import os
import glob
import sys
import argparse
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

def save_to_excel(results: list[dict], xlsx_path: str, scale: float, swap: bool = False) -> None:
    """Write evaluation results to a scale-specific sheet in an Excel file."""
    if os.path.exists(xlsx_path):
        wb = openpyxl.load_workbook(xlsx_path)
    else:
        wb = openpyxl.Workbook()
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    sheet_name = f"Evaluate_{int(scale * 100)}pct" + ("_swap" if swap else "")
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    headers = ["Image", "Sobel PA", "Canny PA", "Sobel IoU", "Canny IoU",
               "PA Winner", "IoU Winner"]
    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill(fill_type="solid", fgColor="7030A0")

    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row, r in enumerate(results, 2):
        ws.cell(row=row, column=1, value=r["stem"]).font = Font(name="Arial")
        for col, key in [(2, "sobel_pa"), (3, "canny_pa"),
                         (4, "sobel_iou"), (5, "canny_iou")]:
            cell = ws.cell(row=row, column=col, value=round(r[key], 4))
            cell.font = Font(name="Arial")
            cell.number_format = "0.0%"
        pa_w  = "Sobel" if r["sobel_pa"]  > r["canny_pa"]  else "Canny" if r["canny_pa"]  > r["sobel_pa"]  else "Tie"
        iou_w = "Sobel" if r["sobel_iou"] > r["canny_iou"] else "Canny" if r["canny_iou"] > r["sobel_iou"] else "Tie"
        ws.cell(row=row, column=6, value=pa_w).font  = Font(name="Arial")
        ws.cell(row=row, column=7, value=iou_w).font = Font(name="Arial")

    avg_row = len(results) + 2
    ws.cell(row=avg_row, column=1, value="AVERAGE").font = Font(name="Arial", bold=True)
    for col, key in [(2, "sobel_pa"), (3, "canny_pa"),
                     (4, "sobel_iou"), (5, "canny_iou")]:
        val = round(float(np.mean([r[key] for r in results])), 4)
        cell = ws.cell(row=avg_row, column=col, value=val)
        cell.font = Font(name="Arial", bold=True)
        cell.number_format = "0.0%"

    avg_sobel_pa  = np.mean([r["sobel_pa"]  for r in results])
    avg_canny_pa  = np.mean([r["canny_pa"]  for r in results])
    avg_sobel_iou = np.mean([r["sobel_iou"] for r in results])
    avg_canny_iou = np.mean([r["canny_iou"] for r in results])
    ws.cell(row=avg_row, column=6,
            value="Sobel" if avg_sobel_pa  > avg_canny_pa  else "Canny" if avg_canny_pa  > avg_sobel_pa  else "Tie"
            ).font = Font(name="Arial", bold=True)
    ws.cell(row=avg_row, column=7,
            value="Sobel" if avg_sobel_iou > avg_canny_iou else "Canny" if avg_canny_iou > avg_sobel_iou else "Tie"
            ).font = Font(name="Arial", bold=True)

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 20

    try:
        wb.save(xlsx_path)
    except PermissionError:
        import time
        base, ext = os.path.splitext(xlsx_path)
        xlsx_path = f"{base}_{time.strftime('%H%M%S')}{ext}"
        wb.save(xlsx_path)
        print("\n  WARNING: results.xlsx is open in Excel, so this run was saved to a new file.")
    print(f"\n  Saved Evaluate results ({int(scale*100)}%) : {xlsx_path}")

def labelme_to_mask(json_path: str, save_path: str | None = None) -> np.ndarray:
    with open(json_path, "r") as f:
        data = json.load(f)

    H = data["imageHeight"]
    W = data["imageWidth"]
    mask = np.zeros((H, W), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["label"].lower().startswith("drivable"):
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    if save_path:
        cv2.imwrite(save_path, mask)
        print(f"  Saved ground truth mask: {save_path}")

    return mask

def pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    return np.sum(pred == gt) / gt.size

def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    gt   = (gt_mask   > 0).astype(np.uint8)
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0

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

def evaluate_image(image_path: str, mask_dir: str | None, scale_tag: str = "") -> dict | None:
    """
    For a given image path, locate:
      - Ground truth: <stem>.json  (same dir as image, or --mask-dir)
      - Sobel mask  : <stem><scale_tag>_sobel_mask.png
      - Canny mask  : <stem><scale_tag>_canny_mask.png
    Returns a dict with the results, or None if files are missing.
    """
    stem     = os.path.splitext(os.path.basename(image_path))[0]
    img_dir  = os.path.dirname(image_path) or "."
    search   = mask_dir if mask_dir else img_dir

    json_path   = os.path.join(img_dir, f"{stem}.json")
    sobel_path  = os.path.join(search,  f"{stem}{scale_tag}_sobel_mask.png")
    canny_path  = os.path.join(search,  f"{stem}{scale_tag}_canny_mask.png")

    missing = []
    if not os.path.isfile(json_path):  missing.append(json_path)
    if not os.path.isfile(sobel_path): missing.append(sobel_path)
    if not os.path.isfile(canny_path): missing.append(canny_path)

    if missing:
        print(f"  [SKIP] {stem} — missing files:")
        for m in missing:
            print(f"    {m}")
        return None

    gt_save    = os.path.join(search, f"{stem}_gt_mask.png")
    gt_mask    = labelme_to_mask(json_path, save_path=gt_save)
    sobel_mask = cv2.imread(sobel_path, cv2.IMREAD_GRAYSCALE)
    canny_mask = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)

    if sobel_mask.shape != gt_mask.shape:
        sobel_mask = cv2.resize(sobel_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
    if canny_mask.shape != gt_mask.shape:
        canny_mask = cv2.resize(canny_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    sobel_pa = pixel_accuracy(sobel_mask, gt_mask)
    canny_pa = pixel_accuracy(canny_mask, gt_mask)
    sobel_iou = iou(sobel_mask, gt_mask)
    canny_iou = iou(canny_mask, gt_mask)

    return {"stem": stem, "sobel_pa": sobel_pa, "canny_pa": canny_pa,
            "sobel_iou": sobel_iou, "canny_iou": canny_iou}

def load_mask(path: str, gt_mask: np.ndarray) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m.shape != gt_mask.shape:
        m = cv2.resize(m, (gt_mask.shape[1], gt_mask.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    return m

def pick_winner(a: float, b: float, name_a: str, name_b: str) -> str:
    return name_a if a > b else name_b if b > a else "Tie"

# Labels: CC = largest connected component, FF = flood fill.
# Sobel's own method is CC, Canny's own method is FF; --swap masks use the other.
METHOD_KEYS = ["sobel_be", "sobel_ff", "canny_ff", "canny_be"]

def evaluate_methods(image_path: str, mask_dir: str | None, base_tag: str = "") -> dict | None:
    """
    Evaluate all four masks for one image against the ground truth:
      Sobel + BE
      Sobel + FF
      Canny + FF
      Canny + BE
    """
    stem    = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.dirname(image_path) or "."
    search  = mask_dir if mask_dir else img_dir

    json_path = os.path.join(img_dir, f"{stem}.json")
    paths = {
        "sobel_be": os.path.join(search, f"{stem}{base_tag}_sobel_mask.png"),
        "sobel_ff": os.path.join(search, f"{stem}{base_tag}_swap_sobel_mask.png"),
        "canny_ff": os.path.join(search, f"{stem}{base_tag}_canny_mask.png"),
        "canny_be": os.path.join(search, f"{stem}{base_tag}_swap_canny_mask.png"),
    }

    missing = [p for p in [json_path, *paths.values()] if not os.path.isfile(p)]
    if missing:
        print(f"  [SKIP] {stem} — missing files:")
        for m in missing:
            print(f"    {m}")
        return None

    gt_mask = labelme_to_mask(json_path, save_path=os.path.join(search, f"{stem}_gt_mask.png"))
    r = {"stem": stem}
    for key, p in paths.items():
        m = load_mask(p, gt_mask)
        r[f"{key}_pa"]  = pixel_accuracy(m, gt_mask)
        r[f"{key}_iou"] = iou(m, gt_mask)
    return r

def save_methods_to_excel(results: list[dict], xlsx_path: str, scale: float) -> None:
    """Write the method-vs-method comparison to a Methods_<scale>pct sheet."""
    if os.path.exists(xlsx_path):
        wb = openpyxl.load_workbook(xlsx_path)
    else:
        wb = openpyxl.Workbook()
        if "Sheet" in wb.sheetnames:
            del wb["Sheet"]

    sheet_name = f"Methods_{int(scale * 100)}pct"
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)

    headers = ["Image",
               "Sobel BE PA", "Sobel FF PA", "Sobel BE IoU", "Sobel FF IoU", "Sobel PA Winner", "Sobel IoU Winner",
               "Canny FF PA", "Canny BE PA", "Canny FF IoU", "Canny BE IoU", "Canny PA Winner", "Canny IoU Winner"]
    keys = [("sobel_be", "pa"), ("sobel_ff", "pa"), ("sobel_be", "iou"), ("sobel_ff", "iou")]
    keys2 = [("canny_ff", "pa"), ("canny_be", "pa"), ("canny_ff", "iou"), ("canny_be", "iou")]

    header_font = Font(name="Arial", bold=True, color="FFFFFF")
    header_fill = PatternFill(fill_type="solid", fgColor="C55A11")
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    def fill_row(row, vals, bold=False):
        f = Font(name="Arial", bold=bold)
        ws.cell(row=row, column=1, value=vals["stem"]).font = f
        for i, (k, m) in enumerate(keys):
            c = ws.cell(row=row, column=2 + i, value=round(vals[f"{k}_{m}"], 4))
            c.font, c.number_format = f, "0.0%"
        ws.cell(row=row, column=6, value=pick_winner(vals["sobel_be_pa"],  vals["sobel_ff_pa"],  "BE", "FF")).font = f
        ws.cell(row=row, column=7, value=pick_winner(vals["sobel_be_iou"], vals["sobel_ff_iou"], "BE", "FF")).font = f
        for i, (k, m) in enumerate(keys2):
            c = ws.cell(row=row, column=8 + i, value=round(vals[f"{k}_{m}"], 4))
            c.font, c.number_format = f, "0.0%"
        ws.cell(row=row, column=12, value=pick_winner(vals["canny_ff_pa"],  vals["canny_be_pa"],  "FF", "BE")).font = f
        ws.cell(row=row, column=13, value=pick_winner(vals["canny_ff_iou"], vals["canny_be_iou"], "FF", "BE")).font = f

    for row, r in enumerate(results, 2):
        fill_row(row, r)

    avg = {"stem": "AVERAGE"}
    for key in METHOD_KEYS:
        for m in ("pa", "iou"):
            avg[f"{key}_{m}"] = float(np.mean([r[f"{key}_{m}"] for r in results]))
    fill_row(len(results) + 2, avg, bold=True)

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 18

    try:
        wb.save(xlsx_path)
    except PermissionError:
        import time
        base, ext = os.path.splitext(xlsx_path)
        xlsx_path = f"{base}_{time.strftime('%H%M%S')}{ext}"
        wb.save(xlsx_path)
        print("\n  WARNING: results.xlsx is open in Excel, so this run was saved to a new file.")
    print(f"\n  Saved Methods results ({int(scale*100)}%) : {xlsx_path}")

def run_method_comparison(image_paths: list[str], args, base_tag: str) -> None:
    results = []
    for path in image_paths:
        r = evaluate_methods(path, args.mask_dir, base_tag)
        if r:
            results.append(r)
    if not results:
        print("No images could be evaluated (run Sobel.py and Canny.py both with and without --swap first).")
        return

    col = 22
    def block(title, k1, n1, k2, n2):
        print("\n" + "=" * 86)
        print(title)
        print("=" * 86)
        print(f"{'Image':<{col}}  {n1+' PA':>12}  {n2+' PA':>12}  {n1+' IoU':>12}  {n2+' IoU':>12}  {'IoU win':>8}")
        for r in results:
            print(f"{r['stem']:<{col}}  {r[k1+'_pa']:>11.1%}  {r[k2+'_pa']:>11.1%}"
                  f"  {r[k1+'_iou']:>11.1%}  {r[k2+'_iou']:>11.1%}"
                  f"  {pick_winner(r[k1+'_iou'], r[k2+'_iou'], n1.split()[1], n2.split()[1]):>8}")
        if len(results) > 1:
            avg = lambda k, m: float(np.mean([r[f"{k}_{m}"] for r in results]))
            print("-" * 86)
            print(f"{'AVERAGE':<{col}}  {avg(k1, 'pa'):>11.1%}  {avg(k2, 'pa'):>11.1%}"
                  f"  {avg(k1, 'iou'):>11.1%}  {avg(k2, 'iou'):>11.1%}")

    block("SOBEL: connected components (own) vs flood fill (swap)", "sobel_be", "Sobel BE", "sobel_ff", "Sobel FF")
    block("CANNY: flood fill (own) vs connected components (swap)", "canny_ff", "Canny FF", "canny_be", "Canny BE")

    mask_dir = args.mask_dir if args.mask_dir else "."
    save_methods_to_excel(results, os.path.join(mask_dir, "results.xlsx"), args.scale)

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Sobel and Canny drivable-region masks against LabelMe ground truth — batch mode"
    )
    p.add_argument(
        "inputs", nargs="*", default=["images"],
        help="One or more original image files, directories, or glob patterns. "
             "Defaults to the 'images' folder in the current directory if nothing "
             "is given. Ground truth JSON and masks are looked up by stem name."
    )
    p.add_argument(
        "--mask-dir", default=None, metavar="PATH",
        help="Directory containing the *_sobel_mask.png and *_canny_mask.png files "
             "(default: same directory as each input image)"
    )
    p.add_argument(
        "--scale", type=float, default=1.0, metavar="S",
        help="Scale factor used when generating masks, e.g. 0.5. "
             "Tells the evaluator to look for scale-tagged filenames (default: 1.0)"
    )
    p.add_argument(
        "--swap", action="store_true",
        help="Evaluate the masks produced with --swap (looks for _swap-tagged filenames)"
    )
    p.add_argument(
        "--compare-methods", action="store_true",
        help="Compare Sobel vs Sobel --swap and Canny vs Canny --swap against the ground "
             "truth (needs masks from both normal and --swap runs at this --scale)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        if args.inputs == ["images"]:
            sys.exit(
                "Error: no images found in './images' (default folder). "
                "Create it and add your image files, or pass a folder/file path instead, "
                "e.g. python evaluate.py path/to/your/folder"
            )
        sys.exit("Error: no valid image files found.")

    print(f"Found {len(image_paths)} image(s) to evaluate.\n")

    scale_tag = f"_scale{int(args.scale * 100)}pct" if args.scale != 1.0 else ""
    if args.compare_methods:
        run_method_comparison(image_paths, args, scale_tag)
        return
    if args.swap:
        scale_tag += "_swap"

    results = []
    for path in image_paths:
        r = evaluate_image(path, args.mask_dir, scale_tag)
        if r:
            results.append(r)

    if not results:
        print("No images could be evaluated (check that JSON and mask files exist).")
        return

    # Per-image results
    col = 28
    print("\n" + "=" * 85)
    print(f"{'Image':<{col}}  {'Sobel PA':>10}  {'Canny PA':>10}  {'Sobel IoU':>10}  {'Canny IoU':>10}")
    print("=" * 85)
    for r in results:
        print(f"{r['stem']:<{col}}  {r['sobel_pa']:>9.1%}  {r['canny_pa']:>9.1%}"
              f"  {r['sobel_iou']:>9.1%}  {r['canny_iou']:>9.1%}")
    print("=" * 85)

    # Aggregate summary
    if len(results) > 1:
        avg_sobel_pa  = np.mean([r["sobel_pa"]  for r in results])
        avg_canny_pa  = np.mean([r["canny_pa"]  for r in results])
        avg_sobel_iou = np.mean([r["sobel_iou"] for r in results])
        avg_canny_iou = np.mean([r["canny_iou"] for r in results])
        print(f"\n{'AVERAGE':<{col}}  {avg_sobel_pa:>9.1%}  {avg_canny_pa:>9.1%}"
              f"  {avg_sobel_iou:>9.1%}  {avg_canny_iou:>9.1%}")
        pa_winner  = "Sobel" if avg_sobel_pa  > avg_canny_pa  else "Canny" if avg_canny_pa  > avg_sobel_pa  else "Tie"
        iou_winner = "Sobel" if avg_sobel_iou > avg_canny_iou else "Canny" if avg_canny_iou > avg_sobel_iou else "Tie"
        print(f"\nPA  winner across {len(results)} images: {pa_winner}")
        print(f"IoU winner across {len(results)} images: {iou_winner}")

    mask_dir = args.mask_dir if args.mask_dir else "."
    save_to_excel(results, os.path.join(mask_dir, "results.xlsx"), args.scale, args.swap)

if __name__ == "__main__":
    main()
