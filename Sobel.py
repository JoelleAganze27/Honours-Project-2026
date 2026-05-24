import argparse
import sys
import numpy as np
import cv2

def preprocess(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)

    return image, smooth

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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    drivable_zone = cv2.morphologyEx(drivable_zone, cv2.MORPH_CLOSE, kernel)

    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[H//2:, :] = drivable_zone

    return full_mask

def overlay_and_display(
        image: np.ndarray,
        full_mask: np.ndarray,
        alpha: float = 0.4,
        output_path: str = "result.png",
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
    print(f"Saved result to: {output_path}")

    try:
        cv2.imshow("Drivable Region Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("No display available - result saved to disk only")

    return output

def main():
    parser = argparse.ArgumentParser(
        description="Sobel-based drivable region detector with green overlay."
    )
    parser.add_argument(
        "--input", required=True, metavar="PATH",
        help="Path to the input image (jpg, png, etc.)"
    )
    """
    parser.add_argument(
        "--threshold", type=float, default=1000.0,
        help="Low threshold (default: 50)"
    )
    """
    parser.add_argument(
        "--output", default="result.png", metavar="PATH",
        help="Path to save the output image (default: result.png)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.4, metavar="A",
        help="Green overlay opacity in [0, 1] (default: 0.4)"
    )
    args = parser.parse_args()

    print(f"Loading image: {args.input}")
    image, smooth = preprocess(args.input)
    print(f" Image size : {image.shape[1]}×{image.shape[0]} px")

    print("Applying Sobel operator...")
    mag = apply_sobel(smooth)

    print("Thresholding with Otsu's method...")
    edge_mask = threshold(mag)
    # print(f"Threshold = {args.threshold}")

    print("Finding drivable region...")
    full_mask = find_drivable_region(edge_mask)
    road_px = int(np.sum(full_mask > 0))
    total_px = full_mask.size
    print(f" Road pixels: {road_px:,} / {total_px:,} ({100 * road_px / total_px:.1f}%)")

    # Save the binary mask for evaluation
    cv2.imwrite("sobel_mask.png", full_mask)
    print("Saved binary mask to: sobel_mask.png")

    print(f"Compositing overlay (alpha={args.alpha})...")
    overlay_and_display(image, full_mask, alpha=args.alpha, output_path=args.output)

if __name__ == "__main__":
    main()