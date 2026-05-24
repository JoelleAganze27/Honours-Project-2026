import argparse
import sys
import numpy as np
import cv2

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

def canny_drivable_overlay(
    image: np.ndarray,
    low_thresh:  float = 80.0,
    high_thresh: float = 200.0,
    alpha:       float = 0.45,
    kernel_size: int = 5,
    sigma:       float = 1.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, kernel_size, sigma)
    mag, dire = compute_gradients(blurred)
    nms = non_maximum_suppression(mag, dire)
    edge_map = double_threshold_and_hysteresis(nms, low_thresh, high_thresh)
    drivable = extract_drivable_region(edge_map)
    result = apply_green_overlay(image, drivable, alpha)
    return result, edge_map, drivable

def parse_args():
    p = argparse.ArgumentParser(
        description="Canny edge detection with drivable region green overlay"
    )
    p.add_argument("--image",  required=True, help="Path to input image")
    p.add_argument("--low",    type=float, default=50.0,  help="Low threshold (default: 50)")
    p.add_argument("--high",   type=float, default=100.0, help="High threshold (default: 100)")
    p.add_argument("--alpha",  type=float, default=0.45,  help="Overlay opacity 0-1 (default: 0.45)")
    p.add_argument("--save",   type=str,   default=None,  help="Save result to this path instead of displaying")
    p.add_argument("--no-intermediates", action="store_true",
                   help="Skip showing intermediate edge/mask windows")
    return p.parse_args()

def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Error: could not read image at '{args.image}'")

    print(f"Image loaded: {image.shape[1]}×{image.shape[0]} px")
    print(f"Running Canny (low={args.low}, high={args.high}) alpha={args.alpha}")

    result, edge_map, drivable = canny_drivable_overlay(
        image,
        low_thresh=args.low,
        high_thresh=args.high,
        alpha=args.alpha,
    )

    cv2.imwrite("canny_mask.png", drivable * 255)
    print("Saved binary mask to: canny_mask.png")

    if args.save:
        cv2.imwrite(args.save, result)
        print(f"Saved result to '{args.save}'")
    else:
        cv2.imshow("Original", image)
        if not args.no_intermediates:
            cv2.imshow("Canny edges", edge_map)
            cv2.imshow("Drivable mask", drivable * 255)
        cv2.imshow("Drivable region overlay", result)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()