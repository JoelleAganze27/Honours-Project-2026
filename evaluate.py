import numpy as np
import cv2
import json

def labelme_to_mask(json_path: str, output_path: str = "ground_truth.png") -> np.ndarray:
    with open(json_path, "r") as f:
        data = json.load(f)

    H = data["imageHeight"]
    W = data["imageWidth"]
    mask = np.zeros((H, W), dtype=np.uint8)

    for shape in data["shapes"]:
        if shape["label"] == "drivable":
            points = np.array(shape["points"], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    cv2.imwrite(output_path, mask)
    print(f"Ground truth mask saved to: {output_path}")
    return mask


def pixel_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    correct = np.sum(pred == gt)
    total = gt.size

    return correct / total


def main():
    gt_mask = labelme_to_mask("road.json", "ground_truth.png")

    sobel_mask = cv2.imread("sobel_mask.png", cv2.IMREAD_GRAYSCALE)
    canny_mask = cv2.imread("canny_mask.png", cv2.IMREAD_GRAYSCALE)

    if sobel_mask is None:
        print("Error: sobel_mask.png not found. Run Sobel.py first.")
        return
    if canny_mask is None:
        print("Error: canny_mask.png not found. Run Canny.py first.")
        return

    if sobel_mask.shape != gt_mask.shape:
        sobel_mask = cv2.resize(sobel_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    if canny_mask.shape != gt_mask.shape:
        canny_mask = cv2.resize(canny_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)


    sobel_pa = pixel_accuracy(sobel_mask, gt_mask)
    canny_pa = pixel_accuracy(canny_mask, gt_mask)

    print("\n========== Pixel Accuracy Results ==========")
    print(f"  Sobel Pixel Accuracy : {sobel_pa:.4f}  ({sobel_pa * 100:.1f}%)")
    print(f"  Canny Pixel Accuracy : {canny_pa:.4f}  ({canny_pa * 100:.1f}%)")
    print("============================================")

    if sobel_pa > canny_pa:
        print("  Winner: Sobel")
    elif canny_pa > sobel_pa:
        print("  Winner: Canny")
    else:
        print("  Result: Tie")


if __name__ == "__main__":
    main()