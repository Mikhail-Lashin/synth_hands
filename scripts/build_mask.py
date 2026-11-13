from pathlib import Path
import cv2
import numpy as np
import tyro


def load_cached_depth(file_path: Path) -> np.ndarray:
    if file_path.suffix == ".npz":
        with np.load(file_path) as data:
            return data["d"].astype(np.float32)
    return np.load(file_path).astype(np.float32)


def extract_hand_component(binary_frame: np.ndarray) -> np.ndarray:
    """Finds the largest connected component corresponding to the hand."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_frame)

    H, W = binary_frame.shape
    best_label = -1
    max_area = 0

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        cx, cy = centroids[label_id]
        # Ignore tiny noise and components touching the far sides outside the central region
        if area > 500 and (W * 0.25 <= cx <= W * 0.75):
            if area > max_area:
                max_area = area
                best_label = label_id

    if best_label == -1:
        return np.zeros_like(binary_frame)

    # Flood-fill / morphological closing to seal small internal holes inside the hand
    hand_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    hand_mask = cv2.morphologyEx(hand_mask, cv2.MORPH_CLOSE, kernel)
    return hand_mask


def main(
    cache_bag_dir: Path = Path("cache/ffs_depth/test_iters4_disp128"),
    output_mask: Path = Path("cache/hand_mask_editable.png"),
    output_bg_mask: Path = Path("cache/bg_mask.png"),
    invert_from_editable: bool = False,
    depth_threshold: float = 0.35,
    min_depth: float = 0.05,
    dilate_px: int = 5,
):
    # Mode 2: Convert manually edited PNG into inverted background mask for the main script
    if invert_from_editable:
        if not output_mask.exists():
            raise FileNotFoundError(f"Editable mask not found: {output_mask}")
        edited = cv2.imread(str(output_mask), cv2.IMREAD_GRAYSCALE)
        bg_mask = cv2.bitwise_not(edited)
        cv2.imwrite(str(output_bg_mask), bg_mask)
        np.save(output_bg_mask.with_suffix(".npy"), bg_mask > 127)
        print(f">>> Inverted mask saved to: {output_bg_mask} and {output_bg_mask.with_suffix('.npy')}")
        return

    # Mode 1: Sweep across all cached frames and accumulate hand mask
    depth_files = sorted(list(cache_bag_dir.glob("depth_*.npz")) + list(cache_bag_dir.glob("depth_*.npy")))
    if not depth_files:
        raise FileNotFoundError(f"No cached depth files found in: {cache_bag_dir}")

    first_depth = load_cached_depth(depth_files[0])
    H, W = first_depth.shape
    x1 = (W - H) // 2
    x2 = x1 + H

    accum_mask = np.zeros((H, W), dtype=np.uint8)
    print(f">>> Sweeping {len(depth_files)} frames to build union hand mask...")

    for idx, f_path in enumerate(depth_files):
        depth = load_cached_depth(f_path)

        valid = (depth >= min_depth) & (depth <= depth_threshold) & ~np.isnan(depth) & ~np.isinf(depth)
        valid[:, :x1] = False
        valid[:, x2:] = False

        hand_comp = extract_hand_component(valid.astype(np.uint8) * 255)
        accum_mask = cv2.bitwise_or(accum_mask, hand_comp)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(depth_files)} frames...")

    # Slightly dilate the final union mask so fast finger tips are never clipped at the boundary
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        accum_mask = cv2.dilate(accum_mask, k)

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask), accum_mask)

    # Also save the inverted background mask right away
    bg_mask = cv2.bitwise_not(accum_mask)
    cv2.imwrite(str(output_bg_mask), bg_mask)
    np.save(output_bg_mask.with_suffix(".npy"), bg_mask > 127)

    print(f">>> Editable union mask (WHITE = hand, BLACK = background) saved to: {output_mask}")
    print(f">>> Ready-to-use background mask saved to: {output_bg_mask.with_suffix('.npy')}")


if __name__ == "__main__":
    tyro.cli(main)
