"""Teaching examples from the VidLens YOLOv1 lesson; no model or dataset required."""

from math import sqrt


def decode_box(column, row, x, y, w, h, size=(448, 448), grid=7):
    width, height = size
    cx, cy = (column + x) / grid * width, (row + y) / grid * height
    bw, bh = w * width, h * height
    return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def nms(candidates, threshold=0.5):
    """One class at a time; each row is (name, score, xyxy). Suppress IoU > threshold."""
    remaining = sorted(candidates, key=lambda item: item[1], reverse=True)
    kept = []
    while remaining:
        best, *remaining = remaining
        kept.append(best[0])
        remaining = [item for item in remaining if iou(best[2], item[2]) <= threshold]
    return kept


def main():
    truth = decode_box(2, 3, 0.5, 0.75, 128 / 448, 160 / 448)
    predictors = [(98, 158, 226, 318), (64, 128, 192, 288)]
    candidates = [("A", 0.72, (96, 160, 224, 320)),
                  ("B", 0.64, (104, 168, 232, 328)),
                  ("C", 0.65, (300, 80, 380, 160))]
    return {
        "decoded_xyxy": truth,
        "predictor_iou": [iou(box, truth) for box in predictors],
        "class_scores": [0.8 * 0.9, 0.8 * 0.4],
        "sqrt_width_losses": [(sqrt(0.04) - sqrt(0.09)) ** 2,
                              (sqrt(0.64) - sqrt(0.69)) ** 2],
        "parameters_direct": 3 * 3 * 256 * 256,
        "parameters_bottleneck": 256 * 64 + 3 * 3 * 64 * 256,
        "nms_pair_iou": iou(candidates[0][2], candidates[1][2]),
        "nms_at_0_5": nms(candidates, 0.5),
        "nms_at_0_9": nms(candidates, 0.9),
        "exercise_output": (7, 7, 3 * 5 + 20),
        "exercise_box": decode_box(4, 1, 0.25, 0.5, 0.25, 0.125),
        "exercise_score": 0.6 * 0.8,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(main(), ensure_ascii=False, indent=2))
