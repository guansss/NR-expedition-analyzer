from __future__ import annotations

import argparse
import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import TextRecognition
import yaml
from jsonschema import validate


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_ROWS_PER_SCREENSHOT = 8
NIGHTLORD_REGION_LEFT_RATIO = 0.078
NIGHTLORD_REGION_TOP_RATIO = 0.215
NIGHTLORD_SEARCH_WIDTH_RATIO = 0.45
RECOGNITION_DEBUG_DIRNAME = "debug"
POINTS_ALLOWED_CHARS = set("0123456789")

hide_date = False

OCR_ENGINE: TextRecognition | None = None


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize expeditions from screenshots.")
    parser.add_argument(
        "-n",
        type=int,
        help="Number of screenshots to process (default: all).",
    )
    parser.add_argument(
        "--hide-date",
        action="store_true",
        help="Hide expedition's date in saved images and generate webpage.",
    )
    return parser.parse_args()


def get_ocr_engine() -> TextRecognition:
    global OCR_ENGINE
    if OCR_ENGINE is None:
        OCR_ENGINE = TextRecognition()
    return OCR_ENGINE


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_templates(template_dir: Path, use_color: bool = False) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    read_mode = cv2.IMREAD_UNCHANGED if use_color else cv2.IMREAD_GRAYSCALE
    for path in sorted(template_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        raw_image = cv2.imread(str(path), read_mode)
        if raw_image is None:
            logging.warning("Failed to load template: %s", path)
            continue

        image = raw_image
        mask = None
        if use_color:
            if len(raw_image.shape) == 2:
                image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            elif raw_image.shape[2] == 4:
                image = raw_image[:, :, :3]
                alpha = raw_image[:, :, 3]
                mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)[1]

        h, w = image.shape[:2]
        templates.append({"name": path.stem, "image": image, "mask": mask, "h": h, "w": w})
    return templates


def best_template_match(region: Any, templates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    region_h, region_w = region.shape[:2]

    for template in templates:
        tpl = template["image"]
        mask = template.get("mask")
        th, tw = tpl.shape[:2]
        if th > region_h or tw > region_w:
            continue
        method = cv2.TM_CCORR_NORMED if mask is not None else cv2.TM_CCOEFF_NORMED
        response = cv2.matchTemplate(region, tpl, method, mask=mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        if max_val > best_score:
            best_score = float(max_val)
            best = {
                "name": template["name"],
                "score": best_score,
                "x": int(max_loc[0]),
                "y": int(max_loc[1]),
                "w": int(template["w"]),
                "h": int(template["h"]),
            }

    return best


def base_nightlord_name(name: str) -> str:
    return name[:-9] if name.endswith("-everdark") else name


def direct_similarity_score(patch: Any, template: Any, mask: Any | None) -> float:
    if patch.shape[:2] != template.shape[:2]:
        return -1.0

    patch_f = patch.astype("float32")
    template_f = template.astype("float32")

    if mask is not None:
        masked = mask > 0
        if not masked.any():
            return -1.0
        patch_pixels = patch_f[masked]
        template_pixels = template_f[masked]
    else:
        patch_pixels = patch_f.reshape(-1, patch_f.shape[-1] if len(patch_f.shape) == 3 else 1)
        template_pixels = template_f.reshape(-1, template_f.shape[-1] if len(template_f.shape) == 3 else 1)

    diff = patch_pixels - template_pixels
    mse = float((diff * diff).mean()) if diff.size else 1e9
    return 1.0 / (1.0 + mse)


def resize_with_nearest_mask(mask: Any, shape: tuple[int, int]) -> Any:
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def color_signature(image: Any, mask: Any | None) -> tuple[float, float, float, float, float, float]:
    if mask is not None:
        working_mask = mask > 0
        if not working_mask.any():
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pixels = image[working_mask]
    else:
        pixels = image.reshape(-1, image.shape[-1] if len(image.shape) == 3 else 1)

    pixels = pixels.astype("float32")
    if pixels.ndim == 1:
        pixels = pixels[:, None]

    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)

    if mean.shape[0] == 1:
        mean = np.repeat(mean, 3)
        std = np.repeat(std, 3)
    elif mean.shape[0] == 2:
        mean = np.pad(mean, (0, 3 - mean.shape[0]), mode="edge")
        std = np.pad(std, (0, 3 - std.shape[0]), mode="edge")

    return (
        float(mean[0]),
        float(mean[1]),
        float(mean[2]),
        float(std[0]),
        float(std[1]),
        float(std[2]),
    )


def color_signature_score(patch: Any, template: Any, mask: Any | None) -> float:
    patch_sig = color_signature(patch, mask)
    template_sig = color_signature(template, mask)

    if patch_sig == template_sig == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0):
        return -1.0

    diff = 0.0
    for patch_value, template_value in zip(patch_sig, template_sig):
        diff += abs(patch_value - template_value)

    # Smaller color signature differences should yield higher scores.
    return 1.0 / (1.0 + diff)


def write_debug_image(path: Path, image: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def resolve_nightlord_variant(
    row_band: Any,
    match: dict[str, Any],
    templates: list[dict[str, Any]],
) -> str:
    name = str(match["name"])
    base_name = base_nightlord_name(name)
    variant_names = [base_name, f"{base_name}-everdark"]

    x = int(match["x"])
    y = int(match["y"])
    best_name = name
    best_score = -1.0
    search_offsets = range(-4, 5)

    for variant_name in variant_names:
        variant_template = next((t for t in templates if t["name"] == variant_name), None)
        if not variant_template:
            continue

        h = int(variant_template["h"])
        w = int(variant_template["w"])
        variant_mask = variant_template.get("mask")

        local_best_score = -1.0

        for offset_y in search_offsets:
            for offset_x in search_offsets:
                candidate_x = x + offset_x
                candidate_y = y + offset_y
                if candidate_x < 0 or candidate_y < 0:
                    continue
                if candidate_y + h > row_band.shape[0] or candidate_x + w > row_band.shape[1]:
                    continue

                patch = row_band[candidate_y : candidate_y + h, candidate_x : candidate_x + w]
                patch_mask = variant_mask
                if patch_mask is not None and patch_mask.shape[:2] != patch.shape[:2]:
                    patch_mask = resize_with_nearest_mask(patch_mask, patch.shape[:2])

                color_score = color_signature_score(patch, variant_template["image"], patch_mask)
                shape_score = direct_similarity_score(patch, variant_template["image"], patch_mask)

                # Put more weight on color than exact pixel alignment.
                score = (0.8 * color_score) + (0.2 * shape_score)
                if score > local_best_score:
                    local_best_score = score

        if local_best_score > best_score:
            best_score = local_best_score
            best_name = variant_name

    return best_name


def find_nightlords_in_sequence(
    search_image: Any,
    templates: list[dict[str, Any]],
    threshold: float,
    search_left: int,
    search_top: int,
    search_right: int,
    log_miss: bool,
) -> list[dict[str, Any]]:
    image_h, _ = search_image.shape[:2]
    matches: list[dict[str, Any]] = []
    current_top = search_top
    max_icon_height = max(int(t["h"]) for t in templates) if templates else 20
    estimated_height = max(20, max_icon_height)

    for row_idx in range(EXPECTED_ROWS_PER_SCREENSHOT):
        if current_top >= image_h:
            break

        band_top = current_top
        band_bottom = min(image_h, band_top + int(estimated_height * 1.2))
        if band_bottom <= band_top:
            break

        row_band = search_image[band_top:band_bottom, search_left:search_right]
        match = best_template_match(row_band, templates)
        if not match or match["score"] < threshold:
            if log_miss:
                logging.warning("Nightlord not recognized at row %s", row_idx)
            current_top = min(image_h, band_top + estimated_height)
            continue

        abs_x = search_left + int(match["x"])
        abs_y = band_top + int(match["y"])
        abs_w = int(match["w"])
        abs_h = int(match["h"])

        resolved_name = resolve_nightlord_variant(
            row_band,
            match,
            templates,
        )

        matches.append(
            {
            "name": resolved_name,
                "score": float(match["score"]),
                "x": abs_x,
                "y": abs_y,
                "w": abs_w,
                "h": abs_h,
            }
        )

        # Per spec, next row recognition starts from the current row bottom.
        current_top = min(image_h, abs_y + abs_h)

    return matches


def choose_nightlord_threshold(
    search_image: Any,
    templates: list[dict[str, Any]],
    search_left: int,
    search_top: int,
    search_right: int,
) -> float:
    thresholds = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
    best_threshold = thresholds[-1]
    best_count = -1

    for threshold in thresholds:
        matches = find_nightlords_in_sequence(
            search_image,
            templates,
            threshold,
            search_left,
            search_top,
            search_right,
            log_miss=False,
        )
        if len(matches) > best_count:
            best_count = len(matches)
            best_threshold = threshold

        if len(matches) >= EXPECTED_ROWS_PER_SCREENSHOT:
            return threshold, EXPECTED_ROWS_PER_SCREENSHOT

    return best_threshold, best_count


def preprocess_for_ocr(image: Any) -> Any:
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def parse_date_from_text(text: str) -> str | None:
    date_match = re.search(r"(\d{4}/\d{2}/\d{2})", text)
    time_match = re.search(r"(\d{2}:\d{2}:\d{2})", text)
    if not date_match or not time_match:
        return None
    return f"{date_match.group(1)} {time_match.group(1)}"


def run_paddle_ocr_text(image: Any) -> tuple[str, float]:
    ocr = get_ocr_engine()
    # Paddle text recognizer expects a 3-channel image.
    input_image = image
    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    result = ocr.predict(input_image)
    if not result:
        return "", 0.0

    first = result[0]
    text = str(first.get("rec_text", "")).strip() if hasattr(first, "get") else ""
    score = float(first.get("rec_score", 0.0)) if hasattr(first, "get") else 0.0
    return text, score


def normalize_ocr_digit_text(text: str) -> str:
    # Paddle may confuse visually similar characters in numeric-only regions.
    replacements = {
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Z": "2",
        ",": "",
        ".": "",
        " ": "",
    }
    normalized = text
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def extract_numeric_candidates(text: str) -> list[str]:
    normalized = normalize_ocr_digit_text(text)
    candidates: list[str] = []
    current: list[str] = []

    for char in normalized:
        if char in POINTS_ALLOWED_CHARS:
            current.append(char)
        else:
            if current:
                candidates.append("".join(current))
                current = []
    if current:
        candidates.append("".join(current))

    return [candidate for candidate in candidates if 1 <= len(candidate) <= 5]


def points_candidate_score(candidate: str, confidence: float) -> float:
    # Prefer full-length point values and then confidence.
    length_bias = {1: -1.0, 2: -0.8, 3: 0.0, 4: 0.0}
    score = confidence + length_bias.get(len(candidate), -1.0)
    if not candidate.endswith("0"):
        score -= 1.0
    return score


def recognize_date(date_image: Any) -> str | None:
    preprocessed = preprocess_for_ocr(date_image)
    half = max(1, preprocessed.shape[0] // 2)
    top_text, _ = run_paddle_ocr_text(preprocessed[:half, :])
    bottom_text, _ = run_paddle_ocr_text(preprocessed[half:, :])
    text = f"{top_text}\n{bottom_text}".strip()
    return parse_date_from_text(text)


def recognize_points(points_image: Any) -> int:
    gray = points_image if len(points_image.shape) == 2 else cv2.cvtColor(points_image, cv2.COLOR_BGR2GRAY)

    variants: list[Any] = []
    base = gray
    # variants.append(base)
    variants.append(cv2.bitwise_not(base))

    candidate_scores: dict[str, float] = {}
    best_digits = ""
    best_confidence = -1.0
    best_variant = base

    for variant in variants:
        text, confidence = run_paddle_ocr_text(variant)
        candidates = extract_numeric_candidates(text)
        if not candidates:
            continue

        for candidate in candidates:
            candidate_scores[candidate] = candidate_scores.get(candidate, 0.0) + points_candidate_score(candidate, confidence)

        top_candidate = max(candidates, key=lambda value: points_candidate_score(value, confidence))
        if points_candidate_score(top_candidate, confidence) > points_candidate_score(best_digits, best_confidence):
            best_digits = top_candidate
            best_confidence = confidence
            best_variant = variant

    if candidate_scores:
        best_digits = max(candidate_scores.keys(), key=lambda value: candidate_scores[value])

    if not best_digits:
        return -1, best_variant

    try:
        return int(best_digits), best_variant
    except ValueError:
        return -1, best_variant


def parse_datetime(value: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S")
    except ValueError:
        return dt.datetime.min


def validate_points(expeditions: list[dict[str, Any]]) -> None:
    valid_indexes = [idx for idx, item in enumerate(expeditions) if item.get("points", -1) != -1]
    for current, nxt in zip(valid_indexes, valid_indexes[1:]):
        previous_item = expeditions[current]
        next_item = expeditions[nxt]
        prev_points = int(previous_item["points"])
        next_points = int(next_item["points"])

        if previous_item.get("day") == "day3-victory" and prev_points <= next_points:
            logging.warning("Abnormal points around %s (%s). Expected %d > %d", previous_item["date"], previous_item["image"], prev_points, next_points)
        if previous_item.get("day") != "day3-victory" and prev_points > next_points:
            logging.warning("Abnormal points around %s (%s). Expected %d <= %d", previous_item["date"], previous_item["image"], prev_points, next_points)
        if prev_points - next_points  > 350:
            logging.warning("Abnormal points jump around %s (%s). Expected %d - %d <= 350", previous_item["date"], previous_item["image"], prev_points, next_points)
        if prev_points - next_points  < -800:
            logging.warning("Abnormal points drop around %s (%s). Expected %d - %d >= -800", previous_item["date"], previous_item["image"], prev_points, next_points)


def load_yaml_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    return []


def merge_custom_fields(
    recognized: list[dict[str, Any]],
    previous_results: list[dict[str, Any]],
) -> None:
    previous_by_date: dict[str, dict[str, Any]] = {}
    for item in previous_results:
        date = item.get("date")
        if isinstance(date, str) and date not in previous_by_date:
            previous_by_date[date] = item

    matched_dates: set[str] = set()
    for item in recognized:
        old_item = previous_by_date.get(item["date"])
        if not old_item:
            continue
        matched_dates.add(item["date"])
        item["excluded"] = bool(old_item.get("excluded", False))
        item["status"] = str(old_item.get("status", item["status"]))
        item["note"] = old_item.get("note", "")
        if item["note"] is None:
            item["note"] = ""
        item.pop("outdated", None)

        if item["day"] == "day3-victory" and item["status"] != "victory":
            logging.warning(
                "Status/day mismatch at %s (%s): day is day3-victory but status is '%s'",
                item.get("date"),
                item.get("image"),
                item["status"],
            )

    outdated = []
    for date, item in previous_by_date.items():
        if date not in matched_dates:
            if not item.get("note", "") and item.get("status", "") == get_default_status(item.get("day", "day0")):
                continue
            item["outdated"] = True
            outdated.append(item)

    if len(outdated) > 0:
        logging.warning(
            "Found %d outdated expeditions in previous results that are not present in the current recognition.",
            len(outdated),
        )
        recognized.extend(outdated)


def load_schema(schema_path: Path) -> None:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    return yaml.safe_load(schema_path.read_text(encoding="utf-8"))


def clean_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def detect_day(row_gray: Any, day_templates: list[dict[str, Any]], search_left: int, search_right: int, threshold: float = 0.75) -> str:
    row_h, row_w = row_gray.shape
    left = max(0, min(search_left, row_w - 1))
    right = max(left + 1, min(search_right, row_w))
    day_region = row_gray[:, left:right]
    match = best_template_match(day_region, day_templates)
    if match and match["score"] >= threshold:
        return str(match["name"])
    logging.warning("Day icon not recognized, fallback to day0")
    return "day0"


def get_default_status(day_value: str) -> str:
    if day_value == "day3-victory":
        return "victory"
    return "defeat"


def collect_expeditions_for_screenshot(
    screenshot_path: Path,
    nightlord_templates: list[dict[str, Any]],
    day_templates: list[dict[str, Any]],
    crops_dir: Path,
    debug_dir: Path,
    date_cache: set[str],
) -> list[dict[str, Any]]:
    color_image = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
    if color_image is None:
        logging.warning("Failed to load screenshot: %s", screenshot_path)
        return []

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray_image.shape

    search_left = int(image_w * NIGHTLORD_REGION_LEFT_RATIO)
    search_top = int(image_h * NIGHTLORD_REGION_TOP_RATIO)
    search_right = min(image_w, search_left + int(image_w * NIGHTLORD_SEARCH_WIDTH_RATIO))

    threshold, expected_rows = choose_nightlord_threshold(
        color_image,
        nightlord_templates,
        search_left,
        search_top,
        search_right,
    )
    logging.info("Using nightlord threshold %.2f (%s expected rows)", threshold, expected_rows)

    nightlord_matches = find_nightlords_in_sequence(
        color_image,
        nightlord_templates,
        threshold,
        search_left,
        search_top,
        search_right,
        log_miss=True,
    )
    if len(nightlord_matches) < EXPECTED_ROWS_PER_SCREENSHOT:
        logging.warning(
            "Only recognized %s/%s nightlords in %s",
            len(nightlord_matches),
            EXPECTED_ROWS_PER_SCREENSHOT,
            screenshot_path.name,
        )

    expeditions: list[dict[str, Any]] = []
    for row_idx, nightlord_match in enumerate(nightlord_matches):
        row_key = f"{screenshot_path.stem}_{row_idx}"

        nx = int(nightlord_match["x"])
        ny = int(nightlord_match["y"])
        nw = int(nightlord_match["w"])
        nh = int(nightlord_match["h"])

        row_left = max(0, nx - int(image_w * 0.027))
        row_right = min(image_w, row_left + int(image_w * 0.57))
        row_top = max(0, ny)
        row_bottom = min(image_h, ny + nh)
        if row_right <= row_left or row_bottom <= row_top:
            logging.warning("Invalid row crop at row %s in %s", row_idx, screenshot_path.name)
            continue

        row_color = color_image[row_top:row_bottom, row_left:row_right]
        row_gray = gray_image[row_top:row_bottom, row_left:row_right]
        row_h, row_w = row_gray.shape

        day_search_left = max(0, (nx + nw) - row_left + int(row_w * 0.03))
        day_search_right = min(row_w, day_search_left + int(row_w * 0.25))
        points_left = int(row_w * 0.77)
        points_right = min(row_w, points_left + int(row_w * 0.07))
        points_top = int(row_h * 0.45)
        points_bottom = int(row_h * (1.0 -0.18))
        date_left = int(row_w * 0.87)

        if debug_dir is not None:
            row_vis = row_color.copy()
            nightlord_left = nx - row_left
            cv2.rectangle(
                row_vis,
                (int(nightlord_left), 0),
                (int(nightlord_left + nw), nh),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                row_vis,
                (day_search_left, 0),
                (day_search_right, row_h),
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                row_vis,
                (points_left, points_top),
                (points_right, points_bottom),
                (255, 255, 255),
                2,
            )
            cv2.rectangle(
                row_vis,
                (date_left, 0),
                (row_w, row_h),
                (255, 255, 255),
                2,
            )
            cv2.line(
                row_vis,
                (date_left, int(row_h * 0.5)),
                (row_w, int(row_h * 0.5)),
                (255, 255, 255),
                2,
            )
            write_debug_image(debug_dir / f"{row_key}.jpg", row_vis)

        date_crop = row_gray[:, date_left:row_w]
        recognized_date = recognize_date(date_crop)
        if not recognized_date:
            logging.warning("Date OCR failed at row %s in %s", row_idx, screenshot_path.name)
            continue

        if recognized_date in date_cache:
            logging.info("Duplicate expedition skipped by date %s", recognized_date)
            continue

        day_value = detect_day(row_gray, day_templates, day_search_left, day_search_right)

        points_crop = row_gray[points_top:points_bottom, points_left:points_right]
        points_value, points_preprocessed = recognize_points(points_crop)
        # if debug_dir is not None:
        #     write_debug_image(debug_dir / f"{row_key}_{points_value}.jpg", points_preprocessed)

        if points_value == -1:
            logging.warning("Points OCR failed at row %s in %s", row_idx, screenshot_path.name)

        image_name = f"{row_key}.jpg"
        row_save = row_color[:, :date_left] if hide_date else row_color
        cv2.imwrite(str(crops_dir / image_name), row_save)

        expedition = {
            "nightlord": str(nightlord_match["name"]),
            "day": day_value,
            "points": points_value,
            "date": recognized_date,
            "image": image_name,
            "excluded": False,
            "status": "victory" if day_value == "day3-victory" else "defeat",
            "note": "",
        }
        expeditions.append(expedition)
        date_cache.add(recognized_date)

        logging.info(
            "Recognized %s day=%s points=%s date=%s",
            expedition["nightlord"],
            expedition["day"],
            expedition["points"],
            expedition["date"],
        )

    return expeditions


def save_results(path: Path, expeditions: list[dict[str, Any]]) -> None:
    content = yaml.safe_dump(expeditions, allow_unicode=True, sort_keys=False)
    content = content.replace("note: ''", "note: ")
    if hide_date:
        content = "# hide_date\n" + content
    path.write_text(content, encoding="utf-8")


def main() -> None:
    global hide_date

    args = parse_args()
    num_screenshots = args.n if args.n is not None and args.n > 0 else None
    hide_date = bool(args.hide_date)

    configure_logging()

    root = repo_root()
    screenshots_dir = root / "data" / "expeditions"
    crops_dir = root / "data" / "expeditions-recognized"
    debug_dir = crops_dir / RECOGNITION_DEBUG_DIRNAME
    nightlords_dir = root / "assets" / "expedition-nightlords"
    days_dir = root / "assets" / "days"
    schema_path = root / "assets" / "results-schema.yaml"
    results_path = root / "results.yaml"

    clean_directory(crops_dir)
    clean_directory(debug_dir)

    nightlord_templates = load_templates(nightlords_dir, use_color=True)
    day_templates = load_templates(days_dir)
    if not nightlord_templates:
        raise RuntimeError("No nightlord templates found.")
    if not day_templates:
        raise RuntimeError("No day templates found.")

    screenshots = [
        path
        for path in sorted(screenshots_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ][:num_screenshots]

    all_expeditions: list[dict[str, Any]] = []
    date_cache: set[str] = set()

    for idx, screenshot in enumerate(screenshots, start=1):
        logging.info("============ (%d/%d) %s ============", idx, len(screenshots), screenshot.name)
        all_expeditions.extend(
            collect_expeditions_for_screenshot(
                screenshot,
                nightlord_templates,
                day_templates,
                crops_dir,
                debug_dir,
                date_cache,
            )
        )

    all_expeditions.sort(key=lambda item: parse_datetime(item["date"]), reverse=True)
    logging.info("Validating points consistency across expeditions")
    validate_points(all_expeditions)

    previous_results = load_yaml_list(results_path)
    schema = load_schema(schema_path)

    logging.info("Validating previous results.yaml against schema")
    validate(instance=previous_results, schema=schema)
    logging.info("Merging custom fields from previous results.yaml")
    merge_custom_fields(all_expeditions, previous_results)
    logging.info("Validating final results against schema")
    validate(instance=all_expeditions, schema=schema)

    save_results(results_path, all_expeditions)

    len_outdated_expeditions = sum(1 for item in all_expeditions if item.get("outdated", False))
    if len_outdated_expeditions > 0:
      logging.info("Recognition complete. Total expeditions: %s (+%s outdated)", len(all_expeditions) - len_outdated_expeditions, len_outdated_expeditions)
    else:
      logging.info("Recognition complete. Total expeditions: %s", len(all_expeditions))


if __name__ == "__main__":
    main()
