from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from paddleocr import TextRecognition
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_ROWS_PER_SCREENSHOT = 8
NIGHTLORD_REGION_LEFT_RATIO = 0.078
NIGHTLORD_REGION_TOP_RATIO = 0.215
NIGHTLORD_SEARCH_WIDTH_RATIO = 0.45
NIGHTLORD_DEBUG_DIRNAME = "debug-nightlord-matches"

OCR_ENGINE: TextRecognition | None = None


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
    debug_dir: Path | None = None,
    debug_prefix: str = "",
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
        local_best_patch = None

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
                    local_best_patch = patch

        if debug_dir is not None and local_best_patch is not None:
            safe_score = f"{local_best_score:.6f}".replace(".", "_")
            prefix = debug_prefix or "row"
            write_debug_image(debug_dir / f"{prefix}_{variant_name}_patch_s{safe_score}.png", local_best_patch)
            write_debug_image(
                debug_dir / f"{prefix}_{variant_name}_template_s{safe_score}.png",
                variant_template["image"],
            )
            if variant_mask is not None:
                write_debug_image(
                    debug_dir / f"{prefix}_{variant_name}_mask_s{safe_score}.png",
                    variant_mask,
                )

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
    debug_dir: Path | None = None,
    debug_prefix: str = "",
    save_debug_images: bool = False,
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

        if save_debug_images and debug_dir is not None:
            row_key = f"{debug_prefix}_row{row_idx}"
            write_debug_image(debug_dir / f"{row_key}_band.png", row_band)
            row_vis = row_band.copy()
            cv2.rectangle(
                row_vis,
                (int(match["x"]), int(match["y"])),
                (int(match["x"] + abs_w), int(match["y"] + abs_h)),
                (0, 255, 0),
                2,
            )
            write_debug_image(debug_dir / f"{row_key}_band_match.png", row_vis)

        resolved_name = resolve_nightlord_variant(
            row_band,
            match,
            templates,
            debug_dir=debug_dir if save_debug_images else None,
            debug_prefix=f"{debug_prefix}_row{row_idx}",
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


def run_paddle_ocr_text(image: Any) -> str:
    ocr = get_ocr_engine()
    # Paddle text recognizer expects a 3-channel image.
    input_image = image
    if len(input_image.shape) == 2:
        input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    result = ocr.predict(input_image)
    if not result:
        return ""

    first = result[0]
    text = str(first.get("rec_text", "")).strip() if hasattr(first, "get") else ""
    return text


def recognize_date(date_image: Any) -> str | None:
    preprocessed = preprocess_for_ocr(date_image)
    half = max(1, preprocessed.shape[0] // 2)
    top_text = run_paddle_ocr_text(preprocessed[:half, :])
    bottom_text = run_paddle_ocr_text(preprocessed[half:, :])
    text = f"{top_text}\n{bottom_text}".strip()
    return parse_date_from_text(text)


def recognize_points(points_image: Any) -> int:
    preprocessed = preprocess_for_ocr(points_image)
    text = run_paddle_ocr_text(preprocessed)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return -1
    try:
        return int(digits)
    except ValueError:
        return -1


def parse_datetime(value: str) -> dt.datetime:
    try:
        return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S")
    except ValueError:
        return dt.datetime.min


def validate_points_order(expeditions: list[dict[str, Any]]) -> None:
    valid_indexes = [idx for idx, item in enumerate(expeditions) if item.get("points", -1) != -1]
    for current, nxt in zip(valid_indexes, valid_indexes[1:]):
        previous_item = expeditions[current]
        next_item = expeditions[nxt]
        prev_points = int(previous_item["points"])
        next_points = int(next_item["points"])

        if previous_item.get("day") == "day3-victory" and prev_points < next_points:
            logging.warning("Abnormal points around %s", previous_item.get("date"))
        if previous_item.get("day") != "day3-victory" and prev_points > next_points:
            logging.warning("Abnormal points around %s", previous_item.get("date"))


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
        item["excluded"] = bool(old_item.get("excluded", old_item.get("exclude", False)))
        item["reason"] = old_item.get("reason", "")
        item["note"] = old_item.get("note", "")

    for date in sorted(previous_by_date):
        if date not in matched_dates:
            logging.warning("Custom data discarded for unmatched expedition date: %s", date)


def validate_against_schema(records: list[dict[str, Any]], schema_path: Path) -> None:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    try:
        from jsonschema import validate
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'jsonschema'. Install it before running recognize.py."
        ) from exc

    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    validate(instance=records, schema=schema)


def clean_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def detect_day(row_gray: Any, day_templates: list[dict[str, Any]], search_left: int, threshold: float = 0.75) -> str:
    row_h, row_w = row_gray.shape
    left = max(0, min(search_left, row_w - 1))
    right = max(left + 1, int(row_w * 0.77))
    day_region = row_gray[:, left:right]
    match = best_template_match(day_region, day_templates)
    if match and match["score"] >= threshold:
        return str(match["name"])
    logging.warning("Day icon not recognized, fallback to day0")
    return "day0"


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
    logging.info("Using nightlord threshold %.2f (%s expected rows) for %s", threshold, expected_rows, screenshot_path.name)

    nightlord_matches = find_nightlords_in_sequence(
        color_image,
        nightlord_templates,
        threshold,
        search_left,
        search_top,
        search_right,
        log_miss=True,
        debug_dir=debug_dir,
        debug_prefix=screenshot_path.stem,
        save_debug_images=True,
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
        nx = int(nightlord_match["x"])
        ny = int(nightlord_match["y"])
        nw = int(nightlord_match["w"])
        nh = int(nightlord_match["h"])

        row_left = max(0, nx - int(image_w * 0.016))
        row_right = min(image_w, row_left + int(image_w * 0.57))
        row_top = max(0, ny)
        row_bottom = min(image_h, ny + nh)
        if row_right <= row_left or row_bottom <= row_top:
            logging.warning("Invalid row crop at row %s in %s", row_idx, screenshot_path.name)
            continue

        row_color = color_image[row_top:row_bottom, row_left:row_right]
        row_gray = gray_image[row_top:row_bottom, row_left:row_right]
        row_h, row_w = row_gray.shape

        date_left = int(row_w * 0.87)
        date_crop = row_gray[:, date_left:row_w]
        recognized_date = recognize_date(date_crop)
        if not recognized_date:
            logging.warning("Date OCR failed at row %s in %s", row_idx, screenshot_path.name)
            continue

        if recognized_date in date_cache:
            logging.info("Duplicate expedition skipped by date %s", recognized_date)
            continue

        day_search_left = max(0, (nx + nw) - row_left)
        day_value = detect_day(row_gray, day_templates, day_search_left)

        points_left = int(row_w * 0.77)
        points_right = min(row_w, points_left + int(row_w * 0.07))
        points_crop = row_gray[:, points_left:points_right]
        points_value = recognize_points(points_crop)
        if points_value == -1:
            logging.warning("Points OCR failed at row %s in %s", row_idx, screenshot_path.name)

        image_name = f"{screenshot_path.stem}_{row_idx}.jpg"
        cv2.imwrite(str(crops_dir / image_name), row_color)

        expedition = {
            "nightlord": str(nightlord_match["name"]),
            "day": day_value,
            "points": points_value,
            "date": recognized_date,
            "image": image_name,
            "excluded": False,
            "reason": "",
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
    path.write_text(
        yaml.safe_dump(expeditions, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> None:
    configure_logging()

    root = repo_root()
    screenshots_dir = root / "data" / "expeditions"
    crops_dir = root / "data" / "expeditions-recognized"
    debug_dir = crops_dir / NIGHTLORD_DEBUG_DIRNAME
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
    ]

    all_expeditions: list[dict[str, Any]] = []
    date_cache: set[str] = set()

    for screenshot in screenshots:
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
    validate_points_order(all_expeditions)

    previous_results = load_yaml_list(results_path)
    merge_custom_fields(all_expeditions, previous_results)

    validate_against_schema(all_expeditions, schema_path)
    save_results(results_path, all_expeditions)

    logging.info("Recognition complete. Total expeditions: %s", len(all_expeditions))


if __name__ == "__main__":
    main()
