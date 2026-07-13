from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any

import cv2
from paddleocr import TextRecognition
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXPECTED_ROWS_PER_SCREENSHOT = 8

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


def load_templates(template_dir: Path) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for path in sorted(template_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logging.warning("Failed to load template: %s", path)
            continue
        h, w = image.shape
        templates.append({"name": path.stem, "image": image, "h": h, "w": w})
    return templates


def best_template_match(region: Any, templates: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = -1.0
    region_h, region_w = region.shape

    for template in templates:
        tpl = template["image"]
        th, tw = tpl.shape
        if th > region_h or tw > region_w:
            continue
        response = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
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


def choose_nightlord_threshold(gray_image: Any, templates: list[dict[str, Any]]) -> float:
    h, w = gray_image.shape
    band_right = max(1, int(w * 0.45))
    thresholds = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]

    for threshold in thresholds:
        recognized_rows = 0
        for row_idx in range(EXPECTED_ROWS_PER_SCREENSHOT):
            top = int(row_idx * h / EXPECTED_ROWS_PER_SCREENSHOT)
            bottom = int((row_idx + 1) * h / EXPECTED_ROWS_PER_SCREENSHOT)
            row_band = gray_image[top:bottom, 0:band_right]
            match = best_template_match(row_band, templates)
            if match and match["score"] >= threshold:
                recognized_rows += 1
        if recognized_rows >= 6:
            return threshold

    return 0.55


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
    date_cache: set[str],
) -> list[dict[str, Any]]:
    color_image = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
    if color_image is None:
        logging.warning("Failed to load screenshot: %s", screenshot_path)
        return []

    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    image_h, image_w = gray_image.shape
    threshold = choose_nightlord_threshold(gray_image, nightlord_templates)
    logging.info("Using nightlord threshold %.2f for %s", threshold, screenshot_path.name)

    expeditions: list[dict[str, Any]] = []
    band_right = max(1, int(image_w * 0.45))

    for row_idx in range(EXPECTED_ROWS_PER_SCREENSHOT):
        band_top = int(row_idx * image_h / EXPECTED_ROWS_PER_SCREENSHOT)
        band_bottom = int((row_idx + 1) * image_h / EXPECTED_ROWS_PER_SCREENSHOT)
        row_band = gray_image[band_top:band_bottom, 0:band_right]
        nightlord_match = best_template_match(row_band, nightlord_templates)
        if not nightlord_match or nightlord_match["score"] < threshold:
            logging.warning("Nightlord not recognized at row %s in %s", row_idx, screenshot_path.name)
            continue

        nx = int(nightlord_match["x"])
        ny = int(band_top + nightlord_match["y"])
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
    nightlords_dir = root / "assets" / "expedition-nightlords"
    days_dir = root / "assets" / "days"
    schema_path = root / "assets" / "results-schema.yaml"
    results_path = root / "results.yaml"

    clean_directory(crops_dir)

    nightlord_templates = load_templates(nightlords_dir)
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
