# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
LABEL_MINION = 'minion_monster_tower_kill'
LABEL_HERO = 'hero_kill'
LABEL_KILLED = 'killed'

@dataclass(frozen=True)
class ROIConfig:
    gold_roi: tuple[int, int, int, int] = (60, 183, 44, 15)
    kills_roi: tuple[int, int, int, int] = (783, 15, 18, 15)
    deaths_roi: tuple[int, int, int, int] = (811, 15, 18, 15)
    assists_roi: tuple[int, int, int, int] = (839, 15, 18, 15)

@dataclass
class OCRValue:
    value: Optional[int]
    text: str = ''
    confidence: float = 0.0
    variant: str = ''

@dataclass
class HUDValues:
    gold: OCRValue
    kills: OCRValue
    deaths: OCRValue
    assists: OCRValue

def is_augmented_key(key: str) -> bool:
    return '__aug_' in key or '__aug2_' in key

def source_key(key: str) -> str:
    return re.split('__(?:aug|aug2)_', key, maxsplit=1)[0]

def key_position(key: str) -> Optional[tuple[str, int]]:
    key = source_key(key)
    if key.isdigit():
        return ('pure_digits', int(key))
    match = re.match('^(.*_)(\\d+)$', key)
    if not match:
        return None
    return (match.group(1), int(match.group(2)))

def key_at(group: str, frame_number: int) -> str:
    return str(frame_number) if group == 'pure_digits' else f'{group}{frame_number}'

class NumericOCR:

    def __init__(self, confidence_min: float=0.1) -> None:
        self.engine = RapidOCR()
        self.confidence_min = confidence_min

    @staticmethod
    def _scale_roi(roi: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        (height, width) = frame_shape[:2]
        (sx, sy) = (width / 960.0, height / 432.0)
        (x, y, w, h) = roi
        return (round(x * sx), round(y * sy), max(1, round(w * sx)), max(1, round(h * sy)))

    @staticmethod
    def crop(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        (x, y, w, h) = roi
        (height, width) = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        return frame[y:min(height, y + h), x:min(width, x + w)]

    @staticmethod
    def _variants(patch: np.ndarray) -> list[tuple[str, np.ndarray]]:
        enlarged = cv2.resize(patch, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        return [('raw', patch), ('cubic_x4', enlarged)]

    def read_patch(self, patch: np.ndarray, *, max_digits: int) -> OCRValue:
        if patch.size == 0:
            return OCRValue(None)
        candidates: list[OCRValue] = []
        for (variant_name, variant) in self._variants(patch):
            (result, _) = self.engine(variant, use_det=False, use_cls=False, use_rec=True)
            if not result:
                continue
            for item in result:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                text = str(item[0]).strip()
                try:
                    confidence = float(item[1])
                except (TypeError, ValueError):
                    continue
                digits = ''.join(re.findall('\\d', text))
                if not digits or len(digits) > max_digits:
                    continue
                candidates.append(OCRValue(int(digits), text, confidence, variant_name))
        if not candidates:
            return OCRValue(None)
        best = max(candidates, key=lambda item: item.confidence)
        if best.confidence < self.confidence_min:
            return OCRValue(None, best.text, best.confidence, best.variant)
        return best

    def read_frame(self, frame: np.ndarray, roi_config: ROIConfig=ROIConfig()) -> HUDValues:
        rois = {'gold': (roi_config.gold_roi, 5), 'kills': (roi_config.kills_roi, 2), 'deaths': (roi_config.deaths_roi, 2), 'assists': (roi_config.assists_roi, 2)}
        values: dict[str, OCRValue] = {}
        for (name, (base_roi, max_digits)) in rois.items():
            roi = self._scale_roi(base_roi, frame.shape)
            values[name] = self.read_patch(self.crop(frame, roi), max_digits=max_digits)
        return HUDValues(**values)

def correct_gold_ocr_sequence(frames: list[dict]) -> tuple[list[Optional[int]], list[dict]]:
    values = [row['values'].gold.value for row in frames]
    corrected = list(values)
    corrections = []
    for index in range(1, len(values) - 1):
        (previous, current, following) = values[index - 1:index + 2]
        if previous is None or current is None or following is None:
            continue
        if previous >= 1000 or current < 1000 or following >= 1000:
            continue
        text = str(current)
        alternatives = {int(text[:remove] + text[remove + 1:]) for remove in range(len(text)) if text[:remove] + text[remove + 1:]}
        plausible = [value for value in alternatives if 0 <= value <= 999 and abs(value - previous) <= 500 and (abs(following - value) <= 500) and (previous - 10 <= value <= following + 10)]
        if not plausible:
            continue
        replacement = min(plausible, key=lambda value: abs(value - previous) + abs(following - value))
        corrected[index] = replacement
        corrections.append({'key': frames[index]['key'], 'ocr_value': current, 'corrected_value': replacement, 'reason': 'isolated_extra_digit'})
    return (corrected, corrections)

def find_unannotated_candidates(events: list[dict], annotations: list[tuple[str, str]], gold_delta_min: int) -> tuple[list[dict], dict[str, dict]]:
    frame_map = {row['key']: row for event in events for row in event['frames']}
    annotation_map = {key: label for (key, label) in annotations if not is_augmented_key(key)}
    grouped: dict[str, list[dict]] = {}
    for row in frame_map.values():
        position = key_position(row['key'])
        if position is not None:
            grouped.setdefault(position[0], []).append(row)
    raw_candidates = []
    for (group, rows) in grouped.items():
        rows.sort(key=lambda row: key_position(row['key'])[1])
        (corrected_gold, corrections) = correct_gold_ocr_sequence(rows)
        correction_map = {item['key']: item for item in corrections}
        last_valid: dict[str, Optional[tuple[int, int]]] = {'kills': None, 'deaths': None}
        for (index, row) in enumerate(rows):
            number = key_position(row['key'])[1]
            for (field, label) in (('deaths', LABEL_KILLED), ('kills', LABEL_HERO)):
                value = getattr(row['values'], field).value
                previous = last_valid[field]
                if value is not None:
                    if previous is not None:
                        (previous_number, previous_value) = previous
                        if number - previous_number <= 7 and 1 <= value - previous_value <= 3:
                            raw_candidates.append({'group': group, 'detected_state': label, 'delta': value - previous_value, 'from_frame': previous_number, 'to_frame': number})
                    last_valid[field] = (number, value)
            if index == 0:
                continue
            previous_number = key_position(rows[index - 1]['key'])[1]
            if number != previous_number + 1:
                continue
            (before, after) = corrected_gold[index - 1:index + 1]
            if before is not None and after is not None and (gold_delta_min <= after - before <= 3000):
                raw_candidates.append({'group': group, 'detected_state': LABEL_MINION, 'delta': after - before, 'from_frame': previous_number, 'to_frame': number, 'ocr_correction': correction_map.get(row['key'])})
    counter_events = [row for row in raw_candidates if row['detected_state'] != LABEL_MINION]
    results = []
    annotation_ranges = {LABEL_HERO: (-2, 8), LABEL_KILLED: (-2, 8), LABEL_MINION: (-2, 3)}
    for candidate in raw_candidates:
        group = candidate['group']
        number = candidate['to_frame']
        detected = candidate['detected_state']
        if detected == LABEL_MINION:
            if any((row['group'] == group and abs(row['to_frame'] - number) <= 5 for row in counter_events)):
                continue
            if any((annotation_map.get(key_at(group, nearby)) in (LABEL_HERO, LABEL_KILLED) for nearby in range(number - 8, number + 9))):
                continue
        (before, after) = annotation_ranges[detected]
        nearby_labels = [{'key': key_at(group, nearby), 'label': annotation_map[key_at(group, nearby)]} for nearby in range(number + before, number + after + 1) if key_at(group, nearby) in annotation_map]
        if any((item['label'] == detected for item in nearby_labels)):
            continue
        candidate = dict(candidate)
        candidate['from_key'] = key_at(group, candidate['from_frame'])
        candidate['to_key'] = key_at(group, number)
        candidate['nearby_labels'] = nearby_labels
        results.append(candidate)
    return (results, frame_map)
