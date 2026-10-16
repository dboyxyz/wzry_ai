# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import json
from pathlib import Path
from dataclasses import asdict
import cv2
import numpy as np
import torch
from PIL import Image
from train_status import build_model, build_transforms, LABELS
from hud_ocr import NumericOCR, find_unannotated_candidates
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / 'weights/status_best.pt'
REWARDS = {'normal': 0.01, 'dead': 0.01, 'tower_attack': -0.5, 'minion_monster_tower_kill': 2.0, 'hero_kill': 5.0, 'killed': -2.0}

def aligned_paths(run, actions, vocabulary):
    candidates = [run / '_actions.jsonl', run.parent / (run.name + '_actions.jsonl')]
    op = next((p for p in candidates if p.is_file()), None)
    if op is None:
        raise ValueError(f'{run}: Missing action records; cannot align features')
    (paths, expected) = ([], [])
    move = 'keep-move'
    for line in op.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line.replace("'", '"'))
        path = run / (str(row.get('frame_id', '')) + '.jpg')
        if not path.is_file():
            continue
        if row.get('movement', 'keep-move') != 'keep-move':
            move = row['movement']
        expected.append(int(vocabulary.get(move + '_' + row.get('skill', 'none'), 0)))
        paths.append(path)
    if len(paths) != len(actions) or not np.array_equal(expected, actions):
        raise ValueError(f'{run}: Images/actions do not match NPZ; rebuild features')
    numbers = [int(p.stem) for p in paths]
    if any((b <= a for (a, b) in zip(numbers, numbers[1:]))):
        raise ValueError('Frame IDs must be strictly increasing')
    return paths

class HybridRewards:

    def __init__(self, weights, device, confidence=0.1, gold_min=20):
        checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
        self.provenance = {'state_weights': str(Path(weights).resolve()), 'epoch': checkpoint.get('epoch'), 'labels': checkpoint['labels'], 'image_size': checkpoint['image_size'], 'ocr_confidence_min': confidence, 'gold_delta_min': gold_min, 'reward_values': REWARDS}
        if tuple(checkpoint['labels']) != tuple(LABELS):
            raise ValueError('State label order mismatch')
        (self.model, _) = build_model(checkpoint['backbone'], False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device).eval().requires_grad_(False)
        (_, self.transform) = build_transforms(*checkpoint['image_size'])
        self.device = device
        self.ocr = NumericOCR(confidence)
        self.gold_min = gold_min

    @torch.inference_mode()
    def prepare(self, paths, out):
        (frames, records) = ([], [])
        for (i, path) in enumerate(paths):
            with Image.open(path) as im:
                rgb = im.convert('RGB')
                tensor = self.transform(rgb).unsqueeze(0).to(self.device)
                prob = self.model(tensor).softmax(-1)[0].cpu().tolist()
                values = self.ocr.read_frame(cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
            key = path.parent.name + '_' + path.stem
            frames.append({'key': key, 'values': values})
            label = LABELS[int(np.argmax(prob))]
            records.append({'key': key, 'path': str(path), 'visual_label': label, 'probabilities': dict(zip(LABELS, prob)), 'ocr': asdict(values), 'events': [], 'reward': REWARDS[label]})
            if (i + 1) % 25 == 0:
                print(f'OCR + visual states: {i + 1}/{len(paths)}', flush=True)
        segments = []
        for row in frames:
            if not segments or int(row['key'].rsplit('_', 1)[1]) != int(segments[-1][-1]['key'].rsplit('_', 1)[1]) + 1:
                segments.append([])
            segments[-1].append(row)
        by_key = {r['key']: r for r in records}
        for segment in segments:
            (events, _) = find_unannotated_candidates([{'frames': segment}], [], self.gold_min)
            for event in events:
                by_key[event['to_key']]['events'].append(event)
        for row in records:
            priorities = ['killed', 'hero_kill', 'minion_monster_tower_kill']
            event_labels = {e['detected_state'] for e in row['events']}
            row['reward_label'] = next((l for l in priorities if l in event_labels), row['visual_label'])
            row['reward'] = REWARDS[row['reward_label']]
        out.mkdir(parents=True, exist_ok=True)
        (out / 'reward_config.json').write_text(json.dumps(self.provenance, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'rewards.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        return np.asarray([r['reward'] for r in records], dtype=np.float32)
