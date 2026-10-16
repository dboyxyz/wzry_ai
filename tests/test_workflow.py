"""CPU tests for schema compatibility, temporal events, and causal masking."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_legacy import translate, MAPPING
from hud_ocr import HUDValues, OCRValue, correct_gold_ocr_sequence, find_unannotated_candidates
from masks import create_masks
from policy_model import Transformer
from hybrid_rewards import aligned_paths
from review_states import write_run_pages, write_index
import run_agent


def frame(number, gold, kills=0, deaths=0):
    return {'key': f'123_{number}', 'values': HUDValues(
        OCRValue(gold), OCRValue(kills), OCRValue(deaths), OCRValue(0))}


class WorkflowTests(unittest.TestCase):
    def test_manual_override_and_ai_toggle(self):
        previous = run_agent.STATE
        try:
            run_agent.STATE = run_agent.InputState(ai_enabled=True, a_down=True)
            move, skill, enabled = run_agent.choose_runtime_action('right_attack')
            self.assertEqual((move, skill, enabled), ('left', 'none', True))
            run_agent.STATE = run_agent.InputState(ai_enabled=False)
            self.assertEqual(run_agent.choose_runtime_action('right_attack'),
                             ('keep-move', 'none', False))
            run_agent.STATE = run_agent.InputState(ai_enabled=True, pending_stop=True)
            self.assertEqual(run_agent.choose_runtime_action('right_attack'),
                             ('stop', 'none', True))
        finally:
            run_agent.STATE = previous

    def test_action_ids_round_trip(self):
        root = Path(__file__).resolve().parents[1]
        forward = json.loads((root/'config/action_to_id.json').read_text())
        inverse = json.loads((root/'config/id_to_action.json').read_text())
        self.assertEqual(set(forward.values()), set(range(130)))
        for word, number in forward.items():
            self.assertEqual(inverse[str(number)], word)
            self.assertEqual(len(word.split('_')), 2)

    def test_extra_digit_and_purchase(self):
        values, corrections = correct_gold_ocr_sequence([
            frame(0, 560), frame(1, 2562), frame(2, 564)])
        self.assertEqual(values, [560, 562, 564])
        self.assertEqual(len(corrections), 1)
        events, _ = find_unannotated_candidates([{'frames': [
            frame(0, 100), frame(1, 150), frame(2, 10)]}], [], 20)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['detected_state'], 'minion_monster_tower_kill')

    def test_kill_suppresses_nearby_gold(self):
        events, _ = find_unannotated_candidates([{'frames': [
            frame(0, 100), frame(1, 200, kills=1), frame(2, 200, kills=1)]}], [], 20)
        self.assertEqual([event['detected_state'] for event in events], ['hero_kill'])

    def test_policy_causality_and_shapes(self):
        model = Transformer(130, 12, 1, 3, 0., feature_dim=24).eval()
        x = torch.randn(1, 4, 24)
        actions = torch.full((1, 4), 128)
        _, mask = create_masks(actions, actions, 'cpu')
        with torch.no_grad():
            original, value = model(x, actions, mask)
            changed = x.clone()
            changed[:, 2:] += 100
            result, _ = model(changed, actions, mask)
        self.assertEqual(tuple(original.shape), (1, 4, 130))
        self.assertEqual(tuple(value.shape), (1, 4, 1))
        torch.testing.assert_close(original[:, :2], result[:, :2])

    def test_alignment_and_english_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'123'
            run.mkdir()
            for i in range(2):
                (run/f'{i}.jpg').touch()
            (run/'_actions.jsonl').write_text(
                '{"frame_id":"0","movement":"right","skill":"attack"}\n'
                '{"frame_id":"1","movement":"keep-move","skill":"none"}\n')
            paths = aligned_paths(run, np.array([13, 24]), {'right_attack': 13, 'right_none': 24})
            self.assertEqual(len(paths), 2)
            with self.assertRaises(ValueError):
                aligned_paths(run, np.array([0, 24]), {'right_attack': 13, 'right_none': 24})
            record = {'key': '123_0', 'path': str(paths[0]), 'visual_label': 'normal',
                      'reward_label': 'normal', 'probabilities': {'normal': 1.}, 'events': [],
                      'ocr': {k: {'value': 0, 'confidence': .9} for k in ['gold', 'kills', 'deaths', 'assists']}}
            counts = write_run_pages(run, [record], run, 1)
            write_index([('123', 1, counts)], Path(tmp))
            self.assertIn('normal', (run/'page_001.html').read_text())
            self.assertIn('123/page_001.html', (Path(tmp)/'index.html').read_text())

    def test_legacy_label_conversion(self):
        old = next(key for key, value in MAPPING.items() if value == 'tower_attack')
        self.assertEqual(translate({'123_0': old}), {'123_0': 'tower_attack'})


if __name__ == '__main__':
    unittest.main()
