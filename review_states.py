# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import argparse
import html
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
import torch
from hybrid_rewards import DEFAULT_WEIGHTS, HybridRewards
ROOT = Path(__file__).resolve().parent
ALL_LABELS = ['normal', 'dead', 'tower_attack', 'minion_monster_tower_kill', 'hero_kill', 'killed']

def image_paths(run: Path, limit: int) -> list[Path]:
    paths = [path for path in run.glob('*.jpg') if path.stem.isdigit()]
    paths.sort(key=lambda path: int(path.stem))
    return paths[:limit or None]

def fmt_ocr(ocr: dict) -> str:
    names = [('gold', 'Gold'), ('kills', 'K'), ('deaths', 'D'), ('assists', 'A')]
    values = []
    for (key, title) in names:
        item = ocr[key]
        value = '—' if item['value'] is None else item['value']
        values.append(f"{title}={value} ({item['confidence']:.3f})")
    return '\u3000'.join(map(str, values))

def write_run_pages(run: Path, records: list[dict], output: Path, page_size: int) -> Counter:
    counts = Counter((row['reward_label'] for row in records))
    page_count = max(1, (len(records) + page_size - 1) // page_size)
    nav = ' '.join((f'<a href="page_{i:03d}.html">{i}</a>' for i in range(1, page_count + 1)))
    options = ''.join((f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in ALL_LABELS))
    for page_no in range(1, page_count + 1):
        cards = []
        for row in records[(page_no - 1) * page_size:page_no * page_size]:
            label = row['reward_label']
            confidence = max(row['probabilities'].values())
            image_src = os.path.relpath(row['path'], output)
            events = '；'.join((f"{event['detected_state']}：{event['from_frame']}→{event['to_frame']}，Δ={event['delta']}" for event in row['events'])) or 'None'
            probs = '\u3000'.join((f'{key}={value:.2%}' for (key, value) in row['probabilities'].items()))
            cards.append(f'''<article class="card" data-label="{html.escape(label)}">\n<a href="{html.escape(image_src)}"><img loading="lazy" src="{html.escape(image_src)}"></a>\n<div><strong>{html.escape(row['key'])}</strong></div><div class="final">Final label: {html.escape(label)}</div><div>Visual state: {html.escape(row['visual_label'])} (confidence {confidence:.2%}）</div>\n<div class="small">{html.escape(probs)}</div>\n<div class="small">OCR：{html.escape(fmt_ocr(row['ocr']))}</div><div class="small">Events: {html.escape(events)}</div>\n</article>''')
        page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">\n<title>{html.escape(run.name)} state review, page {page_no}</title><style>\nbody{{font-family:system-ui;margin:18px;background:#f5f6f8;color:#202124}}a{{margin:3px}}\n.toolbar{{position:sticky;top:0;background:#fff;padding:10px;z-index:2;border:1px solid #ddd}}\n.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px;margin-top:12px}}\n.card{{background:white;padding:10px;border:1px solid #ddd;border-radius:8px}}img{{width:100%;height:auto}}\n.final{{font-size:18px;color:#b42318;font-weight:700}}.small{{font-size:12px;overflow-wrap:anywhere;margin-top:4px}}\n</style></head><body><div class="toolbar"><a href="../index.html">Back to overview</a>\u3000\nRun {html.escape(run.name)} page {page_no}/{page_count} | Filter: <select id="filter"><option value="">All</option>{options}</select><div>{nav}</div></div>\n<main class="grid">{''.join(cards)}</main><script>\ndocument.getElementById('filter').onchange=e=>document.querySelectorAll('.card').forEach(\n x=>x.style.display=(!e.target.value||x.dataset.label===e.target.value)?'block':'none');\n</script></body></html>"""
        (output / f'page_{page_no:03d}.html').write_text(page, encoding='utf-8')
    return counts

def write_index(summaries: list[tuple[str, int, Counter]], output: Path) -> None:
    rows = []
    total = Counter()
    total_n = 0
    for (name, count, labels) in summaries:
        total.update(labels)
        total_n += count
        cells = ''.join((f'<td>{labels[label]}</td>' for label in ALL_LABELS))
        rows.append(f'<tr><td><a href="{html.escape(name)}/page_001.html">{html.escape(name)}</a></td><td>{count}</td>{cells}</tr>')
    heads = ''.join((f'<th>{html.escape(label)}</th>' for label in ALL_LABELS))
    totals = ''.join((f'<td>{total[label]}</td>' for label in ALL_LABELS))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Training state review</title>\n<style>body{{font-family:system-ui;margin:30px}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style></head>\n<body><h1>Training state review</h1><p>Labels combine visual states and temporal OCR events. Select a run to inspect images and inference details.</p>\n<table><thead><tr><th>Run</th><th>Images</th>{heads}</tr></thead><tbody>{''.join(rows)}<tr><th>Total</th><th>{total_n}</th>{totals}</tr></tbody></table></body></html>"""
    (output / 'index.html').write_text(page, encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/runs')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run-id', nargs='+')
    selection.add_argument('--all-runs', action='store_true')
    parser.add_argument('--state-weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'training_state_review' / datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--page-size', type=int, default=200)
    parser.add_argument('--max-frames', type=int, default=0, help='Maximum frames per run; 0 means all')
    parser.add_argument('--ocr-confidence-min', type=float, default=0.1)
    parser.add_argument('--gold-delta-min', type=int, default=20)
    parser.add_argument('--reuse', action='store_true', help='Reuse existing rewards.json and rebuild review pages')
    args = parser.parse_args()
    if args.page_size < 1 or args.max_frames < 0:
        parser.error('page-size must be positive and max-frames nonnegative')
    (data_dir, output) = (args.data_dir.resolve(), args.output_dir.resolve())
    names = sorted((path.name for path in data_dir.iterdir() if path.is_dir())) if args.all_runs else list(dict.fromkeys(args.run_id))
    runs = [(data_dir / name, image_paths(data_dir / name, args.max_frames)) for name in names]
    empty = [run.name for (run, paths) in runs if not paths]
    if empty:
        raise ValueError('No numeric JPG filenames in runs: ' + ', '.join(empty))
    if output.exists() and (not args.reuse):
        raise FileExistsError(f'Output directory exists: {output}; choose a new directory or use --reuse')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable')
    recognizer = None
    summaries = []
    for (run, paths) in runs:
        run_output = output / run.name
        reward_json = run_output / 'rewards.json'
        if args.reuse and reward_json.is_file():
            records = json.loads(reward_json.read_text(encoding='utf-8'))
            print(f'Reusing {run.name}: {len(records)} frames', flush=True)
        else:
            if recognizer is None:
                recognizer = HybridRewards(args.state_weights.resolve(), device, args.ocr_confidence_min, args.gold_delta_min)
            recognizer.prepare(paths, run_output)
            records = json.loads(reward_json.read_text(encoding='utf-8'))
        counts = write_run_pages(run, records, run_output, args.page_size)
        summaries.append((run.name, len(records), counts))
        write_index(summaries, output)
        print(f'Review generated: {run.name}，{dict(counts)}', flush=True)
    print(f"Complete: {output / 'index.html'}", flush=True)
if __name__ == '__main__':
    main()
