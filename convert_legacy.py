"""Convert original datasets and checkpoints into the English repository schema."""
import argparse
import ast
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAPPING = json.loads((ROOT / 'config/legacy_mapping.json').read_text())


def translate(value):
    if isinstance(value, str):
        for old in sorted(MAPPING, key=len, reverse=True):
            if not old.startswith('_'):
                value = value.replace(old, MAPPING[old])
        return value
    if isinstance(value, dict):
        return {translate(k): translate(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [translate(item) for item in value]
    return value


def json_lines(source, target):
    with source.open(encoding='utf-8') as src, target.open('x', encoding='utf-8') as dst:
        for number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = ast.literal_eval(line)
            dst.write(json.dumps(translate(item), ensure_ascii=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['runs', 'labels', 'policy', 'status'])
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    args = parser.parse_args()
    if args.target.exists():
        raise FileExistsError(f'Refusing to overwrite {args.target}')
    if args.kind == 'runs':
        args.target.mkdir(parents=True)
        for run in sorted(p for p in args.source.iterdir() if p.is_dir()):
            frames = sorted(run.glob('*.jpg'))
            if not frames:
                continue
            candidates = [run / MAPPING['_record_inside'],
                          args.source / (run.name + MAPPING['_record_suffix'])]
            record = next((p for p in candidates if p.is_file()), None)
            if record is None:
                raise FileNotFoundError(f'Missing action record for {run.name}')
            out = args.target / run.name
            out.mkdir()
            for image in frames:
                shutil.copy2(image, out / image.name)
            json_lines(record, out / '_actions.jsonl')
            # Features are intentionally rebuilt to verify image/action alignment.
            print(f'Converted {run.name}: {len(frames)} images')
    elif args.kind == 'labels':
        args.target.parent.mkdir(parents=True, exist_ok=True)
        json_lines(args.source, args.target)
    else:
        import torch
        checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
        if args.kind == 'policy':
            converted = {}
            for key, tensor in checkpoint.items():
                for old, new in MAPPING['_policy_prefixes'].items():
                    if key.startswith(old):
                        key = new + key[len(old):]
                        break
                converted[key] = tensor
            checkpoint = converted
        else:
            # Keep an inference checkpoint with English metadata only.
            checkpoint = {key: translate(checkpoint[key]) for key in
                          ('epoch', 'labels', 'backbone', 'image_size', 'best_macro_f1')
                          if key in checkpoint}
            original = torch.load(args.source, map_location='cpu', weights_only=False)
            checkpoint['model_state'] = original['model_state']
        args.target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.target)
    print(f'Written: {args.target}')


if __name__ == '__main__':
    main()
