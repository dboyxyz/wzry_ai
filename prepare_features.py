# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import torch
import torchvision
import numpy as np
import os
import json
from PIL import Image
from resnet_utils import myResnet, load_policy_backbone
import argparse
from config import ROOT

def _load_json_lines(path: str) -> list[dict]:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        rows: list[dict] = []
        for line in f:
            s = line.strip()
            if not s:
                continue
            s = s.replace("'", '"')
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
        return rows

def _find_op_json(data_dir: str, run_dir: str, run_id: str) -> str | None:
    candidates = [os.path.join(run_dir, '_actions.jsonl'), os.path.join(data_dir, f'{run_id}_actions.jsonl')]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def main() -> int:
    parser = argparse.ArgumentParser(description='Precompute policy features from JPG frames and action records')
    parser.add_argument('--data-dir', default='./data/runs', help='Root directory of recorded runs')
    parser.add_argument('--run-id', default=None, help='Process one run directory')
    parser.add_argument('--device', default='cuda:0', help='Device, for example cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size; start with 4 on a GPU')
    parser.add_argument('--feat-dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--max-frames', type=int, default=0, help='Frame limit per run; 0 means all')
    args = parser.parse_args()
    record_root = args.data_dir
    if not os.path.exists(record_root):
        os.makedirs(record_root)
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        print('CUDA unavailable; using CPU')
        device = torch.device('cpu')
    backbone = load_policy_backbone()
    resnet101 = myResnet(backbone.eval()).to(device).requires_grad_(False)
    vocabulary_path = ROOT / 'config/action_to_id.json'
    with open(vocabulary_path, encoding='utf-8') as f:
        vocabulary = json.load(f)
    for (root, dirs, files) in os.walk(record_root):
        if len(dirs) > 0:
            break
    run_names = [args.run_id] if args.run_id else dirs
    for run_id in run_names:
        run_dir = os.path.join(record_root, run_id)
        if not os.path.isdir(run_dir):
            continue
        record_path = _find_op_json(record_root, run_dir, run_id)
        if record_path is None:
            print(f'Skipping run without action records: {run_id}')
            continue
        cache_path = os.path.join(run_dir, 'features.npz')
        if os.path.isfile(cache_path):
            continue
        records = _load_json_lines(record_path)
        if not records:
            print(f'Skipping empty action file: {run_id}')
            continue
        valid_rows: list[tuple[str, dict]] = []
        for df in records:
            img_path = os.path.join(run_dir, '{}.jpg'.format(df.get('frame_id', '')))
            if os.path.isfile(img_path):
                valid_rows.append((img_path, df))
            if args.max_frames and len(valid_rows) >= args.max_frames:
                break
        if not valid_rows:
            print(f'Skipping run without images: {run_id}')
            continue
        n = len(valid_rows)
        feat_dim = 6 * 6 * 2048
        feat_np_dtype = np.float16 if args.feat_dtype == 'float16' else np.float32
        features = np.empty((n, feat_dim), dtype=feat_np_dtype)
        actions = np.empty((n,), dtype=np.int64)
        control_modes = np.empty((n,), dtype=np.int64)
        movement = 'keep-move'
        print('Processing {}'.format(run_id))

        def _iter_batches():
            bs = max(1, int(args.batch_size))
            for start in range(0, n, bs):
                yield (start, valid_rows[start:start + bs])
        use_amp = device.type == 'cuda' and args.feat_dtype == 'float16'
        out_t_dtype = torch.float16 if args.feat_dtype == 'float16' else torch.float32
        row_index = 0
        with torch.inference_mode():
            for (start, batch) in _iter_batches():
                imgs: list[np.ndarray] = []
                dfs: list[dict] = []
                for (img_path, df) in batch:
                    img = Image.open(img_path).convert('RGB')
                    imgs.append(np.asarray(img))
                    dfs.append(df)
                img_t = torch.tensor(np.stack(imgs, axis=0), device=device, dtype=torch.float32)
                img_t = img_t.permute(0, 3, 2, 1) / 255.0
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        (_, out) = resnet101(img_t)
                else:
                    (_, out) = resnet101(img_t)
                if out.dim() == 3:
                    out = out.unsqueeze(0)
                elif out.dim() != 4:
                    raise RuntimeError(f'unexpected feature shape: {tuple(out.shape)}')
                out = out.reshape(out.size(0), -1).to(dtype=out_t_dtype)
                out_cpu = out.detach().to('cpu').numpy()
                bsz = out_cpu.shape[0]
                features[row_index:row_index + bsz] = out_cpu
                for j in range(bsz):
                    df = dfs[j]
                    movementa = df.get('movement', 'keep-move')
                    if movementa != 'keep-move':
                        movement = movementa
                    key = movement + '_' + df.get('skill', 'none')
                    actions[row_index + j] = int(vocabulary.get(key, 0))
                    control_modes[row_index + j] = int(df.get('control_mode', 0))
                row_index += bsz
                if row_index % 200 == 0 or row_index == n:
                    print(f'run={run_id} processed={row_index}/{n}')
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
        np.savez(cache_path, features=features, actions=actions, control_modes=control_modes)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
