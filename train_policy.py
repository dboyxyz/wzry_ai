# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from hybrid_rewards import HybridRewards, DEFAULT_WEIGHTS, aligned_paths
ROOT = Path(__file__).resolve().parent

def discounted_returns(rewards, gamma=0.9999 * 0.985, horizon=202):
    result = np.zeros(len(rewards), dtype=np.float32)
    for t in range(len(rewards)):
        stop = min(len(rewards), t + horizon)
        result[t] = np.dot(rewards[t:stop], gamma ** np.arange(stop - t))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/runs')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run-id', nargs='+', help='Select one or more run directories')
    selection.add_argument('--all-runs', action='store_true', help='Select every run directory under data-dir')
    parser.add_argument('--state-weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'hybrid_policy_runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--policy-weights', type=Path, default=ROOT / 'weights/policy.pt')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--chunk-size', type=int, default=600)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--max-updates', type=int, default=0, help='Maximum updates; 0 means unlimited')
    parser.add_argument('--prepare-only', action='store_true', help='Only generate rewards')
    parser.add_argument('--max-frames', type=int, default=0, help='First N frames per run for a smoke test; 0 means all')
    parser.add_argument('--ocr-confidence-min', type=float, default=0.1)
    parser.add_argument('--gold-delta-min', type=int, default=20)
    parser.add_argument('--reward-alignment', choices=['same-frame', 'next-frame'], default='same-frame', help='Reward alignment; next-frame assigns the following observation reward to the current action')
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.chunk_size <= 1000 or min(args.save_every, args.max_updates, args.max_frames) < 0:
        parser.error('Invalid epoch count, chunk size (1..1000), or negative limits')
    if not 0 <= args.ocr_confidence_min <= 1 or args.gold_delta_min <= 0:
        parser.error('OCR threshold must be in [0,1] and gold threshold positive')
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.state_weights = args.state_weights.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite output directory: {args.output_dir}')
    device = torch.device(args.device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA unavailable; use --device cpu --prepare-only for reward preparation')
    if not args.prepare_only and str(device) != 'cuda:0':
        raise ValueError('Policy training requires cuda:0')
    vocabulary = json.loads((ROOT / 'config/action_to_id.json').read_text(encoding='utf-8'))
    if args.all_runs:
        run_names = sorted((path.name for path in args.data_dir.iterdir() if path.is_dir()))
        if not run_names:
            raise ValueError(f'No run directories under: {args.data_dir}')
    else:
        run_names = list(dict.fromkeys(args.run_id))
    missing_npz = [name for name in run_names if not (args.data_dir / name / 'features.npz').is_file()]
    if missing_npz:
        raise FileNotFoundError('Missing features.npz for runs: ' + ', '.join(missing_npz) + '. Run prepare_features.py first.')
    datasets = []
    for name in run_names:
        if Path(name).name != name:
            raise ValueError('run-id must be a direct child directory name')
        run = args.data_dir / name
        with np.load(run / 'features.npz', allow_pickle=False) as data:
            actions = data['actions']
            paths = aligned_paths(run, actions, vocabulary)
            if not paths or len(data['features']) != len(paths):
                raise ValueError(f'{run}: Empty data or mismatched feature count')
        if args.max_frames:
            paths = paths[:args.max_frames]
        datasets.append((run, paths))
    print(f'Selected {len(datasets)} runs, {sum((len(paths) for (_, paths) in datasets))} frames: ' + ', '.join((run.name for (run, _) in datasets)), flush=True)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    recognizer = HybridRewards(args.state_weights, device, args.ocr_confidence_min, args.gold_delta_min)
    rewards = {}
    for (run, paths) in datasets:
        rewards[run.name] = recognizer.prepare(paths, args.output_dir / run.name)
    del recognizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if args.prepare_only:
        print(f'Rewards prepared: {args.output_dir}', flush=True)
        return
    args.policy_weights = args.policy_weights.resolve()
    from policy_model import PolicyAgent
    from masks import create_masks
    agent = PolicyAgent(args.policy_weights, device)
    updates = 0
    for epoch in range(1, args.epochs + 1):
        for (run, paths) in datasets:
            with np.load(run / 'features.npz', allow_pickle=False) as data:
                features = data['features'][:len(paths)]
                targets = data['actions'][:len(paths)]
            inputs = np.concatenate(([128], targets[:-1]))
            breaks = [0] + [i for i in range(1, len(paths)) if int(paths[i].stem) != int(paths[i - 1].stem) + 1] + [len(paths)]
            for (left, right) in zip(breaks, breaks[1:]):
                inputs[left] = 128
                reward = rewards[run.name][left:right].copy()
                if args.reward_alignment == 'next-frame':
                    reward = np.concatenate((reward[1:], [0])).astype(np.float32)
                returns = discounted_returns(reward)
                for start in range(left, right, args.chunk_size):
                    end = min(right, start + args.chunk_size)
                    x = torch.from_numpy(features[None, start:end].astype(np.float32)).to(device)
                    a = torch.from_numpy(inputs[None, start:end].astype(np.int64)).to(device)
                    y = torch.from_numpy(targets[None, start:end].astype(np.int64)).to(device)
                    (_, mask) = create_masks(a, a, device)
                    weights = torch.from_numpy(returns[start - left:end - left]).to(device)
                    agent.model.train().requires_grad_(True)
                    for _ in range(3):
                        (logits, _) = agent.model(x, a, mask)
                        log_prob = logits.log_softmax(-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
                        loss = -(log_prob * weights).mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError('Nonfinite loss; stopping training')
                        agent.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
                        agent.optimizer.step()
                    updates += 1
                    print(f'Epoch {epoch} {run.name} [{start}:{end}] loss={loss.item():.6f}', flush=True)
                    with (args.output_dir / 'training.jsonl').open('a', encoding='utf-8') as log:
                        log.write(json.dumps(dict(epoch=epoch, run=run.name, start=start, end=end, update=updates, loss=loss.item())) + '\n')
                    if args.max_updates and updates >= args.max_updates:
                        torch.save(agent.model.state_dict(), args.output_dir / 'policy_last.pt')
                        return
        if epoch == args.epochs or (args.save_every and epoch % args.save_every == 0):
            torch.save(agent.model.state_dict(), args.output_dir / f'policy_epoch_{epoch:03d}.pt')
if __name__ == '__main__':
    main()
