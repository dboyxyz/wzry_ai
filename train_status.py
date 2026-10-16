# Adapted from WZCQ and the local Jetson project; reorganized for this repository.
from __future__ import annotations
import argparse
import csv
import html
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from config import WEIGHTS_DIR
ImageFile.LOAD_TRUNCATED_IMAGES = True
LABELS = ('normal', 'dead', 'tower_attack')
LABEL_TO_ID = {label: index for (index, label) in enumerate(LABELS)}
DEFAULT_JSON = Path('data/status/labels.jsonl')
DEFAULT_IMAGE_DIR = DEFAULT_JSON.parent
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / 'status_training'

@dataclass(frozen=True)
class Sample:
    key: str
    source_key: str
    label: str
    label_id: int
    group: str
    path: Path
    augmented: bool

def is_augmented(key: str) -> bool:
    return '__aug_' in key or '__aug2_' in key

def get_source_key(key: str) -> str:
    return re.split('__(?:aug|aug2)_', key, maxsplit=1)[0]

def get_group(source_key: str) -> str:
    match = re.match('^(\\d+)_(\\d+)$', source_key)
    if match:
        return match.group(1)
    if source_key.isdigit():
        return 'pure_digits'
    return 'misc'

def resolve_image(image_dir: Path, key: str) -> Optional[Path]:
    for suffix in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        path = image_dir / f'{key}{suffix}'
        if path.is_file():
            return path
    return None

def load_samples(json_path: Path, image_dir: Path) -> list[Sample]:
    samples = []
    seen = set()
    with json_path.open('r', encoding='utf-8') as stream:
        for (line_number, line) in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{json_path}:{line_number}: {exc}') from exc
            for (key, label) in item.items():
                (key, label) = (str(key), str(label))
                if label not in LABEL_TO_ID or key in seen:
                    continue
                path = resolve_image(image_dir, key)
                if path is None:
                    continue
                source_key = get_source_key(key)
                samples.append(Sample(key=key, source_key=source_key, label=label, label_id=LABEL_TO_ID[label], group=get_group(source_key), path=path, augmented=is_augmented(key)))
                seen.add(key)
    return samples

def choose_validation_groups(originals: list[Sample], validation_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    groups = sorted({sample.group for sample in originals})
    if len(groups) < 2:
        raise ValueError('At least two runs are required for a grouped split')
    target_size = len(originals) * validation_ratio
    total_counts = Counter((sample.label_id for sample in originals))
    rng = random.Random(seed)
    best: Optional[tuple[float, set[str]]] = None
    validation_group_count = max(1, round(len(groups) * validation_ratio))
    for _ in range(5000):
        chosen = set(rng.sample(groups, validation_group_count))
        subset = [sample for sample in originals if sample.group in chosen]
        counts = Counter((sample.label_id for sample in subset))
        if any((counts[index] == 0 for index in range(len(LABELS)))):
            continue
        size_error = abs(len(subset) - target_size) / max(target_size, 1)
        ratio_error = sum((abs(counts[index] / len(subset) - total_counts[index] / len(originals)) for index in range(len(LABELS))))
        score = size_error + 2.0 * ratio_error
        if best is None or score < best[0]:
            best = (score, chosen)
    if best is None:
        raise ValueError('No grouped validation split contains all three classes')
    validation_groups = best[1]
    return (set(groups) - validation_groups, validation_groups)

def limit_per_class(samples: list[Sample], maximum: Optional[int], seed: int) -> list[Sample]:
    if maximum is None or maximum <= 0:
        return samples
    rng = random.Random(seed)
    result = []
    for label_id in range(len(LABELS)):
        rows = [sample for sample in samples if sample.label_id == label_id]
        rng.shuffle(rows)
        result.extend(rows[:maximum])
    rng.shuffle(result)
    return result

class StatusDataset(Dataset):

    def __init__(self, samples: list[Sample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = ImageOps.exif_transpose(image).convert('RGB')
            tensor = self.transform(image)
        return (tensor, sample.label_id)

def build_transforms(height: int, width: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([transforms.Resize((height, width), antialias=True), transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.06), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    validation_transform = transforms.Compose([transforms.Resize((height, width), antialias=True), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return (train_transform, validation_transform)

def build_model(backbone_name: str, pretrained: bool) -> tuple[nn.Module, bool]:
    constructors = {'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT), 'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT), 'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT), 'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT)}
    (constructor, weights) = constructors[backbone_name]
    loaded_pretrained = False
    if pretrained:
        try:
            model = constructor(weights=weights)
            loaded_pretrained = True
        except Exception as exc:
            print(f'Pretrained weights unavailable; random initialization: {exc}')
            model = constructor(weights=None)
    else:
        model = constructor(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(input_features, len(LABELS)))
    return (model, loaded_pretrained)

def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    model = model._orig_mod if hasattr(model, '_orig_mod') else model
    for (name, parameter) in model.named_parameters():
        is_classifier = name.startswith('fc.') or '.fc.' in name
        parameter.requires_grad = trainable or is_classifier

def set_frozen_backbone_eval(model: nn.Module) -> None:
    model = model._orig_mod if hasattr(model, '_orig_mod') else model
    for (name, module) in model.named_children():
        if name != 'fc':
            module.eval()

def create_balanced_sampler(samples: list[Sample]) -> WeightedRandomSampler:
    counts = Counter((sample.label_id for sample in samples))
    weights = [1.0 / counts[sample.label_id] for sample in samples]
    return WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)

def confusion_metrics(confusion: np.ndarray) -> dict:
    per_class = {}
    (recalls, f1_values) = ([], [])
    for (index, label) in enumerate(LABELS):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1_values.append(f1)
        per_class[label] = {'support': support, 'precision': precision, 'recall': recall, 'f1': f1}
    return {'accuracy': float(np.trace(confusion) / max(confusion.sum(), 1)), 'balanced_accuracy': float(np.mean(recalls)), 'macro_f1': float(np.mean(f1_values)), 'per_class': per_class, 'confusion': confusion.tolist()}

def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: Optional[torch.optim.Optimizer], scaler: torch.amp.GradScaler, amp_enabled: bool, grad_clip: float, backbone_frozen: bool=False) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    if training and backbone_frozen:
        set_frozen_backbone_eval(model)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    loss_sum = 0.0
    sample_count = 0
    for (images, targets) in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        predictions = logits.argmax(dim=1)
        truth_np = targets.detach().cpu().numpy()
        prediction_np = predictions.detach().cpu().numpy()
        np.add.at(confusion, (truth_np, prediction_np), 1)
        batch_size = len(targets)
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
    return (loss_sum / max(sample_count, 1), confusion_metrics(confusion))

def print_metrics(prefix: str, loss: float, metrics: dict) -> None:
    print(f"{prefix}: loss={loss:.5f} acc={metrics['accuracy']:.2%} balanced_acc={metrics['balanced_accuracy']:.2%} macro_f1={metrics['macro_f1']:.2%}")
    for label in LABELS:
        row = metrics['per_class'][label]
        print(f"  {label}: n={row['support']} P={row['precision']:.2%} R={row['recall']:.2%} F1={row['f1']:.2%}")

@torch.no_grad()
def export_validation_errors(model: nn.Module, loader: DataLoader, samples: list[Sample], device: torch.device, amp_enabled: bool, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / 'images'
    image_dir.mkdir(exist_ok=True)
    model.eval()
    rows = []
    offset = 0
    for (images, targets) in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        target_values = targets.numpy()
        for (local_index, (target, prediction, probability)) in enumerate(zip(target_values, predictions, probabilities)):
            sample = samples[offset + local_index]
            sorted_probability = np.sort(probability)
            rows.append({'key': sample.key, 'path': str(sample.path), 'true_label': LABELS[int(target)], 'predicted_label': LABELS[int(prediction)], 'correct': bool(target == prediction), 'confidence': float(probability[prediction]), 'margin': float(sorted_probability[-1] - sorted_probability[-2]), **{f'prob_{label}': float(probability[label_id]) for (label_id, label) in enumerate(LABELS)}})
        offset += len(targets)
    field_names = list(rows[0]) if rows else []
    with (output_dir / 'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / 'predictions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    errors = [row for row in rows if not row['correct']]
    html_rows = []
    error_counts = Counter()
    for (index, row) in enumerate(errors, 1):
        category = f"{row['true_label']} → {row['predicted_label']}"
        error_counts[category] += 1
        source = Path(row['path'])
        filename = f"{index:03d}_{row['key']}{source.suffix.lower()}"
        shutil.copy2(source, image_dir / filename)
        probability_text = '<br>'.join((f"{html.escape(label)}={row[f'prob_{label}']:.4f}" for label in LABELS))
        html_rows.append(f"""<tr><td>{index}</td><td><code>{html.escape(row['key'])}</code></td><td>{html.escape(row['true_label'])}</td><td>{html.escape(row['predicted_label'])}</td><td>{probability_text}<br>margin={row['margin']:.4f}</td><td><a href="images/{html.escape(filename)}"><img src="images/{html.escape(filename)}"></a></td></tr>""")
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8">\n<title>Three-state validation errors</title><style>body{font-family:sans-serif;margin:20px;background:#f5f5f5}\ntable{border-collapse:collapse;background:white}td,th{border:1px solid #aaa;padding:6px;text-align:center}\nimg{width:480px}th{position:sticky;top:0;background:#eee}</style></head><body>\n<h1>Best model validation errors</h1>'
    page += f'<p>Validation images={len(rows)}, errors={len(errors)}；{html.escape(str(dict(error_counts)))}</p>'
    page += '<table><tr><th>#</th><th>Key</th><th>Truth</th><th>Prediction</th><th>Probabilities</th><th>Image</th></tr>' + ''.join(html_rows) + '</table></body></html>'
    (output_dir / 'index.html').write_text(page, encoding='utf-8')
    summary = {'validation_samples': len(rows), 'errors': len(errors), 'accuracy': 1.0 - len(errors) / max(len(rows), 1), 'error_distribution': dict(error_counts)}
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def choose_device(device_name: str, require_cuda: bool) -> torch.device:
    if device_name == 'auto':
        device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device_name.startswith('cuda') and (not torch.cuda.is_available()):
        if require_cuda:
            raise SystemExit('CUDA required but unavailable')
        print('CUDA unavailable; using CPU')
        device_name = 'cpu'
    return torch.device(device_name)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a three-class visual state classifier')
    parser.add_argument('--mode', choices=('quick', 'full'), default='quick')
    parser.add_argument('--label-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--image-dir', type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device', default='auto', help='auto, cpu, or cuda:0')
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--backbone', choices=('resnet18', 'resnet34', 'resnet50', 'resnet101'), default='resnet50')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--include-augmented', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile; start without it on Jetson')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--image-width', type=int, default=384)
    parser.add_argument('--image-height', type=int, default=192)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--backbone-lr-mult', type=float, default=0.1, help='Backbone learning-rate multiplier')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--validation-ratio', type=float, default=0.28)
    parser.add_argument('--validation-groups', nargs='+', default=None, help='Hold out these runs for validation')
    parser.add_argument('--seed', type=int, default=20260907)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=None)
    parser.add_argument('--sampling', choices=('balanced', 'natural'), default='balanced')
    parser.add_argument('--max-train-per-class', type=int, default=None)
    parser.add_argument('--max-val-per-class', type=int, default=None)
    parser.add_argument('--early-stopping', type=int, default=7)
    parser.add_argument('--save-history', action=argparse.BooleanOptionalAction, default=True, help='Save epoch checkpoints at the requested interval')
    parser.add_argument('--save-every', type=int, default=1, help='Checkpoint interval in epochs')
    parser.add_argument('--history-with-optimizer', action='store_true', help='Include optimizer state in epoch checkpoints')
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--eval-all', action='store_true', help='Evaluate all original images; requires --eval-only')
    parser.add_argument('--export-errors-dir', type=Path, default=None, help='Export per-image probabilities and an HTML error report during evaluation')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.eval_only and args.resume is None:
        args.resume = WEIGHTS_DIR / 'status_best.pt'
    if args.save_every <= 0:
        raise SystemExit('--save-every must be positive')
    if args.eval_all and (not args.eval_only):
        raise SystemExit('--eval-all requires --eval-only')
    if args.mode == 'quick':
        args.epochs = 2 if args.epochs is None else args.epochs
        args.batch_size = 16 if args.batch_size is None else args.batch_size
        args.workers = 2 if args.workers is None else args.workers
        args.freeze_backbone_epochs = 1 if args.freeze_backbone_epochs is None else args.freeze_backbone_epochs
        if args.max_train_per_class is None:
            args.max_train_per_class = 128
        if args.max_val_per_class is None:
            args.max_val_per_class = 64
    else:
        args.epochs = 30 if args.epochs is None else args.epochs
        args.batch_size = 32 if args.batch_size is None else args.batch_size
        args.workers = 4 if args.workers is None else args.workers
        args.freeze_backbone_epochs = 2 if args.freeze_backbone_epochs is None else args.freeze_backbone_epochs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    device = choose_device(args.device, args.require_cuda)
    amp_enabled = args.amp and device.type == 'cuda'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.label_json, args.image_dir)
    originals = [sample for sample in samples if not sample.augmented]
    if args.eval_all:
        train_groups: set[str] = set()
        validation_groups = {sample.group for sample in originals}
        train_samples: list[Sample] = []
        validation_samples = originals
    else:
        if args.validation_groups:
            all_groups = {sample.group for sample in originals}
            validation_groups = set(args.validation_groups)
            if not validation_groups <= all_groups:
                raise ValueError('Requested validation run does not exist')
            train_groups = all_groups - validation_groups
        else:
            (train_groups, validation_groups) = choose_validation_groups(originals, args.validation_ratio, args.seed)
        train_samples = [sample for sample in samples if sample.group in train_groups and (args.include_augmented or not sample.augmented)]
        validation_samples = [sample for sample in originals if sample.group in validation_groups]
        train_samples = limit_per_class(train_samples, args.max_train_per_class, args.seed)
    validation_samples = limit_per_class(validation_samples, args.max_val_per_class, args.seed + 1)
    if not validation_samples or (not args.eval_only and (not train_samples)):
        raise SystemExit('Training or validation set is empty')
    print(f'Device: {device}; AMP: {amp_enabled}; mode: {args.mode}')
    if args.eval_all:
        print('Test distribution:', dict(Counter((sample.label for sample in validation_samples))))
    else:
        print('Training runs:', sorted(train_groups))
        print('Validation runs:', sorted(validation_groups))
        print('Training distribution:', dict(Counter((sample.label for sample in train_samples))))
        print('Validation distribution:', dict(Counter((sample.label for sample in validation_samples))))
        print('Augmented training images:', sum((sample.augmented for sample in train_samples)))
    (train_transform, validation_transform) = build_transforms(args.image_height, args.image_width)
    validation_dataset = StatusDataset(validation_samples, validation_transform)
    loader_options = {'num_workers': args.workers, 'pin_memory': device.type == 'cuda', 'persistent_workers': args.workers > 0}
    train_loader = None
    if train_samples:
        train_dataset = StatusDataset(train_samples, train_transform)
        sampler = create_balanced_sampler(train_samples) if args.sampling == 'balanced' else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, drop_last=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **loader_options)
    (model, loaded_pretrained) = build_model(args.backbone, args.pretrained)
    model.to(device)
    if args.compile:
        model = torch.compile(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    classifier_parameters = list(base_model.fc.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [parameter for parameter in base_model.parameters() if id(parameter) not in classifier_ids]
    optimizer = torch.optim.AdamW([{'params': backbone_parameters, 'lr': args.lr * args.backbone_lr_mult}, {'params': classifier_parameters, 'lr': args.lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    (start_epoch, best_macro_f1) = (0, -1.0)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint['model_state'])
        if not args.eval_only:
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                print('Checkpoint has no optimizer state; using a new optimizer')
            start_epoch = int(checkpoint['epoch']) + 1
            best_macro_f1 = float(checkpoint.get('best_macro_f1', -1.0))
        print(f'Loaded: {args.resume}')
    history_path = args.output_dir / 'history.json'
    if args.resume is not None and history_path.is_file():
        try:
            history = json.loads(history_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            history = []
    else:
        history = []
    epochs_without_improvement = 0
    epoch_range = range(start_epoch, start_epoch + 1) if args.eval_only else range(start_epoch, args.epochs)
    for epoch in epoch_range:
        started = time.perf_counter()
        frozen = loaded_pretrained and epoch < args.freeze_backbone_epochs
        set_backbone_trainable(model, not frozen)
        if args.eval_only:
            (train_loss, train_metrics) = (math.nan, {})
        else:
            assert train_loader is not None
            (train_loss, train_metrics) = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp_enabled, grad_clip=5.0, backbone_frozen=frozen)
        (validation_loss, validation_metrics) = run_epoch(model, validation_loader, criterion, device, None, scaler, amp_enabled, grad_clip=5.0)
        if not args.eval_only:
            scheduler.step()
            print_metrics(f'Epoch {epoch + 1} train', train_loss, train_metrics)
        print_metrics(f'Epoch {epoch + 1} val', validation_loss, validation_metrics)
        if args.eval_only and args.export_errors_dir is not None:
            error_summary = export_validation_errors(model, validation_loader, validation_samples, device, amp_enabled, args.export_errors_dir)
            print(f"Error report: {args.export_errors_dir / 'index.html'}")
            print(f"Error distribution: {error_summary['error_distribution']}")
        print(f'  frozen_backbone={frozen} elapsed={time.perf_counter() - started:.1f}s')
        row = {'epoch': epoch, 'train_loss': train_loss, 'train_metrics': train_metrics, 'validation_loss': validation_loss, 'validation_metrics': validation_metrics}
        history.append(row)
        if args.eval_only:
            break
        current_score = validation_metrics['macro_f1']
        checkpoint = {'epoch': epoch, 'model_state': base_model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'best_macro_f1': max(best_macro_f1, current_score), 'labels': LABELS, 'backbone': args.backbone, 'image_size': (args.image_height, args.image_width), 'config': vars(args), 'train_metrics': train_metrics, 'validation_metrics': validation_metrics}
        torch.save(checkpoint, args.output_dir / 'last.pt')
        if current_score > best_macro_f1:
            best_macro_f1 = current_score
            epochs_without_improvement = 0
            torch.save(checkpoint, args.output_dir / 'best.pt')
            print(f"  Saved best checkpoint: {args.output_dir / 'best.pt'}")
        else:
            epochs_without_improvement += 1
        if args.save_history and (not args.eval_only) and ((epoch + 1) % args.save_every == 0):
            epoch_dir = args.output_dir / 'epochs'
            epoch_dir.mkdir(exist_ok=True)
            history_checkpoint = checkpoint if args.history_with_optimizer else {key: value for (key, value) in checkpoint.items() if key != 'optimizer_state'}
            epoch_path = epoch_dir / f'epoch_{epoch + 1:03d}.pt'
            torch.save(history_checkpoint, epoch_path)
            print(f'  Saved epoch checkpoint: {epoch_path}')
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        if epochs_without_improvement >= args.early_stopping:
            print(f'Consecutive {args.early_stopping} epochs without improvement; early stopping')
            break
    if args.eval_only:
        print('Evaluation complete; training checkpoints and metadata unchanged')
        return 0
    metadata = {'labels': LABELS, 'train_groups': sorted(train_groups), 'validation_groups': sorted(validation_groups), 'train_distribution': dict(Counter((sample.label for sample in train_samples))), 'validation_distribution': dict(Counter((sample.label for sample in validation_samples))), 'train_augmented': sum((sample.augmented for sample in train_samples)), 'loaded_pretrained': loaded_pretrained, 'best_macro_f1': best_macro_f1, 'arguments': vars(args)}
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print(f"Training complete; best checkpoint: {args.output_dir / 'best.pt'}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
