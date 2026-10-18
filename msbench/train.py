"""
Trains one reference network on one task and scores it (docs/tasks-and-metrics.md):

    python -m msbench.train <dataset folder> <run folder> --task T2-1mm --model unet --seed 0
                            --uncertainty references/label_uncertainty.parquet [--epochs 40 --batch 4 --hours 20]

The run folder holds last.pt (to continue an interrupted run: the same command again; given again after the last
epoch, it only scores best.pt, e.g. with a newer uncertainty file), best.pt (the best validation
error), log.csv (one line per epoch) and, at the end, results/<model>-s<seed>/<task>.json in the format of
msbench.evaluate, with the cost of the run (GPU-hours, inference time per case, GPU, code revision).

Training: AdamW with a one-cycle schedule (5 % warm-up, cosine decay), mixed precision (bfloat16) on GPUs, the mean squared error of the normalised
fields over the fluid points; the prediction is base + network output (base: the interpolated coarse fields for T2,
0 for T1). Validation: relative L2 error of the velocity on the val split after every epoch. A run stops after the
given epochs, or before the next epoch when --hours would be exceeded; it then scores the best checkpoint.
"""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from msbench.data import SCORING_SPLITS, Dataset
from msbench.evaluate import read_uncertainty, score_predictors
from msbench.models import MODELS
from msbench.torch_data import TaskData


def revision():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:
        return ''


def masked_mse(prediction, target, fluid):
    return (((prediction - target) ** 2) * fluid).sum() / (fluid.sum() * prediction.shape[1])


def velocity_error(prediction, target, fluid):
    """relative L2 error of (u_x, u_y) per sample, normalised units"""
    difference = ((prediction[:, :2] - target[:, :2]) ** 2 * fluid).flatten(1).sum(1)
    norm = ((target[:, :2]) ** 2 * fluid).flatten(1).sum(1)
    return torch.sqrt(difference / norm.clamp_min(1e-12))


def build(args, in_channels):
    options = {'unet': {}, 'fno': {'modes_x': args.modes_x, 'modes_y': args.modes_y}}[args.model]
    if args.width:
        options['width'] = args.width
    return MODELS[args.model](in_channels, 3, **options)


def validate(model, loader, device, amp):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, target, base, fluid in loader:
            inputs, target, base, fluid = (t.to(device, non_blocking=True) for t in (inputs, target, base, fluid))
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(inputs)
            errors.append(velocity_error(base + output.float(), target, fluid).cpu())
    model.train()
    return float(torch.cat(errors).mean())


def main():
    parser = argparse.ArgumentParser(description="Train and score a reference network")
    parser.add_argument('dataset')
    parser.add_argument('run')
    parser.add_argument('--task', required=True, choices=['T1', 'T2-2mm', 'T2-1mm-FO', 'T2-1mm'])
    parser.add_argument('--model', required=True, choices=sorted(MODELS))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--uncertainty', required=True)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--width', type=int, default=None, help="default: 32 for the UNet, 64 for the FNO")
    parser.add_argument('--modes-x', type=int, default=48)
    parser.add_argument('--modes-y', type=int, default=16)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--hours', type=float, default=None, help="stop before the next epoch after this many hours")
    parser.add_argument('--max-train', type=int, default=None, help="use only the first cases (smoke tests)")
    parser.add_argument('--splits', nargs='*', default=SCORING_SPLITS)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = device.type == 'cuda'
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    train = TaskData(args.dataset, args.task, 'train')
    val = TaskData(args.dataset, args.task, 'val')
    train_set = torch.utils.data.Subset(train, range(args.max_train)) if args.max_train else train
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              pin_memory=amp, drop_last=True, generator=generator, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, batch_size=args.batch, num_workers=args.workers, pin_memory=amp)

    model = build(args, train.in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(steps, 1),
                                                    pct_start=0.05)
    state = {'epoch': 0, 'best': float('inf'), 'seconds': 0.0, 'revision': revision()}
    if (run / 'last.pt').exists():                                   # continue an interrupted run
        checkpoint = torch.load(run / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        state = checkpoint['state']
        print(f"continuing after epoch {state['epoch']}", flush=True)

    log_exists = (run / 'log.csv').exists()
    with open(run / 'log.csv', 'a', newline='') as log:
        writer = csv.writer(log)
        if not log_exists:
            writer.writerow(['epoch', 'train_loss', 'val_velocity_rel_l2', 'seconds'])
        while state['epoch'] < args.epochs:
            if args.hours and state['seconds'] / 3600 * (1 + 1 / max(state['epoch'], 1)) > args.hours:
                print(f"stopping before epoch {state['epoch'] + 1}: the time budget of {args.hours} h", flush=True)
                break
            start, losses = time.time(), []
            for inputs, target, base, fluid in train_loader:
                inputs, target, base, fluid = (t.to(device, non_blocking=True) for t in (inputs, target, base, fluid))
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    output = model(inputs)
                loss = masked_mse(base + output.float(), target, fluid)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            error = validate(model, val_loader, device, amp)
            state['epoch'] += 1
            state['seconds'] += time.time() - start
            writer.writerow([state['epoch'], float(np.mean(losses)), error, round(state['seconds'], 1)])
            log.flush()
            if error < state['best']:
                state['best'] = error
                torch.save(model.state_dict(), run / 'best.pt')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'state': state}, run / 'last.pt')
            print(f"epoch {state['epoch']}: train loss {np.mean(losses):.4g}, val velocity rel L2 {error:.4g}, "
                  f"{state['seconds'] / 3600:.2f} h", flush=True)

    model.load_state_dict(torch.load(run / 'best.pt', map_location=device, weights_only=True))
    model.eval()
    dataset, uncertainty = Dataset(args.dataset), read_uncertainty(args.uncertainty)
    method = f"{args.model}-s{args.seed}"
    results = {'task': args.task, 'method': method, 'splits': {}}
    timings = []
    for split in args.splits:
        data = TaskData(args.dataset, args.task, split)

        def predict(i, case, label, data=data):
            inputs, _, base, _ = data[i]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                output = model(inputs[None].to(device))
            prediction = (base.to(device) + output[0].float())
            if device.type == 'cuda':
                torch.cuda.synchronize()
            timings.append(time.time() - start)
            return data.denormalise_fields(prediction.cpu().numpy())

        results['splits'][split] = score_predictors(dataset, split, {method: predict}, uncertainty,
                                                    with_attributes=args.task.startswith('T2'))[method]
        print(f"{split}: scored", flush=True)
    results['cost'] = {'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
                       'training_gpu_hours': round(state['seconds'] / 3600, 3), 'epochs': state['epoch'],
                       'inference_seconds_per_case_median': float(np.median(timings)) if timings else None,
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'best_val_velocity_rel_l2': state['best']}
    # a finished run given again only scores (to rescore with a newer evaluation): the training revision is kept
    results['config'] = {**vars(args), 'revision': revision(), 'training_revision': state.get('revision', ''),
                         'slurm_job_id': os.environ.get('SLURM_JOB_ID', '')}
    output = run / 'results' / method / f"{args.task}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1) + '\n')
    print(f"written to {output}")


if __name__ == "__main__":
    main()
