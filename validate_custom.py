import os
import glob
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import scipy.io as sio
import cv2
from tqdm import tqdm

from models import nlost
from util.CustomDataset import NLOSTCustomDataset, _get_file_list
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from metric import RMSE, PSNR, SSIM, AverageMeter

cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',       type=str, required=True,
                        help='Root folder with transient/, img/ subfolders')
    parser.add_argument('--checkpoint',     type=str, required=True,
                        help='Path to .pth checkpoint file')
    parser.add_argument('--output_dir',     type=str, default='output_custom',
                        help='Where to save predicted intensity images')
    parser.add_argument('--spatial',        type=int, default=64)
    parser.add_argument('--tlen',           type=int, default=256)
    parser.add_argument('--bin_len',        type=float, default=0.0192)
    parser.add_argument('--target_size',    type=int, default=64)
    parser.add_argument('--num_coders',     type=int, default=1)
    parser.add_argument('--batch_size',     type=int, default=1)
    parser.add_argument('--num_workers',    type=int, default=0)
    parser.add_argument('--no_amp',         action='store_true', default=False,
                        help='Disable mixed precision inference')
    parser.add_argument('--split',          type=str, default='all',
                        choices=['all', 'val'],
                        help='"all" runs on every sample; "val" uses same 20%% val split as training')
    parser.add_argument('--train_ratio',    type=float, default=0.8,
                        help='Must match the value used during training (only used when --split val)')
    parser.add_argument('--seed',           type=int, default=42,
                        help='Must match the value used during training (only used when --split val)')
    return parser.parse_args()


def load_model(args, device):
    model = nlost.NLOST(
        ch_in       = 1,
        num_coders  = args.num_coders,
        spatial     = args.spatial,
        tlen        = args.tlen,
        bin_len     = args.bin_len,
        target_size = args.target_size,
    )
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)

    # handle DataParallel prefix 'module.'
    new_state = {}
    for k, v in state.items():
        new_state[k.replace('module.', '')] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f'[WARN] Missing keys  : {missing}')
    if unexpected:
        print(f'[WARN] Unexpected keys: {unexpected}')

    print(f'Loaded checkpoint: {args.checkpoint}')
    model.eval()
    return model


def get_file_list(args):
    transient_dir = os.path.join(args.data_dir, 'transient')
    all_files = _get_file_list(transient_dir)

    if args.split == 'val':
        rng     = np.random.default_rng(args.seed)
        indices = rng.permutation(len(all_files))
        n_train = int(len(all_files) * args.train_ratio)
        files   = [all_files[i] for i in indices[n_train:]]
        print(f'Val split  : {len(files)} / {len(all_files)} samples')
    else:
        files = all_files
        print(f'All samples: {len(files)}')

    return files


def main():
    args  = parse_args()
    use_amp = not args.no_amp
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # ── model ────────────────────────────────────────────────────────────────
    model = load_model(args, device)

    # ── dataset ──────────────────────────────────────────────────────────────
    file_list     = get_file_list(args)
    transient_dir = os.path.join(args.data_dir, 'transient')
    img_dir       = os.path.join(args.data_dir, 'img')
    vol_dir       = os.path.join(args.data_dir, 'vol')

    dataset = NLOSTCustomDataset(transient_dir, img_dir, vol_dir, file_list)
    loader  = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # ── metrics ──────────────────────────────────────────────────────────────
    rmse_fn = RMSE().to(device)
    psnr_fn = PSNR().to(device)
    ssim_fn = SSIM().to(device)
    meters  = {k: AverageMeter() for k in ['rmse', 'psnr', 'ssim']}

    print(f'\nRunning inference on {len(dataset)} samples...')
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            M_mea  = batch['ds_meas'].to(device)           # (B,1,512,64,64)
            img_gt = batch['img_gt'].to(device).unsqueeze(1)  # (B,1,64,64)

            with autocast(enabled=use_amp):
                vlo_re, inten_re, _ = model(M_mea)
                inten_re = (inten_re + 1) / 2

            inten_re = inten_re.float()
            vlo_re   = vlo_re.float()

            pred   = torch.clamp(inten_re.detach(), 0, 1)
            target = torch.clamp(img_gt,            0, 1)

            meters['rmse'].update(rmse_fn(pred, target).cpu().item())
            meters['psnr'].update(psnr_fn(pred, target).cpu().item())
            meters['ssim'].update(ssim_fn(pred, target).cpu().item())

            # save each image in the batch
            B = pred.shape[0]
            for b in range(B):
                name    = file_list[sample_idx]
                pred_np = pred[b, 0].cpu().numpy()    # (H,W) range [0,1]
                gt_np   = target[b, 0].cpu().numpy()  # (H,W) range [0,1]

                # normalise pred intensity for display
                pmax = pred_np.max()
                pred_disp = pred_np / pmax if pmax > 0 else pred_np

                cv2.imwrite(
                    os.path.join(args.output_dir, f'{name}_pred_int.png'),
                    (pred_disp * 255).astype(np.uint8),
                )
                cv2.imwrite(
                    os.path.join(args.output_dir, f'{name}_gt_int.png'),
                    (gt_np * 255).astype(np.uint8),
                )

                # save predicted volume
                vol_np = vlo_re[b, 0].cpu().numpy()   # (D, H, W)

                # depth max-projection → PNG (what the 3D volume looks like from top)
                vol_proj = np.max(vol_np, axis=0)      # (H, W)
                vmax = vol_proj.max()
                vol_disp = vol_proj / vmax if vmax > 0 else vol_proj
                cv2.imwrite(
                    os.path.join(args.output_dir, f'{name}_pred_vol_proj.png'),
                    (vol_disp * 255).astype(np.uint8),
                )

                # save full 3D volume as .mat
                sio.savemat(
                    os.path.join(args.output_dir, f'{name}_pred_vol.mat'),
                    {'pred_vol': vol_np},
                )

                sample_idx += 1

    # ── print summary ────────────────────────────────────────────────────────
    print('\n' + '=' * 50)
    print(f'Results on {len(dataset)} samples:')
    print(f'  RMSE : {meters["rmse"].item():.4f}')
    print(f'  PSNR : {meters["psnr"].item():.2f} dB')
    print(f'  SSIM : {meters["ssim"].item():.4f}')
    print('=' * 50)
    print(f'Output images saved to: {args.output_dir}')

    # save metrics to txt
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'Checkpoint : {args.checkpoint}\n')
        f.write(f'Data dir   : {args.data_dir}\n')
        f.write(f'Split      : {args.split}\n')
        f.write(f'Samples    : {len(dataset)}\n')
        f.write(f'RMSE       : {meters["rmse"].item():.4f}\n')
        f.write(f'PSNR       : {meters["psnr"].item():.2f} dB\n')
        f.write(f'SSIM       : {meters["ssim"].item():.4f}\n')
    print(f'Metrics saved to    : {metrics_path}')


if __name__ == '__main__':
    main()
