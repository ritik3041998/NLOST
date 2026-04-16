import os
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from models import nlost
from util.CustomDataset import build_dataloaders
from util.SaveChkp import save_checkpoint
from pro.Loss import criterion_L2
from metric import RMSE, PSNR, SSIM, AverageMeter

cudnn.benchmark = True


# argument parser

def parse_args():
    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument('--data_dir',     type=str, required=True,
                        help='Root folder containing transient/, img/, vol/')
    parser.add_argument('--model_dir',    type=str, default='checkpoints/custom',
                        help='Where to save checkpoints and logs')
    # data
    parser.add_argument('--batch_size',   type=int,   default=2)
    parser.add_argument('--num_workers',  type=int,   default=0)
    parser.add_argument('--train_ratio',  type=float, default=0.8)
    parser.add_argument('--seed',         type=int,   default=42)
    # model
    parser.add_argument('--spatial',      type=int,   default=64)
    parser.add_argument('--tlen',         type=int,   default=256)
    parser.add_argument('--bin_len',      type=float, default=0.0192,
                        help='Sensor bin length in metres (c * dt, after temporal binning)')
    parser.add_argument('--target_size',  type=int,   default=64)
    parser.add_argument('--num_coders',   type=int,   default=1)
    # training
    parser.add_argument('--num_epoch',    type=int,   default=50)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_save',     type=int,   default=100,
                        help='Validate and save checkpoint every N iterations')
    # mixed precision
    parser.add_argument('--amp',          action='store_true', default=True,
                        help='Enable mixed precision training (default: on)')
    parser.add_argument('--no_amp',       action='store_false', dest='amp',
                        help='Disable mixed precision training')
    # resume
    parser.add_argument('--resume',       type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


# validation

def validate(model, val_loader, n_iter, logWriter, device, use_amp):
    metric_list = ['rmse', 'psnr', 'ssim']
    rmse_fn = RMSE().to(device)
    psnr_fn = PSNR().to(device)
    ssim_fn = SSIM().to(device)
    meters  = {k: AverageMeter() for k in metric_list}

    l_int = []
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader, desc='  val'):
            M_mea  = sample['ds_meas'].cuda()          # (B,1,512,64,64)
            img_gt = sample['img_gt'].cuda().unsqueeze(1)  # (B,1,64,64)

            with autocast(enabled=use_amp):
                _, inten_re, _ = model(M_mea)
                inten_re = (inten_re + 1) / 2

            inten_re = inten_re.float()

            l_int.append(criterion_L2(inten_re, img_gt).item())

            pred   = torch.clamp(inten_re.detach(), 0, 1)
            target = torch.clamp(img_gt,            0, 1)
            meters['rmse'].update(rmse_fn(pred, target).cpu().item())
            meters['psnr'].update(psnr_fn(pred, target).cpu().item())
            meters['ssim'].update(ssim_fn(pred, target).cpu().item())

    logWriter.add_scalar('val/loss_int', np.mean(l_int), n_iter)
    for k in metric_list:
        logWriter.add_scalar(f'val/{k}', meters[k].item(), n_iter)

    return {k: meters[k].item() for k in metric_list}


# main

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.model_dir, 'train.log')),
            logging.StreamHandler(),
        ],
        format='%(asctime)s %(levelname)s: %(message)s',
    )
    logging.info('=' * 60)
    logging.info(f'Args: {args}')
    logging.info(f'Mixed precision (AMP): {args.amp}')
    logging.info('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Building dataloaders...')
    train_loader, val_loader = build_dataloaders(
        data_root   = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        train_ratio = args.train_ratio,
        seed        = args.seed,
    )
    logging.info(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')

    logging.info('Building model...')
    model = nlost.NLOST(
        ch_in       = 1,
        num_coders  = args.num_coders,
        spatial     = args.spatial,
        tlen        = args.tlen,
        bin_len     = args.bin_len,
        target_size = args.target_size,
    )
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logging.info(f'Using {torch.cuda.device_count()} GPUs')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Trainable parameters: {n_params / 1e6:.2f}M')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 1
    n_iter      = 0
    if args.resume is not None:
        logging.info(f'Resuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt.get('epoch', 1) + 1
        n_iter      = ckpt.get('n_iter', 0)
        logging.info(f'Resumed at epoch {start_epoch}, iter {n_iter}')

    logWriter = SummaryWriter(args.model_dir)

    logging.info('Starting training...')
    for epoch in range(start_epoch, args.num_epoch + 1):
        model.train()
        logging.info(f'Epoch {epoch}/{args.num_epoch}  lr={optimizer.param_groups[0]["lr"]:.2e}')

        for sample in tqdm(train_loader, desc=f'  epoch {epoch}'):
            M_mea  = sample['ds_meas'].cuda()              # (B,1,512,64,64)
            img_gt = sample['img_gt'].cuda().unsqueeze(1)  # (B,1,64,64)

            with autocast(enabled=args.amp):
                _, inten_re, _ = model(M_mea)
                inten_re = (inten_re + 1) / 2
                loss = criterion_L2(inten_re.float(), img_gt)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            n_iter += 1

            logWriter.add_scalar('train/loss',      loss.item(),        n_iter)
            logWriter.add_scalar('train/amp_scale', scaler.get_scale(), n_iter)

            if n_iter % args.num_save == 0:
                logging.info(f'[iter {n_iter}] Running validation...')
                val_metrics = validate(
                    model, val_loader, n_iter, logWriter, device, args.amp
                )
                model.train()
                log_str = ' | '.join(f'{k} {v:.4f}' for k, v in val_metrics.items())
                logging.info(f'[iter {n_iter}] val: {log_str}')
                save_checkpoint(
                    n_iter, epoch, model, optimizer,
                    file_path=os.path.join(
                        args.model_dir, f'epoch{epoch}_iter{n_iter}.pth'
                    ),
                )

        ckpt_path = os.path.join(args.model_dir, f'epoch{epoch}_end.pth')
        save_checkpoint(n_iter, epoch, model, optimizer, file_path=ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt['scaler'] = scaler.state_dict()
        torch.save(ckpt, ckpt_path)

        logging.info(f'Epoch {epoch} done. Checkpoint saved to {ckpt_path}')

    logWriter.close()
    logging.info('Training complete.')


if __name__ == '__main__':
    main()
