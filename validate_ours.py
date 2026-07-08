import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import scipy.io as scio
import cv2
import util.SetDistTrain as utils
import cv2
cudnn.benchmark = True
from models import nlost
def main(args):
    
    # baseline  -- config MUST match how the checkpoint was trained (see train.py)
    model = nlost.NLOST(ch_in=1, num_coders=1,spatial=args.spatial,tlen=256,bin_len=args.bin_len,target_size=args.target_size)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model_path = args.pretrained_model
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location="cpu")
        ckpt_dict = checkpoint['state_dict']
        # our checkpoint has no 'module.' prefix; the DataParallel model expects one
        new_dict = {}
        for k, v in ckpt_dict.items():
            nk = k if k.startswith('module.') else 'module.' + k
            new_dict[nk] = v
        missing, unexpected = model.load_state_dict(new_dict, strict=False)
        print('Loaded', model_path, '| missing:', len(missing), 'unexpected:', len(unexpected))
    else:
        print('Loading Failed', model_path)

    print("Start eval...")
    all_file = []
    files = os.listdir(args.fk_data_path)
    for fi in files:
        fi_d = os.path.join(args.fk_data_path, fi)
        all_file.append(fi_d)

    out_path = args.output_path
    if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)

    for i in range(len(all_file)): 
        transient_data = scio.loadmat(all_file[i])
        transient_data = transient_data['final_meas']  # h w t
        M_wnoise = np.asarray(transient_data).astype(np.float32).reshape([1, 256, 256, -1])  # 1, H, W, T
        if args.target_size == 128:
            M_wnoise = M_wnoise[:, ::2, :, :] + M_wnoise[:, 1::2, :, :]
            M_wnoise = M_wnoise[:, :, ::2, :] + M_wnoise[:, :, 1::2, :]
        M_wnoise = np.ascontiguousarray(M_wnoise)
        M_wnoise = np.transpose(M_wnoise, (0, 3, 1, 2))  # 1, T, H, W
        M_mea = torch.from_numpy(M_wnoise[None])  # 1, 1, T, H, W
        print(M_mea.shape)
        with torch.no_grad():
            model.eval()
            vlo_re, im_re, dep_re = model(M_mea)
            im_re = (im_re + 1) / 2
            dep_re = (dep_re + 1) / 2
            front_view = im_re.detach().cpu().numpy()[0, 0]
            front_dep = dep_re.detach().cpu().numpy()[0, 0]
            name = files[i][:-4]
            cv2.imwrite(out_path + f'/{name}_int.png', (front_view / np.max(front_view)) * 255)
            cv2.imwrite(out_path + f'/{name}_dep.png', front_dep * 255)
            del vlo_re, im_re, dep_re
        del M_mea
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fk_data_path", type=str, default=r"D:\NLOST\dataset\align_fk_256_512_meas_10min", help="Path to the fk dataset.")
    parser.add_argument("--target_size", type=int, default=128, help="The spatial resolution")
    parser.add_argument("--spatial", type=int, default=64, help="model spatial grid (must match training)")
    parser.add_argument("--bin_len", type=float, default=0.01, help="temporal bin length (must match training)")
    parser.add_argument("--output_path", type=str, default=r"D:\NLOST\output_ours", help="Path to output.")
    parser.add_argument("--pretrained_model", type=str, default=r"D:\NLOST\checkpointsnlost_2026_78\epoch_4_176_END.pth", help="Pretrained Model Path.")
    args = parser.parse_args()

    return args

def test():
    args = parse_args()
    main(args)
    


if __name__=="__main__":
    test()
    




