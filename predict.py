"""
Standalone inference / reconstruction for a trained NLOST checkpoint.

Given a single transient measurement file, load the trained model and predict
the front-view intensity image and the depth map, then save them as PNG (and
raw .npy).

The model config here MUST match how the checkpoint was trained (see
run_train_nlost.bat / train.py): spatial=64, tlen=256, bin_len=0.01,
target_size=128, and a measurement that enters the network as
[B, 1, T, H, W] with H=W=128 (256x256 inputs are 2x2 spatially binned to 128,
exactly like util/LFEDataset.py and pro/Validate.test_on_align_fk).

Examples
--------
  python predict.py --checkpoint checkpointsnlost_2026_78/epoch_4_176_END.pth \
      --input path/to/transient.mat --out predictions/

  python predict.py --checkpoint ckpt5_nlost_2026_78/epoch_5_220_END.pth \
      --input meas.mat --key final_meas --out predictions/
"""
import os
import argparse
import numpy as np
import torch
import cv2
import scipy.io as scio

from models import nlost


# Candidate variable names for the transient array inside a .mat file.
MAT_KEYS = ['data', 'final_meas', 'meas', 'measlr', 'sig', 'transient', 'M', 'x']


def parse_args():
    p = argparse.ArgumentParser("NLOST reconstruction on a transient file")
    p.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    p.add_argument("--input", required=True, help="transient file (.mat/.npy/.hdr/.mp4)")
    p.add_argument("--out", default="predictions", help="output directory")
    p.add_argument("--key", default="auto",
                   help="variable name inside a .mat file (default: auto-detect)")
    # model config -- keep these matched to training unless you know better
    p.add_argument("--target_size", type=int, default=128)
    p.add_argument("--spatial", type=int, default=64)
    p.add_argument("--tlen", type=int, default=256)
    p.add_argument("--bin_len", type=float, default=0.01)
    p.add_argument("--clip", type=int, default=512, help="temporal bins fed to the net")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _pick_mat_array(mat, key):
    if key != "auto":
        if key not in mat:
            raise KeyError("key '%s' not in mat file; available: %s"
                           % (key, [k for k in mat.keys() if not k.startswith('__')]))
        return np.asarray(mat[key])
    for k in MAT_KEYS:
        if k in mat:
            return np.asarray(mat[k])
    # fall back: the largest non-private array
    cands = [(k, np.asarray(v)) for k, v in mat.items()
             if not k.startswith('__') and hasattr(v, 'shape')]
    if not cands:
        raise KeyError("no usable array found in mat file")
    cands.sort(key=lambda kv: kv[1].size, reverse=True)
    print("[predict] auto-picked mat key '%s' %s" % (cands[0][0], cands[0][1].shape))
    return cands[0][1]


def _load_mat_array(path, key):
    """Load the transient array from a .mat file, supporting both classic (<=v7.2)
    and v7.3 (HDF5) formats. Returns a numpy array."""
    try:
        return _pick_mat_array(
            scio.loadmat(path, verify_compressed_data_integrity=False), key)
    except NotImplementedError:
        # MATLAB v7.3 files are HDF5 -> read with h5py
        import h5py
        with h5py.File(path, 'r') as f:
            names = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
            if key != "auto":
                if key not in f:
                    raise KeyError("key '%s' not in v7.3 mat; available: %s" % (key, names))
                name = key
            else:
                # pick the largest dataset (the transient cube)
                name = max(names, key=lambda k: int(np.prod(f[k].shape)))
                print("[predict] auto-picked v7.3 mat key '%s' %s" % (name, f[name].shape))
            return np.array(f[name])


def _to_HWT(arr):
    """Reshape any 3D transient into (H, W, T) with H == W (square SPAD grid)."""
    arr = np.squeeze(np.asarray(arr)).astype(np.float32)
    if arr.ndim == 4:                       # e.g. (1,H,W,T) or (1,T,H,W)
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError("expected a 3D transient, got shape %s" % (arr.shape,))
    h, w, d = arr.shape
    # Identify the temporal axis as the one that does NOT equal the (square) spatial dims.
    if h == w:                              # already (H, W, T)
        return arr
    if w == d:                              # (T, H, W)
        return np.transpose(arr, (1, 2, 0))
    if h == d:                              # (H, T, W)
        return np.transpose(arr, (0, 2, 1))
    raise ValueError("cannot infer H,W,T from shape %s (no two equal spatial dims)"
                     % (arr.shape,))


def load_measurement(path, key, target_size, clip):
    """Return a torch tensor shaped [1, 1, T, H, W] ready for the model."""
    ext = path.split('.')[-1].lower()
    if ext == 'mat':
        arr = _load_mat_array(path, key)
        hwt = _to_HWT(arr)
    elif ext == 'npy':
        hwt = _to_HWT(np.load(path))
    elif ext == 'mp4':
        cap = cv2.VideoCapture(path); assert cap.isOpened(), "cannot open %s" % path
        frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        hwt = np.stack(frames, axis=-1).astype(np.float32) / 255.0   # (H, W, T)
    else:
        raise ValueError("unsupported input extension: .%s" % ext)

    H = hwt.shape[0]
    meas = hwt[None]                        # (1, H, W, T)

    # Bring the measurement onto the model's spatial grid (= target_size).
    # Any exact integer factor is handled by block-sum binning (photon-count preserving),
    # e.g. 256->128 (bike, x2) or 512->64 (meas_10min, x8). Non-integer -> interpolate.
    if H == target_size:
        pass
    elif H % target_size == 0:
        f = H // target_size
        _, hh, ww, tt = meas.shape
        meas = meas.reshape(1, target_size, f, target_size, f, tt).sum(axis=(2, 4))
        print("[predict] %dx%d spatial block-binning: %d -> %d" % (f, f, H, target_size))
    else:
        print("[predict] WARNING: input spatial %d not a multiple of %d; resizing." % (H, target_size))
        meas = np.stack([cv2.resize(meas[0, :, :, t], (target_size, target_size))
                         for t in range(meas.shape[-1])], axis=-1)[None]

    # Temporal: crop or zero-pad to `clip` bins.
    T = meas.shape[-1]
    if T >= clip:
        meas = meas[..., :clip]
        if T > clip:
            print("[predict] temporal crop: %d -> %d bins" % (T, clip))
    else:
        pad = np.zeros(meas.shape[:-1] + (clip - T,), dtype=np.float32)
        meas = np.concatenate([meas, pad], axis=-1)
        print("[predict] temporal zero-pad: %d -> %d bins" % (T, clip))

    meas = np.ascontiguousarray(np.transpose(meas, (0, 3, 1, 2)))   # (1, T, H, W)
    return torch.from_numpy(meas[None].astype(np.float32))          # (1, 1, T, H, W)


def load_model(checkpoint, opt, device):
    model = nlost.NLOST(ch_in=1, num_coders=1, spatial=opt.spatial,
                        tlen=opt.tlen, bin_len=opt.bin_len, target_size=opt.target_size)
    ck = torch.load(checkpoint, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[predict] missing keys: %d (e.g. %s)" % (len(missing), missing[:3]))
    if unexpected:
        print("[predict] unexpected keys: %d (e.g. %s)" % (len(unexpected), unexpected[:3]))
    ep = ck.get("epoch", "?") if isinstance(ck, dict) else "?"
    print("[predict] loaded checkpoint '%s' (epoch %s)" % (checkpoint, ep))
    return model.to(device).eval()


def save_gray(path, arr):
    a = arr.astype(np.float32)
    mx = np.max(a)
    a = a / mx if mx > 0 else a
    cv2.imwrite(path, (a * 255).clip(0, 255).astype(np.uint8))


def main():
    opt = parse_args()
    os.makedirs(opt.out, exist_ok=True)
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    model = load_model(opt.checkpoint, opt, device)
    meas = load_measurement(opt.input, opt.key, opt.target_size, opt.clip).to(device)
    print("[predict] model input shape:", tuple(meas.shape))

    with torch.no_grad():
        _, im_re, dep_re = model(meas)
        im_re = (im_re + 1) / 2
        dep_re = (dep_re + 1) / 2

    intensity = im_re.detach().cpu().numpy()[0, 0]
    depth = dep_re.detach().cpu().numpy()[0, 0]

    stem = os.path.splitext(os.path.basename(opt.input))[0]
    save_gray(os.path.join(opt.out, stem + "_intensity.png"), intensity)
    save_gray(os.path.join(opt.out, stem + "_depth.png"), depth)
    np.save(os.path.join(opt.out, stem + "_intensity.npy"), intensity)
    np.save(os.path.join(opt.out, stem + "_depth.npy"), depth)
    scio.savemat(os.path.join(opt.out, stem + "_recon.mat"),
                 {"intensity": intensity, "depth": depth})

    print("[predict] intensity range [%.4f, %.4f] shape %s"
          % (intensity.min(), intensity.max(), intensity.shape))
    print("[predict] depth     range [%.4f, %.4f] shape %s"
          % (depth.min(), depth.max(), depth.shape))
    print("[predict] saved reconstruction to %s/" % opt.out)


if __name__ == "__main__":
    main()
