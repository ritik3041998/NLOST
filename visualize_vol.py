import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os


def load_vol(path):
    if path.endswith('.mat'):
        d = sio.loadmat(path)
        key = [k for k in d.keys() if not k.startswith('_')][0]
        vol = d[key].astype(np.float32)
    else:
        raise ValueError('Only .mat files supported')

    # normalize to [0, 1]
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    print(f'Loaded: {path}')
    print(f'Shape : {vol.shape}  (D x H x W)')
    print(f'Range : [{vmin:.4f}, {vmax:.4f}]')
    return vol


def show_max_projections(vol, title='Volume Max Projections'):
    """Show max projections along all 3 axes — like MATLAB volume viewer overview."""
    xy = np.max(vol, axis=0)   # top view    (H, W)
    xz = np.max(vol, axis=1)   # front view  (D, W)
    yz = np.max(vol, axis=2)   # side view   (D, H)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].imshow(xy,  cmap='hot', origin='upper')
    axes[0].set_title('Top view  (max over depth)')
    axes[0].set_xlabel('W'); axes[0].set_ylabel('H')

    axes[1].imshow(xz,  cmap='hot', origin='upper')
    axes[1].set_title('Front view  (max over H)')
    axes[1].set_xlabel('W'); axes[1].set_ylabel('Depth')

    axes[2].imshow(yz,  cmap='hot', origin='upper')
    axes[2].set_title('Side view  (max over W)')
    axes[2].set_xlabel('H'); axes[2].set_ylabel('Depth')

    for ax in axes:
        plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def show_slice_viewer(vol, title='Slice Viewer'):
    """Interactive slice viewer — drag slider to move through depth slices."""
    D, H, W = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.2)
    fig.suptitle(title, fontsize=13)

    d_init = D // 2
    h_init = H // 2
    w_init = W // 2

    im0 = axes[0].imshow(vol[d_init, :, :], cmap='hot', vmin=0, vmax=1, origin='upper')
    axes[0].set_title(f'XY slice  (depth={d_init})')
    axes[0].set_xlabel('W'); axes[0].set_ylabel('H')

    im1 = axes[1].imshow(vol[:, h_init, :], cmap='hot', vmin=0, vmax=1, origin='upper')
    axes[1].set_title(f'XZ slice  (H={h_init})')
    axes[1].set_xlabel('W'); axes[1].set_ylabel('Depth')

    im2 = axes[2].imshow(vol[:, :, w_init], cmap='hot', vmin=0, vmax=1, origin='upper')
    axes[2].set_title(f'YZ slice  (W={w_init})')
    axes[2].set_xlabel('H'); axes[2].set_ylabel('Depth')

    for ax, im in zip(axes, [im0, im1, im2]):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # sliders
    ax_d = plt.axes([0.10, 0.08, 0.22, 0.03])
    ax_h = plt.axes([0.42, 0.08, 0.22, 0.03])
    ax_w = plt.axes([0.72, 0.08, 0.22, 0.03])

    s_d = Slider(ax_d, 'Depth', 0, D - 1, valinit=d_init, valstep=1)
    s_h = Slider(ax_h, 'H',     0, H - 1, valinit=h_init, valstep=1)
    s_w = Slider(ax_w, 'W',     0, W - 1, valinit=w_init, valstep=1)

    def update(_):
        d = int(s_d.val)
        h = int(s_h.val)
        w = int(s_w.val)
        im0.set_data(vol[d, :, :])
        im1.set_data(vol[:, h, :])
        im2.set_data(vol[:, :, w])
        axes[0].set_title(f'XY slice  (depth={d})')
        axes[1].set_title(f'XZ slice  (H={h})')
        axes[2].set_title(f'YZ slice  (W={w})')
        fig.canvas.draw_idle()

    s_d.on_changed(update)
    s_h.on_changed(update)
    s_w.on_changed(update)


def compare_two(vol1, vol2, label1='Predicted', label2='Ground Truth'):
    """Side-by-side max projection comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f'{label1}  vs  {label2}', fontsize=13)

    for row, (vol, lbl) in enumerate([(vol1, label1), (vol2, label2)]):
        xy = np.max(vol, axis=0)
        xz = np.max(vol, axis=1)
        yz = np.max(vol, axis=2)
        for col, (proj, view) in enumerate(zip([xy, xz, yz],
                                               ['Top (max depth)', 'Front (max H)', 'Side (max W)'])):
            im = axes[row, col].imshow(proj, cmap='hot', vmin=0, vmax=1, origin='upper')
            axes[row, col].set_title(f'{lbl} — {view}')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vol',    type=str, required=True,
                        help='Path to pred_vol.mat or gt vol.mat')
    parser.add_argument('--gt',     type=str, default=None,
                        help='Optional: path to GT vol.mat for comparison')
    parser.add_argument('--mode',   type=str, default='both',
                        choices=['proj', 'slice', 'both'],
                        help='proj=max projections, slice=interactive slicer, both=show all')
    args = parser.parse_args()

    vol = load_vol(args.vol)
    name = os.path.basename(args.vol)

    if args.mode in ('proj', 'both'):
        show_max_projections(vol, title=f'Max Projections — {name}')

    if args.mode in ('slice', 'both'):
        show_slice_viewer(vol, title=f'Slice Viewer — {name}')

    if args.gt is not None:
        gt = load_vol(args.gt)
        compare_two(vol, gt, label1='Predicted', label2='Ground Truth')

    plt.show()


if __name__ == '__main__':
    main()
