import os
import numpy as np
import torch
import scipy.io as scio
import cv2
from models import nlost

# ── config ──────────────────────────────────────────────────────────────────
MAT_PATH   = r"D:\NLOST\dataset\align_fk_256_512_meas_10min\bike_10min.mat"
CKPT_PATH  = r"D:\NLOST\checkpointsnlost_2026_518\epoch_4_176_END.pth"
OUT_DIR    = r"D:\NLOST\output_test_mat"
TARGET_SIZE = 128
# ────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = nlost.NLOST(ch_in=1, num_coders=1, spatial=64, tlen=256,
                    bin_len=0.01, target_size=TARGET_SIZE)
ckpt = torch.load(CKPT_PATH, map_location=device)
state = ckpt["state_dict"]
# strip 'module.' prefix if saved from DDP
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state)
model.to(device).eval()
print("Model loaded.")

# load & preprocess transient data
# raw shape: (256, 256, 512)  →  (1, 256, 256, 512)
raw = scio.loadmat(MAT_PATH)["final_meas"].astype(np.float32)
M = raw.reshape(1, 256, 256, -1)               # (1, 256, 256, 512)

# spatial 2× downsample to match training (size=256, target_size=128)
M = M[:, ::2, :, :] + M[:, 1::2, :, :]        # (1, 128, 256, 512)
M = M[:, :, ::2, :] + M[:, :, 1::2, :]        # (1, 128, 128, 512)

M = np.ascontiguousarray(M)
M = np.transpose(M, (0, 3, 1, 2))             # (1, 512, 128, 128)
M_mea = torch.from_numpy(M[None]).to(device)  # (1, 1, 512, 128, 128)
print("Input shape:", M_mea.shape)

# inference
with torch.no_grad():
    _, inten_re, dep_re = model(M_mea)
    inten_re = (inten_re + 1) / 2
    dep_re   = (dep_re   + 1) / 2

# save outputs
inten = inten_re.squeeze().cpu().numpy()
dep   = dep_re.squeeze().cpu().numpy()

cv2.imwrite(os.path.join(OUT_DIR, "intensity.png"),
            (inten / (inten.max() + 1e-8) * 255).astype(np.uint8))
cv2.imwrite(os.path.join(OUT_DIR, "depth.png"),
            (dep.clip(0, 1) * 255).astype(np.uint8))

print(f"Saved to {OUT_DIR}")
