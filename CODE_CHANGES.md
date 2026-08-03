# Code Changes — exact file & line reference

Every source change made to support multi-size training (Bunny 64×64×2048,
256×256×2048, …) and reconstruction, with exact locations, before/after code,
and reason. Line numbers are for the committed state on `main`.

Commits: `4bc1f46` (core changes), `9bdc2eb` / `da3652b` (docs & launchers).

---

## 1. `models/nlost.py`  —  remove the hard-coded transformer temporal dim

**Location:** `NLOST.__init__`, lines **65-71**.

**Before:**
```python
self.loc_encds = modelClone(WindowEncoderSep(..., input_resolution=[self.spatial,self.spatial,16], ...), self.coders)
self.glb_encds = modelClone(GlobalEncoderSep(..., input_resolution=[self.spatial//2,self.spatial//2,16], ...), self.coders)
self.locglb_inte = LocGlbInteNBlks_LCGC_l1d2(channels_m * 4, [self.spatial,self.spatial,16], 8, self.coders)
```

**After:**
```python
t_feat = self.tlen // 16          # line 65
self.loc_encds  = modelClone(WindowEncoderSep(..., input_resolution=[self.spatial,self.spatial,t_feat], ...), self.coders)      # line 67
self.glb_encds  = modelClone(GlobalEncoderSep(..., input_resolution=[self.spatial//2,self.spatial//2,t_feat], ...), self.coders) # line 69
self.locglb_inte = LocGlbInteNBlks_LCGC_l1d2(channels_m * 4, [self.spatial,self.spatial,t_feat], 8, self.coders)                 # line 71
```

**Why:** The FK volume depth equals `tlen`, and the conv stack
(`msfeat`→`sig_feat`→`dsfusion`) reduces it by a fixed factor of 16, so the
transformer temporal resolution is `tlen//16`. Hard-coding `16` assumed `tlen=256`.
Now any `tlen` works (Bike 256→16, Bunny/256-data 1024→64). Bike unchanged.

---

## 2. `util/LFEDataset.py`  —  fix duplicated temporal down-sampling

**Location:** `LFEDataset._load_meas`, block removed at lines **163-168** (comment
now at **163-170**). The kept temporal binning is at lines **139-144**.

**Before (two identical blocks — the transient was binned twice for ds>1):**
```python
# line 139  temporal down-sampling
if self.ds > 1:
    c, t, h, w = raw.shape
    raw = raw.reshape(c, t // self.ds, self.ds, h, w).sum(axis=2)
raw = raw[:, :self.clip]                       # clip
... color handling ...
# spatial down-sampling   <-- MISLABELED; actually a 2nd temporal binning
if self.ds > 1:
    c, t, h, w = raw.shape
    raw = raw.reshape(c, t // self.ds, self.ds, h, w).sum(axis=2)
```

**After:** the second block is deleted, leaving only the first (lines 139-144);
a NOTE comment replaces it at lines 163-170.

**Why:** With `ds>1` the transient was down-sampled twice (e.g. Bunny `ds=4`
would give 2048→512→128 instead of 2048→512). Removing the duplicate makes `ds`
a single temporal binning factor. Bike uses `ds=1`, so it is byte-for-byte
unchanged.

---

## 3. `util/ParseArgs.py`  —  new geometry arguments

**Location:** `get_args_parser`, lines **26-34** (inserted before the model params).

**Added:**
```python
# line 27  parser.add_argument("--dataset",       type=str,   default="bike")
# line 28  parser.add_argument("--meas_size",     type=int,   default=256)   # loader spatial size
# line 29  parser.add_argument("--ds",            type=int,   default=1)     # temporal binning factor
# line 30  parser.add_argument("--clip",          type=int,   default=512)   # temporal bins kept
# line 32  parser.add_argument("--model_spatial", type=int,   default=64)    # model spatial grid
# line 33  parser.add_argument("--tlen",          type=int,   default=256)   # FK crop (power of 2)
# line 34  parser.add_argument("--bin_len",       type=float, default=0.01)  # FK bin length
```

**Why:** Exposes the dataset/model geometry through the CLI. Every default equals
the original Bike value, so existing commands behave identically.

---

## 4. `train.py`  —  build loader and model from the new args

**Location A — dataloader** (both `train_data` and `val_data`), lines **68-70**
and **80-82**.

**Before:** `ds=1, clip=512, size=256`
**After:**  `ds=opt.ds, clip=opt.clip, size=opt.meas_size`

**Location B — model**, line **107**.

**Before:**
```python
model = nlost.NLOST(ch_in=1, num_coders=1, spatial=64, tlen=256, bin_len=0.01, target_size=opt.target_size)
```
**After:**
```python
model = nlost.NLOST(ch_in=1, num_coders=1, spatial=opt.model_spatial, tlen=opt.tlen, bin_len=opt.bin_len, target_size=opt.target_size)
```

**Why:** The single training script now drives any dataset size from the CLI
args instead of hard-coded literals.

---

## 5. `predict.py`  —  new reconstruction script (v7.3/HDF5 + any-factor binning)

New file. Key pieces:

* **`_load_mat_array(path, key)`** — lines **72-90**: loads classic `.mat` via
  `scipy.io`, and MATLAB **v7.3/HDF5** via `h5py` (line 80) when scipy raises
  `NotImplementedError`. Auto-picks the largest dataset if no key is given.
* **Block-sum spatial binning** — lines **142-146**:
  ```python
  elif H % target_size == 0:                     # line 142
      f = H // target_size
      meas = meas.reshape(1, target_size, f, target_size, f, tt).sum(axis=(2, 4))   # line 145
  ```
  Handles any integer factor (256→128 ×2, 512→64 ×8), photon-count preserving.
* Temporal crop / zero-pad to `--clip`; saves `*_intensity.png/.npy`,
  `*_depth.png/.npy`, `*_recon.mat`.

**Why:** Reconstruct an arbitrary transient (`.mat`/`.npy`/`.mp4`, any spatial
size) with a trained checkpoint, matching the model geometry via CLI flags.

---

## 6. `validate_ours.py`  —  load our checkpoints with the correct geometry

**Location:** `main`, lines **18** and **26-30**; args at lines **74-77**.

* **Line 18** — model built from args instead of a fixed 256-grid config:
  ```python
  model = nlost.NLOST(ch_in=1, num_coders=1, spatial=args.spatial, tlen=256, bin_len=args.bin_len, target_size=args.target_size)
  ```
* **Lines 26-30** — add the `module.` prefix so our (non-DataParallel) checkpoint
  loads into the `DataParallel` model (previously the weights stayed random):
  ```python
  new_dict = {}
  for k, v in ckpt_dict.items():
      nk = k if k.startswith('module.') else 'module.' + k
      new_dict[nk] = v
  missing, unexpected = model.load_state_dict(new_dict, strict=False)
  ```
* **Lines 74-77** — new `--spatial` / `--bin_len` args + default paths pointing
  at the test set.

**Why:** The original script built a mismatched model and never loaded the trained
weights; it now reconstructs correctly in the same batch/output format.

---

## Files NOT changed (proof the design is reusable)

The feature/attention/reconstruction modules were **not** modified — they already
infer sizes dynamically: `models/nlost_modules.py`, `models/modules.py`,
`models/utils_pytorch/fk_1_10.py`, `pro/Train.py`, `pro/Validate.py`,
`util/SaveChkp.py`. All new dataset sizes are handled by arguments alone.
