# NLOST — Bunny (64×64×2048) support & reconstruction

This document records the changes made to train NLOST on the **Bunny** dataset
(transient shape **64×64×2048**) in addition to the existing **Bike** dataset
(transient shape **256×256×512**), and how to run training and reconstruction.

---

## 1. Summary

The original pipeline assumed a **256×256×512** measurement that the data loader
binned to **128×128×512** and the model processed at `spatial=64, tlen=256`
(transformer temporal resolution hard-coded to `16`).

The Bunny dataset is **64×64×2048** — smaller spatially, 4× deeper temporally.
The model was already parameterised by `spatial`, so no structural redesign was
needed; the work was:

* remove the one hard-coded tensor dimension (temporal `16`),
* fix a data-loader bug so temporal binning works,
* expose the dataset/model geometry through CLI arguments,
* add a Bunny training launcher and generalise the reconstruction script.

Bunny is trained **natively at full 64×64×2048** (no spatial or temporal
down-sampling): `spatial=32, tlen=1024, target_size=64`, transformer temporal =
`1024/16 = 64`. Peak GPU memory ≈ **4.8 GB** at batch size 1 (fits a 6 GB GPU).

**Bike behaviour is fully preserved** — every new argument defaults to the
original Bike values.

---

## 2. Architecture changes — `models/nlost.py`

* Replaced the hard-coded temporal resolution `16` in the three transformer
  `input_resolution` lists with a value **inferred from `tlen`**:

  ```python
  t_feat = self.tlen // 16   # FK volume depth (=tlen) is reduced by a fixed /16
                             # by the msfeat/sig_feat/dsfusion conv stack
  # loc_encds / glb_encds / locglb_inte now use [spatial, spatial, t_feat]
  ```

  * Bike (`tlen=256`)  → `t_feat = 16` (unchanged).
  * Bunny (`tlen=1024`) → `t_feat = 64`.

* No other model code changed. `spatial`, `target_size`, and all reconstruction
  / projection / rendering layers already infer their sizes dynamically.

---

## 3. Dataloader changes — `util/LFEDataset.py`

* Removed a **duplicated temporal down-sampling block** (mislabeled
  "spatial down-sampling") that applied the `ds` factor a second time, which
  double-binned the transient for any `ds > 1`.
  * `ds` now means a single temporal binning factor.
  * Bike uses `ds=1` (unchanged). Bunny can use `ds` for optional temporal
    binning; the native run uses `ds=1` and keeps all 2048 bins.
* The existing file lister already handles Bunny — its folder layout
  (`<root>/0/<model>/shine_.../{video-confocalspad*.mp4, confocal-0-*.hdr,
  depth-0-*.hdr}`) is identical to Bike.

---

## 4. Configuration / argument changes

### `util/ParseArgs.py` — new arguments (defaults reproduce Bike)

| Argument          | Default | Bike | Bunny (native) | Meaning                                   |
|-------------------|---------|------|----------------|-------------------------------------------|
| `--dataset`       | `bike`  | bike | bunny          | dataset name (informational)              |
| `--meas_size`     | `256`   | 256  | 64             | raw measurement spatial size (loader)     |
| `--ds`            | `1`     | 1    | 1              | temporal binning factor in the loader     |
| `--clip`          | `512`   | 512  | 2048           | temporal bins kept                         |
| `--model_spatial` | `64`    | 64   | 32             | model spatial grid (= measurement / 2)    |
| `--tlen`          | `256`   | 256  | 1024           | FK crop / temporal length (power of 2)    |
| `--bin_len`       | `0.01`  | 0.01 | 0.01           | FK temporal bin length                     |

### `train.py`

* The `LFEDataset(...)` and `nlost.NLOST(...)` calls now read the values above
  from `opt` instead of hard-coded literals (`size=256, clip=512, spatial=64,
  tlen=256, bin_len=0.01`).

---

## 5. New / updated scripts

* **`run_train_bunny.bat`** (new) — trains Bunny natively at 64×64×2048.
* `run_train_nlost.bat` (Bike) — unchanged behaviour.
* **`predict.py`** (reconstruction) — generalised:
  * reads MATLAB **v7.3 / HDF5** `.mat` files (via `h5py`) as well as classic
    `.mat`, plus `.npy` / `.mp4`;
  * **block-sum spatial binning** for any integer factor
    (256→128 for Bike, 512→64 for the 512-res test file), photon-count
    preserving; non-integer factors fall back to interpolation;
  * temporal crop / zero-pad to `--clip`.
* `validate_ours.py` — fixed to load our (non-DataParallel) checkpoints and use
  the trained model geometry (`spatial`, `bin_len`, `target_size`) instead of the
  previous hard-coded 256-grid config.

---

## 6. How to train

### Bunny (native 64×64×2048)

```bat
run_train_bunny.bat
```

or explicitly:

```bat
python train.py --model_dir "D:\NLOST\checkpoints_bunny" --model_name nlost ^
    --dataset bunny --data_dir "D:\NLOST\bunny" ^
    --meas_size 64 --ds 1 --clip 2048 ^
    --model_spatial 32 --tlen 1024 --bin_len 0.01 --target_size 64 ^
    --bacth_size 1 --num_workers 0 --num_epoch 6 --num_save 999999
```

Notes: `--num_epoch N` trains `N-1` epochs (loop is `range(1, N)`), so pass `6`
for 5 epochs. Checkpoints are written to `checkpoints_bunny<name>_<date>/` as
`epoch_<e>_<iter>_END.pth`.

### Bike (unchanged)

```bat
run_train_nlost.bat
```

---

## 7. How to reconstruct (the Bunny model)

Reconstruct a transient `.mat` with a trained Bunny checkpoint. The script
auto-reads v7.3/HDF5 files and auto-bins the spatial grid down to `target_size`.

```bat
python predict.py ^
    --checkpoint "checkpoints_bunnynlost_2026_78\epoch_5_250_END.pth" ^
    --input "D:\NLOST\dataset\meas_10min.mat" ^
    --spatial 32 --tlen 1024 --clip 2048 --target_size 64 --bin_len 0.01 ^
    --out "predictions_bunny"
```

* A **real 64×64×2048** test file needs no binning (already on the grid) and uses
  the exact same command — just change `--input`.
* A **512×512×2048** file (e.g. `dataset/meas_10min.mat`, key `meas`) is binned
  8×8 → 64 automatically.

Outputs written to the `--out` folder for input `NAME.mat`:

* `NAME_intensity.png` / `NAME_intensity.npy`
* `NAME_depth.png` / `NAME_depth.npy`
* `NAME_recon.mat` (keys: `intensity`, `depth`)

---

## 8. Assumptions & limitations

* **`tlen` must be a power of 2** (FK requirement) and the model input has
  `T = 2·tlen`, `H = W = 2·spatial`.
* **`bin_len = 0.01`** is reused from Bike. It sets the FK depth scaling; since
  Bunny is trained from scratch the network adapts. Tune via `--bin_len` if the
  true bin length is known.
* **Validation split**: `list_file_path_bike` sends model dirs `[:250]` to train
  and `[250:]` to test. Bunny (10 dirs) and Bike (few dirs) therefore have an
  empty val set, so training uses `--num_save 999999` to skip validation. Lower
  the split index to hold out validation data.
* Datasets, checkpoints, logs, and prediction outputs are **git-ignored**
  (large / generated); only code and docs are versioned.
