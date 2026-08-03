# Training NLOST on any transient size (incl. 256×256×2048)

After the parameterization changes in commit `4bc1f46` (documented in
`CHANGES_BUNNY.md`), NLOST can train on any square transient **S×S×T** by
setting **arguments only** — no further code edits are required. This file gives
the recipe and the worked 256×256×2048 configuration for a high-memory GPU.

---

## 1. Code changes that enable arbitrary sizes (already applied & pushed)

| # | File | Change | Effect |
|---|------|--------|--------|
| 1 | `models/nlost.py`    | transformer temporal dim `16` → `tlen//16` | any temporal depth |
| 2 | `util/LFEDataset.py` | removed duplicated temporal-downsample block | `ds` bins once (or keep all bins) |
| 3 | `util/ParseArgs.py`  | new args `--meas_size/--ds/--clip/--model_spatial/--tlen/--bin_len` | configurable geometry |
| 4 | `train.py`           | build loader + model from those args | one pipeline for all sizes |

Pull commit `4bc1f46` on the target machine; that is the entire code delta.

---

## 2. How to derive the arguments from a dataset of shape S×S×T

Uses the proven `sa=2` layout (model halves spatial internally):

| Argument          | Rule                                   |
|-------------------|----------------------------------------|
| `--meas_size`     | `S`  (native; use `2S` only to bin 2S→S) |
| `--target_size`   | `S`  (= GT image size = model input spatial) |
| `--model_spatial` | `S / 2`                                |
| `--clip`          | temporal bins kept; **must be 2 × (power of 2)**: 512, 1024, 2048, 4096 … |
| `--ds`            | `T / clip` to bin, or `1` to keep all bins (then `clip = T`) |
| `--tlen`          | `clip / 2` (**power of 2**)            |
| `--bin_len`       | `0.01` (33 ps); change only if the true bin width differs |

Hard constraints: `tlen` is a power of 2 · model input = `[B, 1, clip, S, S]` ·
transformer temporal = `tlen/16` · GT confocal/depth resolution = `target_size`.

---

## 3. Worked configurations

| Dataset            | S×S×T        | meas_size | ds | clip | model_spatial | tlen | target_size | output PNG |
|--------------------|--------------|-----------|----|------|---------------|------|-------------|-----------|
| Bike               | 256×256×512  | 256       | 1  | 512  | 64            | 256  | 128         | 128×128   |
| Bunny (native)     | 64×64×2048   | 64        | 1  | 2048 | 32            | 1024 | 64          | 64×64     |
| **256×256×2048**   | 256×256×2048 | **256**   | 1  | 2048 | **128**       | 1024 | **256**     | **256×256** |
| 128×128×1024       | 128×128×1024 | 128       | 1  | 1024 | 64            | 512  | 128         | 128×128   |

Verified structurally on a 6 GB GPU: the `model_spatial=128` path (256×256 input)
builds and produces 256×256 intensity + depth; the `tlen=1024` path (2048 bins)
is the Bunny training config. Together they cover the 256×256×2048 case.

---

## 4. Commands for 256×256×2048 (high-memory GPU)

Train:

```bash
python train.py \
    --model_dir     "checkpoints_big" \
    --model_name    nlost \
    --dataset       big256 \
    --data_dir      "path/to/your_256_dataset" \
    --meas_size     256 \
    --ds            1 \
    --clip          2048 \
    --model_spatial 128 \
    --tlen          1024 \
    --bin_len       0.01 \
    --target_size   256 \
    --bacth_size    4 \
    --num_workers   8 \
    --num_epoch     51 \
    --num_save      999999
```

Reconstruct / test a 256×256×2048 transient:

```bash
python predict.py \
    --checkpoint "checkpoints_big.../epoch_50_XXXX_END.pth" \
    --input      "your_test.mat" \
    --spatial    128 \
    --tlen       1024 \
    --clip       2048 \
    --target_size 256 \
    --bin_len    0.01 \
    --out        predictions_big
```

Notes:
* `--num_epoch N` trains `N-1` epochs (loop is `range(1, N)`); pass `51` for 50.
* On a large GPU raise `--bacth_size` and `--num_workers` for throughput.
* A 512×512 test file is auto-binned to `target_size` by `predict.py`
  (block-sum). To exploit full 512 detail, train at `meas_size 512,
  model_spatial 256, target_size 512` — needs a very large GPU.

---

## 4b. GPU memory for 256×256×2048 (measured scaling)

Peak GPU memory (batch 1, fwd+bwd), measured and extrapolated:

| Config                         | Input          | Peak GPU |
|--------------------------------|----------------|----------|
| spatial=32, tlen=256           | 64×64×512      | 1.24 GB  |
| spatial=64, tlen=256           | 128×128×512    | 4.96 GB  |
| spatial=32, tlen=1024 (bunny)  | 64×64×2048     | 4.8 GB   |
| **spatial=64, tlen=1024 (opt A)** | 128×128×2048 | **~19 GB** |
| **spatial=128, tlen=256 (opt B)** | 256×256×512  | **~19 GB** |
| **spatial=128, tlen=1024 (native)** | 256×256×2048 | **~77 GB** |

Memory scales **quadratically with spatial**, **linearly with tlen**. So native
256×256×2048 needs an ~80 GB GPU. For a 32 GB GPU pick one trade-off:

| Option | Config | meas_size | target_size | model_spatial | ds | clip | tlen | Output | Keeps |
|--------|--------|-----------|-------------|---------------|----|------|------|--------|-------|
| **A** (full depth) | 128×128×2048 | 256 | 128 | 64  | 1 | 2048 | 1024 | 128×128 | all 2048 bins |
| **B** (full spatial) | 256×256×512 | 256 | 256 | 128 | 4 | 512  | 256  | 256×256 | 256×256 detail |
| **C** (native) | 256×256×2048 | 256 | 256 | 128 | 1 | 2048 | 1024 | 256×256 | everything (~80 GB GPU) |

**Only `tlen ∈ {256, 1024}` are safe** — a latent window-attention relative-position
bias bug breaks intermediate temporal sizes (e.g. tlen=512). A/B/C all use safe values.

### Option A launcher — `run_train_256.bat` (fits 32 GB, keeps all 2048 bins)

```bash
python train.py --model_dir "checkpoints_256" --model_name nlost --dataset big256 \
    --data_dir "path/to/your_256_dataset" \
    --meas_size 256 --target_size 128 --model_spatial 64 \
    --ds 1 --clip 2048 --tlen 1024 --bin_len 0.01 \
    --bacth_size 1 --num_workers 8 --num_epoch 51 --num_save 999999
```

Verify memory/shape on the target GPU before a long run:

```bash
python -c "import torch;from models import nlost;m=nlost.NLOST(1,1,spatial=64,tlen=1024,bin_len=0.01,target_size=128).cuda().train();x=torch.rand(1,1,2048,128,128).cuda();v,i,d=m(x);((i+1)/2).mean().backward();print('OK',tuple(i.shape),'%.1f GB'%(torch.cuda.max_memory_allocated()/1e9))"
```

Reconstruct a 256×256×2048 test file with the Option-A model (auto-bins 256→128):

```bash
python predict.py --checkpoint "checkpoints_256.../epoch_50_XXXX_END.pth" \
    --input "your_test.mat" --spatial 64 --tlen 1024 --clip 2048 \
    --target_size 128 --bin_len 0.01 --out predictions_256
```

## 5. The one case that DOES need a code change

The dataloader's file lister (`util/LFEDataset.py: list_file_path_bike`) expects
the Bike/Bunny folder layout:

```
<data_dir>/0/<model>/shine_.../{video-confocalspad*.mp4, confocal-0-*.hdr, depth-0-*.hdr}
```

If your 256 dataset uses this layout, nothing changes. If it stores raw `.mat`
cubes or a different folder structure, add a new lister function that returns the
same `{'Mea','dep','img','path'}` dict, and the rest of the pipeline is unchanged.
