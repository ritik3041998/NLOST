@echo off
REM ============================================================================
REM Train NLOST on a 256x256x512 dataset  (full 256x256 output, ~19 GB)
REM   No code changes -- arguments only. This is the 256-spatial config with the
REM   original 512-bin temporal depth (tlen=256 -> transformer temporal = 16).
REM     --meas_size 256, --target_size 256, --model_spatial 128 : native 256 grid
REM     --ds 1, --clip 512, --tlen 256                          : 512 time bins
REM   Peak GPU memory ~19 GB at batch size 1.
REM
REM   >>> EDIT --data_dir to your 256x256x512 dataset (bike/bunny folder layout).
REM   If you still hit CUDA OOM, switch to the lighter 128x128 output:
REM     --target_size 128  --model_spatial 64   (~5 GB)
REM ============================================================================
python train.py ^
    --model_dir     "checkpoints_256x512" ^
    --model_name    nlost ^
    --dataset       big256 ^
    --data_dir      "path\to\your_256_dataset" ^
    --meas_size     256 ^
    --target_size   256 ^
    --model_spatial 128 ^
    --ds            1 ^
    --clip          512 ^
    --tlen          256 ^
    --bin_len       0.01 ^
    --bacth_size    1 ^
    --num_workers   8 ^
    --num_epoch     51 ^
    --num_save      999999
pause
