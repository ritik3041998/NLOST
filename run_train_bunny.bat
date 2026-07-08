@echo off
REM Train NLOST on the Bunny dataset at NATIVE full resolution 64x64x2048.
REM   - meas_size 64 : native spatial grid (no spatial binning)
REM   - ds 1, clip 2048 : keep ALL 2048 temporal bins (no temporal binning)
REM   - model_spatial 32, target_size 64 : FK/model run natively on the 64-grid
REM   - tlen 1024    : FK crop (power of 2) = 2048/2 after sig_expand; transformer temporal = 1024/16 = 64
REM Peak GPU memory ~4.8 GB at batch size 1 (fits a 6 GB GPU).
d:\NLOST\nlost_env\Scripts\python.exe d:/NLOST/train.py ^
    --model_dir     "D:\NLOST\checkpoints_bunny" ^
    --model_name    nlost ^
    --dataset       bunny ^
    --data_dir      "D:\NLOST\bunny" ^
    --meas_size     64 ^
    --ds            1 ^
    --clip          2048 ^
    --model_spatial 32 ^
    --tlen          1024 ^
    --bin_len       0.01 ^
    --target_size   64 ^
    --bacth_size    1 ^
    --num_workers   0 ^
    --num_epoch     6 ^
    --num_save      999999
pause
