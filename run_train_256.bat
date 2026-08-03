@echo off
REM ============================================================================
REM Train NLOST on a 256x256x2048 dataset  --  Option A (fits a 32 GB GPU)
REM   Keeps ALL 2048 time bins (full depth resolution); bins spatial 256 -> 128.
REM     --meas_size 256, --target_size 128  : loader bins 256 -> 128 (2x2 sum)
REM     --ds 1, --clip 2048                 : keep every time bin
REM     --model_spatial 64, --tlen 1024     : bike spatial + bunny temporal (proven)
REM   Output images: 128x128.  Peak GPU memory ~19 GB at batch size 1.
REM
REM   >>> EDIT --data_dir to point at your 256x256x2048 dataset (bike/bunny layout).
REM   NOTE: for native 256x256 OUTPUT keeping all 2048 bins you need ~77 GB (80 GB GPU);
REM         see TRAIN_ANY_SIZE.md option C.
REM ============================================================================
python train.py ^
    --model_dir     "checkpoints_256" ^
    --model_name    nlost ^
    --dataset       big256 ^
    --data_dir      "path\to\your_256_dataset" ^
    --meas_size     256 ^
    --target_size   128 ^
    --model_spatial 64 ^
    --ds            1 ^
    --clip          2048 ^
    --tlen          1024 ^
    --bin_len       0.01 ^
    --bacth_size    1 ^
    --num_workers   8 ^
    --num_epoch     51 ^
    --num_save      999999
pause
