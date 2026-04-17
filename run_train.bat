@echo off
d:\NLOST\nlost_env\Scripts\python.exe d:/NLOST/train_custom.py ^
    --data_dir      "D:\NLOST\data" ^
    --model_dir     "D:\NLOST\checkpoints\run5_fixed" ^
    --spatial       64 ^
    --tlen          256 ^
    --bin_len       0.0192 ^
    --target_size   64 ^
    --batch_size    1 ^
    --num_workers   0 ^
    --num_epoch     10 ^
    --int_weight    1.0 ^
    --vol_weight    0.5 ^
    --no_amp
pause
