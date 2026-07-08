@echo off
d:\NLOST\nlost_env\Scripts\python.exe d:/NLOST/train.py ^
    --model_dir     "D:\NLOST\checkpoints" ^
    --model_name    nlost ^
    --data_dir      "D:\NLOST\bike" ^
    --bacth_size    1 ^
    --target_size   128 ^
    --num_workers   0 ^
    --num_epoch     5 ^
    --num_save      999999
pause
