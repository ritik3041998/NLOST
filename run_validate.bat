@echo off
d:\NLOST\nlost_env\Scripts\python.exe d:/NLOST/validate_custom.py ^
    --data_dir    "D:\NLOST\data" ^
    --checkpoint  "D:\NLOST\checkpoints\run5_fixed\iter100.pth" ^
    --output_dir  "D:\NLOST\output_run5_fixed" ^
    --spatial     64 ^
    --tlen        256 ^
    --bin_len     0.0192 ^
    --target_size 64 ^
    --split       val ^
    --no_amp
pause
