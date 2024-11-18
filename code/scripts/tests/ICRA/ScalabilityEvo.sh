# Eval
python3 evaluation/loss_evo/compareLosses.py \
    --trainLosses evaluation/loss_evo/FS4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    --descriptions "FS"

python3 evaluation/loss_evo/compareLosses.py \
    --trainLosses evaluation/loss_evo/TVS4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    --descriptions "TVS"

python3 evaluation/loss_evo/compareLosses.py \
    --trainLosses evaluation/loss_evo/Flocking4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    --descriptions "FLocking"

python3 evaluation/loss_evo/compareLosses.py \
    --trainLosses evaluation/loss_evo/FS4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    evaluation/loss_evo/FS4LEMURS_3_20000_20000_2000_50_50_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    evaluation/loss_evo/TVS4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    evaluation/loss_evo/Flocking4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    --descriptions "FS_Old" "FS_New" "TVS" "Flocking"

python3 evaluation/loss_evo/compareLosses.py \
    --trainLosses evaluation/loss_evo/FS4LEMURS_3_20000_20000_2000_50_50_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    evaluation/loss_evo/TVS4LEMURS_3_40000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    --descriptions "FS_New" "TVS"