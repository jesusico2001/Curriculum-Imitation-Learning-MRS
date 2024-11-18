policy=$1

python3 evaluation/loss_evo/compareLosses.py \
    --trainLosses evaluation/loss_evo/"$policy"4LEMURS_3_20000_20000_2000_5_42_42_fixed_0.0_fixed_0.0/trainLosses.pth \
    evaluation/loss_evo/"$policy"4LEMURS_3_20000_20000_2000_50_42_42_fixed_800.0_fixed_1.0/trainLosses.pth \
    --descriptions NOCL CL