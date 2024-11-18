# bash scripts/trainBabySteps.sh FS GNN 3 40000 50 4 fixed 800 fixed 1
# bash scripts/trainBabySteps.sh FS MLP 3 40000 50 4 fixed 800 fixed 1
# bash scripts/evalTraining.sh FS GNN 3 40000 50 4 fixed 800 fixed 1
# bash scripts/evalTraining.sh FS MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/trainBabySteps.sh TVS GNN 3 40000 50 4 fixed 800 fixed 1
bash scripts/trainBabySteps.sh TVS MLP 3 40000 50 4 fixed 800 fixed 1
bash scripts/evalTraining.sh TVS GNN 3 40000 50 4 fixed 800 fixed 1
bash scripts/evalTraining.sh TVS MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/trainBabySteps.sh Flocking GNN 3 40000 50 4 fixed 800 fixed 1
bash scripts/trainBabySteps.sh Flocking MLP 3 40000 50 4 fixed 800 fixed 1
bash scripts/evalTraining.sh Flocking GNN 3 40000 50 4 fixed 800 fixed 1
bash scripts/evalTraining.sh Flocking MLP 3 40000 50 4 fixed 800 fixed 1
