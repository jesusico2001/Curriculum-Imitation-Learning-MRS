bash scripts/trainModel.sh FS GNN 3 40000 5 4 fixed 0 fixed 0
bash scripts/trainModel.sh FS GNN 3 40000 50 4 fixed 800 fixed 1

bash scripts/trainModel.sh TVS GNN 3 40000 5 4 fixed 0 fixed 0
bash scripts/trainModel.sh TVS GNN 3 40000 50 4 fixed 800 fixed 1

bash scripts/trainModel.sh Flocking GNN 3 40000 5 4 fixed 0 fixed 0
bash scripts/trainModel.sh Flocking GNN 3 40000 50 4 fixed 800 fixed 1

bash scripts/evalTraining.sh FS GNN 3 40000 5 4 fixed 0 fixed 0
bash scripts/evalTraining.sh FS GNN 3 40000 50 4 fixed 800 fixed 1

bash scripts/evalTraining.sh TVS GNN 3 40000 5 4 fixed 0 fixed 0
bash scripts/evalTraining.sh TVS GNN 3 40000 50 4 fixed 800 fixed 1

bash scripts/evalTraining.sh Flocking GNN 3 40000 5 4 fixed 0 fixed 0
bash scripts/evalTraining.sh Flocking GNN 3 40000 50 4 fixed 800 fixed 1

