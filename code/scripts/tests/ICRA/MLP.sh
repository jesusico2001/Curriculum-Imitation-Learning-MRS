bash scripts/trainModel.sh FS MLP 3 40000 5 4 fixed 0 fixed 0
bash scripts/trainModel.sh FS MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/trainModel.sh TVS MLP 3 40000 5 4 fixed 0 fixed 0
bash scripts/trainModel.sh TVS MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/trainModel.sh Flocking MLP 3 40000 5 4 fixed 0 fixed 0
bash scripts/trainModel.sh Flocking MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/evalTraining.sh FS MLP 3 40000 5 4 fixed 0 fixed 0
bash scripts/evalTraining.sh FS MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/evalTraining.sh TVS MLP 3 40000 5 4 fixed 0 fixed 0
bash scripts/evalTraining.sh TVS MLP 3 40000 50 4 fixed 800 fixed 1

bash scripts/evalTraining.sh Flocking MLP 3 40000 5 4 fixed 0 fixed 0
bash scripts/evalTraining.sh Flocking MLP 3 40000 50 4 fixed 800 fixed 1

