# Curriculum Imitation Learning of Distributed Multi-Robot Policies

This repository contains the implementation of a *Curriculum Learning* algorithm for distributed navigation policies in Multi-Robots Systems. This algorithm schedules the difficulty of training trajectories and dynamically fragments demonstrations to match the scheduled difficulty.

The project also includes a module for analytical ground truth data generation in 3 scenarios: *Fixed Swapping*, *Time-Varying Swapping* and *Flocking*. Finally, the evaluation setup contains various neural network architectures (MLP, GNN, LEMURS), on which the performance of the algorithm is evaluated.

## Dependencies

Our code is tested with Ubuntu 20.04 and Python 3.11.5. With the following dependencies: 

```torch 1.13.0``` 

```torchdiffeq 0.2.3```

```numpy 1.24.3```

```matplotlib 3.7.2```

```similaritymeasures 1.1.0```

## Usage

In this file we explain the more direct way to run the code, using shell scripts that abstract some of the pyton script parameters. A finer parametrization of every process can be done executing the equivalent .py files.

> [!IMPORTANT]
> Before running any commands, ensure that your **current working directory** in the shell is set to **`/code`**.

### Dataset generation

To generate the training and evaluation datasets, run
```bash
./scripts/generateDataset.sh policy numAgents numSamples
```

where


```policy (str)```: The scenario to generate data about.  ["FS", "TVS", "Flocking"].

```numAgents (int)```: The scenario to generate data about.  ["FS", "TVS", "Flocking"].

### Training

### Evaluation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
