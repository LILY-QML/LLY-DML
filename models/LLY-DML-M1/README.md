# LLY-DML Model M1

This example model demonstrates a multi-matrix optimization use case for the LLY-DML quantum circuit implementation.

## Overview

The M1 model implements a quantum circuit with L-Gates structure and uses various gradient-based optimizers to train the circuit parameters for multiple input matrices. Each matrix is assigned a unique target quantum state, and the optimization process aims to maximize the probability of measuring the corresponding target state when the circuit is executed with the respective input matrix.

## Features

- Uses 5 qubits and circuit depth of 3
- Implements L-Gate structure (TP0 → IP0 → H → TP1 → IP1 → H → TP2 → IP2)
- Supports multiple optimizers:
  - Adam
  - SGD
  - Momentum
  - RMSprop
  - Adagrad
  - Nadam
- Loads matrices and state mappings from local configuration files
- Visualizes training progress and optimization results

## Directory Structure

```
M1/
├── README.md              # This file
├── example.py             # Main program file
├── logs/                  # Directory for log files
├── results/               # Directory for results and visualizations
└── var/                   # Configuration and data files
    ├── config.json        # Model configuration
    └── data.json          # Input matrices and state mappings
```

## Usage

To run the model:

```bash
cd /path/to/LLY-DML/models/M1
python example.py
```

The program will:
1. Load input matrices and state mappings from `var/data.json`
2. Create a quantum circuit with L-Gates structure
3. Run optimization for each matrix using multiple optimizers
4. Generate visualizations and save results in the `results/` directory

## Configuration

You can modify the model behavior by editing the configuration files:

- `var/config.json`: Change parameters like number of qubits, circuit depth, and training iterations
- `var/data.json`: Define input matrices and their target states

## Results

The model generates several visualization files in the `results/` directory:

- Training progress for each optimizer
- Comparison of optimizers for each matrix
- Overall comparison of all optimizers and matrices

It also saves the complete optimization results in a JSON file for further analysis.

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Qiskit
- LLY-DML src library

## Implementation Details

The model uses the quantum circuit implementation from the src directory, with the L-Gates structure and the optimizers framework. It supports a multi-matrix optimization scenario where different input matrices are assigned to different target quantum states.

The optimization process aims to find training parameters that, when combined with each input matrix, maximize the probability of measuring the corresponding target state from the quantum circuit.