# An Analytical Theory of Creativity in Discrete Diffusion Models — Uniform Diffusion Experiments

This repository contains the code for the **D3PM-uniform diffusion experiments** presented in our NeurIPS 2026 submission.

## Project Description

We extend the locality-equivariance theory of creativity in diffusion models (Kamb & Ganguli, 2025) to discrete state spaces and graph-structured data. We derive an analytical Equivariant-Local Discrete Posterior (ELDP) and validate it experimentally using CNNs and GNNs on a locally consistent binary image task, under D3PM-uniform diffusion. For the masked diffusion experiments, see the companion repository.

## Repository Structure

```
.
├── models.py                        # CNN (MinimalResNet) architecture
├── utils/
│   ├── data.py                      # Dataset utilities
│   ├── idealscore.py                # Ideal score computation
│   ├── noise_schedules.py           # Noise schedule definitions
│   └── train.py                     # Training utilities
├── toy_experiment_discrete.ipynb    # CNN sweep under D3PM-uniform diffusion
├── toy_experiment_graph.ipynb       # GNN sweep on image grid (uniform diffusion)
├── toy_experiment_graph5.ipynb      # 5-node joint node+edge diffusion experiment
├── requirements.txt
└── README.md
```

## Experiments

### CNN sweep under D3PM-uniform diffusion
`toy_experiment_discrete.ipynb` trains CNN models with varying depth L on a 24x24 toroidal binary image task and evaluates local consistency violations under valid sampling (T=400) and invalid sampling (T=3). Demonstrates the collapse phenomenon at L=8.

### GNN sweep on image grid
`toy_experiment_graph.ipynb` repeats the same task with a GNN on a 4-connected pixel grid, sweeping L across a wider range. Demonstrates that the GNN is more resistant to collapse than the CNN due to its diamond-shaped receptive field.

### 5-node joint node and edge diffusion
`toy_experiment_graph5.ipynb` trains a GNN on two 5-node graphs (a cycle with all-white nodes and a line with all-black nodes), simultaneously denoising binary node features and binary edge existence with D3PM-uniform. Demonstrates over-smoothing at L=4.

## Requirements

Python 3.10 or higher is recommended.

```bash
pip install -r requirements.txt
```

## Running the Experiments

Each experiment is self-contained in its notebook. Open the relevant notebook and run all cells in order. Training runs on CPU or GPU automatically. GPU is recommended for the full sweeps.

```bash
jupyter notebook
```

Results are logged to CSV files and sample images are saved automatically.

## Notes

- The CNN uses circular (toroidal) padding to match the periodic boundary conditions of the dataset.
- The GNN uses a 4-connected grid graph (up, down, left, right) with periodic boundary conditions.
- Local consistency violations are computed using a 3x3 majority-vote criterion.
- All experiments use embedding dimension 256 and a cosine noise schedule.
