# An Analytical Theory of Creativity in Discrete Diffusion Models — Masked Diffusion Experiments

This folder contains the code for the **masked diffusion experiments** from our University of Amsterdam Deep Learning 2 group research project.

## Project Description

We extend the locality-equivariance theory of creativity in diffusion models (Kamb & Ganguli, 2025) to discrete state spaces and graph-structured data. We derive an analytical Equivariant-Local Discrete Posterior (ELDP) and validate it experimentally using CNNs and GNNs on a locally consistent binary image task, under masked (absorbing) diffusion. For the D3PM-uniform diffusion experiments, see [../uniform_diffusion/README.md](../uniform_diffusion/README.md).

## Repository Structure

```
.
├── convolutional_diffusion/
│   ├── models.py                        # CNN (MinimalResNet) architecture
│   ├── models_gnn.py                    # GNN (MinimalResGNN) architecture
│   ├── discrete_training_script.py      # CLI training script
│   ├── discrete_els_script.py           # Discrete ELS generation
│   ├── discrete_eval_script.py          # r² evaluation
│   └── utils/
│       ├── discrete_noise.py            # Masked forward process and noise schedule
│       ├── discrete_train.py            # Training loop
│       ├── discrete_idealscore.py       # Discrete ELS machine
│       └── graph_utils.py              # Graph utilities for GNN experiments
├── colab_scripts/
│   ├── colab_discrete_diffusion.py      # CNN sweep + discrete ELS (toy dataset)
│   ├── colab_gnn_standalone.py          # GNN sweep, 3x3 violation metric
│   ├── colab_gnn_4conn_eval.py          # GNN re-evaluation, 4-connected metric
│   ├── colab_mnist_standalone.py        # MNIST training + discrete ELS + r²
│   └── colab_r2_eval.py                 # Standalone r² evaluation
├── requirements.txt
└── README.md
```

## Experiments

### CNN sweep under masked diffusion
`colab_discrete_diffusion.py` trains CNN models (L=3,6,8) on a 24x24 toroidal binary image task under masked diffusion and runs the discrete ELS machine. Evaluates local consistency violations under valid sampling (T=400) and invalid sampling (T=3).

### GNN sweep under masked diffusion
`colab_gnn_standalone.py` trains GNN models (L=1,2,4,8,10,12,16) on the same task. Violations are evaluated using the 3x3 neighbourhood metric to allow direct comparison with the CNN results. `colab_gnn_4conn_eval.py` re-evaluates the saved checkpoints using the 4-connected neighbourhood metric, which matches the GNN's theoretical receptive field and avoids tie-breaking ambiguity.

### MNIST + r² evaluation
`colab_mnist_standalone.py` trains a CNN (L=6, emb_dim=256) on binary MNIST under masked diffusion and computes r² between CNN outputs and the discrete ELS machine, analogous to the continuous result of Kamb & Ganguli.

## Requirements

Python 3.10 or higher is recommended.

```bash
pip install -r requirements.txt
```

## Running the Experiments

### Google Colab
Upload the relevant script from `colab_scripts/` and run:

```python
%run colab_discrete_diffusion.py
```

The scripts include their model definitions. Check the paths near the top before running: the GNN and standalone MNIST training scripts mount Google Drive, while `colab_discrete_diffusion.py` saves to `/content/checkpoints/`. Evaluation scripts require existing checkpoints; `colab_r2_eval.py` defaults to `/content/mnist_cnn_L6_final.pt`.

### Snellius
The commands below describe the original cluster workflow, run from `convolutional_diffusion/`. They currently require `utils/data.py` and `utils/noise_schedules.py`, which are absent from this folder. The `dl2025` environment is cluster-specific and no environment definition is included. Use the Colab scripts above as the starting point for this checkout.

```bash
conda activate dl2025

# Train
python discrete_training_script.py --dataset mnist --epochs 150 --layers 6 --resnet --homedir ./checkpoints

# Generate ELS samples
python discrete_els_script.py --expname my_experiment --dataset mnist --numiters 100 --nsteps 20

# Evaluate r² (replace model.pt with the actual saved model filename)
python discrete_eval_script.py --exp_fname ./experiments/my_experiment/ --model_fname ./checkpoints/model.pt --dsname mnist
```

For GPU jobs, submit via `sbatch` using your existing job script.

## Notes

- The toy CNN uses circular (toroidal) padding. The modular training script defaults to zero padding; `--mode circular` selects periodic boundaries.
- The GNN uses a 4-connected grid graph (up, down, left, right) without periodic boundary conditions.
- Local consistency violations are computed using a 3x3 majority-vote criterion for CNN-GNN comparison, and additionally using a 4-connected criterion for the GNN to match its theoretical receptive field.
- All experiments use embedding dimension 256 and a cosine noise schedule.
