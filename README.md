# An Analytical Theory of Creativity in Discrete Diffusion Models

University of Amsterdam · Deep Learning 2 group research project.

## Research question

How can discrete diffusion models generate new combinations of familiar local
patterns, rather than simply reproduce their training examples? We investigate
how locality, equivariance, and receptive-field size influence this behaviour
in convolutional neural networks (CNNs) and graph neural networks (GNNs).

## Approach

The project extends the locality-equivariance account of diffusion-model
creativity to discrete state spaces and graph-structured data, using an
Equivariant-Local Discrete Posterior (ELDP) as the analytical framework.
The masked experiments implement a discrete equivariant-local score (ELS)
machine for comparison with learned denoisers.

The main toy task uses two binary training images: one entirely black and one
entirely white. We vary network depth and compare 400-step and 3-step sampling,
measuring local majority-rule violations in generated images. Additional
experiments consider joint node-and-edge generation and binary MNIST.

## Experiments

- **[Uniform diffusion](uniform_diffusion/README.md):** D3PM-uniform corruption,
  CNN and GNN depth sweeps on a 24×24 image grid, and joint diffusion of binary
  node features and edges on five-node graphs.
- **[Masked diffusion](masked_diffusion/README.md):** absorbing-mask corruption,
  CNN and GNN sweeps on the binary image task, alternative neighbourhood
  consistency metrics, and CNN–ELS sample comparisons on binary MNIST using r².

## Findings reported in the experiment documentation

- The uniform CNN sweep reports collapse at depth L=8, linking the loss of
  diverse local combinations to increasing receptive-field size.
- The uniform grid GNN sweep reports greater resistance to collapse than the
  CNN, interpreted in terms of its diamond-shaped receptive field.

These are qualitative findings reported in the existing
[uniform experiment README](uniform_diffusion/README.md). The notebooks in this
repository have no saved outputs; numerical results are not reproduced here.

## Repository guide

| Location | Contents |
| --- | --- |
| `uniform_diffusion/toy_experiment_*.ipynb` | Main uniform CNN, grid GNN, and five-node graph experiments |
| `uniform_diffusion/models.py`, `uniform_diffusion/utils/` | Model and supporting utilities used by the notebooks |
| `masked_diffusion/colab_scripts/` | Colab training sweeps and evaluation scripts |
| `masked_diffusion/convolutional_diffusion/` | Modular masked-diffusion models, ELS implementation, and command-line scripts |

See the experiment READMEs linked above for setup and execution details.
Each experiment folder has its own `requirements.txt`. The masked README also
notes missing utilities required by the modular command-line scripts.

The research report is not included in this repository.
