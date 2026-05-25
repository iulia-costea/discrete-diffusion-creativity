import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.discrete_noise import cosine_alpha_bar, MASK_VALUE, full_reverse
from utils.data import get_dataset

def main():
    parser = argparse.ArgumentParser(description='Discrete eval')
    parser.add_argument('--exp_fname', type=str, default='./')
    parser.add_argument('--model_fname', type=str, required=True)
    parser.add_argument('--outputname', type=str, default='els_outputs/')
    parser.add_argument('--dsname', type=str, default='mnist')
    parser.add_argument('--conditional', action='store_true', default=False)
    parser.add_argument('--nsteps', type=int, default=20)
    parser.add_argument('--figname', type=str, default='scatter_discrete.png')
    parser.add_argument('--title', type=str, default='Discrete ELS vs CNN')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model_fname, map_location=device)
    model.eval()

    _, metadata = get_dataset(args.dsname)

    SEEDPATH = os.path.join(args.exp_fname, 'seeds/')
    OUTPATH = os.path.join(args.exp_fname, args.outputname)
    LPATH = os.path.join(args.exp_fname, 'labels/')

    els_corrs = []
    n = 0
    while os.path.exists(f'{SEEDPATH}{n:04d}.pt'):
        seed = torch.load(f'{SEEDPATH}{n:04d}.pt', map_location=device)
        label = None
        if args.conditional:
            label = torch.load(f'{LPATH}{n:04d}.pt', map_location=device)

                                                         
        cnn_out = full_reverse(
            model,
            nsteps=args.nsteps,
            batch_size=1,
            in_channels=metadata['num_channels'],
            image_size=metadata['image_size'],
            schedule=cosine_alpha_bar,
            stochastic=False,
            device=device,
            label=label,
        )

                             
        els_out = torch.load(f'{OUTPATH}{n:04d}.pt', map_location=device)

                                                                       
        def norm(x):
            x = x.float().flatten()
            x = x - x.mean()
            nrm = x.norm()
            return x / nrm if nrm > 1e-8 else x

        r2 = torch.dot(norm(cnn_out), norm(els_out)).item() ** 2
        els_corrs.append(r2)
        n += 1

    print(f'Samples evaluated: {n}')
    print(f'Median r²  CNN vs discrete ELS: {np.median(els_corrs):.4f}')

    fig, ax = plt.subplots()
    ax.hist(els_corrs, bins=20, range=(0, 1))
    ax.set_xlabel('r²  (CNN vs discrete ELS)')
    ax.set_ylabel('Count')
    ax.set_title(args.title)
    fig.savefig(f'scatter_{args.figname}', bbox_inches='tight')
    plt.show()
    print(f'Saved figure to scatter_{args.figname}')

if __name__ == '__main__':
    main()
