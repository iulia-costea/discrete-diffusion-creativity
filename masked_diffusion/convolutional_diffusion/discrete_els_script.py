import argparse
import os
import shutil
import torch

from utils.discrete_noise import cosine_alpha_bar, MASK_VALUE
from utils.discrete_idealscore import (
    DiscreteIdealScoreModule,
    DiscreteLocalScoreModule,
    DiscreteLocalEquivScoreModule,
    DiscreteLocalEquivBordersScoreModule,
    ScheduledDiscreteScoreMachine,
)
from utils.data import get_dataset

def main():
    parser = argparse.ArgumentParser(description='Discrete ELS generation')
    parser.add_argument('--expname', type=str, default=None)
    parser.add_argument('--idealname', type=str, default='els_outputs')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--scoremoduletype', type=str, default='bbELS',
                        help='One of: bbELS, ELS, LS, IS')
    parser.add_argument('--conditional', action='store_true', default=False)
    parser.add_argument('--scalesfile', type=str, default=None)
    parser.add_argument('--scorebatchsize', type=int, default=256)
    parser.add_argument('--numiters', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=20)
    parser.add_argument('--nlabels', type=int, default=10)
    parser.add_argument('--kernelsize', type=int, default=3)
    parser.add_argument('--force_overwrite', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--max_samples', type=int, default=100000)
    args = parser.parse_args()

    dataset, metadata = get_dataset(args.dataset, root='./data', binary=True)
    in_channels = metadata['num_channels']
    image_size = metadata['image_size']

    device = torch.device('cpu' if args.cpu else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    expname = args.expname or f'discrete_{metadata["name"]}_{args.scoremoduletype}'
    if args.conditional:
        expname += '_conditional'

                            
    common = dict(
        dataset=dataset,
        batch_size=args.scorebatchsize,
        image_size=image_size,
        channels=in_channels,
        schedule=cosine_alpha_bar,
        max_samples=args.max_samples,
    )
    if args.scoremoduletype == 'ELS':
        mod = DiscreteLocalEquivScoreModule(kernel_size=args.kernelsize, **common)
    elif args.scoremoduletype == 'bbELS':
        mod = DiscreteLocalEquivBordersScoreModule(kernel_size=args.kernelsize, **common)
    elif args.scoremoduletype == 'LS':
        mod = DiscreteLocalScoreModule(kernel_size=args.kernelsize, **common)
    elif args.scoremoduletype == 'IS':
        mod = DiscreteIdealScoreModule(**common)
    else:
        raise ValueError(f'Unknown scoremoduletype: {args.scoremoduletype}')

    scales = None
    if args.scalesfile is not None:
        scales = list(torch.load(args.scalesfile).int().numpy())

    machine = ScheduledDiscreteScoreMachine(
        mod, in_channels=in_channels, imsize=image_size,
        schedule=cosine_alpha_bar, scales=scales, nsteps=args.nsteps,
    ).to(device)

    DPATH = f'./experiments/{expname}'
    SEEDPATH = f'{DPATH}/seeds/'
    SPATH = f'{DPATH}/{args.idealname}/'
    LPATH = f'{DPATH}/labels/'

    if os.path.isdir(DPATH) and not args.force_overwrite:
        print(f'Directory {DPATH} exists; resuming from existing seeds.')
    else:
        if os.path.isdir(DPATH):
            shutil.rmtree(DPATH)
        os.makedirs(SEEDPATH)
        os.makedirs(SPATH)
        if args.conditional:
            os.makedirs(LPATH)

    for i in range(args.numiters):
        seed_path = f'{SEEDPATH}{i:04d}.pt'
        out_path = f'{SPATH}{i:04d}.pt'

        if os.path.exists(seed_path) and os.path.exists(out_path):
            continue

                                    
        seed = torch.full((1, in_channels, image_size, image_size),
                          MASK_VALUE, device=device)
        label = torch.randint(0, args.nlabels, (1,)) if args.conditional else None

        output = machine(seed.clone(), device=device,
                         label=label.to(device) if label is not None else None)

        torch.save(seed.cpu(), seed_path)
        torch.save(output.cpu(), out_path)
        if args.conditional:
            torch.save(label, f'{LPATH}{i:04d}.pt')

        if (i + 1) % 10 == 0:
            print(f'  Generated {i + 1}/{args.numiters}')

if __name__ == '__main__':
    main()
