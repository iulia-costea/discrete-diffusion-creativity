import argparse
import torch
from torch.utils.data import DataLoader

from models import MinimalResNet, MinimalUNet
from utils.data import get_dataset
from utils.discrete_noise import cosine_alpha_bar
from utils.discrete_train import train_discrete_diffusion

def main():
    parser = argparse.ArgumentParser(description='Discrete absorbing diffusion training')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--conditional', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='zeros',
                        help='Padding mode: zeros (bbELS) or circular (ELS)')
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--mult', type=int, default=2)
    parser.add_argument('--layers', type=int, default=8)
    parser.add_argument('--resnet', action='store_true', default=False)
    parser.add_argument('--homedir', type=str, default='./checkpoints_discrete')
    parser.add_argument('--gamma', type=float, default=0.999965)
    parser.add_argument('--saveinterval', type=int, default=5)
    parser.add_argument('--maxsamps', type=int, default=100000)
    args = parser.parse_args()

    dataset, metadata = get_dataset(args.dataset, root='./data', binary=True)
    if args.maxsamps < len(dataset):
        dataset = torch.utils.data.Subset(
            dataset, list(range(args.maxsamps))
        )
    train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.resnet:
        backbone = MinimalResNet(
            channels=metadata['num_channels'],
            emb_dim=128 * args.mult,
            mode=args.mode,
            conditional=args.conditional,
            num_classes=metadata.get('num_classes'),
            kernel_size=3,
            num_layers=args.layers,
            normalization=None,
            lastksize=3,
        )
        tag = 'DiscreteResNet'
    else:
        backbone = MinimalUNet(
            channels=metadata['num_channels'],
            fsizes=[args.mult * 32 * (2 ** i) for i in range(args.layers)],
            mode=args.mode,
            conditional=args.conditional,
            num_classes=metadata.get('num_classes'),
            normalization=None,
            lastksize=3,
        )
        tag = 'DiscreteUNet'

    import os
    os.makedirs(args.homedir, exist_ok=True)
    fname = os.path.join(
        args.homedir,
        f'{tag}_{metadata["name"]}_{args.mode}_lr{args.lr}_bs{args.batchsize}'
        + ('_conditional' if args.conditional else '')
    )

    backbone = backbone.to(device)

    train_discrete_diffusion(
        backbone,
        train_loader,
        device,
        schedule=cosine_alpha_bar,
        num_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        wd=args.wd,
        in_channels=metadata['num_channels'],
        conditional=args.conditional,
        fname=fname,
        save_interval=args.saveinterval,
    )

if __name__ == '__main__':
    main()
