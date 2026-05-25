import torch
from torch import optim, nn
from models import MinimalUNet, MinimalResNet, DDIM
from utils.noise_schedules import cosine_noise_schedule
import argparse
from utils.data import get_dataset
from torch.utils.data import DataLoader
from utils.train import train_diffusion

def main():

	parser = argparse.ArgumentParser(description='DDIM training')
	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--batchsize', type=int, default=128)
	parser.add_argument('--dataset', type=str)
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--conditional', action="store_true", default=False)
	parser.add_argument('--mode', type=str, default='circular')
	parser.add_argument('--wd', type=float, default=0)
	parser.add_argument('--mult', type=int, default=2)
	parser.add_argument('--nonorm', action="store_true", default=True)
	parser.add_argument('--saveinterval', type=int, default=5)
	parser.add_argument('--layers', type=int, default=3)
	parser.add_argument('--resnet', action="store_true", default=False)
	parser.add_argument('--homedir', type=str, default='./model_checkpoints')
	parser.add_argument('--suppress', action="store_true", default=False)
	parser.add_argument('--gamma', type=float, default=0.999965)
	parser.add_argument('--maxsamps', type=int, default=100000)

	args = parser.parse_args()

	dataset, metadata = get_dataset(args.dataset, root='./data')

	subset_flag = args.maxsamps < len(dataset)
	factor = 1
	if subset_flag:
		factor = len(dataset)//args.maxsamps
		dataset = torch.utils.data.Subset(dataset, [i for i in range(args.maxsamps)])

	train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

	if args.resnet:
		fname = args.homedir + '/MinimalResNet_'
	else:
		fname = args.homedir + '/MinimalUNet_'

	fname += metadata['name'] + f'_{args.mode}_lr_' + str(args.lr) + '_batchsize_' + str(args.batchsize) + '_wd_' + str(args.wd)

	if subset_flag:
		fname += '_maxsamps_' + str(args.maxsamps)

	if args.conditional:
		fname += '_conditional'
	if args.nonorm:
		fname += '_nonorm'
	if args.mult != 1:
		fname += '_mult_' + str(args.mult)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	normal = None if args.nonorm else 'GroupNorm'

	if args.resnet:
		backbone = MinimalResNet(channels=metadata['num_channels'],
								emb_dim=128*args.mult,
								mode=args.mode,
								conditional=args.conditional,
								num_classes=metadata['num_classes'],
								kernel_size=3,
								num_layers=args.layers,
								normalization=normal,
								lastksize=3)
	else: # UNET
		backbone = MinimalUNet(channels=metadata['num_channels'],
								fsizes=[args.mult*32*(2**i) for i in range(args.layers)],
								mode=args.mode,
								conditional=args.conditional,
								num_classes=metadata['num_classes'],
								normalization=normal,
								lastksize=3)

	model = DDIM(pretrained_backbone=backbone,
				default_imsize=metadata['image_size'],
				in_channels=metadata['num_channels'],
				noise_schedule=cosine_noise_schedule)

	model.to(device)
	

	train_diffusion(model, train_loader, cosine_noise_schedule, device,
					max_t=1000,
					num_epochs=args.epochs*factor,
					lr=args.lr,
					in_channels=metadata['num_channels'],
					gamma=args.gamma,
					fname=fname,
					conditional=args.conditional,
					save_interval=args.saveinterval*factor,
					wd=args.wd)



if __name__ == "__main__":
	main()
