import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from utils.noise_schedules import cosine_noise_schedule

class DDIM(nn.Module):

	def __init__(self, backbone=None,
				pretrained_backbone=None,
				in_channels=3,
				noise_schedule=cosine_noise_schedule,
				default_imsize=32):

		super().__init__()
		self.in_channels = in_channels
		self.default_imsize = default_imsize
		if pretrained_backbone is not None:
			self.backbone = pretrained_backbone
		else:
			self.backbone = backbone(in_channels=in_channels, out_channels=in_channels, imsize=default_imsize)
		self.noise_schedule = noise_schedule

	def forward(self, t, x, label=None):
		return self.backbone(t,x,label=label)

	def sample(self, batch_size=1, x=None, nsteps=20, label=None, device=None, breakstep=-1, ddpm=False):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.eval()
		self.backbone.eval()
		if x is None:
			x = torch.normal(0,1,(batch_size,self.in_channels,self.default_imsize,self.default_imsize))

		if ddpm:
			for i in range(nsteps, 0, -1):
				if i == breakstep:
					return x

				t = i*torch.ones(batch_size, device=device)/nsteps
				beta_t = self.noise_schedule(t)  # Determine the noise level for the current step
				score = self(t, x, label=label)

				alpha_t = 1 - beta_t
				alpha_t_prev = 1 - self.noise_schedule(t - 1/nsteps)
				beta_t_prev = 1 - alpha_t_prev
				a_t = alpha_t/alpha_t_prev
				sigma_t = torch.sqrt(beta_t_prev/beta_t)*torch.sqrt(1 - alpha_t/alpha_t_prev)
				sigma_t = sigma_t[:,None,None,None]

				x = torch.sqrt(alpha_t_prev[:,None,None,None])*(x - torch.sqrt(beta_t[:,None,None,None])*score)/torch.sqrt(alpha_t[:,None,None,None]) + torch.sqrt(1 - alpha_t_prev[:,None,None,None] - sigma_t**2)*score + sigma_t*torch.randn_like(x)
				
		else:

			for i in range(nsteps, 0, -1):

				if i == breakstep:
					return x

				t = i*torch.ones(batch_size, device=device)/nsteps
				beta_t = self.noise_schedule(t)  # Determine the noise level for the current step
				score = self(t,x,label=label)
				
				alpha_t = 1 - beta_t
				beta_t_prev = self.noise_schedule(t - 1/nsteps)
				alpha_t_prev = 1 - beta_t_prev

				x *= torch.sqrt(alpha_t_prev/alpha_t)[:,None,None,None]
				score_correction = (torch.sqrt(beta_t_prev[:,None,None,None])-torch.sqrt(alpha_t_prev/alpha_t)[:,None,None,None]*torch.sqrt(beta_t[:,None,None,None]))*score
				x += score_correction

		return x


class EmbeddingModule(nn.Module):

	def __init__(self, fdim, channels, conditional=False, num_classes=None):
		super().__init__()
		self.fdim = fdim
		self.conditional = conditional
		
		if conditional:
			self.class_embeddings = nn.Embedding(num_classes, fdim)
		self.channels = channels

	def forward(self,t,label=None):

		d = self.fdim//2
		targ = t[:,None]/(10000**(torch.arange(d, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))/(d-1)))[None,:]
		emb = torch.cat((torch.sin(targ), torch.cos(targ)),dim=1)

		if self.conditional:
			emb += self.class_embeddings(label)

		return emb


class MinimalResNet(nn.Module):

	def __init__(self, channels=3,
						emb_dim=128,
						mode='circular',
						normalization=None,
						conditional=False,
						num_classes=None,
						kernel_size=3,
						num_layers=6,
						lastksize=1,
						add_one=True):

		super().__init__()
		self.channels = channels
		self.emb_dim = emb_dim
		self.mode = mode
		self.conditional = conditional
		self.num_layers = num_layers
		self.num_classes = num_classes
		self.normalization = normalization
		self.lastksize = lastksize

		self.embedding = EmbeddingModule(emb_dim, channels, conditional=conditional, num_classes=num_classes)

		self.up_projection = nn.Conv2d(channels, emb_dim, kernel_size, padding='same', padding_mode=mode)

		if add_one:
			self.embs = nn.ModuleList([nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GroupNorm(8,emb_dim), nn.ReLU()) for i in range(num_layers+1)])
		else:
			self.embs = nn.ModuleList([nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GroupNorm(8,emb_dim), nn.ReLU()) for i in range(num_layers)])


		if normalization is None:
			self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size, padding='same', padding_mode=mode), nn.ReLU()) for i in range(num_layers)])
		else:
			self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size, padding='same', padding_mode=mode), nn.GroupNorm(8,emb_dim), nn.ReLU()) for i in range(num_layers)])

		if normalization is None:
			self.down_projection = nn.Conv2d(emb_dim, channels, lastksize, padding='same', padding_mode=mode)
		else:
			self.down_projection = nn.Sequential(nn.GroupNorm(8,emb_dim), nn.Conv2d(emb_dim, channels, lastksize, padding='same', padding_mode=mode))

	def forward(self, t, x, label=None):

		embedding_vec = self.embedding(t,label=label)
		state = self.up_projection(x)

		for i in range(self.num_layers):
			delta = self.convs[i](state + self.embs[i](embedding_vec)[:,:,None,None])
			state = state + delta

		delta = self.embs[-1](embedding_vec)[:,:,None,None] if len(self.embs) > self.num_layers else state
		nextstate = state + delta

		return self.down_projection(nextstate)

class MinimalUNet(nn.Module):


	def __init__(self, channels=3,
						fsizes=[32, 64, 128, 256],
						mode='circular',
						conditional=False,
						num_classes=None,
						emb_dim=256,
						normalization=None,
						last_norm=False,
						kernel_size=3,
						lastksize=1):

		super().__init__()

		self.fsizes = fsizes
		self.channels = channels
		self.conditional = conditional
		self.emb_dim = emb_dim
		self.kernel_size = kernel_size
		self.lastksize = lastksize

		self.embedding = EmbeddingModule(emb_dim, channels, conditional=conditional, num_classes=num_classes)

		in_channels = channels
		self.feature_blocks = nn.ModuleList()
		for f in fsizes[:-1]:
			self.feature_blocks.append(UBlock(in_channels, f, normalization=normalization, kernel_size=kernel_size, padding_mode=mode, emb_dim=emb_dim))
			in_channels = f

		self.bottleneck = UBlock(fsizes[-2], fsizes[-1], normalization=normalization, kernel_size=kernel_size, padding_mode=mode, emb_dim=emb_dim)
		self.output_blocks = nn.ModuleList()
		self.upsamples = nn.ModuleList()
		for i in range(len(fsizes)-1,0,-1):
			self.upsamples.append(nn.ConvTranspose2d(fsizes[i], fsizes[i-1], kernel_size=2, stride=2))
			self.output_blocks.append(UBlock(2*fsizes[i-1], fsizes[i-1], normalization=normalization, padding_mode=mode, emb_dim=emb_dim))

		self.last_emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, fsizes[0]))
		self.output_conv = nn.Conv2d(fsizes[0], self.channels, kernel_size=lastksize, padding='same', padding_mode=mode)

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

		self.last_norm = last_norm
		if last_norm:
			if normalization == 'GroupNorm':
				self.last_normalizer = nn.GroupNorm(min(32,fsizes[0]),fsizes[0]) 
			elif normalization == 'BatchNorm':
				self.last_normalizer = nn.BatchNorm2d(fsizes[0])

	def forward(self, t, x, label=None):

		embedding_vec = self.embedding(t,label=label)

		skip_connections = []

		for down in self.feature_blocks:
			x = down(x,embedding_vec)
			skip_connections.append(x)
			x = self.pool(x)
			
		x = self.bottleneck(x,embedding_vec)

		skip_connections = skip_connections[::-1]

		for i in range(len(self.upsamples)):
			upconv = self.upsamples[i](x)
			skip = skip_connections[i]
			x = torch.cat((skip, upconv),dim=1)
			x = self.output_blocks[i](x,embedding_vec)

		try:
			if self.last_norm:
				return self.output_conv(self.last_normalizer(x+self.last_emb(embedding_vec)[:,:,None,None]))
			else:
				return self.output_conv(x+self.last_emb(embedding_vec)[:,:,None,None])	
		except:
			return self.output_conv(x+self.last_emb(embedding_vec)[:,:,None,None])


class UBlock(nn.Module):

	def __init__(self, infeatures, outfeatures,
						depth=2,
						kernel_size=3,
						normalization=None,
						padding_mode='circular',
						emb_dim=32):

		super().__init__()

		self.emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, infeatures))

		module_list = []
		for i in range(depth):
			if i == 0:
				x = infeatures
			else:
				x = outfeatures

			module_list.append(nn.Conv2d(x, outfeatures, kernel_size=kernel_size, padding='same', padding_mode=padding_mode))
			if normalization == 'GroupNorm':
				module_list.append(nn.GroupNorm(min(32,outfeatures),outfeatures))
			elif normalization == 'BatchNorm':
				module_list.append(nn.BatchNorm2d(outfeatures))
			module_list.append(nn.ReLU())
		
		self.model = nn.Sequential(*module_list)

	def forward(self, x, embedding):
		return self.model(x+self.emb(embedding)[:,:,None,None])
