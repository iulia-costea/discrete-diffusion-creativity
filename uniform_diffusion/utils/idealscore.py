import torch
import numpy as np
from torch.nn import functional as F

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def denormalize(image, means, stds):
	if len(image.shape) == 3:
		return (image*torch.tensor(stds)[:,None,None]) + torch.tensor(means)[:,None,None]
	return (image*torch.tensor(stds)[None,:,None,None]) + torch.tensor(means)[None,:,None,None]

def denormalize_imshow(image, means, stds):
	image2 = denormalize(image, means, stds)
	if len(image.shape) == 4:
		plt.imshow(image2.detach().numpy()[0,:,:,:].transpose(1,2,0),cmap=cm.Greys_r)
	else:
		plt.imshow(image2.detach().numpy().transpose(1,2,0),cmap=cm.Greys_r)
	plt.axis('off')
	plt.show()

def circular_convolution_native(input_signal, kernel):
	pad_h = kernel.size(2) // 2
	pad_w = kernel.size(3) // 2
	
	input_padded = F.pad(input_signal, (pad_w, pad_w, pad_h, pad_h), mode='circular')
	
	result = F.conv2d(input_padded, kernel, padding=0)
	
	return result

def exponential_schedule(t):
	return 1 - torch.exp(-2*t)

def linear_noise_schedule(t):
	# returns beta
	return 0.01+0.97*t

def cosine_noise_schedule(t, mode='legacy'):
	# returns beta
	if mode == 'legacy':
		return 1-torch.cos((t) / 1.008 * math.pi / 2) ** 2
	return 1-torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2


class ScheduledScoreMachine(nn.Module):

	def __init__(self, backbone,
					in_channels=3,
					imsize=32,
					default_time_steps=20,
					noise_schedule=cosine_noise_schedule,
					score_backbone=True,
					scales=None,
					**kwargs):

		super().__init__()

		self.backbone = backbone
		self.default_time_steps = default_time_steps
		self.noise_schedule = noise_schedule
		self.in_channels = in_channels
		self.imsize = imsize
		self.score_backbone = score_backbone
		self.scales = scales

	def forward(self, x, nsteps=None, label=None, device=None, visualize=False):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		x = x.clone()

		if nsteps is None:
			if self.scales is None:
				nsteps = self.default_time_steps
			else:
				nsteps = len(self.scales)

		for i in range(nsteps-1, 0, -1):

			batch_size = x.shape[0]
			t = i*torch.ones(batch_size)/nsteps
			beta_t = self.noise_schedule(t)  # Determine the noise level for the current step
			beta_t = beta_t.to(device)

			k = None if self.scales is None else self.scales[i]
			if label is not None:
				eps = self.backbone(t,x,label=label,device=device,k=k)
			else:
				eps = self.backbone(t,x,device=device,k=k)

			if self.score_backbone:
				eps *= -beta_t**0.5

			if visualize:
				imputed = (x-eps*((beta_t)**0.5))/((1 - beta_t)**0.5)
				denormalize_imshow(imputed,[0.5 for q in range(x.shape[1])], [0.5 for q in range(x.shape[1])])
			
			alpha_t = 1 - beta_t
			beta_t_prev = self.noise_schedule(t - 1/nsteps)
			beta_t_prev = beta_t_prev.to(device)
			alpha_t_prev = 1 - beta_t_prev


			x *= ((alpha_t_prev/alpha_t)**0.5)[:,None,None,None]
			score_correction = ((beta_t_prev[:,None,None,None]**0.5)-((alpha_t_prev/alpha_t)**0.5)[:,None,None,None]*(beta_t[:,None,None,None]**0.5))*eps
			x += score_correction

		return x

	def sample(self, nsteps=None, label=None, device=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		x = torch.randn(1,self.in_channels,self.imsize,self.imsize, device=device)
		return self(x.clone(), nsteps=nsteps, label=label, device=device)


class LocalEquivBordersScoreModule(nn.Module):

	def __init__(self, dataset,
				kernel_size=3,
				batch_size=64,
				image_size=32,
				channels=3,
				schedule=cosine_noise_schedule,
				max_samples=None,
				shuffle=False,
				**kwargs):

		super().__init__()

		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.image_size = image_size
		self.schedule = schedule
		self.max_samples = max_samples
		self.local_module = LocalScoreModule(dataset,
							kernel_size=kernel_size,
							image_size=32,
							batch_size=batch_size,
							mode='zeros',
							schedule=schedule,
							max_samples=max_samples)

	def forward(self, t, x, label=None, device=None, k=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		b,c,h,w = x.shape
		if k is None:
			k = self.kernel_size
		if k >= h:
			return self.local_module(t, x, label=label, device=device, k=k)

		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		at = at.to(device)
		bt = bt.to(device)
		xpadded = F.pad(x,(k//2,k//2,k//2,k//2), value=0)
		xpatches = F.unfold(xpadded, k, stride=1, padding=0) 
		xnorms = torch.norm(xpatches, dim=1)**2
		xnorms = xnorms.reshape(b, h, w)


		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b,h,w, device=device)

		subtraction = torch.zeros(b,h,w, device=device)

		q = 0
		updated = False
		for images, labels in self.trainloader:

			if self.max_samples is not None and q > self.max_samples:
				break

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]
			if images.shape[0] == 0:
				q += self.batch_size
				continue


			images = images.to(device)
			labels = labels.to(device)

			bsize = images.shape[0]

			# CORNERS
			# topleft, topright, bottomleft, bottomright

			dk = k + k//2

			padded_xcorners = [F.pad(x[:,:,:k-1,:k-1], (k//2,0,k//2,0)),
								F.pad(x[:,:,:k-1,w-k+1:], (0,k//2,k//2,0)),
								F.pad(x[:,:,h-k+1:,:k-1], (k//2,0,0,k//2)),
								F.pad(x[:,:,h-k+1:,w-k+1:], (0,k//2,0,k//2))]
			

			padded_imcorners = [F.pad(images[:,:,:k-1,:k-1], (k//2,0,k//2,0)),
					F.pad(images[:,:,:k-1,-k+1:], (0,k//2,k//2,0)),
					F.pad(images[:,:,-k+1:,:k-1], (k//2,0,0,k//2)),
					F.pad(images[:,:,-k+1:,-k+1:], (0,k//2,0,k//2))]


			corner_args = []
			corner_vals = []
			lpatch = k-1-k//2

			for i in range(4):
				xpad = padded_xcorners[i]
				ipad = padded_imcorners[i]

				pwise_diffs = xpad[:,None,:,:,:]-at*ipad[None,:,:,:,:] # [b, NP, c, dk, dk]
				pwise_normsquares = torch.sum(pwise_diffs**2, dim=2) # sum over channel dimenions [b, NP, dk, dk]
				
				patches = F.unfold(pwise_normsquares, k, stride=1, padding=0)				
				patches = patches.view(b, bsize, k**2, lpatch, lpatch) # [b, NP, k^2, lpatch, lpatch]
				weight_args = -torch.sum(patches, dim=2)/(2*bt**2) # [b, NP, h, w]

				corner_val = pwise_diffs[:,:,:,k//2:k//2+lpatch,k//2:k//2+lpatch]
				corner_args.append(weight_args)
				corner_vals.append(corner_val)



			# MIDDLE
			middle_patches = F.unfold(images, k, stride=1, padding=0)
			middle_patches = torch.permute(middle_patches, (2,0,1)) # [h*w, 64, k^2 *c]
			middle_patches = middle_patches.reshape(middle_patches.shape[0]*middle_patches.shape[1], c, k, k) # [NP, c, k, k]			
			mpnorms = torch.sum(middle_patches**2, dim=(1,2,3)) # [NP]
			mpcenters = middle_patches[:,:,k//2,k//2] # [NP, c]

			mpdotx = F.conv2d(x, middle_patches, padding='valid')


			center_exp_args = -(xnorms[:,None,k//2:-(k//2),k//2:-(k//2)] - 2*at*mpdotx + (at**2)*mpnorms[None,:,None,None])/(2*bt**2) # [b, NP, h,w]

			center_vals = x[:,None:,k//2:-(k//2),k//2:-(k//2)] - at*mpcenters[None,:,:,None,None]


			# EDGES
			# top, right, bottom, left
			edge_args = [torch.zeros(b,bsize*(h-2*(k//2)),lpatch,(h-2*(k//2)), device=device) for j in range(4)]
			edge_vals = [torch.zeros(b,bsize*(h-2*(k//2)),c,lpatch,(h-2*(k//2)), device=device) for j in range(4)]


			padded_xedges = [F.pad(x[:,:,:k-1,:], (0,0,k//2,0)),
							F.pad(x[:,:,:,-k+1:],(0,k//2,0,0)).transpose(-2,-1),
							F.pad(x[:,:,-k+1:,:],(0,0,0,k//2)),
							F.pad(x[:,:,:,:k-1],(k//2,0,0,0)).transpose(-2,-1)]
			

			xedge_norms = [xnorms[:,:lpatch,k//2:-(k//2)], xnorms[:,k//2:-(k//2),-lpatch:].transpose(-2,-1), xnorms[:,-lpatch:,k//2:-(k//2)], xnorms[:,k//2:-(k//2),:lpatch].transpose(-2,-1)]
			
			padded_iedges = [F.pad(images[:,:,:k-1,:], (0,0,k//2,0)),
							F.pad(images[:,:,:,-k+1:],(0,k//2,0,0)).transpose(-2,-1),
							F.pad(images[:,:,-k+1:,:],(0,0,0,k//2)),
							F.pad(images[:,:,:,:k-1],(k//2,0,0,0)).transpose(-2,-1)]

			for i in range(4):
				xedge = padded_xedges[i]
				iedge = padded_iedges[i]
				for j in range(lpatch):
					xslice = xedge[:,:,j:k+j,:]
					islice = iedge[:,:,j:k+j,:] # [NP, c, k, L]
					filters = torch.cat([islice[:,:,:,a:a+k] for a in range(islice.shape[-1]-k+1)], axis=0) # [bNP, c, k, k]
					fnorms = torch.sum(filters**2, dim=(1,2,3))


					epnorms = torch.sum(filters**2, dim=(1,2,3)) # [NP]
					epdotx = F.conv2d(xslice, filters, padding='valid') # [b, NP, l]
					exnorms = xedge_norms[i][:,j,:] # [b, l]

					edge_args[i][:,:,j,:] = -(exnorms[:,None,:] - 2*at*epdotx[:,:,0,:] + (at**2)*fnorms[None,:,None])/(2*bt**2)  
					edge_vals[i][:,:,:,j,:] = (xslice[:,None,:,k//2,k//2:-(k//2)]-at*filters[None,:,:,k//2,k//2,None])


			if not updated:
				updated = True
				# Center
				subtraction[:,k//2:-(k//2),k//2:-(k//2)] = torch.amax(center_exp_args, dim=1)

				# Corners
				subtraction[:,:k//2,:k//2] = torch.amax(corner_args[0], dim=1)
				subtraction[:,:k//2,-(k//2):] = torch.amax(corner_args[1], dim=1)
				subtraction[:,-(k//2):,:k//2] = torch.amax(corner_args[2], dim=1)
				subtraction[:,-(k//2):,-(k//2):] = torch.amax(corner_args[3], dim=1)

				# Edges
				subtraction[:,:k//2,k//2:-(k//2)] = torch.amax(edge_args[0], dim=1)
				subtraction[:,k//2:-(k//2),-(k//2):] = torch.amax(edge_args[1].transpose(-2,-1), dim=1)
				subtraction[:,-(k//2):,k//2:-(k//2)] = torch.amax(edge_args[2], dim=1)
				subtraction[:,k//2:-(k//2),:k//2] = torch.amax(edge_args[3].transpose(-2,-1), dim=1)

			else:

				new_subtraction = torch.zeros(subtraction.shape, device=device)

				# Center
				new_subtraction[:,k//2:-(k//2),k//2:-(k//2)] = torch.amax(center_exp_args, dim=1)

				# Corners
				new_subtraction[:,:k//2,:k//2] = torch.amax(corner_args[0], dim=1)
				new_subtraction[:,:k//2,-(k//2):] = torch.amax(corner_args[1], dim=1)
				new_subtraction[:,-(k//2):,:k//2] = torch.amax(corner_args[2], dim=1)
				new_subtraction[:,-(k//2):,-(k//2):] = torch.amax(corner_args[3], dim=1)

				# Edges
				new_subtraction[:,:k//2,k//2:-(k//2)] = torch.amax(edge_args[0], dim=1)
				new_subtraction[:,k//2:-(k//2),-(k//2):] = torch.amax(edge_args[1].transpose(-2,-1), dim=1)
				new_subtraction[:,-(k//2):,k//2:-(k//2)] = torch.amax(edge_args[2], dim=1)
				new_subtraction[:,k//2:-(k//2),:k//2] = torch.amax(edge_args[3].transpose(-2,-1), dim=1)


				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)[:,None,:,:]
				denominator /= torch.exp(delta_subtraction-subtraction)[:,:,:]
				subtraction = delta_subtraction


			# Center
			center_exp_vals = torch.exp(center_exp_args-subtraction[:,None,k//2:-(k//2),k//2:-(k//2)])
			numerator[:,:,k//2:-(k//2),k//2:-(k//2)] += torch.sum(center_exp_vals[:,:,None,:,:]*center_vals ,dim=1)
			denominator[:,k//2:-(k//2),k//2:-(k//2)] += torch.sum(center_exp_vals, dim=1)

			# Corners
			corner_subtractions = [subtraction[:,:k//2,:k//2], subtraction[:,:k//2,-(k//2):], subtraction[:,-(k//2):,:k//2], subtraction[:,-(k//2):,-(k//2):]]
			corner_exp_vals = [torch.exp(corner_args[i]-corner_subtractions[i][:,None,:,:]) for i in range(4)]

			numerator[:,:,:k//2,:k//2] += torch.sum(corner_exp_vals[0][:,:,None,:,:]*corner_vals[0], dim=1)
			numerator[:,:,:k//2,-(k//2):] += torch.sum(corner_exp_vals[1][:,:,None,:,:]*corner_vals[1], dim=1)
			numerator[:,:,-(k//2):,:k//2] += torch.sum(corner_exp_vals[2][:,:,None,:,:]*corner_vals[2], dim=1)
			numerator[:,:,-(k//2):,-(k//2):] += torch.sum(corner_exp_vals[3][:,:,None,:,:]*corner_vals[3], dim=1)

			denominator[:,:k//2,:k//2] += torch.sum(corner_exp_vals[0], dim=1)
			denominator[:,:k//2,-(k//2):] += torch.sum(corner_exp_vals[1], dim=1)
			denominator[:,-(k//2):,:k//2] += torch.sum(corner_exp_vals[2], dim=1)
			denominator[:,-(k//2):,-(k//2):] += torch.sum(corner_exp_vals[3], dim=1)

			# Edges
			edge_subtractions = [subtraction[:,:k//2,k//2:-(k//2)], subtraction[:,k//2:-(k//2),-(k//2):], subtraction[:,-(k//2):,k//2:-(k//2)], subtraction[:,k//2:-(k//2),:k//2]]
			edge_args = [edge_args[0], edge_args[1].transpose(-2,-1), edge_args[2], edge_args[3].transpose(-2,-1)]
			edge_vals = [edge_vals[0], edge_vals[1].transpose(-2,-1), edge_vals[2], edge_vals[3].transpose(-2,-1)]

			edge_exp_vals = [torch.exp(edge_args[i]-edge_subtractions[i][:,None,:,:]) for i in range(4)]

			numerator[:,:,:k//2,k//2:-(k//2)] += torch.sum(edge_exp_vals[0][:,:,None,:,:]*edge_vals[0], dim=1)
			numerator[:,:,k//2:-(k//2),-(k//2):] += torch.sum(edge_exp_vals[1][:,:,None,:,:]*edge_vals[1], dim=1)
			numerator[:,:,-(k//2):,k//2:-(k//2)] += torch.sum(edge_exp_vals[2][:,:,None,:,:]*edge_vals[2], dim=1)
			numerator[:,:,k//2:-(k//2),:k//2] += torch.sum(edge_exp_vals[3][:,:,None,:,:]*edge_vals[3], dim=1)

			denominator[:,:k//2,k//2:-(k//2)] += torch.sum(edge_exp_vals[0], dim=1)
			denominator[:,k//2:-(k//2),-(k//2):] += torch.sum(edge_exp_vals[1], dim=1)
			denominator[:,-(k//2):,k//2:-(k//2)] += torch.sum(edge_exp_vals[2], dim=1)
			denominator[:,k//2:-(k//2),:k//2] += torch.sum(edge_exp_vals[3], dim=1)

			q += self.batch_size

		return -numerator/denominator[:,None,:,:]/bt**2


class LocalEquivScoreModule(nn.Module):

	def __init__(self, dataset,
				kernel_size=3,
				batch_size=64,
				image_size=32,
				channels=3,
				schedule=cosine_noise_schedule,
				max_samples=None,
				shuffle=False,
				**kwargs):

		super().__init__()

		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.image_size = image_size
		self.schedule = schedule
		self.max_samples = max_samples

	def forward(self, t, x, label=None, device=None, k=None):
		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


		b,c,h,w = x.shape
		if k is None:
			k = self.kernel_size

		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		bt = bt.to(device)
		at = at.to(device)

		d = k//2

		xpadded = F.pad(x, (d, d, d, d), mode='circular')

		xpatches = F.unfold(xpadded, k, stride=1, padding=0) 
		xnorms = torch.norm(xpatches, dim=1)**2
		xnorms = xnorms.reshape(b, h, w) # [b, h, w] lol

		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b,h,w, device=device)

		subtraction = None

		i = 0
		samps = 0
		max_exp_args = None
		next_exp_args = None

		for images, labels in self.trainloader:

			i += images.shape[0]
			if self.max_samples is not None and i > self.max_samples:
				break

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]
			if images.shape[0] == 0:
				continue

			images = images.to(device)
			labels = labels.to(device)

			samps += images.shape[0]

			bsize = images.shape[0]
			patches = F.unfold(images, k, stride=1, padding=0)

			patches = torch.permute(patches, (2,0,1)) # [h*w, 64, k^2 *c]
			patches = patches.reshape(patches.shape[0]*patches.shape[1], c, k, k) # [NP, c, k, k]			
			pnorms = torch.sum(patches**2, dim=(1,2,3)) # [NP]
			pcenters = patches[:,:,k//2,k//2] # [NP, c]
			
			pdotx = circular_convolution_native(x, patches)

			exp_args = -(xnorms[:,None,:,:] - 2*at*pdotx + (at**2)*pnorms[None,:,None,None])/(2*bt**2) # [b, NP, h,w]

			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
			else:
				new_subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)
				denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:]
				subtraction = delta_subtraction

			exp_vals = torch.exp(exp_args - subtraction) #[b, NP, h, w]
			num_vals = (x[:,None,:,:,:] - at*pcenters[None,:,:,None,None]) #[b,NP,c,h,w]

			numerator += torch.mean(exp_vals[:,:,None,:,:]*num_vals, dim=1)
			denominator += torch.mean(exp_vals, dim=1)

		return -numerator/denominator[:,None,:,:]/bt**2


class LocalScoreModule(nn.Module):

	def __init__(self, dataset,
				kernel_size=3,
				image_size=32,
				batch_size=256,
				show_plots=False,
				schedule=exponential_schedule,
				max_samples=None,
				**kwargs):

		super().__init__()
		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
		self.batch_size = batch_size
		self.kernel_size = kernel_size
		self.image_size = image_size
		self.show_plots = show_plots
		self.schedule = schedule
		self.max_samples = max_samples

	def forward(self, t, x, label=None, device=None, k=None):
		if k is None:
			k = self.kernel_size

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		x = x.to(device)

		b,c,h,w = x.shape
		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		at = at.to(device)
		bt = bt.to(device)

		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b,h,w, device=device)

		subtraction = None

		i = 0


		for images, labels in self.trainloader:

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]

			if images.shape[0] == 0:
				continue

			images = images.to(device)
			labels = labels.to(device)
			bsize = images.shape[0]

			i += bsize
			if self.max_samples is not None and i > self.max_samples:
				break

			pwise_diffs = x[:,None,:,:,:]-at*images[None,:,:,:,:] # [b, NP, c, h, w]
			pwise_normsquares = torch.sum(pwise_diffs**2, dim=2) # [b, NP, h, w]
			patches = F.unfold(pwise_normsquares, k, stride=1, padding=k//2) # [b, NP*k^2, h*w]
			patches = patches.view(b, bsize, k**2, h, w) # [b, NP, k^2, h, w]
			exp_args = -torch.sum(patches, dim=2)/(2*bt**2) # [b, NP, h, w]

			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
			else:
				new_subtraction = torch.amax(exp_args, dim=(0,1), keepdim=True)
				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)
				denominator /= torch.exp(delta_subtraction-subtraction)[:,0,:,:]
				subtraction = delta_subtraction

			exp_vals = torch.exp(exp_args - subtraction) #[b, NP, h, w]
			numerator += torch.mean(exp_vals[:,:,None,:,:]*pwise_diffs, dim=1)
			denominator += torch.mean(exp_vals, dim=1)


		return -numerator/denominator/bt**2


class IdealScoreModule(nn.Module):

	def __init__(self, dataset,
					image_size=32,
					batch_size=128,
					schedule=cosine_noise_schedule,
					max_samples=None,
					shuffle=False,
					**kwargs):

		super().__init__()
		self.dataset = dataset
		self.trainloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
		self.batch_size = batch_size
		self.image_size = image_size
		self.schedule = schedule
		self.max_samples = max_samples

	def forward(self, t, x, label=None, device=None, **kwargs):

		if device is None:
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		x = x.to(device)

		b,c,h,w = x.shape
		bt = (self.schedule(t))**0.5
		at = (1-self.schedule(t))**0.5

		at = at.to(device)
		bt = bt.to(device)

		numerator = torch.zeros(x.shape, device=device)
		denominator = torch.zeros(b, device=device)

		subtraction = None

		i = 0


		for images, labels in self.trainloader:

			if label is not None:
				images = images[(labels==label).squeeze(),:,:,:]

			if images.shape[0] == 0:
				continue

			images = images.to(device)
			labels = labels.to(device)

			bsize = images.shape[0]

			i += bsize
			if self.max_samples is not None and i > self.max_samples:
				break

			pwise_diffs = x[:,None,:,:,:]-at*images[None,:,:,:,:]
			exp_args = -torch.sum(pwise_diffs**2, dim=(2,3,4))/(2*bt**2) 
			

			if subtraction is None:
				subtraction = torch.amax(exp_args, dim=(0,1), keepdim=False)
			else:
				new_subtraction = torch.amax(exp_args, dim=(0,1), keepdim=False)
				delta_subtraction = (new_subtraction>subtraction)*new_subtraction+(subtraction>=new_subtraction)*subtraction
				numerator /= torch.exp(delta_subtraction-subtraction)
				denominator /= torch.exp(delta_subtraction-subtraction)
				subtraction = delta_subtraction

			exp_vals = torch.exp(exp_args - subtraction) 

			numerator += torch.mean(exp_vals[:,:,None,None,None]*pwise_diffs, dim=1)
			denominator += torch.mean(exp_vals, dim=1)


		return -numerator/denominator/bt**2
