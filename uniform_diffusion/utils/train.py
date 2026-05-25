import torch
from torch import optim, nn
from tqdm import tqdm
from torch.nn.functional import mse_loss
import time	

def train_diffusion(model, train_loader, noise_schedule, device,
					max_t=1000,
					num_epochs=100,
					lr=2e-4,
					samp_sched=30,
					in_channels=3,
					gamma=0.99995,
					fname='./model_checkpoints/test',
					conditional=False,
					show_plots=False,
					wd=0.001,
					save_interval=1):

	model.train()
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

	# Default scheduler we use is exponential
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

	for epoch in range(num_epochs):
		model.train()
		loop = tqdm(train_loader, leave=True)
		for images, labels in loop:
			optimizer.zero_grad()

			model.train()

			images = images.to(device)
			labels = labels.to(device)

			batch_size, _, _, _ = images.shape
			t = torch.randint(0, max_t, (batch_size,), device=device).float() / max_t  # Random timestep per image
			beta_t = noise_schedule(t)  # Get noise level from schedule

			noise = torch.normal(0,1,images.shape, device=device)
			noised_images = torch.sqrt(1 - beta_t)[:, None, None, None] * images + torch.sqrt(beta_t)[:, None, None, None] * noise
			
			if conditional:
				predicted_noise = model(t, noised_images, label=labels)
			else:
				predicted_noise = model(t, noised_images)

			loss = mse_loss(predicted_noise, noise)

			
			loss.backward()
			optimizer.step()

			loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
			loop.set_postfix(loss=loss.item())

			scheduler.step()

		if epoch % save_interval == save_interval-1:
			torch.save(model, fname + f'_epoch' + str(epoch) + '.pt')

