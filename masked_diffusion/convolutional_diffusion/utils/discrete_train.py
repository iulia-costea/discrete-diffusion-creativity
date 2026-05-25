
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from .discrete_noise import mask_forward, cosine_alpha_bar

def train_discrete_diffusion(
    model,
    train_loader,
    device,
    schedule=cosine_alpha_bar,
    num_epochs=300,
    lr=1e-4,
    gamma=0.999965,
    wd=0.0,
    in_channels=1,
    conditional=False,
    fname='discrete_model',
    save_interval=5,
):
       
    model.train()
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch}/{num_epochs}]')

        for batch in pbar:
            if conditional:
                x0, labels = batch
                labels = labels.to(device)
            else:
                x0, _ = batch
                labels = None

            x0 = x0.to(device)
            B = x0.shape[0]

                                                       
            t = torch.rand(B, device=device)

                                    
            x_t, unmasked = mask_forward(x0, t, schedule=schedule)

                                                         
            if conditional:
                logits = model(t, x_t, label=labels)
            else:
                logits = model(t, x_t)

                                               
            masked = ~unmasked                
            if masked.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits[masked], x0[masked]
                )
            else:
                                                             
                loss = F.binary_cross_entropy_with_logits(logits, x0)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

            epoch_loss = loss.item()
            pbar.set_postfix(loss=f'{epoch_loss:.4f}')

        if epoch % save_interval == 0:
            torch.save(model, f'{fname}_epoch{epoch}.pt')

    torch.save(model, f'{fname}_final.pt')
    return model
