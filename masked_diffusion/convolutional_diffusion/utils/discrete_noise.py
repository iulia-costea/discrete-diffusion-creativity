import math
import torch

MASK_VALUE = 0.5                                                           

def cosine_alpha_bar(t):
   
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    return torch.cos(t.clamp(0.0, 1.0) * math.pi / 2.0).pow(2)

def mask_forward(x0, t, schedule=cosine_alpha_bar):
  
    device = x0.device
    alpha_t = schedule(t).to(device).view(-1, 1, 1, 1)
    keep = torch.bernoulli(alpha_t.expand_as(x0)).bool()
    x_t = torch.where(keep, x0, torch.full_like(x0, MASK_VALUE))
    return x_t, keep

def mask_reverse_step(x_t, x0_pred, t_val, t_prev_val, schedule=cosine_alpha_bar,
                      stochastic=True):
   
    device = x_t.device
    alpha_t = schedule(torch.tensor(t_val)).to(device)
    alpha_prev = schedule(torch.tensor(t_prev_val)).to(device)

    unmasked = (x_t != MASK_VALUE)                     

                                                        
    x0_sample = torch.bernoulli(x0_pred).float()

    x_prev = x_t.clone()
    masked = ~unmasked

    if stochastic:
                                                    
                                                                   
                                                                  
                                                            
        p_reveal = ((alpha_prev - alpha_t) / (1.0 - alpha_t + 1e-8)).clamp(0.0, 1.0)
        reveal = torch.bernoulli(torch.full_like(x0_pred, p_reveal.item())).bool()
        x_prev[masked & reveal] = x0_sample[masked & reveal]
    else:
                                                                                 
                                                                                
                                                                             
        x0_binary = (x0_pred > 0.5).float()
        keep = torch.bernoulli(
            torch.full_like(x0_pred, alpha_prev.clamp(0.0, 1.0).item())
        ).bool()
        x_prev[masked & keep] = x0_binary[masked & keep]
                                                          

    return x_prev

def full_reverse(model, nsteps=20, batch_size=1, in_channels=1, image_size=28,
                 schedule=cosine_alpha_bar, stochastic=True, device=None,
                 label=None):

       
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        
    x_t = torch.full((batch_size, in_channels, image_size, image_size),
                     MASK_VALUE, device=device)

    timesteps = torch.linspace(1.0, 0.0, nsteps + 1)               

    for i in range(nsteps):
        t_val = timesteps[i].item()
        t_prev_val = timesteps[i + 1].item()

        t_tensor = torch.full((batch_size,), t_val, device=device)
        with torch.no_grad():
            logits = model(t_tensor, x_t, label=label)
            x0_pred = torch.sigmoid(logits)

        x_t = mask_reverse_step(x_t, x0_pred, t_val, t_prev_val,
                                 schedule=schedule, stochastic=stochastic)

    return x_t
