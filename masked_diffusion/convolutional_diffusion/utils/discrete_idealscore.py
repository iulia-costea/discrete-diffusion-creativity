import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .discrete_noise import cosine_alpha_bar, MASK_VALUE

def _circular_conv(x, filters):
    
    b, c, h, w = x.shape
    nf, _, k, _ = filters.shape
    d = k // 2
    x_pad = F.pad(x, (d, d, d, d), mode='circular')
    return F.conv2d(x_pad, filters, padding=0)

                                                                             
                                   
                                                                             

class DiscreteIdealScoreModule(nn.Module):

    def __init__(self, dataset, batch_size=128, schedule=cosine_alpha_bar, **kw):
        super().__init__()
        self.trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.schedule = schedule

    def forward(self, t, x_t, unmasked=None, device=None, k=None, **kw):
      
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x_t = x_t.to(device)

        if unmasked is None:
            unmasked = (x_t != MASK_VALUE)
        m = unmasked.float()
        x_obs = x_t * m                             

        b, c, h, w = x_t.shape
        if k is None:
            k = h                                  
        alpha_t = self.schedule(t).to(device)        
        T_raw = (-torch.log(alpha_t.clamp(min=1e-8))).mean().item()
        log_odds = T_raw * (k ** 2)

                                                                                
        x_obs_norm_sq = (x_obs ** 2).sum(dim=(1, 2, 3))        

        numerator = torch.zeros(b, c, h, w, device=device)
        denominator = torch.zeros(b, device=device)
        subtraction = None

        for images, _ in self.trainloader:
            images = images.to(device)                       

                                                   
                                                                          
            mx_dot_mphi = (x_obs[:, None] * m[:, None] * images[None]).sum(dim=(2, 3, 4))          
            mphi_norm_sq = (m[:, None] * images[None] ** 2).sum(dim=(2, 3, 4))          

            exp_args = -log_odds * (
                x_obs_norm_sq[:, None] - 2 * mx_dot_mphi + mphi_norm_sq
            )          

            if subtraction is None:
                subtraction = exp_args.max(dim=1, keepdim=True).values
            else:
                new_sub = exp_args.max(dim=1, keepdim=True).values
                delta = torch.max(subtraction, new_sub)
                numerator /= torch.exp(delta - subtraction).view(b, 1, 1, 1)
                denominator /= torch.exp(delta - subtraction).squeeze(1)
                subtraction = delta

            exp_vals = torch.exp(exp_args - subtraction)          

                                     
            numerator += torch.einsum('bn,nchw->bchw', exp_vals, images)
            denominator += exp_vals.sum(dim=1)

        return numerator / denominator.view(b, 1, 1, 1).clamp(min=1e-12)

                                                                             
                                                
                                                                             

class DiscreteLocalEquivScoreModule(nn.Module):

    def __init__(self, dataset, kernel_size=3, batch_size=64,
                 image_size=32, channels=1, schedule=cosine_alpha_bar,
                 max_samples=None, shuffle=False, **kw):
        super().__init__()
        self.trainloader = DataLoader(dataset, batch_size=batch_size,
                                      shuffle=shuffle)
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.channels = channels
        self.schedule = schedule
        self.max_samples = max_samples

    def forward(self, t, x_t, unmasked=None, device=None, k=None, **kw):

           
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if k is None:
            k = self.kernel_size
        x_t = x_t.to(device)

        if unmasked is None:
            unmasked = (x_t != MASK_VALUE)
        m = unmasked.float()                     
        x_obs = x_t * m                                                          

        b, c, h, w = x_t.shape
        alpha_t = self.schedule(t).to(device)        
                                                 
                                                                             
        T_raw = (-torch.log(alpha_t.clamp(min=1e-8))).mean().item()
        log_odds = T_raw * (k ** 2)

        d = k // 2
                                                                         
                                         
        xobs_pad = F.pad(x_obs, (d, d, d, d), mode='circular')
        m_pad = F.pad(m, (d, d, d, d), mode='circular')                                                     
                                                 
        def local_sum_sq(tensor_pad):
            unf = F.unfold(
                tensor_pad.view(b * c, 1, h + 2 * d, w + 2 * d),
                k, stride=1, padding=0
            ).view(b, c, k * k, h * w)                   
            return unf.sum(dim=2).view(b, c, h, w)                

                                                                         
        xnorms = local_sum_sq(xobs_pad ** 2).sum(dim=1)              
                                                                                
        m_counts = local_sum_sq(m_pad).sum(dim=1)                    

        numerator = torch.zeros(b, c, h, w, device=device)
        denominator = torch.zeros(b, h, w, device=device)
        subtraction = None

        n_seen = 0
        for images, _ in self.trainloader:
            if self.max_samples is not None and n_seen >= self.max_samples:
                break
            images = images.to(device)                             
            N = images.shape[0]
            n_seen += N

                                                                  
                                                   
            img_pad = F.pad(images, (d, d, d, d), mode='circular')
                                     
            patches_flat = F.unfold(
                img_pad.view(N * c, 1, h + 2 * d, w + 2 * d),
                k, stride=1, padding=0
            ).view(N, c, k * k, h * w)
                                                       
            patches = patches_flat.permute(0, 3, 1, 2).reshape(N * h * w, c, k, k)
            pcenters = patches[:, :, k // 2, k // 2]                            
            pnorms = (patches ** 2).sum(dim=(1, 2, 3))               
                                    
                                  
            pdotx = _circular_conv(x_obs, patches)                 

                                         
                                                      
            mphi_norm = _circular_conv(m, patches ** 2)                 
                                          
                                                                          
            exp_args = -log_odds * (
                xnorms[:, None, :, :] - 2 * pdotx + mphi_norm
            )                 

                                                                   
            if subtraction is None:
                subtraction = exp_args.amax(dim=1)             
            else:
                new_sub = exp_args.amax(dim=1)
                delta = torch.max(subtraction, new_sub)
                numerator /= torch.exp(delta - subtraction)[:, None, :, :]
                denominator /= torch.exp(delta - subtraction)
                subtraction = delta

            exp_vals = torch.exp(exp_args - subtraction[:, None, :, :])                 

                                             
                                                                        
            pc = pcenters.permute(1, 0).view(1, N * h * w, c, 1, 1)                
            numerator += (exp_vals[:, :, None, :, :] * pc).sum(dim=1)                 
            denominator += exp_vals.sum(dim=1)                                      

        return numerator / denominator[:, None, :, :].clamp(min=1e-12)

                                                                             
                                                             
                                                                             

class DiscreteLocalEquivBordersScoreModule(nn.Module):

    def __init__(self, dataset, kernel_size=3, batch_size=64,
                 image_size=32, channels=1, schedule=cosine_alpha_bar,
                 max_samples=None, **kw):
        super().__init__()
        self.k = kernel_size
        self.els = DiscreteLocalEquivScoreModule(
            dataset, kernel_size=kernel_size, batch_size=batch_size,
            image_size=image_size, channels=channels, schedule=schedule,
            max_samples=max_samples)
        self.ls = DiscreteLocalScoreModule(
            dataset, kernel_size=kernel_size, batch_size=batch_size,
            image_size=image_size, channels=channels, schedule=schedule,
            max_samples=max_samples)

    def forward(self, t, x_t, unmasked=None, device=None, k=None, **kw):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if k is None:
            k = self.k
        b, c, h, w = x_t.shape
        if k >= h:
            return self.ls(t, x_t, unmasked=unmasked, device=device, k=k)
        return self.els(t, x_t, unmasked=unmasked, device=device, k=k)

                                                                             
                                                                              
                                                                             

class DiscreteLocalScoreModule(nn.Module):

    def __init__(self, dataset, kernel_size=3, batch_size=256,
                 image_size=32, channels=1, schedule=cosine_alpha_bar,
                 max_samples=None, **kw):
        super().__init__()
        self.trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.kernel_size = kernel_size
        self.schedule = schedule
        self.max_samples = max_samples
        self.channels = channels

    def forward(self, t, x_t, unmasked=None, device=None, k=None, **kw):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if k is None:
            k = self.kernel_size
        x_t = x_t.to(device)

        if unmasked is None:
            unmasked = (x_t != MASK_VALUE)
        m = unmasked.float()
        x_obs = x_t * m

        b, c, h, w = x_t.shape
        alpha_t = self.schedule(t).to(device)
                                               
                                                                             
        T_raw = (-torch.log(alpha_t.clamp(min=1e-8))).mean().item()
        log_odds = T_raw * (k ** 2)

        numerator = torch.zeros(b, c, h, w, device=device)
        denominator = torch.zeros(b, h, w, device=device)
        subtraction = None

        n_seen = 0
        for images, _ in self.trainloader:
            if self.max_samples is not None and n_seen >= self.max_samples:
                break
            images = images.to(device)
            N = images.shape[0]
            n_seen += N

                                                            
            pwise_diffs = x_obs[:, None] - images[None]                           
            m_expanded = m[:, None].expand_as(pwise_diffs)

                                                                     
            masked_diffs_sq = (m_expanded * pwise_diffs) ** 2                     
            mdsq_flat = masked_diffs_sq.view(b * N, c, h, w)
            local_norms = F.unfold(
                F.pad(mdsq_flat, (k//2, k//2, k//2, k//2), mode='constant', value=0),
                k, stride=1, padding=0
            )                    
            local_norms = local_norms.view(b, N, c * k * k, h * w).sum(dim=2).view(b, N, h, w)

            exp_args = -log_odds * local_norms                

            if subtraction is None:
                subtraction = exp_args.amax(dim=1)
            else:
                new_sub = exp_args.amax(dim=1)
                delta = torch.max(subtraction, new_sub)
                numerator /= torch.exp(delta - subtraction)[:, None]
                denominator /= torch.exp(delta - subtraction)
                subtraction = delta

            exp_vals = torch.exp(exp_args - subtraction[:, None])                
            numerator += (exp_vals[:, :, None] * images[None]).sum(dim=1)
            denominator += exp_vals.sum(dim=1)

        return numerator / denominator[:, None].clamp(min=1e-12)

class ScheduledDiscreteScoreMachine(nn.Module):


    def __init__(self, backbone, in_channels=1, imsize=28,
                 schedule=cosine_alpha_bar, scales=None, nsteps=20):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.imsize = imsize
        self.schedule = schedule
        self.scales = scales
        self.nsteps = nsteps

    def forward(self, x_t, nsteps=None, label=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if nsteps is None:
            nsteps = self.nsteps if self.scales is None else len(self.scales)

        x_t = x_t.clone().to(device)
        timesteps = torch.linspace(1.0, 0.0, nsteps + 1)

        for i in range(nsteps):
            t_val = timesteps[i].item()
            t_prev_val = timesteps[i + 1].item()
            t_tensor = torch.full((x_t.shape[0],), t_val, device=device)
            k = None if self.scales is None else int(self.scales[i])

            unmasked = (x_t != MASK_VALUE)
            x0_pred = self.backbone(t_tensor, x_t, unmasked=unmasked,
                                    device=device, k=k)

            from .discrete_noise import mask_reverse_step
            x_t = mask_reverse_step(x_t, x0_pred, t_val, t_prev_val,
                                    schedule=self.schedule, stochastic=True)

        return x_t

    def sample(self, nsteps=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
        x_t = torch.full((1, self.in_channels, self.imsize, self.imsize),
                         MASK_VALUE, device=device)
        return self(x_t, nsteps=nsteps, device=device)
