import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import (
    TensorDataset, DataLoader, RandomSampler, Subset
)
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

                                                                   
                                                                 
                                                     
SAVE_DIR = '/content/checkpoints/'
os.makedirs(SAVE_DIR, exist_ok=True)

                                                                               
MASK_VALUE = 0.5                                                   

def cosine_alpha_bar(t):
                                                                 
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    return torch.cos(t.clamp(0.0, 1.0) * math.pi / 2.0).pow(2)

def mask_forward(x0, t, schedule=cosine_alpha_bar):
                                                                     
    alpha_t = schedule(t).to(x0.device).view(-1, 1, 1, 1)
    keep    = torch.bernoulli(alpha_t.expand_as(x0)).bool()
    x_t     = torch.where(keep, x0, torch.full_like(x0, MASK_VALUE))
    return x_t, keep

def mask_reverse_step(x_t, x0_pred, t_val, t_prev_val,
                      schedule=cosine_alpha_bar, stochastic=True):
                                                                
    device     = x_t.device
    alpha_t    = schedule(torch.tensor(t_val)).to(device)
    alpha_prev = schedule(torch.tensor(t_prev_val)).to(device)
    unmasked   = (x_t != MASK_VALUE)
    x0_sample  = torch.bernoulli(x0_pred).float()
    x_prev     = x_t.clone()
    masked     = ~unmasked

    if stochastic:
        p_reveal = ((alpha_prev - alpha_t) / (1.0 - alpha_t + 1e-8)).clamp(0., 1.)
        reveal   = torch.bernoulli(torch.full_like(x0_pred, p_reveal.item())).bool()
        x_prev[masked & reveal] = x0_sample[masked & reveal]
    else:
        x0_bin = (x0_pred > 0.5).float()
        keep   = torch.bernoulli(
            torch.full_like(x0_pred, alpha_prev.clamp(0., 1.).item())
        ).bool()
        x_prev[masked & keep] = x0_bin[masked & keep]
    return x_prev

def full_reverse(model, nsteps=20, batch_size=1, in_channels=1,
                 image_size=28, schedule=cosine_alpha_bar,
                 stochastic=True, device=None, label=None):
                                                                        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.full((batch_size, in_channels, image_size, image_size),
                     MASK_VALUE, device=device)
    timesteps = torch.linspace(1.0, 0.0, nsteps + 1)
    for i in range(nsteps):
        t_val      = timesteps[i].item()
        t_prev_val = timesteps[i + 1].item()
        t_tensor   = torch.full((batch_size,), t_val, device=device)
        with torch.no_grad():
            logits  = model(t_tensor, x_t, label=label)
            x0_pred = torch.sigmoid(logits)
        x_t = mask_reverse_step(x_t, x0_pred, t_val, t_prev_val,
                                 schedule=schedule, stochastic=stochastic)
    return x_t

                                                                               

class EmbeddingModule(nn.Module):
                                                               
    def __init__(self, fdim, conditional=False, num_classes=None):
        super().__init__()
        self.fdim = fdim
        self.conditional = conditional
        if conditional:
            self.class_embeddings = nn.Embedding(num_classes, fdim)

    def forward(self, t, label=None):
        d    = self.fdim // 2
        freq = 10000 ** (torch.arange(d, device=t.device) / max(d - 1, 1))
        args = t[:, None] / freq[None, :]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.conditional and label is not None:
            emb = emb + self.class_embeddings(label)
        return emb

class MinimalResNet(nn.Module):
       
    def __init__(self, channels=1, emb_dim=256, mode='circular',
                 conditional=False, num_classes=None,
                 kernel_size=3, num_layers=6, lastksize=3):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim    = emb_dim
        self.embedding  = EmbeddingModule(emb_dim, conditional, num_classes)

        self.up_projection = nn.Conv2d(
            channels, emb_dim, kernel_size, padding='same', padding_mode=mode)

        self.embs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GroupNorm(8, emb_dim),
                nn.ReLU(),
            )
            for _ in range(num_layers + 1)
        ])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, kernel_size,
                          padding='same', padding_mode=mode),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        ])
        self.down_projection = nn.Conv2d(
            emb_dim, channels, lastksize, padding='same', padding_mode=mode)

    @property
    def receptive_field(self):
        return 2 * self.num_layers + 5

    def forward(self, t, x, label=None):
        emb   = self.embedding(t, label=label)
        state = self.up_projection(x)
        for i in range(self.num_layers):
            delta = self.convs[i](state + self.embs[i](emb)[:, :, None, None])
            state = state + delta
        state = state + self.embs[-1](emb)[:, :, None, None]
        return self.down_projection(state)

class MPNNLayer(nn.Module):

       
    def __init__(self, dim):
        super().__init__()
        self.lin_self  = nn.Linear(dim, dim, bias=False)
        self.lin_neigh = nn.Linear(dim, dim, bias=True)
        self.act       = nn.ReLU()

    def forward(self, x, edge_index):
                                             
        src, dst = edge_index
        B, N, D  = x.shape
        agg = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        agg.index_add_(1, dst, x[:, src, :])
        deg = torch.bincount(dst, minlength=N).float().clamp(min=1).view(1, N, 1)
        agg = agg / deg
        return self.act(self.lin_self(x) + self.lin_neigh(agg))

def make_grid_edges(H, W):
                                                          
    idx = torch.arange(H * W).reshape(H, W)
    pairs = [
        (idx[:, :-1], idx[:, 1:]),
        (idx[:, 1:],  idx[:, :-1]),
        (idx[:-1, :], idx[1:, :]),
        (idx[1:, :],  idx[:-1, :]),
    ]
    src = torch.cat([s.flatten() for s, _ in pairs])
    dst = torch.cat([d.flatten() for _, d in pairs])
    return torch.stack([src, dst])

class MinimalResGNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, emb_dim=256,
                 num_layers=6, conditional=False, num_classes=None):
        super().__init__()
        self.num_layers      = num_layers
        self.embedding       = EmbeddingModule(emb_dim, conditional, num_classes)
        self.up_projection   = nn.Linear(in_channels, emb_dim)
        self.embs = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())
            for _ in range(num_layers + 1)
        ])
        self.layers = nn.ModuleList([MPNNLayer(emb_dim) for _ in range(num_layers)])
        self.down_projection = nn.Linear(emb_dim, out_channels)

    @property
    def receptive_field_radius(self):
        return self.num_layers                  

    def forward(self, t, x, edge_index, label=None):
        emb   = self.embedding(t, label=label)
        state = self.up_projection(x)
        for i in range(self.num_layers):
            time_bias = self.embs[i](emb).unsqueeze(1)
            delta     = self.layers[i](state + time_bias, edge_index)
            state     = state + delta
        state = state + self.embs[-1](emb).unsqueeze(1)
        return self.down_projection(state)

class GNNImageWrapper(nn.Module):
    def __init__(self, gnn, edge_index, H, W):
        super().__init__()
        self.gnn = gnn
        self.H   = H
        self.W   = W
        self.register_buffer('edge_index', edge_index)

    def forward(self, t, x, label=None):
        B, C, H, W  = x.shape
        x_nodes     = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        out_nodes   = self.gnn(t, x_nodes, self.edge_index, label=label)
        return out_nodes.reshape(B, H, W, C).permute(0, 3, 1, 2)

                                                                              

def _circ_conv(x, filters):
                                                                      
    k   = filters.shape[-1]
    d   = k // 2
    xp  = F.pad(x, (d, d, d, d), mode='circular')
    return F.conv2d(xp, filters, padding=0)

class DiscreteELS(nn.Module):
  
    def __init__(self, dataset, kernel_size=3, batch_size=64,
                 image_size=32, channels=1, schedule=cosine_alpha_bar,
                 max_samples=None):
        super().__init__()
        self.loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.k           = kernel_size
        self.H           = image_size
        self.C           = channels
        self.schedule    = schedule
        self.max_samples = max_samples

    def forward(self, t, x_t, unmasked=None, device=None, **kw):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k  = self.k
        d  = k // 2
        x_t = x_t.to(device)

        if unmasked is None:
            unmasked = (x_t != MASK_VALUE)
        m     = unmasked.float()
        x_obs = x_t * m

        b, c, h, w = x_t.shape

                                                                          
        alpha_t = self.schedule(t).to(device)
        log_T   = (-torch.log(alpha_t.clamp(min=1e-8))).mean().item() * (k ** 2)

                                                              
        xobs_pad = F.pad(x_obs, (d,d,d,d), mode='circular')
        m_pad    = F.pad(m,     (d,d,d,d), mode='circular')

        def local_sq_sum(tp):
            u = F.unfold(tp.view(b*c, 1, h+2*d, w+2*d), k, stride=1, padding=0)
            return u.view(b, c, k*k, h*w).sum(2).view(b, c, h, w)

        xnorms = local_sq_sum(xobs_pad ** 2).sum(1)              

        numerator   = torch.zeros(b, c, h, w, device=device)
        denominator = torch.zeros(b, h, w,    device=device)
        subtraction = None
        n_seen = 0

        for images, _ in self.loader:
            if self.max_samples is not None and n_seen >= self.max_samples:
                break
            images = images.to(device)
            N      = images.shape[0]
            n_seen += N

                                                                        
            img_pad = F.pad(images, (d,d,d,d), mode='circular')
            pf = F.unfold(img_pad.view(N*c, 1, h+2*d, w+2*d),
                          k, stride=1, padding=0).view(N, c, k*k, h*w)
            patches  = pf.permute(0,3,1,2).reshape(N*h*w, c, k, k)
            pcenters = patches[:, :, k//2, k//2]                               

                                                                     
            pdotx    = _circ_conv(x_obs, patches)                      
            mphi_n   = _circ_conv(m,     patches ** 2)                 

                                       
            exp_args = -log_T * (xnorms[:,None] - 2*pdotx + mphi_n)              

                                                                   
            new_sub = exp_args.amax(dim=1)
            if subtraction is None:
                subtraction = new_sub
            else:
                delta = torch.max(subtraction, new_sub)
                numerator   /= torch.exp(delta - subtraction)[:, None]
                denominator /= torch.exp(delta - subtraction)
                subtraction  = delta

            w_vals = torch.exp(exp_args - subtraction[:, None])                      
            pc     = pcenters.permute(1,0).view(1, N*h*w, c, 1, 1)
            numerator   += (w_vals[:,:,None] * pc).sum(1)
            denominator += w_vals.sum(1)

        return numerator / denominator[:, None].clamp(min=1e-12)

class ELSMachine(nn.Module):
                                                                    
    def __init__(self, backbone, in_channels=1, imsize=24,
                 schedule=cosine_alpha_bar, nsteps=20):
        super().__init__()
        self.backbone  = backbone
        self.C         = in_channels
        self.H         = imsize
        self.schedule  = schedule
        self.nsteps    = nsteps

    @torch.no_grad()
    def sample(self, nsteps=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ns  = nsteps or self.nsteps
        x_t = torch.full((1, self.C, self.H, self.H), MASK_VALUE, device=device)
        ts  = torch.linspace(1.0, 0.0, ns + 1)
        for i in range(ns):
            t_val  = ts[i].item()
            tp_val = ts[i+1].item()
            tt     = torch.full((1,), t_val, device=device)
            unm    = (x_t != MASK_VALUE)
            p0     = self.backbone(tt, x_t, unmasked=unm, device=device)
            x_t    = mask_reverse_step(x_t, p0, t_val, tp_val, stochastic=True)
        return x_t

                                                                               

def train_discrete(model, loader, device, schedule=cosine_alpha_bar,
                   num_epochs=100, lr=1e-4, gamma=1.0, fname=None,
                   save_every=25, verbose=True):
                                                
    model.train()
    opt  = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    log  = []

    for epoch in range(1, num_epochs + 1):
        ep_loss = 0.0
        for batch in loader:
            x0  = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            B   = x0.shape[0]
            t   = torch.rand(B, device=device)
            x_t, unmasked = mask_forward(x0, t, schedule)
            logits = model(t, x_t)
            masked = ~unmasked
            loss = (F.binary_cross_entropy_with_logits(logits[masked], x0[masked])
                    if masked.any()
                    else F.binary_cross_entropy_with_logits(logits, x0))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss = loss.item()
        log.append(ep_loss)
        if verbose and epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/{num_epochs}  loss={ep_loss:.4f}')
        if fname and epoch % save_every == 0:
            torch.save(model.state_dict(), f'{fname}_ep{epoch}.pt')

    if fname:
        torch.save(model.state_dict(), f'{fname}_final.pt')
        print(f'Saved → {fname}_final.pt')
    return log

                                                                              

def local_violations(img_binary, periodic=True):
       
    img = img_binary.cpu().float()
    if img.dim() == 4:
        img = img[:, 0]
    if img.dim() == 2:
        img = img.unsqueeze(0)
    B, H, W = img.shape
    result = []
    for b in range(B):
        im    = img[b]
        count = 0
        for i in range(H):
            for j in range(W):
                if periodic:
                    patch = [im[(i+di)%H, (j+dj)%W].item()
                             for di in range(-1,2) for dj in range(-1,2)]
                else:
                    r0,r1 = max(0,i-1), min(H,i+2)
                    c0,c1 = max(0,j-1), min(W,j+2)
                    patch = im[r0:r1, c0:c1].flatten().tolist()
                majority = 1.0 if sum(patch) >= len(patch)/2 else 0.0
                if majority != im[i, j].item():
                    count += 1
        result.append(count)
    return result

def r2_score(a, b):
                                                                    
    def norm(x):
        x  = x.float().flatten()
        x  = x - x.mean()
        nrm = x.norm()
        return x / nrm if nrm > 1e-8 else x
    return torch.dot(norm(a), norm(b)).item() ** 2

def get_samples(model, n, nsteps, image_size, in_channels=1,
                device=None, base_seed=42, stochastic=True):
                                                                              
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    outs, seeds = [], []
    with torch.no_grad():
        for i in range(n):
            s = base_seed + i
            torch.manual_seed(s)
            out = full_reverse(model, nsteps=nsteps, batch_size=1,
                               in_channels=in_channels, image_size=image_size,
                               device=device, stochastic=stochastic)
            outs.append(out.cpu())
            seeds.append(s)
    return torch.cat(outs, dim=0), seeds

def show_samples(imgs_dict, title='', save_path=None, n_cols=8):
                                                              
    rows   = list(imgs_dict.items())
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.6))
    if n_rows == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if c < len(imgs):
                ax.imshow(imgs[c, 0].numpy(), cmap='gray', vmin=0, vmax=1,
                          interpolation='nearest')
            ax.axis('off')
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=60, va='center')
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved → {save_path}')
    plt.show()

                                                                               
print('\n' + '='*60)
print('SECTION 6 — TOY EXPERIMENT (24×24 binary, toroidal)')
print('='*60)

H_TOY = W_TOY = 24

                                         
black     = torch.zeros(1, H_TOY, W_TOY)
white     = torch.ones(1,  H_TOY, W_TOY)
X_toy     = torch.stack([black, white])                      
y_toy     = torch.tensor([0, 1])
toy_ds    = TensorDataset(X_toy, y_toy)

                                                                  
sampler    = RandomSampler(toy_ds, replacement=True, num_samples=6400)
toy_loader = DataLoader(toy_ds, batch_size=64, sampler=sampler)
print(f'Toy dataset: {len(toy_ds)} images  |  {len(toy_loader)} batches/epoch')

                                                                               
                                                     
L_VALUES   = [3, 6, 8]
cnn_models = {}

for L in L_VALUES:
    rf = 2 * L + 5
    print(f'\n--- CNN  L={L}  RF={rf}/{H_TOY} ({100*rf/H_TOY:.0f}%) ---')
    model = MinimalResNet(
        channels=1, emb_dim=256, mode='circular',
        num_layers=L, kernel_size=3, lastksize=3,
    ).to(device)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
    train_discrete(
        model, toy_loader, device,
        num_epochs=100, lr=1e-4, gamma=1.0,
        fname=os.path.join(SAVE_DIR, f'toy_cnn_L{L}'),
        save_every=1000, verbose=True,
    )
    cnn_models[L] = model

                                                                                
print('\n--- Generating samples ---')
cnn_results = {}

for L in L_VALUES:
    rf    = 2 * L + 5
    model = cnn_models[L]

                                                                       
    valid, seeds_v = get_samples(model, n=16, nsteps=400,
                                  image_size=H_TOY, base_seed=42, device=device)
    valid_bin = (valid > 0.5).float()
    v_viols   = local_violations(valid_bin)

                                                                         
    invalid, seeds_i = get_samples(model, n=16, nsteps=3,
                                    image_size=H_TOY, base_seed=42, device=device)
    invalid_bin = (invalid > 0.5).float()
    i_viols     = local_violations(invalid_bin)

    cnn_results[L] = {
        'valid': valid_bin, 'invalid': invalid_bin,
        'v_viols': v_viols, 'i_viols': i_viols,
        'seeds': seeds_v,
    }
    print(f'L={L} RF={rf}: valid violations={np.mean(v_viols):.1f}  '
          f'invalid violations={np.mean(i_viols):.1f}')

                                                                               
print('\n--- Discrete ELS (analytic, no training) ---')
els_mod = DiscreteELS(
    toy_ds, kernel_size=3, batch_size=2,
    image_size=H_TOY, channels=1,
    schedule=cosine_alpha_bar, max_samples=2,
)
els_machine = ELSMachine(els_mod, in_channels=1, imsize=H_TOY,
                          schedule=cosine_alpha_bar, nsteps=400)

els_out_list = []
for i in range(16):
    torch.manual_seed(42 + i)
    out = els_machine.sample(nsteps=400, device=device)
    els_out_list.append((out > 0.5).float().cpu())
els_out = torch.cat(els_out_list, dim=0)                   
els_viols = local_violations(els_out)
print(f'ELS (T=400): mean violations = {np.mean(els_viols):.1f}')

                                                                               
print('\n--- r² (CNN vs discrete ELS, same seeds) ---')
for L in L_VALUES:
    model = cnn_models[L]
    model.eval()
    r2s   = []
    with torch.no_grad():
        for i in range(16):
            torch.manual_seed(42 + i)
            cnn_sample = full_reverse(
                model, nsteps=400, batch_size=1,
                in_channels=1, image_size=H_TOY,
                device=device, stochastic=True,
            )
            els_sample = els_out_list[i].to(device)
            r2s.append(r2_score(cnn_sample, els_sample))
    print(f'  L={L} RF={2*L+5}: median r² = {np.median(r2s):.3f}  '
          f'mean r² = {np.mean(r2s):.3f}')

                                                                                
                      
L_list = L_VALUES
x      = np.arange(len(L_list))
w      = 0.3
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - w, [np.mean(cnn_results[L]['v_viols']) for L in L_list],
       w, label='CNN valid (T=400)', color='steelblue')
ax.bar(x,     [np.mean(cnn_results[L]['i_viols']) for L in L_list],
       w, label='CNN invalid (T=3)', color='coral')
ax.axhline(np.mean(els_viols), color='green', ls='--',
           label=f'Discrete ELS (T=400): {np.mean(els_viols):.1f}')
ax.set_xticks(x)
ax.set_xticklabels([f'L={L}\nRF={2*L+5}/{H_TOY}' for L in L_list])
ax.set_ylabel('Mean local consistency violations')
ax.set_title('Discrete masking diffusion — CNN violations vs receptive field\n'
             '24×24 toroidal binary images')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'toy_violations.png'), dpi=150, bbox_inches='tight')
plt.show()

              
imgs_to_show = {f'CNN L={L} RF={2*L+5} (valid)': cnn_results[L]['valid']
                for L in L_VALUES}
imgs_to_show['Discrete ELS (valid)'] = els_out
show_samples(imgs_to_show,
             title='Valid samples (T=400) — masking diffusion, 24×24 toroidal',
             save_path=os.path.join(SAVE_DIR, 'toy_samples.png'))

print('\n✓ Toy experiment done. Results saved to', SAVE_DIR)

                                                                              
                                                                

                
RUN_GNN = False

if RUN_GNN:
    print('\n' + '='*60)
    print('SECTION 7 — GNN on TOY (optional)')
    print('='*60)

    edge_index_toy = make_grid_edges(H_TOY, W_TOY).to(device)
                                                           
                                                  
    L_GNN = [3, 6, 8, 12]
    gnn_models = {}

    for L in L_GNN:
        n_pixels_diamond = 2*L**2 + 2*L + 1
        print(f'\n--- GNN  L={L}  RF={n_pixels_diamond} pixels '
              f'(CNN L={L} RF={(2*L+5)**2} px²) ---')
        gnn_raw = MinimalResGNN(
            in_channels=1, out_channels=1, emb_dim=256, num_layers=L,
        )
        gnn = GNNImageWrapper(gnn_raw, edge_index_toy, H_TOY, W_TOY).to(device)
        print(f'  Params: {sum(p.numel() for p in gnn.parameters()):,}')
        train_discrete(
            gnn, toy_loader, device,
            num_epochs=100, lr=1e-4, gamma=1.0,
            fname=os.path.join(SAVE_DIR, f'toy_gnn_L{L}'),
            save_every=1000, verbose=True,
        )
        gnn_models[L] = gnn

    gnn_results = {}
    for L in L_GNN:
        gnn = gnn_models[L]
        valid, _   = get_samples(gnn, 16, 400, H_TOY, base_seed=42, device=device)
        invalid, _ = get_samples(gnn, 16,   3, H_TOY, base_seed=42, device=device)
        valid_bin   = (valid > 0.5).float()
        invalid_bin = (invalid > 0.5).float()
        gnn_results[L] = {
            'valid': valid_bin, 'invalid': invalid_bin,
            'v_viols': local_violations(valid_bin),
            'i_viols': local_violations(invalid_bin),
        }
        n_pix = 2*L**2 + 2*L + 1
        print(f'GNN L={L} ({n_pix}px diamond): '
              f'valid={np.mean(gnn_results[L]["v_viols"]):.1f}  '
              f'invalid={np.mean(gnn_results[L]["i_viols"]):.1f}')

                                            
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, (models_dict, label, L_vals) in zip(axes, [
        (cnn_results, 'CNN (square RF = 2L+5)',  L_VALUES),
        (gnn_results, 'GNN (diamond RF, L-hop)', L_GNN),
    ]):
        x = np.arange(len(L_vals))
        ax.bar(x - 0.18, [np.mean(models_dict[L]['v_viols']) for L in L_vals],
               0.35, label='valid (T=400)', color='steelblue')
        ax.bar(x + 0.18, [np.mean(models_dict[L]['i_viols']) for L in L_vals],
               0.35, label='invalid (T=3)', color='coral')
        ax.set_xticks(x); ax.set_xticklabels([f'L={l}' for l in L_vals])
        ax.set_title(label); ax.legend()
        ax.set_ylabel('Mean violations')
    fig.suptitle('CNN vs GNN — masking diffusion, 24×24 toroidal', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'cnn_vs_gnn.png'), dpi=150, bbox_inches='tight')
    plt.show()

                                                                               
                                                                           

RUN_MNIST = False

if RUN_MNIST:
    print('\n' + '='*60)
    print('SECTION 8 — MNIST 32×32 binary (optional)')
    print('='*60)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])
    mnist_ds     = torchvision.datasets.MNIST('./data', train=True,
                                               download=True, transform=transform)
    mnist_loader = DataLoader(mnist_ds, batch_size=128, shuffle=True, num_workers=2)
    print(f'MNIST: {len(mnist_ds)} images (32×32 binary)')

    mnist_model = MinimalResNet(
        channels=1, emb_dim=256, mode='circular',
        num_layers=6, kernel_size=3, lastksize=3,
    ).to(device)
    print(f'Params: {sum(p.numel() for p in mnist_model.parameters()):,}')

    print('Training MNIST CNN (~2 hrs on T4)...')
    mnist_losses = train_discrete(
        mnist_model, mnist_loader, device,
        num_epochs=150, lr=1e-4, gamma=0.999965,
        fname=os.path.join(SAVE_DIR, 'mnist_cnn_L6_circular'),
        save_every=10, verbose=True,
    )

                                                           
    mnist_subset = Subset(mnist_ds, list(range(1000)))
    mnist_els    = DiscreteELS(
        mnist_subset, kernel_size=3, batch_size=64,
        image_size=32, channels=1, schedule=cosine_alpha_bar,
        max_samples=1000,
    )
    mnist_machine = ELSMachine(mnist_els, in_channels=1, imsize=32,
                                schedule=cosine_alpha_bar, nsteps=20)

    N_EVAL = 30
    print(f'\nComputing r² on {N_EVAL} samples...')
    r2_mnist = []
    mnist_model.eval()

    for i in range(N_EVAL):
        seed = 2000 + i
        torch.manual_seed(seed)
        els_sample = mnist_machine.sample(nsteps=20, device=device)
        torch.manual_seed(seed)
        cnn_sample = full_reverse(
            mnist_model, nsteps=20, batch_size=1,
            in_channels=1, image_size=32,
            device=device, stochastic=False,
        )
        r2_mnist.append(r2_score(cnn_sample.cpu(), els_sample.cpu()))
        if (i + 1) % 5 == 0:
            print(f'  {i+1}/{N_EVAL}  running median r²={np.median(r2_mnist):.3f}')

    print(f'\nMNIST  median r² = {np.median(r2_mnist):.4f}')
    print(f'       mean   r² = {np.mean(r2_mnist):.4f}')
    print(f'(continuous paper: ~0.94 on MNIST)')

                  
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(r2_mnist, bins=15, range=(0, 1), color='steelblue', edgecolor='white')
    ax.axvline(np.median(r2_mnist), color='red', ls='--',
               label=f'median={np.median(r2_mnist):.3f}')
    ax.set_xlabel('r²  (CNN vs discrete ELS)')
    ax.set_ylabel('Count')
    ax.set_title('MNIST 32×32 — discrete masking diffusion')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'mnist_r2.png'), dpi=150, bbox_inches='tight')
    plt.show()

print('\nAll done.')
