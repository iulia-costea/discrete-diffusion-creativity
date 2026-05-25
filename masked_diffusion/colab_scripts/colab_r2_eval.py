import math, os, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision, torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

MASK_VALUE   = 0.5
CHECKPOINT   = '/content/mnist_cnn_L6_final.pt'                                
N_EVAL       = 30                                                           
SAVE_DIR     = '/content/'

                                                                                

def cosine_alpha_bar(t):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    return torch.cos(t.clamp(0., 1.) * math.pi / 2.).pow(2)

def mask_reverse_step(x_t, x0_pred, t_val, t_prev_val, stochastic=True):
    device     = x_t.device
    alpha_t    = cosine_alpha_bar(torch.tensor(t_val)).to(device)
    alpha_prev = cosine_alpha_bar(torch.tensor(t_prev_val)).to(device)
    x_prev     = x_t.clone()
    masked     = (x_t == MASK_VALUE)
    if stochastic:
        p_reveal = ((alpha_prev - alpha_t) / (1. - alpha_t + 1e-8)).clamp(0., 1.)
        reveal   = torch.bernoulli(torch.full_like(x0_pred, p_reveal.item())).bool()
        x_prev[masked & reveal] = torch.bernoulli(x0_pred[masked & reveal])
    else:
        x0_bin = (x0_pred > 0.5).float()
        keep   = torch.bernoulli(
            torch.full_like(x0_pred, alpha_prev.clamp(0., 1.).item())
        ).bool()
        x_prev[masked & keep] = x0_bin[masked & keep]
    return x_prev

def full_reverse(model, nsteps=20, image_size=32, in_channels=1,
                 stochastic=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.full((1, in_channels, image_size, image_size), MASK_VALUE, device=device)
    ts  = torch.linspace(1., 0., nsteps + 1)
    for i in range(nsteps):
        tt = torch.full((1,), ts[i].item(), device=device)
        with torch.no_grad():
            x0_pred = torch.sigmoid(model(tt, x_t))
        x_t = mask_reverse_step(x_t, x0_pred, ts[i].item(), ts[i+1].item(),
                                  stochastic=stochastic)
    return x_t

def r2_score(a, b):
    def norm(x):
        x   = x.float().flatten()
        x   = x - x.mean()
        nrm = x.norm()
        return x / nrm if nrm > 1e-8 else x
    return torch.dot(norm(a), norm(b)).item() ** 2

                                                                                

class EmbeddingModule(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.fdim = fdim

    def forward(self, t, label=None):
        d    = self.fdim // 2
        freq = 10000 ** (torch.arange(d, device=t.device) / max(d - 1, 1))
        args = t[:, None] / freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

class MinimalResNet(nn.Module):
    def __init__(self, channels=1, emb_dim=256, mode='circular',
                 num_layers=6, lastksize=3):
        super().__init__()
        self.num_layers    = num_layers
        self.embedding     = EmbeddingModule(emb_dim)
        self.up_projection = nn.Conv2d(channels, emb_dim, 3,
                                        padding='same', padding_mode=mode)
        self.embs = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim),
                          nn.GroupNorm(8, emb_dim), nn.ReLU())
            for _ in range(num_layers + 1)
        ])
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(emb_dim, emb_dim, 3,
                                     padding='same', padding_mode=mode), nn.ReLU())
            for _ in range(num_layers)
        ])
        self.down_projection = nn.Conv2d(emb_dim, channels, lastksize,
                                          padding='same', padding_mode=mode)

    def forward(self, t, x, label=None):
        emb   = self.embedding(t)
        state = self.up_projection(x)
        for i in range(self.num_layers):
            state = state + self.convs[i](state + self.embs[i](emb)[:, :, None, None])
        return self.down_projection(state + self.embs[-1](emb)[:, :, None, None])

                                                                                

def _circ_conv(x, filters):
    k = filters.shape[-1]
    d = k // 2
    return F.conv2d(F.pad(x, (d, d, d, d), mode='circular'), filters, padding=0)

class DiscreteELS(nn.Module):
    def __init__(self, dataset, kernel_size=3, batch_size=64,
                 image_size=32, channels=1, max_samples=None):
        super().__init__()
        self.loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.k           = kernel_size
        self.H           = image_size
        self.C           = channels
        self.max_samples = max_samples

    def forward(self, t, x_t, unmasked=None, device=None, **kw):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k = self.k; d = k // 2
        x_t = x_t.to(device)
        if unmasked is None:
            unmasked = (x_t != MASK_VALUE)
        m     = unmasked.float()
        x_obs = x_t * m
        b, c, h, w = x_t.shape

        log_T = (-torch.log(
            cosine_alpha_bar(t).to(device).clamp(min=1e-8)
        )).mean().item() * (k ** 2)

        def lss(tp):
            u = F.unfold(tp.view(b*c, 1, h+2*d, w+2*d), k, stride=1, padding=0)
            return u.view(b, c, k*k, h*w).sum(2).view(b, c, h, w)

        xnorms = lss(F.pad(x_obs, (d,d,d,d), mode='circular') ** 2).sum(1)
        num = torch.zeros(b, c, h, w, device=device)
        den = torch.zeros(b, h, w,    device=device)
        sub = None; n = 0

        for imgs, _ in self.loader:
            if self.max_samples and n >= self.max_samples:
                break
            imgs = imgs.to(device); N = imgs.shape[0]; n += N
            pf = F.unfold(
                F.pad(imgs, (d,d,d,d), mode='circular').view(N*c, 1, h+2*d, w+2*d),
                k, stride=1, padding=0
            ).view(N, c, k*k, h*w)
            patches  = pf.permute(0,3,1,2).reshape(N*h*w, c, k, k)
            pc       = patches[:, :, k//2, k//2]
            ea       = -log_T * (xnorms[:,None]
                                 - 2*_circ_conv(x_obs, patches)
                                 + _circ_conv(m, patches**2))
            ns = ea.amax(1)
            if sub is None:
                sub = ns
            else:
                d2  = torch.max(sub, ns)
                num /= torch.exp(d2 - sub)[:, None]
                den /= torch.exp(d2 - sub)
                sub  = d2
            w2   = torch.exp(ea - sub[:, None])
            num += (w2[:,:,None] * pc.permute(1,0).view(1,N*h*w,c,1,1)).sum(1)
            den += w2.sum(1)

        return num / den[:, None].clamp(min=1e-12)

class ELSMachine(nn.Module):
    def __init__(self, backbone, in_channels=1, imsize=32, nsteps=20):
        super().__init__()
        self.backbone = backbone
        self.C        = in_channels
        self.H        = imsize
        self.nsteps   = nsteps

    @torch.no_grad()
    def sample(self, nsteps=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ns  = nsteps or self.nsteps
        x_t = torch.full((1, self.C, self.H, self.H), MASK_VALUE, device=device)
        ts  = torch.linspace(1., 0., ns + 1)
        for i in range(ns):
            tt  = torch.full((1,), ts[i].item(), device=device)
            unm = (x_t != MASK_VALUE)
            p0  = self.backbone(tt, x_t, unmasked=unm, device=device)
            x_t = mask_reverse_step(x_t, p0, ts[i].item(), ts[i+1].item(),
                                     stochastic=True)
        return x_t

                                                                                

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
])
mnist_ds     = torchvision.datasets.MNIST('./data', train=True,
                                           download=True, transform=transform)
mnist_subset = Subset(mnist_ds, list(range(1000)))
els_mod      = DiscreteELS(mnist_subset, kernel_size=3, batch_size=64,
                            image_size=32, channels=1, max_samples=1000)
els_machine  = ELSMachine(els_mod, in_channels=1, imsize=32, nsteps=20)

mnist_model = MinimalResNet(channels=1, emb_dim=256,
                             mode='circular', num_layers=6).to(device)
mnist_model.load_state_dict(
    torch.load(CHECKPOINT, map_location=device)
)
mnist_model.eval()
print('Model loaded (150 epochs).')

                                                                                

r2_scores = []
print(f'\nComputing r² on {N_EVAL} samples...\n')

for i in range(N_EVAL):
    seed = 3000 + i
    torch.manual_seed(seed)
    els_out = els_machine.sample(nsteps=20, device=device)
    torch.manual_seed(seed)
    cnn_out = full_reverse(mnist_model, nsteps=20, image_size=32,
                            device=device, stochastic=True)
    r2_scores.append(r2_score(cnn_out.cpu(), els_out.cpu()))
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/{N_EVAL}  median r² = {np.median(r2_scores):.3f}')

print(f'\n{"="*40}')
print(f'Median r² = {np.median(r2_scores):.4f}')
print(f'Mean   r² = {np.mean(r2_scores):.4f}')
print(f'Min    r² = {np.min(r2_scores):.4f}')
print(f'Max    r² = {np.max(r2_scores):.4f}')
print(f'{"="*40}')
print('(continuous paper baseline: ~0.94–0.96 on MNIST)')

                                                                                

              
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(r2_scores, bins=20, range=(0, 1), color='#2d6a9f', edgecolor='white')
ax.axvline(np.median(r2_scores), color='red', ls='--', lw=1.8,
           label=f'median = {np.median(r2_scores):.3f}')
ax.set_xlabel('r²  (discrete CNN vs discrete ELS)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('MNIST 32×32 — discrete masked diffusion\n'
             'Analytic ELS vs trained CNN', fontsize=11)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'mnist_r2_histogram.png'), dpi=180, bbox_inches='tight')
plt.show()

              
n_show = 6
fig, axes = plt.subplots(2, n_show, figsize=(n_show*2.2, 5),
                          gridspec_kw={'hspace': 0.08, 'wspace': 0.04})
for col in range(n_show):
    seed = 3000 + col
    torch.manual_seed(seed)
    els_s = els_machine.sample(nsteps=20, device=device)
    torch.manual_seed(seed)
    cnn_s = full_reverse(mnist_model, nsteps=20, image_size=32,
                          device=device, stochastic=True)
    r2 = r2_score(cnn_s.cpu(), els_s.cpu())
    for row, (img, lbl) in enumerate([(els_s, 'ELS\n(analytic)'),
                                       (cnn_s, 'CNN\n(trained)')]):
        ax = axes[row, col]
        ax.imshow(img[0,0].cpu().numpy(), cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(f'r²={r2:.2f}', fontsize=9)
        if col == 0: ax.set_ylabel(lbl, fontsize=9)
fig.suptitle('MNIST — discrete ELS (top) vs CNN (bottom), same seed', fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'mnist_sample_pairs.png'), dpi=180, bbox_inches='tight')
plt.show()

print(f'\nFigures saved to {SAVE_DIR}')
