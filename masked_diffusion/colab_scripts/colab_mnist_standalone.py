import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

                                        
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/dl2_mnist/'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f'Saving to: {SAVE_DIR}')

                                                                                
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
                 image_size=32, schedule=cosine_alpha_bar,
                 stochastic=False, device=None, label=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.full((batch_size, in_channels, image_size, image_size),
                     MASK_VALUE, device=device)
    ts  = torch.linspace(1.0, 0.0, nsteps + 1)
    for i in range(nsteps):
        t_val      = ts[i].item()
        t_prev_val = ts[i + 1].item()
        t_tensor   = torch.full((batch_size,), t_val, device=device)
        with torch.no_grad():
            x0_pred = torch.sigmoid(model(t_tensor, x_t, label=label))
        x_t = mask_reverse_step(x_t, x0_pred, t_val, t_prev_val,
                                 schedule=schedule, stochastic=stochastic)
    return x_t

                                                                                
class EmbeddingModule(nn.Module):
    def __init__(self, fdim, conditional=False, num_classes=None):
        super().__init__()
        self.fdim        = fdim
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
        self.num_layers    = num_layers
        self.embedding     = EmbeddingModule(emb_dim, conditional, num_classes)
        self.up_projection = nn.Conv2d(channels, emb_dim, kernel_size,
                                       padding='same', padding_mode=mode)
        self.embs = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim),
                          nn.GroupNorm(8, emb_dim), nn.ReLU())
            for _ in range(num_layers + 1)
        ])
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(emb_dim, emb_dim, kernel_size,
                                    padding='same', padding_mode=mode), nn.ReLU())
            for _ in range(num_layers)
        ])
        self.down_projection = nn.Conv2d(emb_dim, channels, lastksize,
                                          padding='same', padding_mode=mode)

    def forward(self, t, x, label=None):
        emb   = self.embedding(t, label=label)
        state = self.up_projection(x)
        for i in range(self.num_layers):
            delta = self.convs[i](state + self.embs[i](emb)[:, :, None, None])
            state = state + delta
        state = state + self.embs[-1](emb)[:, :, None, None]
        return self.down_projection(state)

                                                                                
def train_discrete(model, loader, device, schedule=cosine_alpha_bar,
                   num_epochs=150, lr=1e-4, gamma=0.999965,
                   fname=None, save_every=10, verbose=True):
    model.train()
    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    log   = []
    for epoch in range(1, num_epochs + 1):
        ep_loss = 0.0
        for batch in loader:
            x0  = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            B   = x0.shape[0]
            t   = torch.rand(B, device=device)
            x_t, unmasked = mask_forward(x0, t, schedule)
            logits = model(t, x_t)
            masked = ~unmasked
            loss   = (F.binary_cross_entropy_with_logits(logits[masked], x0[masked])
                      if masked.any()
                      else F.binary_cross_entropy_with_logits(logits, x0))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            ep_loss = loss.item()
        log.append(ep_loss)
        if verbose and epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/{num_epochs}  loss={ep_loss:.4f}')
        if fname and epoch % save_every == 0:
            torch.save(model.state_dict(), f'{fname}_ep{epoch}.pt')
            print(f'  Checkpoint saved (epoch {epoch})')
    if fname:
        torch.save(model.state_dict(), f'{fname}_final.pt')
        print(f'Final model saved → {fname}_final.pt')
    return log

                                                                                
def _circ_conv(x, filters):
    k  = filters.shape[-1]
    d  = k // 2
    xp = F.pad(x, (d, d, d, d), mode='circular')
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
        k   = self.k
        d   = k // 2
        x_t = x_t.to(device)

        if unmasked is None:
            unmasked = (x_t != MASK_VALUE)
        m     = unmasked.float()
        x_obs = x_t * m

        b, c, h, w = x_t.shape
        alpha_t = self.schedule(t).to(device)
        log_T   = (-torch.log(alpha_t.clamp(min=1e-8))).mean().item() * (k ** 2)

        xobs_pad = F.pad(x_obs, (d, d, d, d), mode='circular')
        m_pad    = F.pad(m,     (d, d, d, d), mode='circular')

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

            img_pad = F.pad(images, (d, d, d, d), mode='circular')
            pf = F.unfold(img_pad.view(N*c, 1, h+2*d, w+2*d),
                          k, stride=1, padding=0).view(N, c, k*k, h*w)
            patches  = pf.permute(0, 3, 1, 2).reshape(N*h*w, c, k, k)
            pcenters = patches[:, :, k//2, k//2]

            pdotx  = _circ_conv(x_obs, patches)
            mphi_n = _circ_conv(m,     patches ** 2)

            exp_args = -log_T * (xnorms[:, None] - 2*pdotx + mphi_n)

            new_sub = exp_args.amax(dim=1)
            if subtraction is None:
                subtraction = new_sub
            else:
                delta = torch.max(subtraction, new_sub)
                numerator   /= torch.exp(delta - subtraction)[:, None]
                denominator /= torch.exp(delta - subtraction)
                subtraction  = delta

            w_vals = torch.exp(exp_args - subtraction[:, None])
            pc     = pcenters.permute(1, 0).view(1, N*h*w, c, 1, 1)
            numerator   += (w_vals[:, :, None] * pc).sum(1)
            denominator += w_vals.sum(1)

        return numerator / denominator[:, None].clamp(min=1e-12)

class ELSMachine(nn.Module):
    def __init__(self, backbone, in_channels=1, imsize=32,
                 schedule=cosine_alpha_bar, nsteps=20):
        super().__init__()
        self.backbone = backbone
        self.C        = in_channels
        self.H        = imsize
        self.schedule = schedule
        self.nsteps   = nsteps

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

                                                                                
def r2_score(a, b):
    def norm(x):
        x   = x.float().flatten()
        x   = x - x.mean()
        nrm = x.norm()
        return x / nrm if nrm > 1e-8 else x
    return torch.dot(norm(a), norm(b)).item() ** 2

                                                                                
print('\n' + '='*60)
print('PART 1 — Train MNIST CNN (~2 hrs on T4)')
print('='*60)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float()),
])
mnist_ds     = torchvision.datasets.MNIST('./data', train=True,
                                           download=True, transform=transform)
mnist_loader = DataLoader(mnist_ds, batch_size=128, shuffle=True, num_workers=0)
print(f'MNIST: {len(mnist_ds)} images  (32×32 binary)')

mnist_model = MinimalResNet(
    channels=1, emb_dim=256, mode='circular',
    num_layers=6, kernel_size=3, lastksize=3,
).to(device)
print(f'Params: {sum(p.numel() for p in mnist_model.parameters()):,}')

mnist_losses = train_discrete(
    mnist_model, mnist_loader, device,
    num_epochs=150, lr=1e-4, gamma=0.999965,
    fname=os.path.join(SAVE_DIR, 'mnist_cnn_L6'),
    save_every=10, verbose=True,
)

                
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(mnist_losses, color='steelblue', lw=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('BCE loss')
ax.set_title('MNIST discrete CNN — training loss')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(SAVE_DIR, 'mnist_loss_curve.png'), dpi=150)
plt.show()
print(f'Final loss: {mnist_losses[-1]:.4f}')

                                                                               
                                                                                  
 
                                                                         
                                                                                    
                                
                                                                            
                      

print('\n' + '='*60)
print('PART 2 — ELS generation + r² (~20 min on T4)')
print('='*60)

mnist_subset  = Subset(mnist_ds, list(range(1000)))
mnist_els_mod = DiscreteELS(
    mnist_subset, kernel_size=3, batch_size=64,
    image_size=32, channels=1,
    schedule=cosine_alpha_bar, max_samples=1000,
)
mnist_machine = ELSMachine(
    mnist_els_mod, in_channels=1, imsize=32,
    schedule=cosine_alpha_bar, nsteps=20,
)

N_EVAL    = 50
r2_scores = []
mnist_model.eval()
print(f'Evaluating {N_EVAL} samples...\n')

for i in range(N_EVAL):
    seed = 3000 + i
    torch.manual_seed(seed)
    els_out = mnist_machine.sample(nsteps=20, device=device)

    torch.manual_seed(seed)
    cnn_out = full_reverse(
        mnist_model, nsteps=20, batch_size=1,
        in_channels=1, image_size=32,
        device=device, stochastic=False,
    )
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

                                                                                
print('\nGenerating figures...')

              
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(r2_scores, bins=20, range=(0, 1), color='#2d6a9f', edgecolor='white')
ax.axvline(np.median(r2_scores), color='red', ls='--', lw=1.8,
           label=f'median = {np.median(r2_scores):.3f}')
ax.set_xlabel('r²  (discrete CNN vs discrete ELS)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('MNIST 32×32 — discrete masked diffusion\n'
             'Does the analytic ELS predict the trained CNN?', fontsize=11)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
path = os.path.join(SAVE_DIR, 'mnist_r2_histogram.png')
fig.savefig(path, dpi=180, bbox_inches='tight')
plt.show()
print(f'→ {path}')

                                                 
n_show = 6
fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.2, 5),
                          gridspec_kw={'hspace': 0.08, 'wspace': 0.04})
mnist_model.eval()
for col in range(n_show):
    seed = 3000 + col
    torch.manual_seed(seed)
    els_s = mnist_machine.sample(nsteps=20, device=device)
    torch.manual_seed(seed)
    cnn_s = full_reverse(mnist_model, nsteps=20, batch_size=1,
                          in_channels=1, image_size=32,
                          device=device, stochastic=False)
    r2 = r2_score(cnn_s.cpu(), els_s.cpu())

    for row, (img, lbl) in enumerate([(els_s, 'ELS\n(analytic)'),
                                       (cnn_s, 'CNN\n(trained)')]):
        ax = axes[row, col]
        ax.imshow(img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f'r²={r2:.2f}', fontsize=9)
        if col == 0:
            ax.set_ylabel(lbl, fontsize=9)

fig.suptitle('MNIST sample pairs — discrete ELS (top) vs CNN (bottom)\n'
             'same fully-masked seed, masked diffusion', fontsize=11)
plt.tight_layout()
path = os.path.join(SAVE_DIR, 'mnist_sample_pairs.png')
fig.savefig(path, dpi=180, bbox_inches='tight')
plt.show()
print(f'→ {path}')

print(f'\nAll done. Files saved to {SAVE_DIR}')
