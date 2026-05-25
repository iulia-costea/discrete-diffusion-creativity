import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/dl2_gnn/'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f'Saving to: {SAVE_DIR}')

                                                                                
H_TOY  = W_TOY = 24
L_GNN  = [1, 2, 4, 8, 10, 12, 16]                                       
EPOCHS = 100
EMB    = 256
LR     = 1e-4
MASK_VALUE = 0.5

                                                         
CNN_L      = [3, 6, 8]
CNN_VALID  = [11.2, 15.0, 20.5]
CNN_INVALID= [71.0, 44.5, 91.0]

                                                                                
def cosine_alpha_bar(t):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    return torch.cos(t.clamp(0., 1.) * math.pi / 2.).pow(2)

def mask_forward(x0, t):
    alpha_t = cosine_alpha_bar(t).to(x0.device).view(-1, 1, 1, 1)
    keep    = torch.bernoulli(alpha_t.expand_as(x0)).bool()
    return torch.where(keep, x0, torch.full_like(x0, MASK_VALUE)), keep

def mask_reverse_step(x_t, x0_pred, t_val, t_prev_val, stochastic=True):
    device     = x_t.device
    alpha_t    = cosine_alpha_bar(torch.tensor(t_val)).to(device)
    alpha_prev = cosine_alpha_bar(torch.tensor(t_prev_val)).to(device)
    x_prev     = x_t.clone()
    masked     = (x_t == MASK_VALUE)
    if stochastic:
        p = ((alpha_prev - alpha_t) / (1. - alpha_t + 1e-8)).clamp(0., 1.)
        reveal = torch.bernoulli(torch.full_like(x0_pred, p.item())).bool()
        x_prev[masked & reveal] = torch.bernoulli(x0_pred[masked & reveal])
    else:
        x0_bin = (x0_pred > 0.5).float()
        keep   = torch.bernoulli(
            torch.full_like(x0_pred, alpha_prev.clamp(0., 1.).item())
        ).bool()
        x_prev[masked & keep] = x0_bin[masked & keep]
    return x_prev

def full_reverse(model, nsteps=400, image_size=24, stochastic=True, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_t = torch.full((1, 1, image_size, image_size), MASK_VALUE, device=device)
    ts  = torch.linspace(1., 0., nsteps + 1)
    model.eval()
    with torch.no_grad():
        for i in range(nsteps):
            tt      = torch.full((1,), ts[i].item(), device=device)
            x0_pred = torch.sigmoid(model(tt, x_t))
            x_t     = mask_reverse_step(x_t, x0_pred, ts[i].item(),
                                         ts[i+1].item(), stochastic=stochastic)
    return x_t

                                                                                
class EmbeddingModule(nn.Module):
    def __init__(self, fdim):
        super().__init__(); self.fdim = fdim
    def forward(self, t, label=None):
        d    = self.fdim // 2
        freq = 10000 ** (torch.arange(d, device=t.device) / max(d-1, 1))
        args = t[:, None] / freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

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
        return self.act(self.lin_self(x) + self.lin_neigh(agg / deg))

class MinimalResGNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, emb_dim=256, num_layers=6):
        super().__init__()
        self.num_layers      = num_layers
        self.embedding       = EmbeddingModule(emb_dim)
        self.up_projection   = nn.Linear(in_channels, emb_dim)
        self.embs = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())
            for _ in range(num_layers + 1)
        ])
        self.layers = nn.ModuleList([MPNNLayer(emb_dim) for _ in range(num_layers)])
        self.down_projection = nn.Linear(emb_dim, out_channels)

    def forward(self, t, x_img, label=None):
                                                          
        B, C, H, W = x_img.shape
        x     = x_img.permute(0, 2, 3, 1).reshape(B, H*W, C)
        emb   = self.embedding(t)
        state = self.up_projection(x)
        for i in range(self.num_layers):
            tb    = self.embs[i](emb).unsqueeze(1)
            delta = self.layers[i](state + tb, self.edge_index)
            state = state + delta
        state = state + self.embs[-1](emb).unsqueeze(1)
        out   = self.down_projection(state)                     
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)                

    def set_edge_index(self, edge_index):
        self.edge_index = edge_index

                                                                                
def train(model, loader, device, num_epochs=100, lr=1e-4, fname=None):
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, num_epochs + 1):
        ep_loss = 0.0
        for batch in loader:
            x0 = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            B  = x0.shape[0]
            t  = torch.rand(B, device=device)
            x_t, unmasked = mask_forward(x0, t)
            logits = model(t, x_t)
            masked = ~unmasked
            loss   = (F.binary_cross_entropy_with_logits(logits[masked], x0[masked])
                      if masked.any()
                      else F.binary_cross_entropy_with_logits(logits, x0))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss = loss.item()
        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}/{num_epochs}  loss={ep_loss:.4f}')
    if fname:
        torch.save(model.state_dict(), fname)
        print(f'  Saved → {fname}')

                                                                                
def local_violations(img_tensor, periodic=True):
    img = img_tensor.cpu().float()
    if img.dim() == 4: img = img[:, 0]
    if img.dim() == 2: img = img.unsqueeze(0)
    B, H, W = img.shape
    result = []
    for b in range(B):
        im = img[b]; count = 0
        for i in range(H):
            for j in range(W):
                if periodic:
                    patch = [im[(i+di)%H, (j+dj)%W].item()
                             for di in range(-1,2) for dj in range(-1,2)]
                else:
                    patch = im[max(0,i-1):i+2, max(0,j-1):j+2].flatten().tolist()
                majority = 1.0 if sum(patch) >= len(patch)/2 else 0.0
                if majority != im[i, j].item(): count += 1
        result.append(count)
    return result

def draw_pixel_image(ax, img_tensor, show_violations=True, periodic=True):
    img = img_tensor.cpu().float()
    if img.dim() == 3: img = img[0]
    H, W = img.shape
    ax.imshow(img.numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='#4466cc', linewidth=0.45, alpha=0.9)
    ax.tick_params(which='both', length=0, labelsize=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    n_viol = 0
    if show_violations:
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                v = img[i, j].item()
                if v == MASK_VALUE: continue
                if periodic:
                    patch = [img[(i+di)%H, (j+dj)%W].item()
                             for di in range(-1,2) for dj in range(-1,2)]
                else:
                    patch = img[max(0,i-1):i+2, max(0,j-1):j+2].flatten().tolist()
                majority = 1.0 if sum(patch) >= len(patch)/2 else 0.0
                if majority != v:
                    overlay[i, j] = [1, 0.08, 0.08, 0.85]; n_viol += 1
        if n_viol: ax.imshow(overlay, interpolation='nearest')
    return n_viol

                                                                                
black   = torch.zeros(1, H_TOY, W_TOY)
white   = torch.ones(1,  H_TOY, W_TOY)
X_toy   = torch.stack([black, white])
toy_ds  = TensorDataset(X_toy, torch.tensor([0, 1]))
sampler = RandomSampler(toy_ds, replacement=True, num_samples=6400)
toy_loader = DataLoader(toy_ds, batch_size=64, sampler=sampler)
edge_index = make_grid_edges(H_TOY, W_TOY).to(device)
print(f'Dataset: {len(toy_ds)} images | {len(toy_loader)} batches/epoch')

                                                                                
def diamond_rf(L):
    return 2 * L**2 + 2 * L + 1

print('\nGNN receptive fields:')
for L in L_GNN:
    print(f'  L={L:2d}: {diamond_rf(L):4d} pixels ({100*diamond_rf(L)/(H_TOY*H_TOY):.0f}% of image)')

gnn_models  = {}
gnn_results = {}

for L in L_GNN:
    drf = diamond_rf(L)
    pct = 100 * drf / (H_TOY * H_TOY)
    print(f'\n=== GNN  L={L}  RF={drf} pixels ({pct:.0f}%) ===')

    gnn = MinimalResGNN(in_channels=1, out_channels=1, emb_dim=EMB, num_layers=L).to(device)
    gnn.set_edge_index(edge_index)
    print(f'  Params: {sum(p.numel() for p in gnn.parameters()):,}')

    train(gnn, toy_loader, device, num_epochs=EPOCHS, lr=LR,
          fname=os.path.join(SAVE_DIR, f'gnn_L{L}_final.pt'))

                      
    valid_samples, invalid_samples = [], []
    for i in range(16):
        torch.manual_seed(42 + i)
        v = full_reverse(gnn, nsteps=400, image_size=H_TOY, device=device)
        torch.manual_seed(42 + i)
        inv = full_reverse(gnn, nsteps=3,   image_size=H_TOY, device=device)
        valid_samples.append((v > 0.5).float().cpu())
        invalid_samples.append((inv > 0.5).float().cpu())

    valid_t   = torch.cat(valid_samples)
    invalid_t = torch.cat(invalid_samples)
    v_viols   = local_violations(valid_t)
    i_viols   = local_violations(invalid_t)

    gnn_results[L] = {
        'valid': valid_t, 'invalid': invalid_t,
        'v_viols': v_viols, 'i_viols': i_viols,
    }
    gnn_models[L] = gnn
    print(f'  Valid (T=400): {np.mean(v_viols):.1f}   Invalid (T=3): {np.mean(i_viols):.1f}')

                                                                                
print('\n' + '='*65)
print(f'{"L":>4}  {"RF (px)":>8}  {"Coverage":>10}  {"Valid (T=400)":>14}  {"Invalid (T=3)":>14}')
print('-'*65)
for L in L_GNN:
    drf = diamond_rf(L)
    pct = 100 * drf / (H_TOY * H_TOY)
    vv  = np.mean(gnn_results[L]['v_viols'])
    iv  = np.mean(gnn_results[L]['i_viols'])
    print(f'{L:>4}  {drf:>8}  {pct:>9.0f}%  {vv:>14.1f}  {iv:>14.1f}')
print('='*65)

                                                                                
for L in L_GNN:
    res = gnn_results[L]
    drf = diamond_rf(L)
    pct = 100 * drf / (H_TOY * H_TOY)

    ratios = [1, 1, 0.25] + [1]*4 + [0.25] + [1]*2
    fig    = plt.figure(figsize=(14, 2.4))
    gs     = gridspec.GridSpec(1, len(ratios), figure=fig,
                               width_ratios=ratios, wspace=0.04)
    train_axes   = [fig.add_subplot(gs[0, i])     for i in [0, 1]]
    valid_axes   = [fig.add_subplot(gs[0, 3 + i]) for i in range(4)]
    invalid_axes = [fig.add_subplot(gs[0, 8 + i]) for i in range(2)]
    fig.add_subplot(gs[0, 2]).axis('off')
    fig.add_subplot(gs[0, 7]).axis('off')

    draw_pixel_image(train_axes[0], torch.zeros(1, H_TOY, H_TOY), show_violations=False)
    draw_pixel_image(train_axes[1], torch.ones(1,  H_TOY, H_TOY), show_violations=False)
    v_v = [draw_pixel_image(ax, res['valid'][i, 0:1])   for i, ax in enumerate(valid_axes)]
    i_v = [draw_pixel_image(ax, res['invalid'][i, 0:1]) for i, ax in enumerate(invalid_axes)]

    cx_t = np.mean([ax.get_position().x0 + ax.get_position().x1 for ax in train_axes])   / 2
    cx_v = np.mean([ax.get_position().x0 + ax.get_position().x1 for ax in valid_axes])   / 2
    cx_i = np.mean([ax.get_position().x0 + ax.get_position().x1 for ax in invalid_axes]) / 2
    fig.text(cx_t, 1.02, 'Training Data',         ha='center', va='bottom',
             fontsize=12, fontweight='bold', transform=fig.transFigure)
    fig.text(cx_v, 1.02, f'Valid (T=400)   mean viol. = {np.mean(v_v):.1f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             transform=fig.transFigure)
    fig.text(cx_i, 1.02, f'Invalid (T=3)   mean viol. = {np.mean(i_v):.1f}',
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             transform=fig.transFigure)
    fig.suptitle(f'GNN  L={L}  RF={drf} pixels ({pct:.0f}%)  — masked diffusion',
                 fontsize=9, y=1.14, color='#333333')

    path = os.path.join(SAVE_DIR, f'samples_gnn_L{L}.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.show(); print(f'→ {path}')

                                                                                
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

     
ax = axes[0]
rf_cnn = [2*L+5 for L in CNN_L]
ax.plot(rf_cnn, CNN_VALID,   'o-', color='#2d6a9f', lw=2, ms=7, label='Valid (T=400)')
ax.plot(rf_cnn, CNN_INVALID, 's--', color='#c0392b', lw=2, ms=7, label='Invalid (T=3)')
ax.set_xticks(rf_cnn)
ax.set_xticklabels([f'{r}\n(L={l})' for r, l in zip(rf_cnn, CNN_L)], fontsize=9)
ax.set_xlabel('Receptive field width (pixels)', fontsize=11)
ax.set_ylabel('Mean local consistency violations', fontsize=11)
ax.set_title('CNN  (square RF = 2L+5)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)

     
ax = axes[1]
drf_gnn  = [diamond_rf(L) for L in L_GNN]
v_gnn    = [np.mean(gnn_results[L]['v_viols'])   for L in L_GNN]
iv_gnn   = [np.mean(gnn_results[L]['i_viols'])   for L in L_GNN]
ax.plot(drf_gnn, v_gnn,  'o-',  color='#2d6a9f', lw=2, ms=7, label='Valid (T=400)')
ax.plot(drf_gnn, iv_gnn, 's--', color='#c0392b', lw=2, ms=7, label='Invalid (T=3)')
ax.set_xticks(drf_gnn)
ax.set_xticklabels([f'{r}\n(L={l})' for r, l in zip(drf_gnn, L_GNN)], fontsize=8)
ax.set_xlabel('Receptive field size (pixels in diamond)', fontsize=11)
ax.set_title('GNN  (diamond RF = 2L²+2L+1)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.spines[['top','right']].set_visible(False)

fig.suptitle('CNN vs GNN — discrete masked diffusion\n'
             '24×24 toroidal binary images', fontsize=12)
plt.tight_layout()
path = os.path.join(SAVE_DIR, 'cnn_vs_gnn.png')
fig.savefig(path, dpi=180, bbox_inches='tight')
plt.show(); print(f'→ {path}')

print(f'\nAll done. Files saved to {SAVE_DIR}')
