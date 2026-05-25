
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

from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/dl2_gnn/'
EVAL_DIR = os.path.join(SAVE_DIR, '4conn_eval/')
os.makedirs(EVAL_DIR, exist_ok=True)
print(f'Loading models from: {SAVE_DIR}')
print(f'Saving figures to:   {EVAL_DIR}')

                                                                                
H_TOY  = W_TOY = 24
L_GNN  = [1, 2, 4, 8, 10, 12, 16]
EMB    = 256
MASK_VALUE = 0.5

                                                                                
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

def local_violations_4conn(img_tensor):
    img = img_tensor.cpu().float()
    if img.dim() == 4: img = img[:, 0]
    if img.dim() == 2: img = img.unsqueeze(0)
    B, H, W = img.shape
    result = []
    for b in range(B):
        im = img[b]; count = 0
        for i in range(H):
            for j in range(W):
                patch = [
                    im[i, j].item(),
                    im[(i-1)%H, j].item(),
                    im[(i+1)%H, j].item(),
                    im[i, (j-1)%W].item(),
                    im[i, (j+1)%W].item(),
                ]
                majority = 1.0 if sum(patch) >= len(patch) / 2 else 0.0
                if majority != im[i, j].item(): count += 1
        result.append(count)
    return result

def draw_pixel_image_4conn(ax, img_tensor, show_violations=True):
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
                patch = [
                    img[i, j].item(),
                    img[(i-1)%H, j].item(),
                    img[(i+1)%H, j].item(),
                    img[i, (j-1)%W].item(),
                    img[i, (j+1)%W].item(),
                ]
                majority = 1.0 if sum(patch) >= len(patch) / 2 else 0.0
                if majority != v:
                    overlay[i, j] = [1, 0.08, 0.08, 0.85]; n_viol += 1
        if n_viol: ax.imshow(overlay, interpolation='nearest')
    return n_viol

                                                                                 
edge_index = make_grid_edges(H_TOY, W_TOY).to(device)

def diamond_rf(L):
    return 2 * L**2 + 2 * L + 1

gnn_results = {}

for L in L_GNN:
    ckpt_path = os.path.join(SAVE_DIR, f'gnn_L{L}_final.pt')
    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path} — skipping L={L}')
        continue

    print(f'\n=== Loading GNN L={L} ===')
    gnn = MinimalResGNN(in_channels=1, out_channels=1, emb_dim=EMB, num_layers=L).to(device)
    gnn.set_edge_index(edge_index)
    gnn.load_state_dict(torch.load(ckpt_path, map_location=device))
    gnn.eval()

    valid_samples, invalid_samples = [], []
    for i in range(16):
        torch.manual_seed(42 + i)
        v = full_reverse(gnn, nsteps=400, image_size=H_TOY, device=device)
        torch.manual_seed(42 + i)
        inv = full_reverse(gnn, nsteps=3, image_size=H_TOY, device=device)
        valid_samples.append((v > 0.5).float().cpu())
        invalid_samples.append((inv > 0.5).float().cpu())

    valid_t   = torch.cat(valid_samples)
    invalid_t = torch.cat(invalid_samples)
    v_viols   = local_violations_4conn(valid_t)
    i_viols   = local_violations_4conn(invalid_t)

    gnn_results[L] = {
        'valid': valid_t, 'invalid': invalid_t,
        'v_viols': v_viols, 'i_viols': i_viols,
    }
    print(f'  Valid (T=400): {np.mean(v_viols):.1f}   Invalid (T=3): {np.mean(i_viols):.1f}')

                                                                                
print('\n' + '='*70)
print('Table 5 (updated) — GNN masked diffusion, 4-connected violation metric')
print(f'{"L":>4}  {"RF (px)":>8}  {"Coverage":>10}  {"Valid (T=400)":>14}  {"Invalid (T=3)":>14}')
print('-'*70)
for L in L_GNN:
    if L not in gnn_results: continue
    drf = diamond_rf(L)
    pct = 100 * drf / (H_TOY * H_TOY)
    vv  = np.mean(gnn_results[L]['v_viols'])
    iv  = np.mean(gnn_results[L]['i_viols'])
    print(f'{L:>4}  {drf:>8}  {pct:>9.0f}%  {vv:>14.1f}  {iv:>14.1f}')
print('='*70)

                                                                                
for L in gnn_results:
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

    draw_pixel_image_4conn(train_axes[0], torch.zeros(1, H_TOY, H_TOY), show_violations=False)
    draw_pixel_image_4conn(train_axes[1], torch.ones(1,  H_TOY, H_TOY), show_violations=False)
    v_v = [draw_pixel_image_4conn(ax, res['valid'][i, 0:1])   for i, ax in enumerate(valid_axes)]
    i_v = [draw_pixel_image_4conn(ax, res['invalid'][i, 0:1]) for i, ax in enumerate(invalid_axes)]

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
    fig.suptitle(
        f'GNN  L={L}  RF={drf} pixels ({pct:.0f}%)  — masked diffusion  '
        f'[4-connected violation metric]',
        fontsize=9, y=1.14, color='#333333'
    )

    path = os.path.join(EVAL_DIR, f'samples_gnn_L{L}_4conn.png')
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.show(); print(f'→ {path}')

print(f'\nDone. Figures saved to {EVAL_DIR}')
