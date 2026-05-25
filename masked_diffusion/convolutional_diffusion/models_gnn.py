import torch
from torch import nn

class SinusoidalEmbedding(nn.Module):
                                                                                                

    def __init__(self, fdim: int, conditional: bool = False, num_classes: int = None):
        super().__init__()
        self.fdim = fdim
        self.conditional = conditional
        if conditional:
            self.class_emb = nn.Embedding(num_classes, fdim)

    def forward(self, t: torch.Tensor, label=None) -> torch.Tensor:
        d = self.fdim // 2
        freq = 10000 ** (torch.arange(d, device=t.device) / max(d - 1, 1))
        args = t[:, None] / freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)             
        if self.conditional and label is not None:
            emb = emb + self.class_emb(label)
        return emb

class MPNNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_self  = nn.Linear(dim, dim, bias=False)
        self.lin_neigh = nn.Linear(dim, dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        src, dst = edge_index                    
        B, N, D = x.shape

                                                                     
        agg = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        agg.index_add_(1, dst, x[:, src, :])                                   
        deg = torch.bincount(dst, minlength=N).float().clamp(min=1)               
        agg = agg / deg.view(1, N, 1)                                            

        return self.act(self.lin_self(x) + self.lin_neigh(agg))

class MinimalResGNN(nn.Module):
    def __init__(
        self,
        in_channels:  int  = 1,
        out_channels: int  = 1,
        emb_dim:      int  = 128,
        num_layers:   int  = 6,
        conditional:  bool = False,
        num_classes:  int  = None,
    ):
        super().__init__()
        self.emb_dim    = emb_dim
        self.num_layers = num_layers

        self.embedding       = SinusoidalEmbedding(emb_dim, conditional, num_classes)
        self.up_projection   = nn.Linear(in_channels, emb_dim)

                                                                                      
        self.embs = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())
            for _ in range(num_layers + 1)
        ])
        self.layers = nn.ModuleList([
            MPNNLayer(emb_dim) for _ in range(num_layers)
        ])
        self.down_projection = nn.Linear(emb_dim, out_channels)

    def forward(
        self,
        t:          torch.Tensor,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        label=None,
    ) -> torch.Tensor:
           
        emb   = self.embedding(t, label)                   
        state = self.up_projection(x)                          

        for i in range(self.num_layers):
            time_bias = self.embs[i](emb).unsqueeze(1)                                        
            delta = self.layers[i](state + time_bias, edge_index)
            state = state + delta                                                           

        state = state + self.embs[-1](emb).unsqueeze(1)
        return self.down_projection(state)                          

class GNNImageWrapper(nn.Module):

    def __init__(self, gnn: MinimalResGNN, edge_index: torch.Tensor, H: int, W: int):
        super().__init__()
        self.gnn = gnn
        self.H = H
        self.W = W
        self.register_buffer('edge_index', edge_index)

    def forward(self, t: torch.Tensor, x: torch.Tensor, label=None) -> torch.Tensor:
        from utils.graph_utils import image_to_graph, graph_to_image
        x_graph, _, _ = image_to_graph(x)                                         
        out_graph = self.gnn(t, x_graph, self.edge_index, label=label)            
        return graph_to_image(out_graph, self.H, self.W)                             
