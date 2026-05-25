
import torch

MASK_VALUE = 0.5                                      

def make_grid_edges(H: int, W: int, connectivity: int = 4) -> torch.Tensor:
       
    idx = torch.arange(H * W).reshape(H, W)
    src_list, dst_list = [], []

    pairs_4 = [
        (idx[:, :-1], idx[:, 1:]),           
        (idx[:, 1:],  idx[:, :-1]),         
        (idx[:-1, :], idx[1:, :]),          
        (idx[1:, :],  idx[:-1, :]),       
    ]
    for s, d in pairs_4:
        src_list.append(s.flatten())
        dst_list.append(d.flatten())

    if connectivity == 8:
        pairs_diag = [
            (idx[:-1, :-1], idx[1:, 1:]),                
            (idx[1:, 1:],   idx[:-1, :-1]),           
            (idx[:-1, 1:],  idx[1:, :-1]),              
            (idx[1:, :-1],  idx[:-1, 1:]),             
        ]
        for s, d in pairs_diag:
            src_list.append(s.flatten())
            dst_list.append(d.flatten())

    return torch.stack([torch.cat(src_list), torch.cat(dst_list)], dim=0)

def image_to_graph(img: torch.Tensor):
       
    if img.dim() == 3:
        C, H, W = img.shape
        return img.permute(1, 2, 0).reshape(H * W, C), H, W
    B, C, H, W = img.shape
    return img.permute(0, 2, 3, 1).reshape(B, H * W, C), H, W

def graph_to_image(node_features: torch.Tensor, H: int, W: int) -> torch.Tensor:

       
    if node_features.dim() == 2:
        N, C = node_features.shape
        return node_features.reshape(H, W, C).permute(2, 0, 1)
    B, N, C = node_features.shape
    return node_features.reshape(B, H, W, C).permute(0, 3, 1, 2)

def mask_forward_graph(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule,
) -> tuple:

    alpha_t = schedule(t).to(x0.device).view(-1, 1, 1)
    keep = torch.bernoulli(alpha_t.expand_as(x0)).bool()
    x_t = torch.where(keep, x0, torch.full_like(x0, MASK_VALUE))
    return x_t, keep
