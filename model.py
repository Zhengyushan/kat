import torch
from torch import nn, einsum
from einops import rearrange, repeat
import copy

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

               
class KernelAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads > 0 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, kx, krd, clst, att_mask=None, l_debug_idx=0):
        c_qkv = self.to_qkv(x).chunk(3, dim = -1)
        k_kqv = self.to_qkv(kx).chunk(3, dim = -1)
        c_kqv = self.to_qkv(clst).chunk(3, dim = -1)

        t_q, t_k, t_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), c_qkv)
        k_q, k_k, k_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), k_kqv)
        c_q, _  , _   = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), c_kqv)

        # information summary flow (ISF) -- Eq.2
        dots = einsum('b h i d, b h j d -> b h i j', t_q, k_k) * self.scale
        if att_mask is not None:
            dots = dots.masked_fill(att_mask, torch.tensor(-1e9))
        attn = self.attend(dots)* krd.permute(0,1,3,2)
        att_out = einsum('b h i j, b h j d -> b h i d', attn, k_v)
        att_out = rearrange(att_out, 'b h n d -> b n (h d)')

        # information distribution flow (IDF) -- Eq.3
        k_dots = einsum('b h i d, b h j d -> b h i j', k_q, t_k) * self.scale
        if att_mask is not None:
            k_dots = k_dots.masked_fill(att_mask.permute(0,1,3,2), torch.tensor(-1e9))
        k_attn = self.attend(k_dots) * krd
        k_out = einsum('b h i j, b h j d -> b h i d', k_attn, t_v)
        k_out = rearrange(k_out, 'b h n d -> b n (h d)')

        # classification token -- Eq.4
        c_dots = einsum('b h i d, b h j d -> b h i j', c_q, k_k) * self.scale
        if att_mask is not None:
            c_dots = c_dots.masked_fill(att_mask[:,:,:1], torch.tensor(-1e9))
        c_attn = self.attend(c_dots)
        c_out = einsum('b h i j, b h j d -> b h i d', c_attn, k_v)
        c_out = rearrange(c_out, 'b h n d -> b n (h d)')

        return self.to_out(att_out), self.to_out(k_out), self.to_out(c_out)


class KATBlocks(nn.Module):
    def __init__(self, npk, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ms = npk # initial scale factor of the Gaussian mask

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                KernelAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
            ]))
        self.h = heads
        self.dim = dim

    def forward(self, x, kx, rd, clst, mask=None, kmask=None):
        kernel_mask = repeat(kmask, 'b i ()  -> b i j', j = self.dim) < 0.5
        att_mask = einsum('b i d, b j d -> b i j', mask.float(), kmask.float())
        att_mask = repeat(att_mask.unsqueeze(1), 'b () i j -> b h i j', h = self.h) < 0.5

        rd = repeat(rd.unsqueeze(1), 'b () i j -> b h i j', h = self.h)
        rd2 = rd * rd

        k_reps = []
        for l_idx, (pn, attn, ff) in enumerate(self.layers):
            x, kx, clst = pn(x), pn(kx), pn(clst)

            soft_mask = torch.exp(-rd2 / (2*self.ms * 2**l_idx))
            x_, kx_, clst_ = attn(x, kx, soft_mask, clst, att_mask, l_idx)
            x = x + x_
            clst = clst + clst_
            kx = kx + kx_
            
            x = ff(x) + x
            clst = ff(clst) + clst
            kx = ff(kx) + kx

            k_reps.append(kx.masked_fill(kernel_mask, 0))

        return k_reps, clst
        

class KAT(nn.Module):
    def __init__(self, num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal=16, pool = 'cls', dim_head = 64, dropout = 0.5, emb_dropout = 0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, dim))
        self.nk = num_kernal

        self.dropout = nn.Dropout(emb_dropout)

        self.kt = KATBlocks(num_pk, dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes) 
        )

    def forward(self, node_features, krd, mask=None, kmask=None):
        x = self.to_patch_embedding(node_features)
        b = x.shape[0]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        kernel_tokens = repeat(self.kernel_token, '() () d -> b k d', b = b, k = self.nk)

        x = self.dropout(x)
        k_reps, clst = self.kt(x, kernel_tokens, krd, cls_tokens, mask, kmask)

        return k_reps, self.mlp_head(clst[:, 0])

    
def kat_inference(kat_model, data):
    feats = data[0].float().cuda(non_blocking=True)
    rd = data[1].float().cuda(non_blocking=True)
    masks = data[2].int().cuda(non_blocking=True)
    kmasks = data[3].int().cuda(non_blocking=True)

    return kat_model(feats, rd, masks, kmasks)
    

class KATCL(nn.Module):
    """
    Build a BYOL model for the kernels.
    """
    def __init__(self, num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal=16, pool = 'cls', dim_head = 64, dropout = 0.5, emb_dropout = 0.,
        byol_hidden_dim=512, byol_pred_dim=256, momentum=0.99):

        super(KATCL, self).__init__()

        self.momentum = momentum
        # create the online encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.online_kat = KAT(num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal, pool, dim_head, dropout, emb_dropout)
        self.online_projector = nn.Sequential(nn.Linear(dim, byol_hidden_dim, bias=False),
                                       nn.LayerNorm(byol_hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(byol_hidden_dim, byol_pred_dim))  # output layer

        # create the target encoder
        self.target_kat = copy.deepcopy(self.online_kat)
        self.target_projector = copy.deepcopy(self.online_projector)

        # freeze target encoder
        for params in self.target_kat.parameters():
            params.requires_grad = False
        for params in self.target_projector.parameters():
            params.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(byol_pred_dim, byol_hidden_dim, bias=False),
                                       nn.LayerNorm(byol_hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(byol_hidden_dim, byol_pred_dim))  # output layer

    @torch.no_grad()
    def _update_moving_average(self):
        for online_params, target_params in zip(self.online_kat.parameters(), self.target_kat.parameters()):
            target_params.data = target_params.data * self.momentum + online_params.data * (1 - self.momentum)

        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = target_params.data * self.momentum + online_params.data * (1 - self.momentum)


    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        online_k1, o1 = kat_inference(self.online_kat, x1)  
        online_k2, o2 = kat_inference(self.online_kat, x2)  
        
        online_z1 = self.online_projector(torch.cat(online_k1, dim=1)) 
        online_z2 = self.online_projector(torch.cat(online_k2, dim=1))  

        p1 = self.predictor(online_z1) 
        p2 = self.predictor(online_z2) 

        with torch.no_grad():
            self._update_moving_average()
            target_k1, _ = kat_inference(self.target_kat, x1)  
            target_k2, _ = kat_inference(self.target_kat, x2)  
            
            target_z1 = self.target_projector(torch.cat(target_k1, dim=1)) 
            target_z2 = self.target_projector(torch.cat(target_k2, dim=1))  

        return p1, p2, o1, o2, target_z1.detach(), target_z2.detach()