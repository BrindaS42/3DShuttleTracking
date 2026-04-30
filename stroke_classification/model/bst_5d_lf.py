import torch
from torch import nn, Tensor
from model.tempose import TCN, MLP, MLP_Head, TransformerEncoder, FeedForward
from positional_encodings.torch_encodings import PositionalEncoding1D

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, drop_p) -> None:
        super().__init__()
        d_cat = d_head * n_head
        self.h = n_head
        self.to_q = nn.Linear(d_model, d_cat, bias=False)
        self.to_kv = nn.Linear(d_model, d_cat * 2, bias=False)
        self.scale = d_head**-0.5
        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(drop_p))
        self.tail = nn.Sequential(nn.Linear(d_cat, d_model), nn.Dropout(drop_p, inplace=True))

    def forward(self, x1, x2, mask=None):
        # x1, x2: (b, t, d_model)
        q, kv = self.to_q(x1), self.to_kv(x2)
        b, t, _ = q.shape
        
        q = q.view(b, t, self.h, -1).transpose(1, 2)
        k, v = kv.view(b, t, self.h, -1).chunk(2, dim=-1)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        
        dots = (q.contiguous() @ k.transpose(-1, -2).contiguous()) * self.scale
        if mask is not None: 
            dots = dots.masked_fill(mask.view(b, 1, 1, t) == 0.0, -torch.inf)
        
        # --- FIX: Reorder Reshape and Tail ---
        # 1. Compute attention weighted values: (b, h, t, d_head)
        attn_out = self.attend(dots) @ v.contiguous()
        
        # 2. Merge Heads: (b, t, h * d_head) -> (b, t, 768)
        merged_heads = attn_out.transpose(1, 2).reshape(b, t, -1)
        
        # 3. Apply Tail Linear Layer: (b, t, 768) -> (b, t, d_model)
        return self.tail(merged_heads)

class CrossTransformerLayer(nn.Module):
    def __init__(self, d_model, d_head, n_head, hd_mlp, drop_p) -> None:
        super().__init__()
        self.ln1_x1, self.ln1_x2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, d_head, n_head, drop_p)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_model, hd_mlp, drop_p)
    def forward(self, x1, x2, mask=None):
        x = self.cross_attn(self.ln1_x1(x1), self.ln1_x2(x2), mask)
        return self.ff(self.ln2(x)) + x

class BST_CG_AP(nn.Module):
    def __init__(self, in_dim, seq_len, n_class=35, n_people=2, d_model=100, d_head=128, n_head=6, depth_tem=2, depth_inter=1, drop_p=0.3, mlp_d_scale=4, tcn_kernel_size=5):
        super().__init__()
        self.mlp_positions = MLP(2, out_dim=in_dim, hd_dim=256, drop_p=drop_p)
        self.tcn_pose = TCN(in_dim, [d_model, d_model], tcn_kernel_size, drop_p)
        
        # LATE FUSION SHUTTLE BACKBONES
        self.tcn_s2d = TCN(2, [d_model // 4, d_model // 2], tcn_kernel_size, drop_p)
        self.tcn_s3d = TCN(3, [d_model // 4, d_model // 2], tcn_kernel_size, drop_p)

        self.learned_token_tem = nn.Parameter(torch.randn(1, d_model))
        self.embedding_tem = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.pre_dropout = nn.Dropout(drop_p, inplace=True)
        self.encoder_tem = TransformerEncoder(d_model, d_head, n_head, depth_tem, d_model * mlp_d_scale, drop_p)
        self.embedding_cross = nn.Parameter(torch.empty(1, seq_len, d_model))
        self.cross_trans = CrossTransformerLayer(d_model, d_head, n_head, d_model * mlp_d_scale, drop_p)
        self.learned_token_inter = nn.Parameter(torch.randn(1, d_model))
        self.embedding_inter = nn.Parameter(torch.empty(1, 1+seq_len, d_model))
        self.encoder_inter = TransformerEncoder(d_model, d_head, n_head, depth_inter, d_model * mlp_d_scale, drop_p)
        
        self.cos_sim = nn.CosineSimilarity()
        self.mlp_clean = MLP(d_model, d_model, d_model, drop_p)
        self.mlp_head = MLP_Head(d_model * 3, n_class, d_model * mlp_d_scale, drop_p)
        self.d_model = d_model
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        p_enc = PositionalEncoding1D(self.d_model)
        self.embedding_tem.copy_(p_enc(self.embedding_tem))
        self.embedding_cross.copy_(p_enc(self.embedding_cross))
        self.embedding_inter.copy_(p_enc(self.embedding_inter))
        nn.init.normal_(self.learned_token_tem, std=0.02)
        nn.init.normal_(self.learned_token_inter, std=0.02)

    def forward(self, JnB, shuttle_5d, pos, video_len):
        b, t, n, in_dim = JnB.shape
        JnB = JnB.permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        pos_imp = self.mlp_positions(pos).permute(0, 2, 3, 1).reshape(b*n, in_dim, t)
        JnB = self.tcn_pose(JnB * pos_imp + JnB).view(b, n, -1, t).transpose(-2, -1)

        # Process 5D input (2D pixels + 3D meters)[cite: 16]
        s2d = self.tcn_s2d(shuttle_5d[:, :, :2].transpose(1, 2).contiguous())
        s3d = self.tcn_s3d(shuttle_5d[:, :, 2:].transpose(1, 2).contiguous())
        shuttle_feat = torch.cat([s2d, s3d], dim=1).unsqueeze(1).transpose(-2, -1)
        
        x = torch.cat((JnB, shuttle_feat), dim=1)
        x = x.view(b*3, t, self.d_model)
        x = torch.cat((self.learned_token_tem.expand(b*3, -1, -1), x), dim=1) + self.embedding_tem
        
        mask = (torch.arange(0, 1+t, device=x.device).unsqueeze(0) < (1 + video_len).unsqueeze(-1))
        x = self.encoder_tem(self.pre_dropout(x), mask.repeat_interleave(3, dim=0))
        p1, p2, s = map(lambda ts: ts.squeeze(1), x.view(b, 3, 1+t, -1).chunk(3, dim=1))
        
        p1_cls, p2_cls, s_cls = p1[:, 0], p2[:, 0], s[:, 0]
        p1, p2, s = p1[:, 1:] + self.embedding_cross, p2[:, 1:] + self.embedding_cross, s[:, 1:] + self.embedding_cross
        
        m_cr = mask[:, 1:].contiguous()
        p1_s, p2_s = self.cross_trans(p1, s, m_cr), self.cross_trans(p2, s, m_cr)
        p1_s = self.encoder_inter(torch.cat((self.learned_token_inter.expand(b, -1, -1), p1_s), dim=1) + self.embedding_inter, mask)
        p2_s = self.encoder_inter(torch.cat((self.learned_token_inter.expand(b, -1, -1), p2_s), dim=1) + self.embedding_inter, mask)
        p1_s_cls, p2_s_cls = p1_s[:, 0], p2_s[:, 0]

        alpha = ((self.cos_sim(p1_s_cls, s_cls) - self.cos_sim(p2_s_cls, s_cls) + 2) / 4).unsqueeze(1)
        p1_con, p2_con = alpha * (p1_cls + p1_s_cls), (1-alpha) * (p2_cls + p2_s_cls)
        s_cls = s_cls - self.mlp_clean(torch.minimum(p1_s_cls, p2_s_cls))
        return self.mlp_head(torch.cat((p1_con, p2_con, s_cls), dim=1))