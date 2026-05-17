import os
import math
import torch
import torch.nn as nn
import articulate as art


def sinusoidal_pos_embed(T: int, D: int, device):
    """[T,D] 标准sin/cos位置编码，不需要设max_len。"""
    pe = torch.zeros(T, D, device=device)
    pos = torch.arange(0, T, device=device).unsqueeze(1)          # [T,1]
    div = torch.exp(torch.arange(0, D, 2, device=device) * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # [T,D]



class IMUSTEncoder(nn.Module):
    """
    SpatioTemporal encoder over tokens (t, k): L = T*K
    """
    def __init__(self, in_dim: int, d_model=256, nhead=8, num_layers=4, dropout=0.1, max_t=512):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.sensor_id_emb = nn.Embedding(10, d_model)   # sensor id 0..9
        self.time_emb = nn.Embedding(max_t, d_model)     # time id 0..max_t-1

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x_vis, vis_ids, src_key_padding_mask=None):
        """
        x_vis: [B,T,K,C]
        vis_ids: [B,K]
        src_key_padding_mask: [B, T*K]  True=padding
        return memory: [B, T*K, D]
        """
        B, T, K, C = x_vis.shape

        # tokens: flatten (t,k)
        x = x_vis.reshape(B, T * K, C)  # [B, L, C], L=T*K

        # sensor ids per token spatial
        sid = vis_ids[:, None, :].expand(B, T, K).reshape(B, T * K)  # [B, L]

        # time ids per token
        tid = torch.arange(T, device=x_vis.device)[None, :, None].expand(B, T, K).reshape(B, T * K)  # [B, L]

        h = self.in_proj(x) + self.sensor_id_emb(sid) + self.time_emb(tid)  # [B,L,D]
        mem = self.enc(h, src_key_padding_mask=src_key_padding_mask)
        return mem

class MLPHeadDecoder(nn.Module):
    """
    Same assumption: K==10 and fixed order.
    """
    def __init__(self, d_model: int, out_dim: int, hidden_ratio: float = 2.0, K_full: int = 10, dropout: float = 0.0):
        super().__init__()
        self.K_full = K_full
        hidden = int(d_model * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, memory: torch.Tensor, T: int):
        B, L, D = memory.shape
        assert L % T == 0
        K = L // T
        assert K == self.K_full
        x = memory.reshape(B, T, K, D)   # [B,T,10,D]
        return self.net(x)               # [B,T,10,C]
# -------------------------
class MaskedIMUAutoEncoder(nn.Module):
    """
    SpatioTemporal token AE:
      encoder sees (t,k) visible tokens
      decoder queries (t,n) for all 10 sensors
    """
    def __init__(self, feat_dim, d_model=384, nhead=8, enc_layers=4, dec_layers=2, dropout=0.1, max_t=512):
        super().__init__()
        self.encoder = IMUSTEncoder(feat_dim, d_model, nhead, enc_layers, dropout, max_t=max_t)
        # self.decoder = IMUSTDecoder(feat_dim, d_model, nhead, dec_layers, dropout, max_t=max_t)
        self.decoder = MLPHeadDecoder(
            d_model = d_model,
            out_dim = feat_dim,
        )
    def forward(self, x_vis, vis_ids, mem_pad_mask=None):
        """
        x_vis: [B,T,K,C]
        vis_ids: [B,K]
        mem_pad_mask: [B,T*K] True=padding
        return pred_full: [B,T,10,C]
        """
        B, T, K, C = x_vis.shape
        memory = self.encoder(x_vis, vis_ids, src_key_padding_mask=mem_pad_mask)  # [B,T*K,D]
        # pred_full = self.decoder(memory, T=T, memory_key_padding_mask=mem_pad_mask)  # [B,T,10,C]
        pred_full = self.decoder(memory, T=T)  # [B,T,10,C]
        return pred_full




def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    half = head_dim // 2
    freqs = torch.exp(-math.log(base) * torch.arange(0, half, device=device).float() / half)  # [half]
    pos = torch.arange(seq_len, device=device).float()  # [seq_len]
    ang = pos[:, None] * freqs[None, :]  # [seq_len, half]
    cos = torch.cos(ang)
    sin = torch.sin(ang)
    return cos, sin

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    q,k: [B, H, L, Dh]
    cos,sin: [L, Dh/2]
    """
    Dh = q.shape[-1]
    half = Dh // 2

    q1, q2 = q[..., :half], q[..., half:half*2]
    k1, k2 = k[..., :half], k[..., half:half*2]

    cos = cos[None, None, :, :]  # [1,1,L,half]
    sin = sin[None, None, :, :]  # [1,1,L,half]

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    # 如果 head_dim 是奇数或不是 2*half（一般不会），把尾巴拼回去
    if Dh > 2 * half:
        q_rot = torch.cat([q_rot, q[..., 2*half:]], dim=-1)
        k_rot = torch.cat([k_rot, k[..., 2*half:]], dim=-1)

    return q_rot, k_rot
class RoPESelfAttention(nn.Module):
    """
    Self-attn with RoPE.
    x: [B, L, D] -> out: [B, L, D]
    """
    def __init__(self, d_model: int, nhead: int, dropout=0.1, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rope_base = rope_base

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)

        self._rope_cache = {}  # key=(L, device.type, device.index)

    def _get_rope(self, L: int, device):
        key = (L, device.type, device.index)
        hit = self._rope_cache.get(key, None)
        if hit is not None:
            cos, sin = hit
            if cos.device == device:
                return cos, sin
        cos, sin = build_rope_cache(L, self.head_dim, device=device, base=self.rope_base)
        self._rope_cache[key] = (cos, sin)
        return cos, sin

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x: [B,L,D]
        attn_mask: [L,L] bool(True=block) or float(additive, e.g. -inf)
        key_padding_mask: [B,L] bool(True=pad)
        """
        B, L, D = x.shape
        qkv = self.qkv(x)                # [B,L,3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B,H,L,Dh]
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self._get_rope(L, x.device)
        q, k = apply_rope(q, k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B,H,L,L]

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask[None, None, :, :], float("-inf"))
            else:
                attn = attn + attn_mask[None, None, :, :]

        w = torch.softmax(attn, dim=-1)
        w = self.drop(w)

        out = w @ v  # [B,H,L,Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.proj(out)
        return out
 


# -------------------------
# Flow matching
# -------------------------
def timestep_sincos_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t: [B] in [0,1]
    return: [B,dim]
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, device=device).float() / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb





class IMUStateEmbed(nn.Module):
    """
    Embed x_t (9-dim) into d_model with sensor/time embeddings.
    """
    def __init__(self, in_dim: int, d_model: int, max_t=512, num_sensors=10):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.sensor_emb = nn.Embedding(num_sensors, d_model)
        self.time_emb = nn.Embedding(max_t, d_model)

    def forward(self, x, vis_ids):
        B, T, K, C = x.shape
        L = T * K
        xt = x.reshape(B, L, C)
        sid = vis_ids[:, None, :].expand(B, T, K).reshape(B, L)
        tid = torch.arange(T, device=x.device)[None, :, None].expand(B, T, K).reshape(B, L)
        return self.in_proj(xt) + self.sensor_emb(sid) + self.time_emb(tid)  # [B,L,D]








# 假定你已有这些工具类 / 函数：
# - RoPESelfAttention
# - IMUSTEncoder
# - IMUStateEmbed
# - build_sliding_window_attn_mask
# - timestep_sincos_embedding

# -------------------------
# Conditional DiT blocks with Cross-Attn
# -------------------------

# class AdaLNMod(nn.Module):
#     """
#     DiT-like AdaLN-Zero for a 4-sub-layer block:
#       (temporal attn, spatial attn, cross attn, mlp).

#     cond c: [B, Dc] -> 12 tensors:
#       (shift_t, scale_t, gate_t,
#        shift_s, scale_s, gate_s,
#        shift_c, scale_c, gate_c,
#        shift_m, scale_m, gate_m), each [B, D]

#     - independent params per sub-layer (DiT-style separation)
#     - gate is NOT sigmoid (allowing sign/magnitude)
#     - last linear is zero-inited (AdaLN-Zero)
#     """
#     def __init__(self, d_model: int, cond_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(cond_dim, 12 * d_model, bias=True),
#         )
#         # AdaLN-Zero init:
#         nn.init.zeros_(self.net[-1].weight)
#         nn.init.zeros_(self.net[-1].bias)

#     def forward(self, c):
#         x = self.net(c)  # [B, 12D]
#         (shift_t, scale_t, gate_t,
#          shift_s, scale_s, gate_s,
#          shift_c, scale_c, gate_c,
#          shift_m, scale_m, gate_m) = x.chunk(12, dim=-1)
#         return (shift_t, scale_t, gate_t,
#                 shift_s, scale_s, gate_s,
#                 shift_c, scale_c, gate_c,
#                 shift_m, scale_m, gate_m)


# class SpatioTemporalDiTBlock(nn.Module):
#     """
#     一个 block 内部结构：
#       1) Temporal self-attn (RoPE)
#       2) Spatial self-attn  (RoPE)
#       3) Cross-attn  (Q = current tok, KV = cond_tok from encoder)
#       4) MLP

#     每个子层都用 AdaLN-Zero 调制 + gate。
#     """
#     def __init__(self, d_model: int, nhead: int, ffn: int, cond_dim: int, dropout=0.1):
#         super().__init__()
#         # 4 个子层各自的 LayerNorm（DiT 通常 LN 不带 affine，由 AdaLN 提供）
#         self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)  # temp
#         self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)  # spatial
#         self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)  # cross
#         self.norm4 = nn.LayerNorm(d_model, elementwise_affine=False)  # mlp

#         # RoPE attention (原样保留)
#         self.temporal_attn = RoPESelfAttention(d_model, nhead, dropout=dropout, rope_base=10000.0)
#         self.spatial_attn  = RoPESelfAttention(d_model, nhead, dropout=dropout, rope_base=10000.0)

#         # Cross-attention：Q = tok, KV = cond_tok (encoder tokens)
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=nhead,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.mlp = nn.Sequential(
#             nn.Linear(d_model, ffn),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(ffn, d_model),
#         )
#         self.drop = nn.Dropout(dropout)

#         # DiT-like modulation: 4 个子层参数
#         self.mod = AdaLNMod(d_model, cond_dim)

#     def forward(self, x, c, cond_tok, T, K, temporal_mask=None, mem_pad_mask=None):
#         """
#         x:        [B, T*K, D]   当前 token 序列
#         c:        [B, D]        全局条件 (t_emb + y_emb)
#         cond_tok: [B, T*K, D]   encoder 输出 token，用于 cross-attn 的 KV
#         mem_pad_mask: [B, T*K]  True=padding，用于 cross-attn 的 key_padding_mask
#         """
#         B, L, D = x.shape
#         assert L == T * K
#         assert cond_tok.shape[:2] == (B, L)

#         (shift_t, scale_t, gate_t,
#          shift_s, scale_s, gate_s,
#          shift_c, scale_c, gate_c,
#          shift_m, scale_m, gate_m) = self.mod(c)  # 每个都是 [B, D]

#         # --- 1) Temporal Attention ---
#         h = self.norm1(x)
#         h = h * (1 + scale_t.unsqueeze(1)) + shift_t.unsqueeze(1)  # AdaLN 调制

#         h_temp = h.view(B, T, K, D).permute(0, 2, 1, 3).reshape(B * K, T, D)
#         h_temp = self.temporal_attn(
#             h_temp,
#             attn_mask=temporal_mask,    # 你的 sliding-window mask（在时间维）
#             key_padding_mask=None,      # 一般无 pad；pad 已体现在 mem_pad_mask
#         )
#         h_temp = h_temp.view(B, K, T, D).permute(0, 2, 1, 3).reshape(B, T * K, D)

#         x = x + self.drop(h_temp * gate_t.unsqueeze(1))

#         # --- 2) Spatial Attention ---
#         h = self.norm2(x)
#         h = h * (1 + scale_s.unsqueeze(1)) + shift_s.unsqueeze(1)

#         h_spat = h.view(B, T, K, D).reshape(B * T, K, D)
#         h_spat = self.spatial_attn(
#             h_spat,
#             attn_mask=None,
#             key_padding_mask=None,
#         )
#         h_spat = h_spat.view(B, T, K, D).reshape(B, T * K, D)

#         x = x + self.drop(h_spat * gate_s.unsqueeze(1))

#         # --- 3) Cross-Attention (Q = x, KV = cond_tok from encoder) ---
#         h = self.norm3(x)
#         h = h * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)

#         # MultiheadAttention(batch_first=True) 输入: [B, L, D]
#         # key_padding_mask: [B, L]，True 表示忽略
#         x_cross, _ = self.cross_attn(
#             query=h,
#             key=cond_tok,
#             value=cond_tok,
#             key_padding_mask=mem_pad_mask,
#             need_weights=False,
#         )
#         x = x + self.drop(x_cross * gate_c.unsqueeze(1))

#         # --- 4) MLP ---
#         h = self.norm4(x)
#         h = h * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)
#         x = x + self.drop(self.mlp(h) * gate_m.unsqueeze(1))

#         return x


# class IMUDenoiseFlowDiT(nn.Module):
#     def __init__(
#         self,
#         pretrained_ae=None,
#         cin_state: int = 9,
#         cin_cond: int = 9,
#         cout: int = 9,
#         depth: int = 6,
#         nhead: int = 4,
#         ffn: int = 512,
#         window_size: int = 0,
#         dropout: float = 0.1,
#         max_T: int = 300,
#         t_embed_dim: int = 256,
#         enc_d_model: int = 384,
#         enc_layers: int = 4,
#         enc_nhead: int = 8,
#     ):
#         super().__init__()

#         # encoder
#         if pretrained_ae is not None:
#             self.encoder = pretrained_ae.encoder
#         else:
#             self.encoder = IMUSTEncoder(
#                 in_dim=cin_cond,
#                 d_model=enc_d_model,
#                 nhead=enc_nhead,
#                 num_layers=enc_layers,
#                 dropout=dropout,
#                 max_t=max_T,
#             )
#             # self.encoder = IMUDSTFormerEncoder(
#             #     in_dim=cin_cond,
#             #     d_model=enc_d_model,
#             #     depth=4,
#             #     num_heads=enc_nhead,
#             #     mlp_ratio=4.0,
#             #     drop_rate=0.1,
#             #     attn_drop_rate=0.1,
#             #     drop_path_rate=0.0,
#             #     max_t=512,
#             #     max_k=10,
#             #     att_fuse=True,
#             # )
#         self.d_model = self.encoder.in_proj.out_features

#         self.state_embed = IMUStateEmbed(cin_state, self.d_model, max_t=max_T, num_sensors=10)

#         # timestep -> d_model
#         self.t_mlp = nn.Sequential(
#             nn.Linear(t_embed_dim, self.d_model),
#             nn.SiLU(),
#             nn.Linear(self.d_model, self.d_model),
#         )

#         # global y projection from pooled encoder tokens (DiT-like class embedding)
#         self.y_proj = nn.Sequential(
#             nn.LayerNorm(self.d_model),
#             nn.Linear(self.d_model, self.d_model),
#         )

#         self.blocks = nn.ModuleList([
#             SpatioTemporalDiTBlock(
#                 self.d_model,
#                 nhead=nhead,
#                 ffn=ffn,
#                 cond_dim=self.d_model,
#                 dropout=dropout,
#             )
#             for _ in range(depth)
#         ])
#         self.final_norm = nn.LayerNorm(self.d_model)

#         self.out = nn.Linear(self.d_model, cout)

#         # 输出头：可以用 DiT 风格 zero-init，这里给了一个小 std
#         # nn.init.zeros_(self.out.weight)
#         # nn.init.zeros_(self.out.bias)
#         nn.init.normal_(self.out.weight, std=1e-3)
#         nn.init.zeros_(self.out.bias)

#         self.window_size = int(window_size)
#         self._mask_cache = {}
#         # self.cond_proj = nn.Sequential(
#         #     nn.LayerNorm(self.d_model),
#         #     nn.Linear(self.d_model, self.d_model),
#         # )
#         # self.cond_gate = nn.Parameter(torch.zeros(1))  # zero-init: start from no-cond

#         self.fusion_proj = nn.Linear(self.d_model * 2, self.d_model)
#         self.cond_gate_net = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model),
#             nn.Sigmoid()
#         )
#         self.adaln_input_proj = nn.Sequential(
#             nn.Linear(self.d_model * 2, self.d_model),
#             nn.SiLU(),
#             nn.Linear(self.d_model, self.d_model),
#         )

#     def _get_attn_mask(self, T: int, device):
#         if self.window_size <= 0 or self.window_size >= T:
#             return None
#         key = (T, self.window_size, device.type)
#         m = self._mask_cache.get(key, None)
#         if m is None or m.device != device:
#             m = build_sliding_window_attn_mask(T, self.window_size, device=device, dtype=torch.bool)
#             self._mask_cache[key] = m
#         return m

#     def forward(self, x_t_9, x_n_9,vis_ids, mem_pad_mask=None, t=None):
#         """
#         x_t_9:       [B, T, K, 9]
#         x_noisy_12:  [B, T, K, 12]
#         vis_ids:     [B, K] (或兼容形状)
#         mem_pad_mask:[B, T*K] True=padding (仅 encoder & cross-attn 使用)
#         t:           [B] in [0,1]
#         """
#         assert t is not None, "t must be provided"
#         B, T, K, _ = x_t_9.shape

#         # encoder token-level condition
#         cond_tok = self.encoder(
#             x_n_9,
#             vis_ids,
#             src_key_padding_mask=mem_pad_mask,   # 你原来的用法
#         )  # [B, T*K, D]

#         # state tokens
#         state_tok = self.state_embed(x_t_9, vis_ids)  # [B, T*K, D]

#         gate = self.cond_gate_net(cond_tok)
#         refined_cond = cond_tok * gate 

#         # 然后再做拼接
#         tok = self.fusion_proj(torch.cat([state_tok, refined_cond], dim=-1))

#         # tok = cond_tok
#         # timestep embedding
#         t_emb = self.t_mlp(timestep_sincos_embedding(t, dim=256))  # [B, D]

#         # === y_emb：从 cond_tok pooled，最好做 masked mean ===
#         # if mem_pad_mask is None:
#         y_pooled = cond_tok.mean(dim=1)  # [B, D]
#         # else:
#         #     valid = (~mem_pad_mask).float()      # [B, L]
#         #     denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
#         #     y_pooled = (cond_tok * valid.unsqueeze(-1)).sum(dim=1) / denom  # [B, D]
        
#         y_emb = self.y_proj(y_pooled)  # [B, D]

#         # global condition for AdaLN (DiT-like)
#         # c = t_emb + y_emb  # [B, D]
#         c = self.adaln_input_proj(torch.cat([t_emb, y_emb], dim=-1)) #

#         temporal_mask = self._get_attn_mask(T, tok.device)

#         # 逐层 block：每层都能看 cond_tok (KV)
#         for blk in self.blocks:
#             tok = blk(
#                 tok,            # x
#                 c,              # global cond
#                 cond_tok=cond_tok,
#                 T=T,
#                 K=K,
#                 temporal_mask=temporal_mask,
#                 mem_pad_mask=mem_pad_mask,
#             )

#         tok = self.final_norm(tok)
#         v = self.out(tok)  # [B, T*K, 9]
#         return v.view(B, T, K, -1)





class SpatioTemporalDiTBlockNoCondTokNoAdaLN(nn.Module):
    """
    Ablation block without token-level condition branch and without AdaLN.

    Block structure:
      1) Temporal self-attn
      2) Spatial self-attn
      3) MLP

    Standard PreNorm residual block.
    """
    def __init__(self, d_model: int, nhead: int, ffn: int, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.temporal_attn = RoPESelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            rope_base=10000.0,
        )

        self.spatial_attn = RoPESelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            rope_base=10000.0,
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, d_model),
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x, T, K, temporal_mask=None):
        """
        x: [B, T*K, D]
        """
        B, L, D = x.shape
        assert L == T * K

        # 1) Temporal attention
        h = self.norm1(x)
        h_temp = h.view(B, T, K, D).permute(0, 2, 1, 3).reshape(B * K, T, D)

        h_temp = self.temporal_attn(
            h_temp,
            attn_mask=temporal_mask,
            key_padding_mask=None,
        )

        h_temp = h_temp.view(B, K, T, D).permute(0, 2, 1, 3).reshape(B, T * K, D)
        x = x + self.drop(h_temp)

        # 2) Spatial attention
        h = self.norm2(x)
        h_spat = h.view(B, T, K, D).reshape(B * T, K, D)

        h_spat = self.spatial_attn(
            h_spat,
            attn_mask=None,
            key_padding_mask=None,
        )

        h_spat = h_spat.view(B, T, K, D).reshape(B, T * K, D)
        x = x + self.drop(h_spat)

        # 3) MLP
        h = self.norm3(x)
        x = x + self.drop(self.mlp(h))

        return x


class IMUDenoiseFlowDiTNoCondTokNoAdaLN(nn.Module):
    """
    Ablation version without token-level condition branch and without AdaLN.

    Difference from IMUDenoiseFlowDiT2:
      - No cross-attention to cond_tok.
      - No fusion between state_tok and cond_tok.
      - No AdaLN modulation.
      - Encoder condition only enters as a simple global bias added to state tokens.

    Output:
        residual correction v = x_clean_9 - x_noisy_9
    """
    def __init__(
        self,
        pretrained_ae=None,
        cin_state: int = 9,
        cin_cond: int = 9,
        cout: int = 9,
        depth: int = 6,
        nhead: int = 4,
        ffn: int = 512,
        window_size: int = 0,
        dropout: float = 0.1,
        max_T: int = 300,
        t_embed_dim: int = 256,
        enc_d_model: int = 384,
        enc_layers: int = 4,
        enc_nhead: int = 8,
    ):
        super().__init__()

        if pretrained_ae is not None:
            self.encoder = pretrained_ae.encoder
        else:
            self.encoder = IMUSTEncoder(
                in_dim=cin_cond,
                d_model=enc_d_model,
                nhead=enc_nhead,
                num_layers=enc_layers,
                dropout=dropout,
                max_t=max_T,
            )

        self.d_model = self.encoder.in_proj.out_features

        self.state_embed = IMUStateEmbed(
            cin_state,
            self.d_model,
            max_t=max_T,
            num_sensors=10,
        )

        # Simple global condition projection.
        # This replaces AdaLN conditioning.
        self.y_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.blocks = nn.ModuleList([
            SpatioTemporalDiTBlockNoCondTokNoAdaLN(
                d_model=self.d_model,
                nhead=nhead,
                ffn=ffn,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)
        self.out = nn.Linear(self.d_model, cout)

        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

        self.window_size = int(window_size)
        self._mask_cache = {}

    def _get_attn_mask(self, T: int, device):
        if self.window_size <= 0 or self.window_size >= T:
            return None

        key = (T, self.window_size, device.type)

        m = self._mask_cache.get(key, None)


        return m

    def forward(self, x_t_9, x_n_9, vis_ids, mem_pad_mask=None):
        """
        x_t_9:       [B, T, K, 9]
        x_n_9:       [B, T, K, 9]
        vis_ids:     [B, K]
        mem_pad_mask:[B, T*K], True means padding
        """
        B, T, K, _ = x_t_9.shape

        # Encoder is only used to extract a global condition vector.
        cond_tok = self.encoder(
            x_n_9,
            vis_ids,
            src_key_padding_mask=mem_pad_mask,
        )  # [B, T*K, D]

        if mem_pad_mask is None:
            y_pooled = cond_tok.mean(dim=1)
        else:
            valid = (~mem_pad_mask).float()
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            y_pooled = (cond_tok * valid.unsqueeze(-1)).sum(dim=1) / denom

        cond_bias = self.y_proj(y_pooled).unsqueeze(1)  # [B, 1, D]

        # State tokens only, plus simple global condition bias.
        tok = self.state_embed(
            x_t_9,
            vis_ids,
        )  # [B, T*K, D]

        tok = tok + cond_bias

        temporal_mask = self._get_attn_mask(T, tok.device)

        for blk in self.blocks:
            tok = blk(
                x=tok,
                T=T,
                K=K,
                temporal_mask=temporal_mask,
            )

        tok = self.final_norm(tok)
        v = self.out(tok)

        return v.view(B, T, K, -1)
        
class IMUDenoiseFlowDiTNoEncoder(nn.Module):
    """
    Ablation version without encoder branch, token-level condition branch, and AdaLN.

    Difference from IMUDenoiseFlowDiT2:
      - No encoder.
      - No cond_tok.
      - No cross-attention.
      - No fusion between state_tok and cond_tok.
      - No AdaLN modulation.
      - The model only uses state_embed(x_t_9).

    Output:
        residual correction v = x_clean_9 - x_noisy_9
    """
    def __init__(
        self,
        pretrained_ae=None,  # kept for compatibility, but not used
        cin_state: int = 9,
        cin_cond: int = 9,   # kept for compatibility, but not used
        cout: int = 9,
        depth: int = 6,
        nhead: int = 4,
        ffn: int = 512,
        window_size: int = 0,
        dropout: float = 0.1,
        max_T: int = 300,
        t_embed_dim: int = 256,  # kept for compatibility, but not used
        enc_d_model: int = 384,
        enc_layers: int = 4,     # kept for compatibility, but not used
        enc_nhead: int = 8,      # kept for compatibility, but not used
    ):
        super().__init__()

        self.d_model = enc_d_model

        self.state_embed = IMUStateEmbed(
            cin_state,
            self.d_model,
            max_t=max_T,
            num_sensors=10,
        )

        self.blocks = nn.ModuleList([
            SpatioTemporalDiTBlockNoCondTokNoAdaLN(
                d_model=self.d_model,
                nhead=nhead,
                ffn=ffn,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)
        self.out = nn.Linear(self.d_model, cout)

        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

        self.window_size = int(window_size)
        self._mask_cache = {}

    def _get_attn_mask(self, T: int, device):
        if self.window_size <= 0 or self.window_size >= T:
            return None

        key = (T, self.window_size, device.type)

        m = self._mask_cache.get(key, None)


        return m

    def forward(self, x_t_9, x_n_9=None, vis_ids=None, mem_pad_mask=None):
        """
        x_t_9:       [B, T, K, 9]
        x_n_9:       unused, kept for API compatibility
        vis_ids:     [B, K], used only by state_embed
        mem_pad_mask: unused
        """
        B, T, K, _ = x_t_9.shape

        tok = self.state_embed(
            x_t_9,
            vis_ids,
        )  # [B, T*K, D]

        temporal_mask = self._get_attn_mask(T, tok.device)

        for blk in self.blocks:
            tok = blk(
                x=tok,
                T=T,
                K=K,
                temporal_mask=temporal_mask,
            )

        tok = self.final_norm(tok)
        v = self.out(tok)

        return v.view(B, T, K, -1)


class AdaLNMod2(nn.Module):
    """
    DiT-like AdaLN-Zero for a 4-sub-layer block:
      (temporal attn, spatial attn, cross attn, mlp).

    cond c: [B, Dc] -> 12 tensors:
      (shift_t, scale_t, gate_t,
       shift_s, scale_s, gate_s,
       shift_c, scale_c, gate_c,
       shift_m, scale_m, gate_m), each [B, D]

    - independent params per sub-layer
    - gate is NOT sigmoid
    - last linear is zero-inited
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 12 * d_model, bias=True),
        )

        # AdaLN-Zero init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        x = self.net(c)  # [B, 12D]

        (
            shift_t, scale_t, gate_t,
            shift_s, scale_s, gate_s,
            shift_c, scale_c, gate_c,
            shift_m, scale_m, gate_m,
        ) = x.chunk(12, dim=-1)

        return (
            shift_t, scale_t, gate_t,
            shift_s, scale_s, gate_s,
            shift_c, scale_c, gate_c,
            shift_m, scale_m, gate_m,
        )


class SpatioTemporalDiTBlock2(nn.Module):
    """
    Block structure:
      1) Temporal self-attn
      2) Spatial self-attn
      3) Cross-attn: Q = current tokens, KV = encoder condition tokens
      4) MLP

    Each sub-layer uses AdaLN-Zero modulation + gate.
    """
    def __init__(self, d_model: int, nhead: int, ffn: int, cond_dim: int, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm4 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.temporal_attn = RoPESelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            rope_base=10000.0,
        )

        self.spatial_attn = RoPESelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            rope_base=10000.0,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn, d_model),
        )

        self.drop = nn.Dropout(dropout)
        self.mod = AdaLNMod2(d_model, cond_dim)

    def forward(self, x, c, cond_tok, T, K, temporal_mask=None, mem_pad_mask=None):
        """
        x:            [B, T*K, D]
        c:            [B, D]
        cond_tok:     [B, T*K, D]
        mem_pad_mask: [B, T*K], True means padding
        """
        B, L, D = x.shape
        assert L == T * K
        assert cond_tok.shape[:2] == (B, L)

        (
            shift_t, scale_t, gate_t,
            shift_s, scale_s, gate_s,
            shift_c, scale_c, gate_c,
            shift_m, scale_m, gate_m,
        ) = self.mod(c)

        # 1) Temporal attention
        h = self.norm1(x)
        h = h * (1 + scale_t.unsqueeze(1)) + shift_t.unsqueeze(1)

        h_temp = h.view(B, T, K, D).permute(0, 2, 1, 3).reshape(B * K, T, D)
        h_temp = self.temporal_attn(
            h_temp,
            attn_mask=temporal_mask,
            key_padding_mask=None,
        )
        h_temp = h_temp.view(B, K, T, D).permute(0, 2, 1, 3).reshape(B, T * K, D)

        x = x + self.drop(h_temp * gate_t.unsqueeze(1))

        # 2) Spatial attention
        h = self.norm2(x)
        h = h * (1 + scale_s.unsqueeze(1)) + shift_s.unsqueeze(1)

        h_spat = h.view(B, T, K, D).reshape(B * T, K, D)
        h_spat = self.spatial_attn(
            h_spat,
            attn_mask=None,
            key_padding_mask=None,
        )
        h_spat = h_spat.view(B, T, K, D).reshape(B, T * K, D)

        x = x + self.drop(h_spat * gate_s.unsqueeze(1))

        # 3) Cross attention
        h = self.norm3(x)
        h = h * (1 + scale_c.unsqueeze(1)) + shift_c.unsqueeze(1)

        x_cross, _ = self.cross_attn(
            query=h,
            key=cond_tok,
            value=cond_tok,
            key_padding_mask=mem_pad_mask,
            need_weights=False,
        )

        x = x + self.drop(x_cross * gate_c.unsqueeze(1))

        # 4) MLP
        h = self.norm4(x)
        h = h * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)

        x = x + self.drop(self.mlp(h) * gate_m.unsqueeze(1))

        return x


class IMUDenoiseFlowDiT2(nn.Module):
    """
    Residual denoising version.

    Input:
        x_t_9:   [B, T, K, 9]
                 For residual denoising, pass x_noisy_9 here.
        x_n_9:   [B, T, K, 9]
                 Condition input, usually also x_noisy_9.
        vis_ids: [B, K]

    Output:
        residual velocity / residual correction:
        v = x_clean_9 - x_noisy_9

    Inference:
        x_denoised_9 = x_noisy_9 + model(x_noisy_9, x_noisy_9, vis_ids)
    """
    def __init__(
        self,
        pretrained_ae=None,
        cin_state: int = 9,
        cin_cond: int = 9,
        cout: int = 9,
        depth: int = 6,
        nhead: int = 4,
        ffn: int = 512,
        window_size: int = 0,
        dropout: float = 0.1,
        max_T: int = 300,
        t_embed_dim: int = 256,  # kept only for backward-compatible initialization; not used
        enc_d_model: int = 384,
        enc_layers: int = 4,
        enc_nhead: int = 8,
    ):
        super().__init__()

        # Encoder
        if pretrained_ae is not None:
            self.encoder = pretrained_ae.encoder
        else:
            self.encoder = IMUSTEncoder(
                in_dim=cin_cond,
                d_model=enc_d_model,
                nhead=enc_nhead,
                num_layers=enc_layers,
                dropout=dropout,
                max_t=max_T,
            )

        self.d_model = self.encoder.in_proj.out_features

        # State embedding for current/noisy IMU tokens
        self.state_embed = IMUStateEmbed(
            cin_state,
            self.d_model,
            max_t=max_T,
            num_sensors=10,
        )

        # Global condition projection from pooled encoder tokens
        self.y_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )

        self.blocks = nn.ModuleList([
            SpatioTemporalDiTBlock2(
                d_model=self.d_model,
                nhead=nhead,
                ffn=ffn,
                cond_dim=self.d_model,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)
        self.out = nn.Linear(self.d_model, cout)

        # Output head init
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

        self.window_size = int(window_size)
        self._mask_cache = {}

        # Fuse state tokens and condition tokens
        self.fusion_proj = nn.Linear(self.d_model * 2, self.d_model)

        self.cond_gate_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid(),
        )

        # Important change:
        # original version used Linear(2D -> D) for [t_emb, y_emb].
        # now there is no t_emb, so use Linear(D -> D).
        self.adaln_input_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def _get_attn_mask(self, T: int, device):
        if self.window_size <= 0 or self.window_size >= T:
            return None

        key = (T, self.window_size, device.type)

        m = self._mask_cache.get(key, None)


        return m

    def forward(self, x_t_9, x_n_9, vis_ids, mem_pad_mask=None):
        """
        x_t_9:       [B, T, K, 9]
                     In residual denoising, this should be x_noisy_9.
        x_n_9:       [B, T, K, 9]
                     Condition input, usually x_noisy_9.
        vis_ids:     [B, K]
        mem_pad_mask:[B, T*K], True means padding.
        """
        B, T, K, _ = x_t_9.shape

        # Encoder token-level condition
        cond_tok = self.encoder(
            x_n_9,
            vis_ids,
            src_key_padding_mask=mem_pad_mask,
        )  # [B, T*K, D]

        # State tokens
        state_tok = self.state_embed(
            x_t_9,
            vis_ids,
        )  # [B, T*K, D]

        # Gated condition refinement
        gate = self.cond_gate_net(cond_tok)
        refined_cond = cond_tok * gate

        # Fuse current/noisy state tokens with condition tokens
        tok = self.fusion_proj(
            torch.cat([state_tok, refined_cond], dim=-1)
        )  # [B, T*K, D]

        # Pooled condition embedding
        if mem_pad_mask is None:
            y_pooled = cond_tok.mean(dim=1)  # [B, D]
        else:
            valid = (~mem_pad_mask).float()  # [B, T*K]
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            y_pooled = (cond_tok * valid.unsqueeze(-1)).sum(dim=1) / denom

        y_emb = self.y_proj(y_pooled)  # [B, D]

        # No timestep embedding.
        # AdaLN global condition only uses the pooled encoder condition.
        c = self.adaln_input_proj(y_emb)  # [B, D]

        temporal_mask = self._get_attn_mask(T, tok.device)

        for blk in self.blocks:
            tok = blk(
                x=tok,
                c=c,
                cond_tok=cond_tok,
                T=T,
                K=K,
                temporal_mask=temporal_mask,
                mem_pad_mask=mem_pad_mask,
            )

        tok = self.final_norm(tok)
        v = self.out(tok)  # [B, T*K, 9]

        return v.view(B, T, K, -1)



