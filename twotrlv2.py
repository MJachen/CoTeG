import torch
import torch.nn as nn
import string
import os


def _parse_spatial_shape(value, default=(64, 64, 64)):
    if not value:
        return default
    parts = [int(x.strip()) for x in value.replace("x", ",").split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("IMAGE_SIZE/OUT_SHAPE must be like 64,64,64")
    return tuple(parts)

def tucker_to_tensor_no_out(core, factors):
    """
    Tucker contraction without output factor.
    core:    (R1, R2, ..., RN, Rout)
    factors: list of N matrices (Ik, Rk)

    returns:
        W ∈ R^{I1 × I2 × ... × IN × Rout}
    """
    N = len(factors)
    letters = string.ascii_lowercase

    assert core.ndim == N + 1, \
        f"core.ndim={core.ndim}, but expected {N+1}"

    r_letters = letters[:N]           # r0 r1 ... r{N-1}
    q_letter = letters[N]             # Rout
    i_letters = letters[N+1:N+1+N]    # i0 i1 ...

    # core: r0 r1 ... r{N-1} q
    core_subs = r_letters + q_letter

    factor_subs = [
        i_letters[k] + r_letters[k] for k in range(N)
    ]

    out_subs = "".join(i_letters) + q_letter

    expr = (
        core_subs + "," +
        ",".join(factor_subs) +
        "->" + out_subs
    )

    return torch.einsum(expr, core, *factors)


class TRL(nn.Module):
    def __init__(self, input_shape, ranks, n_outputs):
        """
        input_shape : (I1, I2, ..., IN)   # Conv 输出的张量形状（不含 batch）
        ranks       : (R1, R2, ..., RN, Rout)
        n_outputs   : 输出类别数 / 任务维度
        """
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.ranks = tuple(ranks)
        self.n_outputs = n_outputs

        N = len(self.input_shape)
        assert len(self.ranks) == N + 1, "ranks 长度必须是 N+1（最后一个是 Rout）"

        # 核心张量 G ∈ R^{R1 × ... × RN × Rout}
        #self.core = nn.Parameter(torch.randn(*self.ranks) * 0.1)
        self.core = nn.Parameter(torch.empty(*self.ranks))
        nn.init.xavier_normal_(self.core)

        # 每个输入 mode 的投影矩阵 U^(k) ∈ R^{Ik × Rk}
        self.factors = nn.ParameterList()
        for i, I in enumerate(self.input_shape):
            #Rk = self.ranks[i]
            #self.factors.append(
            #    nn.Parameter(torch.randn(I, Rk) * 0.1)
            #)
            Rk = self.ranks[i]
            p = nn.Parameter(torch.empty(I, Rk))
            nn.init.xavier_normal_(p)
            self.factors.append(p)

        # 偏置
        self.bias = nn.Parameter(torch.zeros(1, n_outputs))
        self.logit_norm = nn.LayerNorm(n_outputs)

        # ===== Step 2: 独立分类头 =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.ranks[-1]),  # Rout
            nn.Linear(self.ranks[-1], n_outputs)
        )

    def forward(self, x, return_embed=False):
        """
        x: (B, I1, I2, ..., IN)
        return_embed: 是否返回中间 embedding z（用于和 GNN 分支融合）

        返回:
          如果 return_embed=False: logits (B, n_outputs)
          如果 return_embed=True : (logits, z) 其中 z ∈ R^{B × Rout}
        """
        B = x.size(0)
        assert tuple(x.shape[1:]) == self.input_shape, \
            f"TRL 期望输入 shape={self.input_shape}，但得到 {tuple(x.shape[1:])}"

        x_flat = x.reshape(B, -1)
        N = len(self.input_shape)
        Rout = self.ranks[-1]
        device = x.device
        dtype = x.dtype

        # ===== Step 2: 无输出因子的 Tucker contraction =====
        factors_mid = list(self.factors)   # 只包含输入 mode 的 factors
        W_mid = tucker_to_tensor_no_out(self.core, factors_mid)


        W_mid_mat = W_mid.reshape(-1, Rout)                           # (prod(Ik), Rout)

        # 得到 embedding z ∈ R^{B × Rout}
        embed = x_flat @ W_mid_mat

        logits = self.classifier(embed)

        if return_embed:
            return logits, embed
        return logits

class ConvTRLNet(nn.Module):
    def __init__(self, n_outputs, ranks=(16, 4, 4, 4, 8), input_spatial=None):
        """
        n_outputs : 分类类别数
        ranks     : TRL 的秩 (R_c, R_d, R_h, R_w, Rout)
                    对应 input_shape = (C, D, H, W)
        """
        super().__init__()
        if input_spatial is None:
            input_spatial = _parse_spatial_shape(os.environ.get("IMAGE_SIZE") or os.environ.get("OUT_SHAPE"))
        d, h, w = tuple(int(x) for x in input_spatial)
        if min(d, h, w) < 8 or d % 8 != 0 or h % 8 != 0 or w % 8 != 0:
            raise ValueError(f"input_spatial must be divisible by 8 after three pooling layers, got {input_spatial}")

        # 3D 卷积 backbone
        self.features = nn.Sequential(
            # 输入: (B, 4, 64, 64, 64)
            nn.Conv3d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),              # (16, 32, 32, 32)

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),              # (32, 16, 16, 16)

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),              # (64, 8, 8, 8)
        )

        # Conv 输出的张量形状（不含 batch）
        input_shape = (64, d // 8, h // 8, w // 8)

        # TRL 作为最终分类层
        self.trl = TRL(input_shape=input_shape,
                       ranks=ranks,
                       n_outputs=n_outputs)

    def forward(self, x, return_embed=False):
        """
        x: (B, 4, 64, 64, 64)
        """
        x_feat = self.features(x)  # (B, 64, 8, 8, 8)
        return self.trl(x_feat, return_embed=return_embed)


