import torch


# ===================== 公共小工具 =====================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def standardize_features(x: torch.Tensor) -> torch.Tensor:
    """
    对每一列做零均值 + 单位方差标准化
    """
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / y.numel()

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / y.numel()

def build_tensor_features(U_N, G, enhance=True, device="cuda"):
    U_N = U_N.to(device)
    G = G.to(device)

    if not enhance:
        return U_N

    # === Step 1: flatten core G ===
    G_flat = G.reshape(-1)       # shape: (R_total,)

    # === Step 2: project to same dim as U_N ===
    R_total = G_flat.shape[0]
    rN = U_N.shape[1]

    # # fixed random projection (also可换成可训练)
    # W = torch.randn(R_total, rN, device=device) * 0.01
    # [修改] 使用局部随机数生成器，确保 W 永远一致
    g_local = torch.Generator(device=device)
    g_local.manual_seed(12345)  # 固定一个特殊的种子，专门用于生成 W
    W = torch.randn(R_total, rN, generator=g_local, device=device) * 0.01

    G_proj = (G_flat @ W).detach()     # shape: (rN,)

    # === Step 3: enhance ===
    alpha = 0.1
    U_S = U_N + alpha * G_proj

    return U_S

def knn_graph_pytorch(x, k):
    """
    x: (N, d) 节点特征
    返回：edge_index (2, N*k)
    """
    N = x.size(0)
    if N < 2:
        raise ValueError("knn_graph_pytorch requires at least 2 nodes.")
    k = max(1, min(int(k), N - 1))

    # 距离矩阵 (N, N)
    dist = torch.cdist(x, x, p=2)  # GPU 上也能跑

    # 去掉自身
    dist.fill_diagonal_(1e9)

    # 找最近 k 个邻居
    knn = dist.topk(k, largest=False).indices  # (N, k)

    # 构造 edge_index
    row = torch.arange(N, device=x.device).view(-1, 1).repeat(1, k).reshape(-1)
    col = knn.reshape(-1).to(device=x.device)

    edge_index = torch.stack([row, col], dim=0)
    return edge_index

