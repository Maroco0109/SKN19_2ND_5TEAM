"""

모델 생성

- 모든 모델은 nn.module 혹은 BaseEstimator를 기반으로 하는 모델
- 모델 결과를 통해 예측 사망률 계산
- 예측 사망률을 통해 1~100점 사이 위험점수로 변환 (계산식을 이용하여 100점에는 도달하지 않도록 함)
- 예측 비율과 점수로 모두 반환 가능하도록 설계

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepHitSurv(nn.Module) :
    def __init__(self, input_dim, hidden_size=(128, 64), time_bins=50, num_events=4, dropout=0.2) :
        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins

        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(h2, time_bins) for _ in range(num_events)
        ])

    def forward(self, x) :
        s = self.shared(x)
        logits = torch.stack([head(s) for head in self.heads], dim=1)

        pmf = F.softmax(logits, dim=-1)
        cif = torch.cumsum(pmf, dim=-1)

        return logits, pmf, cif

class DeepHitSurvWithSEBlock(nn.Module) :
    def __init__(self, input_dim, hidden_size=(128, 64), time_bins=50, num_events=4, dropout=0.2, se_ratio=0.25) :
        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins
        self.input_dim = input_dim


        se_hidden = max(1, int(input_dim * se_ratio))
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, input_dim),
            nn.Sigmoid()
        )

        se_hidden_shared = max(1, int(h2 * se_ratio))
        self.se_block_event = nn.ModuleList([        
            nn.Sequential(
                nn.Linear(h2, se_hidden_shared),
                nn.ReLU(),
                nn.Linear(se_hidden_shared, h2),
                nn.Sigmoid()
            ) for _ in range(num_events)
        ])


        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(h2, time_bins) for _ in range(num_events)
        ])

    def forward(self, x) :
        scale = self.se_block(x)
        x_scaled = x * scale

        s = self.shared(x_scaled)

        logits_list = []
        for k in range(self.num_events) :
            s_scaled = s * self.se_block_event[k](s)
            logits_list.append(self.heads[k](s_scaled))

        logits = torch.stack(logits_list, dim=1)

        pmf = F.softmax(logits, dim=-1)
        cif = torch.cumsum(pmf, dim=-1)

        return logits, pmf, cif


def deephit_loss(pmf, cif, times, events, alpha=0.5, margin=0.05):
    B, K, T = pmf.shape
    device = pmf.device

    eps = 1e-8
    likelihood_loss = torch.zeros(B, device=device)

    # ===== uncensored =====
    uncensored_mask = (events >= 0)
    if uncensored_mask.any():
        # 항상 1D long tensor로 안전하게
        idx_uncensored = uncensored_mask.nonzero(as_tuple=True)[0]
        t_uncensored = times[idx_uncensored].long()
        e_uncensored = events[idx_uncensored].long()

        pmf_vals = pmf[idx_uncensored, e_uncensored, t_uncensored]
        likelihood_loss[idx_uncensored] = -torch.log(pmf_vals + eps)

    # ===== censored =====
    censored_mask = (events < 0)
    if censored_mask.any():
        idx_censored = censored_mask.nonzero(as_tuple=True)[0]
        t_censored = times[idx_censored].long()

        surv = 1.0 - cif[idx_censored, :, t_censored].sum(dim=1)
        likelihood_loss[idx_censored] = -torch.log(surv + eps)

    L_likelihood = likelihood_loss.mean()

    # ===== ranking loss =====
    L_rank = torch.tensor(0.0, device=device)
    count_pairs = 0

    for k in range(K):
        idx_k = (events == k).nonzero(as_tuple=True)[0]  # 항상 1D long tensor
        if idx_k.numel() == 0:
            continue

        for i in idx_k:
            t_i = int(times[i].item())

            mask_j = (times > t_i)
            if mask_j.sum() == 0:
                continue

            cif_i = cif[i, k, t_i]
            cif_j = cif[mask_j, k, t_i]

            diff = margin + cif_j - cif_i
            loss_pairs = torch.clamp(diff, min=0.0).sum()
            L_rank = L_rank + loss_pairs
            count_pairs += mask_j.sum().item()

    if count_pairs > 0:
        L_rank = L_rank / count_pairs
    else:
        L_rank = torch.tensor(0.0, device=device)

    loss = L_likelihood + alpha * L_rank
    return loss, L_likelihood.detach(), L_rank.detach()
