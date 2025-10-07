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
import torch.optim as optim
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
import random

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
    def __init__(self, input_dim, hidden_size=(128, 64), time_bins=91, num_events=4, dropout=0.2, se_ratio=0.25) :
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

def deephit_loss(pmf, cif, times, events, alpha=0.5, margin=0.05, eps=1e-8):
    """
    DeepHit original-style loss (Lee et al., 2018)
    - Uses PMF for likelihood
    - Uses efficient pairwise ranking loss (vectorized)
    - Handles censored and competing risks properly
    """
    B, K, T = pmf.shape
    device = pmf.device

    # --------------------
    # Likelihood Loss (Negative log-likelihood)
    # --------------------
    likelihood_loss = torch.zeros(B, device=device)

    uncensored_mask = (events >= 0)
    if uncensored_mask.any():
        idx = uncensored_mask.nonzero(as_tuple=True)[0]
        t_idx = times[idx].clamp(min=0, max=T-1).long()
        e_idx = events[idx].long()

        pmf_vals = pmf[idx, e_idx, t_idx].clamp(min=eps, max=1.0)
        likelihood_loss[idx] = -torch.log(pmf_vals + eps)

    censored_mask = (events < 0)
    if censored_mask.any():
        idx = censored_mask.nonzero(as_tuple=True)[0]
        t_idx = times[idx].clamp(min=0, max=T-1).long()
        
        cif_sum = cif[idx, :, t_idx].sum(dim=1).clamp(min=0.0, max=1.0)
        surv_vals = (1.0 - cif_sum).clamp(min=eps)

        likelihood_loss[idx] = -torch.log(surv_vals)

    L_likelihood = likelihood_loss.mean()

    # --------------------
    # Ranking Loss (vectorized, original DeepHit style)
    # --------------------
    # Step 1: event별 mask
    uncensored_idx = (events >= 0).nonzero(as_tuple=True)[0]
    if len(uncensored_idx) == 0:
        return L_likelihood, L_likelihood, torch.tensor(0.0, device=device)

    t_i = times[uncensored_idx].clamp(max=T-1).long()
    e_i = events[uncensored_idx].long()

    # Step 2: 각 샘플의 CIF(time) 추출
    # cif_i: [B_uncensored]
    cif_i = cif[uncensored_idx, e_i, t_i]

    # Step 3: 모든 샘플에 대해 time 비교 (벡터화)
    times_expand = times.unsqueeze(0)
    mask_later = (times_expand > t_i.unsqueeze(1))  # j sample의 time > i sample의 time

    # Step 4: 각 사건별 CIF 비교
    cif_all = cif[:, e_i, t_i]  # shape [B, B_uncensored]
    cif_diff = margin + cif_all.T - cif_i.unsqueeze(1)  # [B_uncensored, B]
    cif_diff = cif_diff * mask_later.float()  # valid pairs만 유지

    L_rank = torch.clamp(cif_diff, min=0).sum() / (mask_later.sum() + eps)

    # --------------------
    # Combine
    # --------------------
    loss = L_likelihood + alpha * L_rank
    return loss, L_likelihood.detach(), L_rank.detach()


class WeightedCoxRiskEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, num_events=4, lr=1e-2, epochs=100, weights=None, verbose=False, device='cpu'):
        self.num_events = num_events
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.device = device
        
        # 가중치는 학습에 사용되지 않음
        if weights is None:
            self.weights = torch.ones(num_events, device=device) / num_events
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def _cox_ph_loss(self, risk_score, times, events):
        risk_score = risk_score.squeeze()
        loss = 0.0
        uncensored_idx = (events >= 0).nonzero()[0]
        if len(uncensored_idx) == 0:
            return torch.tensor(0.0, device=self.device)
        for i in uncensored_idx:
            t_i = times[i]
            mask = times >= t_i
            loss += - (risk_score[i] - torch.log(torch.exp(risk_score[mask]).sum()))
        return loss / len(uncensored_idx)

    def fit(self, X, times, events):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        times = torch.tensor(times, dtype=torch.float32, device=self.device)
        events = torch.tensor(events, dtype=torch.float32, device=self.device)

        B, K = X.shape
        assert K == self.num_events

        # 사건별 단층 선형 회귀(SLP)
        self.event_linears = nn.ModuleList([nn.Linear(1, 1) for _ in range(K)]).to(self.device)

        optimizer = optim.Adam(self.event_linears.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # 사건별 위험점수
            r_list = [self.event_linears[k](X[:, k:k+1]) for k in range(K)]
            r_stack = torch.cat(r_list, dim=1)  # (B, K)

            # 학습 시에는 단순 평균 (weights 미적용)
            risk_score = r_stack.mean(dim=1)

            # Cox loss 계산
            loss = self._cox_ph_loss(risk_score, times, events)
            loss.backward()
            optimizer.step()

            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")

        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        r_list = [self.event_linears[k](X[:, k:k+1]) for k in range(self.num_events)]
        r_stack = torch.cat(r_list, dim=1)

        # 예측 시에만 가중치 적용
        risk_score = (r_stack * self.weights).sum(dim=1)

        # Sigmoid 적용 후 0~100 스케일
        risk_score_scaled = torch.sigmoid(risk_score) * 100

        return risk_score_scaled.detach().cpu().numpy()


def set_seed(seed = 42) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN 비결정적 동작 방지 (연산 재현성 확보)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



