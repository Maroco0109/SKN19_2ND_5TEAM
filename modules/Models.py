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

def smooth_pmf_ema(pmf, alpha=0.3):
    """
    pmf: (B, E, T)
    alpha: EMA smoothing 계수 (0 < alpha < 1)
           alpha가 클수록 최근 값 반영 비중이 높음
    """
    B, E, T = pmf.shape
    pmf_smooth = pmf.clone()

    for t in range(1, T):
        pmf_smooth[:, :, t] = alpha * pmf[:, :, t] + (1 - alpha) * pmf_smooth[:, :, t-1]

    return pmf_smooth

# 기본 DeepHit 모델
class DeepHitSurv(nn.Module) :

    def __init__(self, input_dim, hidden_size=(128, 64), time_bins=91, num_events=4, dropout=0.2) :
        """
        
        input_dim : 특성의 개수
        hidden_size : hidden layer의 node 개수 (h1, h2)
        time_bins : 시간축을 나눌 개수 (출력의 개수)
        num_event : 각각 사건의 개수

        """

        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins
        
        """
        모델의 구조 : 
        features -> h1 -> h2
        h2 -> 4개의 사건 heads -> time bins
        """
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Linear(h2, time_bins) for _ in range(num_events)     # 사건 수만큼의 heads 생성
        ])

    def forward(self, x) :
        s = self.shared(x)
        logits = torch.stack([head(s) for head in self.heads], dim=1)

        batch_size, num_events, time_bins = logits.shape

        # 사건+시간 전체에 대해 softmax 적용
        pmf = F.softmax(logits.view(batch_size, -1), dim=1)
        pmf = pmf.view(batch_size, num_events, time_bins)

        # 시간에 따라 누적합 → CIF 계산
        cif = torch.cumsum(pmf, dim=-1)

        return logits, pmf, cif

# SEBlock을 결합한 Deephit 모델
class DeepHitSurvWithSEBlock(nn.Module):
    def __init__(self, input_dim, hidden_size=(128, 64), time_bins=91, num_events=4, dropout=0.2, se_ratio=0.25):
        """
        input_dim : 특성의 개수
        hidden_size : hidden layer의 node 개수 (h1, h2)
        time_bins : 시간축을 나눌 개수 (출력의 개수)
        num_event : 각각 사건의 개수
        se_ratio : SEBlock에서 사용할 노드의 크기 비율
        """
        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins
        self.input_dim = input_dim

        # 모델 처음에 붙는 SEBlock : Feature weighting의 역할을 함
        se_hidden = max(1, int(input_dim * se_ratio))
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, input_dim),
            nn.Sigmoid()
        )

        # 각 branch에 붙는 SEBlock : 각 사건마다 중요한 Feature를 Selection
        se_hidden_shared = max(1, int(h2 * se_ratio))
        self.se_block_event = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h2, se_hidden_shared),
                nn.ReLU(),
                nn.Linear(se_hidden_shared, h2),
                nn.Sigmoid()
            ) for _ in range(num_events)
        ])

        """
        모델의 구조 :
        features -> SEBlock -> h1 -> h2
        h2 + features -> 4개의 사건 heads -> SEBlock -> time bins
        """
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.LayerNorm(h2),
        )

        self.heads = nn.ModuleList([
            nn.Linear(h2, time_bins) for _ in range(num_events)
        ])

    def forward(self, x):
        scale = self.se_block(x)        # SEBlock 수행
        x_scaled = x * scale            # x를 스케일링

        s = self.shared(x_scaled)       # 공유 브랜치 통과

        logits_list = []
        for k in range(self.num_events):
            s_scaled = s * self.se_block_event[k](s)        # 각 브랜치에 대해 SEBlock 통과
            logits_list.append(self.heads[k](s_scaled))     # 사건별 head 계산

        logits = torch.stack(logits_list, dim=1)

        batch_size, num_events, time_bins = logits.shape

        # 사건+시간 전체에 대해 softmax 적용
        pmf = F.softmax(logits.view(batch_size, -1), dim=1)
        pmf = pmf.view(batch_size, num_events, time_bins)

        # 시간에 따라 누적합 → CIF 계산
        cif = torch.cumsum(pmf, dim=-1)

        return logits, pmf, cif

# Concat을 추가한 모델 구현
class DeepHitSurvWithSEBlockConcat(nn.Module):
    def __init__(self, input_dim, hidden_size=(128, 64), time_bins=91, num_events=4, dropout=0.2, se_ratio=0.25):
        """
        input_dim : 입력 특성 개수
        hidden_size : 공유 레이어 hidden node 수 (h1, h2)
        time_bins : 시간축 분할 수
        num_events : 사건 개수
        se_ratio : SEBlock 내부 차원 비율
        """
        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins
        self.input_dim = input_dim
        self.h2 = h2

        # 초기 SEBlock: 입력 feature weighting
        se_hidden = max(1, int(input_dim * se_ratio))
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, input_dim),
            nn.Sigmoid()
        )

        # 사건별 SEBlock: 사건별 feature weighting
        se_hidden_shared = max(1, int(h2 * se_ratio))
        self.se_block_event = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h2, se_hidden_shared),
                nn.ReLU(),
                nn.Linear(se_hidden_shared, h2),
                nn.Sigmoid()
            ) for _ in range(num_events)
        ])

        # 공유 레이어
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 사건별 head: 입력 dim = h2 + input_dim
        self.heads = nn.ModuleList([
            nn.Linear(h2 + input_dim, time_bins) for _ in range(num_events)
        ])

    def forward(self, x):
        # 1) 입력 SEBlock
        scale = self.se_block(x)
        x_scaled = x * scale

        # 2) 공유 브랜치
        s = self.shared(x_scaled)

        logits_list = []
        for k in range(self.num_events):
            # 사건별 SEBlock 적용
            s_scaled = s * self.se_block_event[k](s)

            # 공유 feature + 원본 feature concatenation
            s_combined = torch.cat([s_scaled, x], dim=-1)  # shape: (batch_size, h2 + input_dim)

            # 사건별 head
            logits_list.append(self.heads[k](s_combined))

        logits = torch.stack(logits_list, dim=1)  # shape: (batch_size, num_events, time_bins)
        batch_size, num_events, time_bins = logits.shape

        # 사건+시간 전체에 대해 softmax 적용
        pmf = F.softmax(logits.view(batch_size, -1), dim=1)
        pmf = pmf.view(batch_size, num_events, time_bins)

        # 시간에 따라 누적합 → CIF 계산
        cif = torch.cumsum(pmf, dim=-1)

        return logits, pmf, cif

# 1D CNN을 추가한 Deephit 모델
class DeepHitSurvWithSEBlockCNN(nn.Module):
    def __init__(self, input_dim, hidden_size=(128,64), time_bins=91, num_events=4, dropout=0.2, se_ratio=0.25, cnn_kernel=3):
        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins
        self.input_dim = input_dim

        # Input SEBlock
        se_hidden = max(1, int(input_dim*se_ratio))
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, input_dim),
            nn.Sigmoid()
        )

        # Branch SEBlock
        se_hidden_shared = max(1, int(h2*se_ratio))
        self.se_block_event = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h2, se_hidden_shared),
                nn.ReLU(),
                nn.Linear(se_hidden_shared, h2),
                nn.Sigmoid()
            ) for _ in range(num_events)
        ])

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Linear heads
        self.heads = nn.ModuleList([
            nn.Linear(h2, time_bins) for _ in range(num_events)
        ])

        # 사건별 1D CNN (시간 축 smoothing)
        self.cnn_blocks = nn.ModuleList([
            nn.Conv1d(in_channels=time_bins, out_channels=time_bins, kernel_size=cnn_kernel, padding=cnn_kernel//2, groups=time_bins)
            for _ in range(num_events)
        ])

    def forward(self, x):
        # Input SEBlock
        scale = self.se_block(x)
        x_scaled = x * scale

        # Shared layers
        s = self.shared(x_scaled)

        logits_list = []
        pmf_list = []
        cif_list = []

        for k in range(self.num_events):
            # Branch SEBlock
            s_scaled = s * self.se_block_event[k](s)

            # Linear head
            logits = self.heads[k](s_scaled)  # (B, T)

            # CNN 적용: 시간축만 smoothing, 채널 유지
            # Conv1d expects (B, C, L), 여기서 C=time_bins, L=1? → groups=time_bins 사용
            logits_cnn = logits.unsqueeze(2)  # (B, T, 1)
            logits_smooth = self.cnn_blocks[k](logits_cnn)  # (B, T, 1)
            logits_smooth = logits_smooth.squeeze(2)         # (B, T)

            # Softmax & CIF
            pmf = F.softmax(logits_smooth, dim=-1)  # (B, T)
            cif = torch.cumsum(pmf, dim=-1)         # (B, T)

            logits_list.append(logits_smooth)
            pmf_list.append(pmf)
            cif_list.append(cif)

        # 사건별 합치기 -> (B, K, T)
        logits = torch.stack(logits_list, dim=1)
        batch_size, num_events, time_bins = logits.shape

        # 사건+시간 전체에 대해 softmax 적용
        pmf = F.softmax(logits.view(batch_size, -1), dim=1)
        pmf = pmf.view(batch_size, num_events, time_bins)

        # 시간에 따라 누적합 → CIF 계산
        cif = torch.cumsum(pmf, dim=-1)

        return logits, pmf, cif

# 2D CNN을 추가한 Deephit 모델
class DeepHitSurvWithSEBlockAnd2DCNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=(128, 64),
        time_bins=91,
        num_events=4,
        dropout=0.2,
        se_ratio=0.25,
        conv_channels=(8, 16),
        conv_kernel_sizes=((2, 5), (2, 3)),
        conv_padding=((1, 2), (0, 1)),
        use_conv_batchnorm=False,
    ):
        super().__init__()
        h1, h2 = hidden_size
        self.num_events = num_events
        self.T = time_bins
        self.input_dim = input_dim

        # ----- [1] 입력 feature weighting용 SEBlock -----
        se_hidden = max(1, int(input_dim * se_ratio))
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, se_hidden),
            nn.ReLU(),
            nn.Linear(se_hidden, input_dim),
            nn.Sigmoid()
        )

        # ----- [2] 사건별 feature weighting용 SEBlock -----
        se_hidden_shared = max(1, int(h2 * se_ratio))
        self.se_block_event = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h2, se_hidden_shared),
                nn.ReLU(),
                nn.Linear(se_hidden_shared, h2),
                nn.Sigmoid()
            ) for _ in range(num_events)
        ])

        # ----- [3] 공유층 -----
        self.shared = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- [4] 사건별 head (각각 time_bins 출력) -----
        self.heads = nn.ModuleList([
            nn.Linear(h2, time_bins) for _ in range(num_events)
        ])

        # ----- [5] 2D CNN block (시간/사건 상관관계 학습) -----
        if len(conv_channels) != 2:
            raise ValueError("conv_channels must be a tuple of length 2, e.g., (8, 16)")
        if len(conv_kernel_sizes) != 2:
            raise ValueError("conv_kernel_sizes must be a tuple of length 2")
        if conv_padding is None:
            conv_padding = tuple((k[0] // 2, k[1] // 2) for k in conv_kernel_sizes)
        if len(conv_padding) != 2:
            raise ValueError("conv_padding must be None or a tuple of length 2")

        conv_layers = [
            nn.Conv2d(1, conv_channels[0], kernel_size=conv_kernel_sizes[0], padding=conv_padding[0]),
        ]
        if use_conv_batchnorm:
            conv_layers.append(nn.BatchNorm2d(conv_channels[0]))
        conv_layers.extend([
            nn.ReLU(),
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=conv_kernel_sizes[1], padding=conv_padding[1]),
        ])
        if use_conv_batchnorm:
            conv_layers.append(nn.BatchNorm2d(conv_channels[1]))
        conv_layers.extend([
            nn.ReLU(),
            nn.Conv2d(conv_channels[1], 1, kernel_size=1),  # 다시 channel=1로 축소
        ])
        self.conv2d_block = nn.Sequential(*conv_layers)

    def forward(self, x):
        # ----- [1] 입력 SEBlock -----
        scale = self.se_block(x)
        x_scaled = x * scale

        # ----- [2] shared representation -----
        s = self.shared(x_scaled)

        # ----- [3] 사건별 branch -----
        logits_list = []
        for k in range(self.num_events):
            s_scaled = s * self.se_block_event[k](s)
            logits_list.append(self.heads[k](s_scaled))

        # ----- [4] stack하여 (B, 4, 91) 만들기 -----
        logits = torch.stack(logits_list, dim=1)  # (B, num_events, time_bins)

        # ----- [5] 2D CNN 적용 -----
        # Conv2d 입력 형태로 reshape: (B, 1, 4, 91)
        x_cnn = logits.unsqueeze(1)
        logits_refined = self.conv2d_block(x_cnn).squeeze(1)  # (B, 4, 91)

        # ----- [6] softmax + cumsum -----
        batch_size, num_events, time_bins = logits_refined.shape

        # 사건+시간 전체에 대해 softmax 적용
        pmf = F.softmax(logits_refined.view(batch_size, -1), dim=1)
        pmf = pmf.view(batch_size, num_events, time_bins)

        # 시간에 따라 누적합 → CIF 계산
        cif = torch.cumsum(pmf, dim=-1)

        return logits_refined, pmf, cif

# 손실 함수 정의 : likelihood + ranking loss
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

def compute_risk_score_sigmoid(pmf, time_lambda=0.05, event_weights=None):
    """
    pmf: torch.Tensor, shape (B, E, T) - 사건별 시간 확률
    time_lambda: float, 지수 감쇠 계수 (시간대 가중치)
    event_weights: list or torch.Tensor, 길이 E, 사건별 가중치
    """
    B, E, T = pmf.shape
    device = pmf.device

    # 시간 가중치
    time_weights = torch.exp(-time_lambda * torch.arange(T, device=device))
    
    # 사건 가중치
    if event_weights is None:
        event_weights = torch.ones(E, device=device)
    else:
        event_weights = torch.tensor(event_weights, device=device, dtype=torch.float32)
    
    # 가중치 적용
    weighted_pmf = pmf * time_weights.view(1, 1, T)
    weighted_pmf = weighted_pmf * event_weights.view(1, E, 1)

    # 가중합 계산
    risk_score_raw = weighted_pmf.sum(dim=(1, 2))

    # 0 기준으로 offset 제거 → 음수도 나오게
    # risk_score_raw = risk_score_raw - risk_score_raw.mean()

    # 시그모이드 + 0~100 스케일
    risk_score = torch.sigmoid(risk_score_raw) * 100

    return risk_score

def set_seed(seed = 42) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN 비결정적 동작 방지 (연산 재현성 확보)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


