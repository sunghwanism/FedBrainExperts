import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class KLIEP(nn.Module):
    def __init__(self, input_dim):
        super(KLIEP, self).__init__()
        # 선형 레이어를 사용해 중요도 가중치 추정
        self.linear = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        return torch.exp(self.linear(x))  # 가중치의 지수값을 사용해 비율을 추정

# 중요도 가중치 함수 정의
def kliep_loss(weights, q_samples):
    # KLIEP의 목적 함수: Q 분포의 로그 우도를 최대화
    return -torch.mean(weights) + torch.log(torch.mean(weights * q_samples))

# 데이터셋 정의 (예: 소스와 타겟 데이터)
np.random.seed(0)
torch.manual_seed(0)

# 소스 분포(P)에서 추출한 샘플
P = torch.tensor(np.random.normal(0, 1, (100, 2)), dtype=torch.float32)

# 타겟 분포(Q)에서 추출한 샘플
Q = torch.tensor(np.random.normal(1, 1, (100, 2)), dtype=torch.float32)

# KLIEP 모델 초기화
input_dim = P.shape[1]
kliep_model = KLIEP(input_dim)

# 옵티마이저 정의
optimizer = optim.SGD(kliep_model.parameters(), lr=0.01)

# 최적화 루프 정의
num_epochs = 500
batch_size = 20
for epoch in range(num_epochs):
    # 소스 데이터(P)에서 미니배치 샘플링
    indices_p = torch.randperm(P.size(0))[:batch_size]
    p_batch = P[indices_p]

    # 타겟 데이터(Q)에서 미니배치 샘플링
    indices_q = torch.randperm(Q.size(0))[:batch_size]
    q_batch = Q[indices_q]

    # 모델을 통해 중요도 가중치 예측
    weights = kliep_model(p_batch)

    # KLIEP 손실 계산
    loss = kliep_loss(weights, q_batch)

    # 역전파 및 옵티마이저 스텝
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 학습 과정 출력
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 최종 중요도 가중치 추정 결과 출력
final_weights = kliep_model(P).detach().numpy()
print("Final Importance Weights:", final_weights)