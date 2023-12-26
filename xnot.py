import torch
import random
import torch.nn as nn
from torch import Tensor


class NegAbs(nn.Module):
    def __init__(self):
        super(NegAbs, self).__init__()

    def forward(self, input):
        return -torch.abs(input)


def sq_cost(X: Tensor, Y: Tensor):
    return (X - Y).square().flatten(start_dim=1).mean(dim=1)


def sample_from_tensor(tensor: Tensor, batch_size: int = 64) -> Tensor:
    indices = random.choices(range(tensor.shape[0]), k=batch_size)
    return tensor[indices]


class XNot(nn.Module):
    def __init__(self, input_shape: int, device: str = 'cpu') -> None:
        super().__init__()
        self.input_shape = input_shape
        self.device = device

        self.T = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, input_shape),
        ).to(device)

        self.f = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            NegAbs(),
        ).to(device)

    def fit(self,
            query_seq: Tensor,
            matching_set: Tensor,
            max_steps: int = 20000,
            t_iters: int = 10,
            cost: str = "sq_cost",
            batch_size: int = 64,
            W: float = 1.0,
            ) -> None:
        if cost == "sq_cost":
            cost_fn = sq_cost
        else:
            raise NotImplementedError(f"Cost function {cost} not implemented.")

        T_opt = torch.optim.Adam(self.T.parameters(), lr=1e-4, weight_decay=1e-10)
        f_opt = torch.optim.Adam(self.f.parameters(), lr=1e-4, weight_decay=1e-10)

        for _ in range(max_steps):  # step
            for _ in range(t_iters):  # t_iter
                X = sample_from_tensor(query_seq, batch_size=batch_size).clone().detach()
                T_loss = cost_fn(X, self.T(X)).mean() - self.f(self.T(X)).mean()
                T_opt.zero_grad();
                T_loss.backward();
                T_opt.step()

            # f optimization
            self.T.eval();
            self.f.train(True)
            X = sample_from_tensor(query_seq, batch_size=batch_size).clone().detach()
            Y = sample_from_tensor(matching_set, batch_size=batch_size).clone().detach()

            X = X.to(device=self.device)
            Y = Y.to(device=self.device)
            f_loss = self.f(self.T(X)).mean() - (W * self.f(Y)).mean()
            f_opt.zero_grad();
            f_loss.backward();
            f_opt.step()

    def predict(self, X: Tensor) -> Tensor:
        return self.T(X)
