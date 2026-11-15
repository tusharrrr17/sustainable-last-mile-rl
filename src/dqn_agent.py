import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """3-Layer MLP Policy/Target Network (18-D State -> 15 Vehicle Archetype Q-Values)."""

    def __init__(self, state_dim: int = 18, action_dim: int = 15, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size Experience Replay Buffer storing masked transitions."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MaskedDQNAgent:
    """
    Covers AI Model Development (Plan IDs 31-35):
    - ID 31: Policy & Target Q-Network setup
    - ID 32: Action-masked epsilon-greedy exploration
    - ID 33: Experience replay & Huber-loss Bellman updates with soft target sync
    """

    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 15,
        lr: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.02,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, valid_mask: np.ndarray, epsilon: float = 0.0) -> int:
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return 0
        if random.random() < epsilon:
            return int(random.choice(valid_indices))

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()
            masked_q = np.where(valid_mask, q_values, -1e9)
            return int(np.argmax(masked_q))

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(self.batch_size)

        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        next_masks_t = torch.tensor(next_masks, device=self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            next_q_all = self.target_net(next_states_t)
            next_q_all = next_q_all.masked_fill(~next_masks_t, -1e9)
            max_next_q = next_q_all.max(dim=1)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Soft target network update
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def save(self, path: str = "models/dqn_model_v1.pth"):
        torch.save(
            {
                "state_dict": self.policy_net.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path: str = "models/dqn_model_v1.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["state_dict"])
        self.target_net.load_state_dict(checkpoint["state_dict"])
        self.policy_net.eval()
