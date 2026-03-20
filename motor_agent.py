"""
motor_agent.py  ——  PPO 智能体（连续动作 + Squashed Gaussian + 多约束 Cost Critics）
=================================================================================
职责：
  - Actor (Squashed Gaussian Policy)
  - Critic (V)
  - 多约束 Cost Critics (V_c_i)
  - PPO update
  - 自适应 entropy / exploration
  - 动作原始值与执行值区分
"""

import math, os, sys
import numpy as np
# Windows + Python 3.13 需要手动注册 torch DLL 目录
if sys.platform == "win32":
    _torch_lib = os.path.join(os.path.dirname(__import__("importlib").import_module("torch").__file__), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(_torch_lib)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional

# ============================================================
#  网络构建工具
# ============================================================
def build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int],
              activation: str = "relu", output_activation: str = "none") -> nn.Sequential:
    """构建多层感知机"""
    act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn.get(activation, nn.ReLU)())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# ============================================================
#  Squashed Gaussian Actor
# ============================================================
class SquashedGaussianActor(nn.Module):
    """
    压缩高斯策略网络：输出连续有界动作 [-1, 1]
    通过 tanh 压缩，前期方差大鼓励探索，后期趋于确定性
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation="relu")
        self.mu_head     = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回 (mu, log_std)"""
        h = self.backbone(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作（带 reparameterization trick）
        返回 (squashed_action, log_prob, raw_action)
        """
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        raw_action = dist.rsample()                          # reparameterized
        squashed_action = torch.tanh(raw_action)             # 压缩到 [-1,1]

        # log_prob with tanh correction
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return squashed_action, log_prob, raw_action

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的 log_prob 和 entropy
        输入原始（未压缩）动作，输出正确的梯度
        返回 (log_prob, entropy, squashed_action)
        """
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        log_prob = dist.log_prob(raw_action)
        squashed = torch.tanh(raw_action)
        log_prob -= torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, squashed

    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """计算策略熵（用于自适应探索）"""
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        return dist.entropy().sum(dim=-1, keepdim=True)


# ============================================================
#  Critic（价值网络）
# ============================================================
class Critic(nn.Module):
    """V(s) 价值网络"""
    def __init__(self, obs_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden_dims, activation="relu")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ============================================================
#  Rollout Buffer
# ============================================================
class RolloutBuffer:
    """PPO 轨迹数据缓冲区"""

    def __init__(self):
        self.obs          = []
        self.actions_raw  = []    # 原始值（未压缩）
        self.actions_exec = []    # 执行值（tanh 压缩后）
        self.rewards      = []
        self.dones         = []
        self.log_probs     = []
        self.values        = []
        self.costs         = {name: [] for name in
                              ["speed", "distance", "smoothness", "projection", "window_energy"]}
        self.cost_values   = {name: [] for name in
                              ["speed", "distance", "smoothness", "projection", "window_energy"]}

    def store(self, obs, action_raw, action_exec, reward, done, log_prob, value,
              costs: dict, cost_values: dict):
        """存储一步交互数据"""
        self.obs.append(obs)
        self.actions_raw.append(action_raw)
        self.actions_exec.append(action_exec)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        for name in costs:
            self.costs[name].append(costs[name])
            self.cost_values[name].append(cost_values[name])

    def clear(self):
        """清空缓冲区"""
        self.__init__()

    def to_tensors(self, device: torch.device) -> dict:
        """转换为 tensor dict"""
        data = {
            "obs":          torch.FloatTensor(np.array(self.obs)).to(device),
            "actions_raw":  torch.FloatTensor(np.array(self.actions_raw)).to(device),
            "actions_exec": torch.FloatTensor(np.array(self.actions_exec)).to(device),
            "rewards":      torch.FloatTensor(np.array(self.rewards)).to(device),
            "dones":        torch.FloatTensor(np.array(self.dones, dtype=np.float32)).to(device),
            "log_probs":    torch.FloatTensor(np.array(self.log_probs)).to(device),
            "values":       torch.FloatTensor(np.array(self.values)).to(device),
        }
        for name in self.costs:
            data[f"cost_{name}"]       = torch.FloatTensor(np.array(self.costs[name])).to(device)
            data[f"cost_value_{name}"] = torch.FloatTensor(np.array(self.cost_values[name])).to(device)
        return data

    def __len__(self):
        return len(self.obs)


# ============================================================
#  PPO Agent（主体）
# ============================================================
class PPOAgent:
    """
    连续动作 PPO 智能体
      - Squashed Gaussian policy
      - 主 Critic + 多约束 Cost Critics
      - PPO-Clip update
      - 自适应 entropy 系数
      - Lagrangian multiplier 由外部 training 模块管理
    """

    CONSTRAINT_NAMES = ["speed", "distance", "smoothness", "projection", "window_energy"]

    def __init__(self,
                 obs_dim: int,
                 act_dim: int = 3,
                 hidden_dims: List[int] = [256, 256],
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_eps: float = 0.2,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 256,
                 max_grad_norm: float = 0.5,
                 target_entropy: Optional[float] = None,
                 entropy_coeff: float = 0.01,
                 entropy_lr: float = 1e-4,
                 device: str = "cpu",
                 ):
        """初始化 PPO Agent"""
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # ---- 网络 ----
        self.actor  = SquashedGaussianActor(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic = Critic(obs_dim, hidden_dims).to(self.device)
        self.cost_critics = nn.ModuleDict({
            name: Critic(obs_dim, hidden_dims).to(self.device)
            for name in self.CONSTRAINT_NAMES
        })

        # ---- 优化器 ----
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.cost_critic_optimizers = {
            name: torch.optim.Adam(self.cost_critics[name].parameters(), lr=lr_critic)
            for name in self.CONSTRAINT_NAMES
        }

        # ---- 自适应 entropy ----
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim * 0.5
        self.log_entropy_coeff = torch.tensor(math.log(entropy_coeff), requires_grad=True, device=self.device)
        self.entropy_optimizer = torch.optim.Adam([self.log_entropy_coeff], lr=entropy_lr)

        # ---- buffer ----
        self.buffer = RolloutBuffer()

    # ----------------------------------------------------------
    @property
    def entropy_coeff(self) -> float:
        return self.log_entropy_coeff.exp().item()

    # ----------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs: np.ndarray,
                      deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float, dict]:
        """
        选择动作（推理时调用）
        返回 (action_exec, action_raw, log_prob, value, cost_values)
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if deterministic:
            mu, log_std = self.actor.forward(obs_t)
            action_raw = mu
            action_exec = torch.tanh(mu)
            dist = Normal(mu, torch.exp(log_std))
            log_prob = dist.log_prob(action_raw)
            log_prob -= torch.log(1 - action_exec.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        else:
            action_exec, log_prob, action_raw = self.actor.sample(obs_t)
        value = self.critic(obs_t).item()

        cost_vals = {}
        for name in self.CONSTRAINT_NAMES:
            cost_vals[name] = self.cost_critics[name](obs_t).item()

        return (
            action_exec.cpu().numpy().flatten(),
            action_raw.cpu().numpy().flatten(),
            log_prob.item(),
            value,
            cost_vals,
        )

    # ----------------------------------------------------------
    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> float:
        """获取状态价值"""
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        return self.critic(obs_t).item()

    # ----------------------------------------------------------
    def compute_gae(self, rewards, values, dones, last_value, gamma, lam) -> Tuple[np.ndarray, np.ndarray]:
        """计算 GAE 优势函数与目标回报"""
        n = len(rewards)
        advantages = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values)
        return advantages, returns

    # ----------------------------------------------------------
    def update(self, last_obs: np.ndarray,
               lagrangian_lambdas: Dict[str, float]) -> dict:
        """
        PPO 更新
        :param last_obs: episode 结束时的最终观测（用于 bootstrap）
        :param lagrangian_lambdas: 各约束的拉格朗日乘子
        :return: 训练统计 dict
        """
        data = self.buffer.to_tensors(self.device)
        n = len(self.buffer)

        if n < self.mini_batch_size:
            self.buffer.clear()
            return {"skipped": True}

        # ---- GAE for reward ----
        rewards_np = data["rewards"].cpu().numpy()
        values_np  = data["values"].cpu().numpy().flatten()
        dones_np   = data["dones"].cpu().numpy().flatten()
        last_value = self.get_value(last_obs)
        advantages, returns = self.compute_gae(rewards_np, values_np, dones_np,
                                               last_value, self.gamma, self.gae_lambda)

        # ---- GAE for each cost ----
        cost_advantages = {}
        cost_returns = {}
        for name in self.CONSTRAINT_NAMES:
            c_rewards = data[f"cost_{name}"].cpu().numpy()
            c_values  = data[f"cost_value_{name}"].cpu().numpy().flatten()
            c_last = 0.0
            c_adv, c_ret = self.compute_gae(c_rewards, c_values, dones_np, c_last,
                                            self.gamma, self.gae_lambda)
            cost_advantages[name] = c_adv
            cost_returns[name] = c_ret

        # to tensors
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        cost_adv_t = {name: torch.FloatTensor(cost_advantages[name]).to(self.device)
                      for name in self.CONSTRAINT_NAMES}
        cost_ret_t = {name: torch.FloatTensor(cost_returns[name]).to(self.device)
                      for name in self.CONSTRAINT_NAMES}

        obs_t          = data["obs"]
        actions_raw_t  = data["actions_raw"]
        old_log_probs  = data["log_probs"].flatten()

        # ---- PPO epochs ----
        indices = np.arange(n)
        stats = {"actor_loss": [], "critic_loss": [], "entropy": [], "entropy_coeff": [],
                 "clip_frac": []}
        for name in self.CONSTRAINT_NAMES:
            stats[f"cost_critic_loss_{name}"] = []

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                mb = indices[start:end]

                mb_obs       = obs_t[mb]
                mb_raw_act   = actions_raw_t[mb]
                mb_old_logp  = old_log_probs[mb]
                mb_adv       = adv_t[mb]
                mb_ret       = ret_t[mb]

                # ---- Actor loss ----
                new_log_prob, entropy, _ = self.actor.evaluate(mb_obs, mb_raw_act)
                new_log_prob = new_log_prob.flatten()
                entropy = entropy.mean()

                ratio = torch.exp(new_log_prob - mb_old_logp)
                clip_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)

                # 主奖励优势
                total_adv = mb_adv.clone()
                # 加上拉格朗日约束惩罚
                for name in self.CONSTRAINT_NAMES:
                    lam_val = lagrangian_lambdas.get(name, 0.0)
                    if lam_val > 0:
                        total_adv = total_adv - lam_val * cost_adv_t[name][mb]

                surr1 = ratio * total_adv
                surr2 = clip_ratio * total_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # 自适应 entropy
                entropy_coeff = self.log_entropy_coeff.exp()
                actor_loss = actor_loss - entropy_coeff.detach() * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # ---- Entropy coeff update ----
                entropy_loss = -(self.log_entropy_coeff * (entropy.detach() - self.target_entropy))
                self.entropy_optimizer.zero_grad()
                entropy_loss.backward()
                self.entropy_optimizer.step()

                # ---- Critic loss ----
                v_pred = self.critic(mb_obs).flatten()
                critic_loss = F.mse_loss(v_pred, mb_ret)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # ---- Cost critics loss ----
                for name in self.CONSTRAINT_NAMES:
                    c_pred = self.cost_critics[name](mb_obs).flatten()
                    c_loss = F.mse_loss(c_pred, cost_ret_t[name][mb])
                    self.cost_critic_optimizers[name].zero_grad()
                    c_loss.backward()
                    nn.utils.clip_grad_norm_(self.cost_critics[name].parameters(), self.max_grad_norm)
                    self.cost_critic_optimizers[name].step()
                    stats[f"cost_critic_loss_{name}"].append(c_loss.item())

                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                stats["actor_loss"].append(actor_loss.item())
                stats["critic_loss"].append(critic_loss.item())
                stats["entropy"].append(entropy.item())
                stats["entropy_coeff"].append(entropy_coeff.item())
                stats["clip_frac"].append(clip_frac)

        self.buffer.clear()

        # 平均
        summary = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in stats.items()}
        return summary

    # ----------------------------------------------------------
    def save(self, path: str):
        """保存模型参数"""
        state = {
            "actor":                self.actor.state_dict(),
            "critic":               self.critic.state_dict(),
            "actor_optimizer":      self.actor_optimizer.state_dict(),
            "critic_optimizer":     self.critic_optimizer.state_dict(),
            "log_entropy_coeff":    self.log_entropy_coeff.detach().cpu(),
        }
        for name in self.CONSTRAINT_NAMES:
            state[f"cost_critic_{name}"] = self.cost_critics[name].state_dict()
            state[f"cost_critic_opt_{name}"] = self.cost_critic_optimizers[name].state_dict()
        torch.save(state, path)

    # ----------------------------------------------------------
    def load(self, path: str):
        """加载模型参数"""
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "log_entropy_coeff" in state:
            self.log_entropy_coeff.data.copy_(state["log_entropy_coeff"])
        for name in self.CONSTRAINT_NAMES:
            if f"cost_critic_{name}" in state:
                self.cost_critics[name].load_state_dict(state[f"cost_critic_{name}"])
            if f"cost_critic_opt_{name}" in state:
                self.cost_critic_optimizers[name].load_state_dict(state[f"cost_critic_opt_{name}"])
