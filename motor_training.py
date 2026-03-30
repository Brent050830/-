"""
motor_training.py  ——  训练模块
================================
职责：
  - 单阶段训练 train_stage(...)
  - 多阶段训练流程衔接（track → energy → polish）
  - rollout 交互数据收集 / eval 评估汇总
  - Lagrangian multiplier 更新（integral / PID 两种方式）
  - checkpoint 保存与恢复（选择最佳模型）
  - 输出评估指标
"""

import os, json, copy, time
import numpy as np
from typing import Dict, List, Optional, Tuple

from motor_env import MotorEnv, generate_road, generate_reference_trajectory, load_efficiency_map, STYLE_PROFILES
from motor_agent import PPOAgent


def score_energy_metrics_for_selection(metrics: dict) -> float:
    """
    选择 energy/checkpoint 时使用的统一评分：
    先保证可用性，再尽量最大化真实节能，同时兼顾控制口径节能。
    """
    total_saving = float(metrics.get("saving_total_pct", -999.0))
    cmp_saving = float(metrics.get("saving_cmp_total_pct", -999.0))
    score = total_saving
    score += 0.20 * cmp_saving

    if not metrics.get("tracking_ok", False):
        score -= 100.0
    if not metrics.get("bias_guard_ok", False):
        score -= 50.0
    if not metrics.get("smooth_ok", True):
        score -= 20.0
    return score

# ============================================================
#  拉格朗日乘子管理器
# ============================================================
class LagrangianManager:
    """
    管理多约束拉格朗日乘子
    支持 integral（默认）和 PID 两种更新方式
    检测到严重违约时临时切换 PID 快速响应
    """

    def __init__(self, constraint_names: List[str],
                 init_lambda: float = 0.1,
                 lr_lambda: float = 0.01,
                 max_lambda: float = 10.0,
                 thresholds: Optional[Dict[str, float]] = None,
                 pid_kp: float = 0.5, pid_ki: float = 0.1, pid_kd: float = 0.05,
                 severe_violation_ratio: float = 3.0):
        """
        初始化拉格朗日乘子管理器
        :param constraint_names: 约束名称列表
        :param init_lambda: 初始乘子值
        :param lr_lambda: 乘子更新学习率
        :param max_lambda: 乘子上界
        :param thresholds: 各约束阈值 dict
        :param pid_kp/ki/kd: PID 参数
        :param severe_violation_ratio: 严重违约检测倍数
        """
        self.names = constraint_names
        self.lambdas = {name: init_lambda for name in constraint_names}
        self.lr = lr_lambda
        self.max_lambda = max_lambda
        self.thresholds = thresholds or {name: 0.05 for name in constraint_names}

        # PID 相关
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.severe_ratio = severe_violation_ratio
        self.integral_accum = {name: 0.0 for name in constraint_names}
        self.prev_error     = {name: 0.0 for name in constraint_names}
        self.mode = {name: "integral" for name in constraint_names}  # integral / pid

    def update(self, avg_costs: Dict[str, float]) -> Dict[str, float]:
        """
        更新拉格朗日乘子
        :param avg_costs: 各约束的平均 cost
        :return: 更新后的 lambda dict
        """
        for name in self.names:
            cost_val = avg_costs.get(name, 0.0)
            threshold = self.thresholds[name]
            error = cost_val - threshold

            # 检测严重违约 → 切换 PID
            if cost_val > threshold * self.severe_ratio:
                self.mode[name] = "pid"
            elif cost_val < threshold * 1.5:
                self.mode[name] = "integral"

            if self.mode[name] == "integral":
                # integral 更新
                self.lambdas[name] += self.lr * error
            else:
                # PID 更新
                self.integral_accum[name] += error
                derivative = error - self.prev_error[name]
                pid_output = (self.pid_kp * error +
                              self.pid_ki * self.integral_accum[name] +
                              self.pid_kd * derivative)
                self.lambdas[name] += self.lr * pid_output
                self.prev_error[name] = error

            # 限幅
            self.lambdas[name] = np.clip(self.lambdas[name], 0.0, self.max_lambda)

        return self.lambdas.copy()

    def get_lambdas(self) -> Dict[str, float]:
        return self.lambdas.copy()

    def state_dict(self) -> dict:
        return {"lambdas": self.lambdas.copy(),
                "integral_accum": self.integral_accum.copy(),
                "prev_error": self.prev_error.copy(),
                "mode": self.mode.copy()}

    def load_state_dict(self, d: dict):
        self.lambdas = d["lambdas"]
        self.integral_accum = d["integral_accum"]
        self.prev_error = d["prev_error"]
        self.mode = d["mode"]


# ============================================================
#  Rollout 收集
# ============================================================
def rollout_episode(env: MotorEnv, agent: PPOAgent) -> Tuple[float, dict, np.ndarray]: # 收集一个 episode 的交互数据到 agent.buffer，返回 (episode_reward, episode_metrics, last_obs)
    """
    运行一个 episode，收集轨迹到 agent.buffer
    返回 (episode_reward, episode_metrics, last_obs)
    """
    obs = env.reset()
    total_reward = 0.0
    done = False
    infos = []

    while not done: # 每一步交互：根据当前 obs 选择 action，执行 env.step(action)，存储 transition 到 buffer，并累积 reward
        action_exec, action_raw, log_prob, value, cost_values = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action_exec)

        costs = info["costs"]
        agent.buffer.store(
            obs=obs,
            action_raw=action_raw,
            action_exec=action_exec,
            reward=reward,
            done=float(done),
            log_prob=log_prob,
            value=value,
            costs=costs,
            cost_values=cost_values,
        )
        total_reward += reward
        infos.append(info)
        obs = next_obs

    metrics = env.get_episode_metrics()
    return total_reward, metrics, obs


# ============================================================
#  Eval（不收集 buffer）
# ============================================================
def eval_episode(env: MotorEnv, agent: PPOAgent) -> Tuple[float, dict, List[dict]]: # 评估一个 episode，不存入 buffer，返回 (episode_reward, episode_metrics, step_infos)，也就是在上一个函数的基础上进行评估
    """
    评估一个 episode（不存入 buffer）
    返回 (episode_reward, metrics, step_infos)
    """
    obs = env.reset()
    total_reward = 0.0
    done = False
    step_infos = []

    while not done:
        action_exec, _, _, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action_exec)
        total_reward += reward
        step_infos.append(info)

    metrics = env.get_episode_metrics()
    return total_reward, metrics, step_infos


# ============================================================
#  单阶段训练
# ============================================================
def train_stage(agent: PPOAgent,
                road: dict,
                ref: dict,
                eff_map: dict,
                style: str = "normal",
                mode: str = "track",
                num_episodes: int = 200,
                energy_weight: float = 0.0,
                energy_weight_schedule: Optional[List[float]] = None,
                residual_torque_scale: float = 6.0,
                residual_torque_scale_schedule: Optional[List[float]] = None,
                lagrangian_mgr: Optional[LagrangianManager] = None,
                eval_interval: int = 20,
                eval_road: Optional[dict] = None,
                eval_ref: Optional[dict] = None,
                checkpoint_dir: str = "checkpoints",
                stage_name: str = "track",
                verbose: bool = True,
                ) -> dict:
    """
    执行单阶段训练
    :param agent: PPO Agent
    :param road: 训练道路
    :param ref: 参考轨迹
    :param eff_map: 效率图
    :param style: 驾驶风格
    :param mode: track / energy
    :param num_episodes: 训练轮数
    :param energy_weight: 能量奖励权重（track 模式为 0）
    :param energy_weight_schedule: 能量权重逐步拉高的 schedule
    :param lagrangian_mgr: 拉格朗日乘子管理器
    :param eval_interval: 评估间隔
    :param eval_road: 评估道路
    :param eval_ref: 评估参考轨迹
    :param checkpoint_dir: checkpoint 保存目录
    :param stage_name: 阶段名称
    :param verbose: 是否打印
    :return: 训练历史 dict
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    if lagrangian_mgr is None:
        lagrangian_mgr = LagrangianManager(agent.CONSTRAINT_NAMES)

    history = {
        "episode_rewards": [],
        "eval_metrics": [],
        "train_stats": [],
        "lambda_history": [],
        "best_eval_saving": -999.0,
        "best_episode": -1,
    }
    eval_interval = max(1, min(eval_interval, num_episodes))

    for ep in range(num_episodes):
        # 能量权重 schedule
        ew = energy_weight
        if energy_weight_schedule is not None and ep < len(energy_weight_schedule):
            ew = energy_weight_schedule[ep]

        torque_scale = residual_torque_scale
        if residual_torque_scale_schedule is not None and ep < len(residual_torque_scale_schedule):
            torque_scale = residual_torque_scale_schedule[ep]

        # 创建环境
        env = MotorEnv(road=road, ref=ref, style=style, mode=mode,
                       eff_map=eff_map, energy_weight=ew,
                       residual_torque_scale=torque_scale)

        # 收集 rollout
        ep_reward, ep_metrics, last_obs = rollout_episode(env, agent)
        history["episode_rewards"].append(ep_reward)

        # 计算平均约束 cost
        avg_costs = {}
        for name in agent.CONSTRAINT_NAMES:
            if len(env.cost_history[name]) > 0:
                avg_costs[name] = float(np.mean(env.cost_history[name]))
            else:
                avg_costs[name] = 0.0

        # 更新拉格朗日乘子
        lambdas = lagrangian_mgr.update(avg_costs)
        history["lambda_history"].append(lambdas.copy())

        # PPO 更新
        train_stat = agent.update(last_obs, lambdas)
        history["train_stats"].append(train_stat)

        # 评估
        if (ep + 1) % eval_interval == 0:
            e_road = eval_road if eval_road is not None else road
            e_ref  = eval_ref if eval_ref is not None else ref
            eval_env = MotorEnv(road=e_road, ref=e_ref, style=style, mode=mode,
                                eff_map=eff_map, energy_weight=ew,
                                residual_torque_scale=torque_scale)
            eval_reward, eval_metrics, eval_infos = eval_episode(eval_env, agent)
            history["eval_metrics"].append({"episode": ep + 1, **eval_metrics})

            # 打印
            if verbose:
                saving = eval_metrics.get("saving_cmp_total_pct", eval_metrics.get("saving_total_pct", 0.0))
                s_mae  = eval_metrics.get("speed_mae", 0.0)
                print(f"  [{stage_name}] ep={ep+1:4d}  reward={ep_reward:8.2f}  "
                      f"eval_ctrl_saving={saving:6.2f}%  speed_mae={s_mae:.3f}  "
                      f"λ_speed={lambdas['speed']:.3f}  λ_energy={lambdas['window_energy']:.3f}  "
                      f"mode={','.join(n for n,m in lagrangian_mgr.mode.items() if m=='pid') or 'integral'}")

            # 保存 best
            if mode == "energy":
                saving_val = score_energy_metrics_for_selection(eval_metrics)
            else:
                saving_val = -eval_metrics.get("speed_mae", 999.0)
            if saving_val > history["best_eval_saving"]:
                history["best_eval_saving"] = saving_val
                history["best_episode"] = ep + 1
                best_path = os.path.join(checkpoint_dir, f"best_{stage_name}_{style}.pt")
                agent.save(best_path)

        # 定期 checkpoint
        if (ep + 1) % (eval_interval * 5) == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{stage_name}_{style}_ep{ep+1}.pt")
            agent.save(ckpt_path)

    # 最终保存
    final_path = os.path.join(checkpoint_dir, f"final_{stage_name}_{style}.pt")
    agent.save(final_path)

    return history


# ============================================================
#  多阶段训练流水线
# ============================================================
def run_multistage_training(agent: PPOAgent,
                            train_road: dict,
                            train_ref: dict,
                            eval_road: dict,
                            eval_ref: dict,
                            eff_map: dict,
                            style: str = "normal",
                            track_episodes: int = 150,
                            energy_episodes: int = 300,
                            polish_episodes: int = 50,
                            energy_weight_final: float = 3.0,
                            residual_torque_final: float = 6.0,
                            track_eval_interval: int = 20,
                            energy_eval_interval: int = 20,
                            polish_eval_interval: int = 10,
                            checkpoint_dir: str = "checkpoints",
                            verbose: bool = True,
                            ) -> dict:
    """
    执行多阶段训练流水线：track → energy → polish
    :return: 完整训练历史
    """
    all_history = {}
    lagrangian_mgr = LagrangianManager(
        agent.CONSTRAINT_NAMES,
        thresholds={
            "speed": 0.05, "distance": 0.05, "smoothness": 0.1,
            "projection": 0.08, "window_energy": 0.1,
        }
    )

    # ==== Stage 1: Track ====
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Stage 1: TRACK  (style={style}, episodes={track_episodes})")
        print(f"{'='*60}")

    track_hist = train_stage(
        agent=agent, road=train_road, ref=train_ref, eff_map=eff_map,
        style=style, mode="track", num_episodes=track_episodes,
        energy_weight=0.0,
        residual_torque_scale=max(residual_torque_final * 0.5, 1.0),
        lagrangian_mgr=lagrangian_mgr,
        eval_interval=track_eval_interval, eval_road=eval_road, eval_ref=eval_ref,
        checkpoint_dir=checkpoint_dir, stage_name="track", verbose=verbose,
    )
    all_history["track"] = track_hist

    # ==== Stage 2: Energy ====
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Stage 2: ENERGY  (style={style}, episodes={energy_episodes})")
        print(f"{'='*60}")

    # 能量权重 schedule：线性拉高
    energy_schedule = np.linspace(0.1, energy_weight_final, energy_episodes).tolist()
    torque_schedule = np.linspace(
        max(residual_torque_final * 0.5, 1.0),
        residual_torque_final,
        energy_episodes,
    ).tolist()

    energy_hist = train_stage(
        agent=agent, road=train_road, ref=train_ref, eff_map=eff_map,
        style=style, mode="energy", num_episodes=energy_episodes,
        energy_weight=energy_weight_final,
        energy_weight_schedule=energy_schedule,
        residual_torque_scale=residual_torque_final,
        residual_torque_scale_schedule=torque_schedule,
        lagrangian_mgr=lagrangian_mgr,
        eval_interval=energy_eval_interval, eval_road=eval_road, eval_ref=eval_ref,
        checkpoint_dir=checkpoint_dir, stage_name="energy", verbose=verbose,
    )
    all_history["energy"] = energy_hist

    # ==== Stage 3: Polish (bias repair + safeguard) ====
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Stage 3: POLISH  (style={style}, episodes={polish_episodes})")
        print(f"{'='*60}")

    # 降低学习率做 fine-tune
    for pg in agent.actor_optimizer.param_groups:
        pg['lr'] *= 0.3
    for pg in agent.critic_optimizer.param_groups:
        pg['lr'] *= 0.3

    polish_hist = train_stage(
        agent=agent, road=train_road, ref=train_ref, eff_map=eff_map,
        style=style, mode="energy", num_episodes=polish_episodes,
        energy_weight=energy_weight_final,
        residual_torque_scale=residual_torque_final,
        residual_torque_scale_schedule=np.linspace(
            max(residual_torque_final * 0.75, 1.0),
            residual_torque_final,
            polish_episodes,
        ).tolist(),
        lagrangian_mgr=lagrangian_mgr,
        eval_interval=polish_eval_interval, eval_road=eval_road, eval_ref=eval_ref,
        checkpoint_dir=checkpoint_dir, stage_name="polish", verbose=verbose,
    )
    all_history["polish"] = polish_hist

    return all_history


# ============================================================
#  Checkpoint 恢复 & Best 选择
# ============================================================
def load_best_checkpoint(agent: PPOAgent, checkpoint_dir: str,
                         stage: str, style: str) -> bool:
    """加载最佳 checkpoint，成功返回 True"""
    path = os.path.join(checkpoint_dir, f"best_{stage}_{style}.pt")
    if os.path.exists(path):
        agent.load(path)
        return True
    return False
