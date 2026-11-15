import os
import json
import numpy as np
import matplotlib.pyplot as plt
from src.env import DeliveryCoordinationEnv
from src.dqn_agent import MaskedDQNAgent


def train_dqn(
    episodes: int = 350,
    episode_length: int = 50,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int = 260,
    save_path: str = "models/dqn_model_v1.pth",
    seed: int = 42,
):
    """
    Covers AI Model Training & Optimization (Plan IDs 34-36):
    - ID 34: Train DQN agent across multi-order Delhi-NCR delivery episodes
    - ID 35: Track episode reward, on-time rate, CO2 emissions, and loss convergence
    - ID 36: Save best trained model weights (models/dqn_model_v1.pth) & training curve
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # Use first 16,000 orders (80%) for training; reserve last 4,000 (20%) for unseen evaluation (ID 37)
    env = DeliveryCoordinationEnv(episode_length=episode_length, seed=seed)
    train_pool = np.arange(0, 16000)

    agent = MaskedDQNAgent(state_dim=env.state_dim, action_dim=env.n_actions)
    rng = np.random.default_rng(seed)

    history = {
        "episode": [],
        "total_reward": [],
        "avg_step_reward": [],
        "on_time_rate": [],
        "avg_co2_kg": [],
        "epsilon": [],
    }

    best_moving_reward = -1e9

    for ep in range(1, episodes + 1):
        sampled_indices = rng.choice(train_pool, size=episode_length, replace=False)
        state, mask = env.reset(order_indices=sampled_indices)

        epsilon = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (ep / eps_decay_episodes),
        )

        ep_reward = 0.0
        on_time_count = 0
        total_co2 = 0.0

        for _ in range(episode_length):
            action = agent.select_action(state, mask, epsilon=epsilon)
            next_state, reward, done, next_mask, info = env.step(action)

            agent.memory.push(state, action, reward, next_state, float(done), next_mask)
            agent.train_step()

            state = next_state
            mask = next_mask
            ep_reward += reward
            total_co2 += info["Estimated_Emissions_kgCO2"]
            if info["Delivery_Status"] == "On Time":
                on_time_count += 1

        avg_step_rew = ep_reward / episode_length
        on_time_pct = (on_time_count / episode_length) * 100.0
        avg_co2 = total_co2 / episode_length

        history["episode"].append(ep)
        history["total_reward"].append(round(ep_reward, 2))
        history["avg_step_reward"].append(round(avg_step_rew, 3))
        history["on_time_rate"].append(round(on_time_pct, 1))
        history["avg_co2_kg"].append(round(avg_co2, 3))
        history["epsilon"].append(round(epsilon, 3))

        if ep >= 20:
            moving_avg = float(np.mean(history["total_reward"][-20:]))
            if moving_avg > best_moving_reward:
                best_moving_reward = moving_avg
                agent.save(save_path)

        if ep % 50 == 0 or ep == 1:
            print(
                f"Episode {ep:3d}/{episodes} | "
                f"Reward: {ep_reward:6.2f} (Avg/step: {avg_step_rew:5.2f}) | "
                f"On-Time: {on_time_pct:5.1f}% | "
                f"CO2/order: {avg_co2:5.3f} kg | "
                f"Eps: {epsilon:.2f}"
            )

    with open("models/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save training convergence plot for web dashboard & project report
    fig, ax = plt.subplots(figsize=(8, 4))
    rewards = np.array(history["total_reward"])
    rolling = np.convolve(rewards, np.ones(15) / 15, mode="valid")
    ax.plot(history["episode"], rewards, alpha=0.35, color="#3b82f6", label="Episode Reward")
    ax.plot(range(15, episodes + 1), rolling, color="#10b981", linewidth=2.2, label="15-Ep Moving Avg")
    ax.set_title("DQN Training Convergence - Sustainable Last-Mile Delivery")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward (50 Orders)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig("static/training_curve.png", dpi=150)
    plt.close()

    print(f"\nTRAINING COMPLETE! Best 20-ep moving reward: {best_moving_reward:.2f}")
    print(f"Saved model weights -> {save_path}")
    print("Saved training plot -> static/training_curve.png")


if __name__ == "__main__":
    train_dqn()
