import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.env import DeliveryCoordinationEnv
from src.dqn_agent import MaskedDQNAgent
from src.baselines import select_random_valid_action, select_rule_based_action


def evaluate_all_policies(
    n_test_episodes: int = 40,
    episode_length: int = 50,
    model_path: str = "models/dqn_model_v1.pth",
    seed: int = 101,
):
    """
    Covers Testing & Benchmarking (Plan IDs 37, 41-45):
    - ID 37: Evaluate on held-out unseen test orders (indices 16,000 to 19,999)
    - ID 41-44: Compare Time, Distance, Cost, CO2, On-Time Rate, Utilization, and RL Reward
    - ID 45: Export comparison report & visual chart for web dashboard and final report
    """
    env = DeliveryCoordinationEnv(episode_length=episode_length, seed=seed)
    test_pool = np.arange(16000, len(env.orders_df))

    agent = MaskedDQNAgent(state_dim=env.state_dim, action_dim=env.n_actions)
    agent.load(model_path)

    policies = ["Random Valid Baseline", "Rule-Based Heuristic", "Trained DQN Agent (Ours)"]
    results = []

    for policy_name in policies:
        rng = np.random.default_rng(seed)
        metrics = {
            "time_min": [],
            "distance_km": [],
            "cost_inr": [],
            "co2_kg": [],
            "on_time": [],
            "utilization_pct": [],
            "reward": [],
        }

        for _ in range(n_test_episodes):
            sampled_idx = rng.choice(test_pool, size=episode_length, replace=False)
            state, mask = env.reset(order_indices=sampled_idx)

            for _ in range(episode_length):
                if policy_name == "Random Valid Baseline":
                    action = select_random_valid_action(mask, rng)
                elif policy_name == "Rule-Based Heuristic":
                    action = select_rule_based_action(mask)
                else:
                    action = agent.select_action(state, mask, epsilon=0.0)

                next_state, reward, done, next_mask, info = env.step(action)

                metrics["time_min"].append(info["Total_Delivery_Time_min"])
                metrics["distance_km"].append(info["Distance_km"])
                metrics["cost_inr"].append(info["Operating_Cost_INR"])
                metrics["co2_kg"].append(info["Estimated_Emissions_kgCO2"])
                metrics["on_time"].append(1.0 if info["Delivery_Status"] == "On Time" else 0.0)
                metrics["utilization_pct"].append(info["Capacity_Utilization_Pct"])
                metrics["reward"].append(reward)

                state, mask = next_state, next_mask

        results.append(
            {
                "Policy": policy_name,
                "Avg_Delivery_Time_min": round(float(np.mean(metrics["time_min"])), 2),
                "Avg_Distance_km": round(float(np.mean(metrics["distance_km"])), 2),
                "Avg_Cost_INR": round(float(np.mean(metrics["cost_inr"])), 2),
                "Avg_CO2_kg": round(float(np.mean(metrics["co2_kg"])), 3),
                "On_Time_Rate_Pct": round(float(np.mean(metrics["on_time"]) * 100.0), 1),
                "Capacity_Utilization_Pct": round(float(np.mean(metrics["utilization_pct"])), 1),
                "Mean_Step_Reward": round(float(np.mean(metrics["reward"])), 3),
            }
        )

    summary_df = pd.DataFrame(results)
    with open("models/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate comparison chart for Web Dashboard & Report (ID 45)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#94a3b8", "#3b82f6", "#10b981"]
    short_labels = ["Random\nValid", "Rule-Based\nHeuristic", "Trained DQN\n(Ours)"]

    axes[0].bar(short_labels, summary_df["Mean_Step_Reward"], color=colors)
    axes[0].set_title("Mean RL Reward (Higher is Better)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(short_labels, summary_df["Avg_CO2_kg"], color=colors)
    axes[1].set_title("Avg CO2 per Order in kg (Lower is Better)")
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(short_labels, summary_df["Avg_Cost_INR"], color=colors)
    axes[2].set_title("Avg Operating Cost in INR (Lower is Better)")
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig("static/benchmark_comparison.png", dpi=150)
    plt.close()

    print("=== BENCHMARK EVALUATION ON 2,000 UNSEEN TEST ORDERS (IDs 37-45) ===")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    evaluate_all_policies()
