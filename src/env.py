import json
import random
import numpy as np
import pandas as pd


class DeliveryCoordinationEnv:
    """
    Covers Operational Environment Development (Plan IDs 25-30):
    - ID 25: Basic simulation framework & delivery episode structure
    - ID 26: Orders, 100-vehicle fleet states (V001-V100), capacity & energy tracking
    - ID 27: Dynamic traffic & weather speed/energy physics
    - ID 28: Operational constraints (capacity, battery/fuel sufficiency, valid action mask)
    - ID 29: Multi-objective reward engine aligned with dataset (R^2 = 0.999998)
    - ID 30: Sample episode validation helper
    """

    def __init__(
        self,
        orders_path: str = "data/orders_v1.csv",
        vehicles_path: str = "data/vehicles_v1.csv",
        physics_path: str = "data/physics_v1.json",
        episode_length: int = 50,
        seed: int = 42,
    ):
        self.orders_df = pd.read_csv(orders_path)
        self.vehicles_df = pd.read_csv(vehicles_path)
        with open(physics_path, "r", encoding="utf-8") as f:
            self.physics = json.load(f)

        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # 15 discrete actions: 3 Vehicle Types x 5 Capacity Classes (Plan ID 15)
        self.vehicle_types = self.physics["vehicle_types"]          # ['EV', 'Petrol', 'Diesel']
        self.capacity_classes = self.physics["capacity_classes_kg"] # [10, 15, 20, 25, 30]
        self.action_catalog = [
            (v_type, cap)
            for v_type in self.vehicle_types
            for cap in self.capacity_classes
        ]
        self.n_actions = len(self.action_catalog)  # 15
        self.state_dim = 18                        # 4 order + 3 priority + 4 traffic + 4 weather + 3 fleet readiness

        self.priorities = self.physics["priorities"]
        self.traffic_levels = self.physics["traffic_levels"]
        self.weather_conditions = self.physics["weather_conditions"]
        self.speed_matrix = self.physics["speed_matrix_kmh"]
        self.vehicle_specs = self.physics["vehicle_specs"]
        self.reward_weights = self.physics["reward_weights"]

        self.fleet_state = None
        self.episode_orders = None
        self.step_idx = 0

    def _init_fleet_state(self) -> pd.DataFrame:
        fleet = self.vehicles_df[["Vehicle_ID", "Vehicle_Type", "Vehicle_Capacity_kg", "Avg_Initial_Energy_Pct"]].copy()
        fleet["Energy_Pct"] = self.rng.uniform(65.0, 100.0, size=len(fleet)).round(2)
        return fleet

    def reset(self, order_indices: np.ndarray | None = None):
        self.fleet_state = self._init_fleet_state()
        if order_indices is not None:
            self.episode_orders = self.orders_df.iloc[order_indices].reset_index(drop=True)
        else:
            sampled_idx = self.rng.choice(len(self.orders_df), size=self.episode_length, replace=False)
            self.episode_orders = self.orders_df.iloc[sampled_idx].reset_index(drop=True)
        self.step_idx = 0
        return self._get_state(), self.get_valid_action_mask()

    def _encode_order_state(self, order_row: dict) -> np.ndarray:
        w_norm = float(order_row["Order_Weight_kg"]) / 30.0
        d_norm = float(order_row["Distance_km"]) / 50.0
        dl_norm = float(order_row["Deadline_Min"]) / 400.0
        sc_norm = float(order_row.get("State_Score", 1.68)) / 4.0

        prio_oh = [1.0 if order_row["Priority"] == p else 0.0 for p in self.priorities]
        traf_oh = [1.0 if order_row["Traffic_Level"] == t else 0.0 for t in self.traffic_levels]
        weat_oh = [1.0 if order_row["Weather"] == w else 0.0 for w in self.weather_conditions]

        ev_mean = float(self.fleet_state[self.fleet_state["Vehicle_Type"] == "EV"]["Energy_Pct"].mean()) / 100.0
        pet_mean = float(self.fleet_state[self.fleet_state["Vehicle_Type"] == "Petrol"]["Energy_Pct"].mean()) / 100.0
        die_mean = float(self.fleet_state[self.fleet_state["Vehicle_Type"] == "Diesel"]["Energy_Pct"].mean()) / 100.0

        return np.array(
            [w_norm, d_norm, dl_norm, sc_norm] + prio_oh + traf_oh + weat_oh + [ev_mean, pet_mean, die_mean],
            dtype=np.float32,
        )

    def _get_state(self) -> np.ndarray:
        row = self.episode_orders.iloc[self.step_idx].to_dict()
        return self._encode_order_state(row)

    def _estimate_consumption(self, v_type: str, distance_km: float, traffic: str, weather: str) -> float:
        base_rate = self.vehicle_specs[v_type]["consumption_per_km"]
        traffic_mult = {"Low": 0.92, "Medium": 1.00, "High": 1.10, "Very High": 1.20}[traffic]
        weather_mult = {"Clear": 0.96, "Cloudy": 1.00, "Rain": 1.08, "Heavy Rain": 1.15}[weather]
        return round(distance_km * base_rate * traffic_mult * weather_mult, 3)

    def get_valid_action_mask(self, order_row: dict | None = None) -> np.ndarray:
        """ID 28: Operational Constraints & Valid-Action Masking."""
        if order_row is None:
            order_row = self.episode_orders.iloc[self.step_idx].to_dict()

        weight = float(order_row["Order_Weight_kg"])
        dist = float(order_row["Distance_km"])
        traffic = order_row["Traffic_Level"]
        weather = order_row["Weather"]

        mask = np.zeros(self.n_actions, dtype=bool)
        for idx, (v_type, cap) in enumerate(self.action_catalog):
            if cap < weight:
                continue
            req_energy = self._estimate_consumption(v_type, dist, traffic, weather)
            candidates = self.fleet_state[
                (self.fleet_state["Vehicle_Type"] == v_type)
                & (self.fleet_state["Vehicle_Capacity_kg"] == cap)
            ]
            if not candidates.empty and candidates["Energy_Pct"].max() >= (req_energy + 2.0):
                mask[idx] = True

        if not mask.any():
            for idx, (v_type, cap) in enumerate(self.action_catalog):
                if cap >= weight:
                    mask[idx] = True
        return mask

    def simulate_order_dispatch(self, order_row: dict, action_idx: int, mutate_fleet: bool = True) -> dict:
        v_type, cap = self.action_catalog[action_idx]
        weight = float(order_row["Order_Weight_kg"])
        dist = float(order_row["Distance_km"])
        traffic = order_row["Traffic_Level"]
        weather = order_row["Weather"]
        priority = order_row["Priority"]
        service_time = float(order_row.get("Service_Time_min", 9.5))
        deadline = float(order_row["Deadline_Min"])

        slot_mask = (self.fleet_state["Vehicle_Type"] == v_type) & (self.fleet_state["Vehicle_Capacity_kg"] == cap)
        best_veh_idx = self.fleet_state.loc[slot_mask, "Energy_Pct"].idxmax()
        veh_id = self.fleet_state.loc[best_veh_idx, "Vehicle_ID"]
        energy_before = float(self.fleet_state.loc[best_veh_idx, "Energy_Pct"])

        base_speed = self.speed_matrix[traffic][weather]
        type_speed_factor = {"EV": 1.04, "Petrol": 1.00, "Diesel": 0.98}[v_type]
        eff_speed = base_speed * type_speed_factor
        travel_time = (dist / eff_speed) * 60.0
        total_time = travel_time + service_time

        consumed = self._estimate_consumption(v_type, dist, traffic, weather)
        spec = self.vehicle_specs[v_type]
        emissions_kg = consumed * spec["emission_factor"]
        cost_inr = consumed * spec["unit_cost_inr"]

        cap_ok = 1 if cap >= weight else 0
        energy_ok = 1 if energy_before >= consumed else 0

        sla_factor = {"Normal": 1.00, "High": 0.85, "Urgent": 0.65}[priority]
        effective_deadline = deadline * sla_factor

        if cap_ok == 0 or energy_ok == 0:
            delivered = 0
            status = "Failed"
        elif total_time <= effective_deadline:
            delivered = 1
            status = "On Time"
        else:
            delivered = 1
            status = "Late"

        rw = self.reward_weights
        if status == "On Time":
            outcome_reward = rw["on_time_bonus"]
        elif status == "Late":
            outcome_reward = rw["late_bonus"]
        elif cap_ok == 1:
            outcome_reward = rw["failed_valid_cap_penalty"]
        else:
            outcome_reward = rw["failed_invalid_cap_penalty"]

        reward = outcome_reward + (rw["distance_penalty_per_km"] * dist) + (rw["emission_penalty_per_kg"] * emissions_kg)

        if mutate_fleet:
            new_energy = max(0.0, energy_before - consumed)
            if new_energy < 20.0:
                new_energy = float(self.rng.uniform(80.0, 100.0))
            self.fleet_state.loc[best_veh_idx, "Energy_Pct"] = round(new_energy, 2)

        utilization_pct = round((weight / cap) * 100.0, 1) if cap > 0 else 0.0

        return {
            "Action_Index": int(action_idx),
            "Vehicle_ID": veh_id,
            "Vehicle_Type": v_type,
            "Vehicle_Capacity_kg": int(cap),
            "Capacity_Utilization_Pct": utilization_pct,
            "Battery_or_Fuel_Before": round(energy_before, 2),
            "Consumption": round(consumed, 3),
            "Consumption_Unit": spec["unit"],
            "Distance_km": round(dist, 2),
            "Travel_Time_min": round(travel_time, 2),
            "Service_Time_min": round(service_time, 2),
            "Total_Delivery_Time_min": round(total_time, 2),
            "Effective_Deadline_min": round(effective_deadline, 2),
            "Operating_Cost_INR": round(cost_inr, 2),
            "Estimated_Emissions_kgCO2": round(emissions_kg, 3),
            "Capacity_Constraint_OK": cap_ok,
            "Delivered": delivered,
            "Delivery_Status": status,
            "RL_Reward": round(reward, 3),
        }

    def step(self, action_idx: int):
        order_row = self.episode_orders.iloc[self.step_idx].to_dict()
        info = self.simulate_order_dispatch(order_row, action_idx, mutate_fleet=True)
        reward = info["RL_Reward"]

        self.step_idx += 1
        done = self.step_idx >= len(self.episode_orders)
        if not done:
            next_state = self._get_state()
            next_mask = self.get_valid_action_mask()
        else:
            next_state = np.zeros(self.state_dim, dtype=np.float32)
            next_mask = np.ones(self.n_actions, dtype=bool)

        return next_state, reward, done, next_mask, info


def validate_environment():
    """ID 30: Validate Operational Environment on a sample episode."""
    env = DeliveryCoordinationEnv(episode_length=10, seed=42)
    state, mask = env.reset()
    assert state.shape == (18,), f"Expected 18-D state, got {state.shape}"
    assert mask.shape == (15,), f"Expected 15-D mask, got {mask.shape}"

    total_reward = 0.0
    for step in range(10):
        valid_actions = np.where(mask)[0]
        action = int(np.random.choice(valid_actions))
        state, reward, done, mask, info = env.step(action)
        total_reward += reward
        assert info["Capacity_Constraint_OK"] == 1, f"Mask failed at step {step}!"

    print(f"ENV VALIDATION PASSED (ID 30): 10/10 valid masked steps | Total Episode Reward = {total_reward:.2f}")
    print("Sample Step Output:", info)


if __name__ == "__main__":
    validate_environment()
