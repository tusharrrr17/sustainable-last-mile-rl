import json
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from src.env import DeliveryCoordinationEnv
from src.dqn_agent import MaskedDQNAgent
from src.baselines import select_random_valid_action, select_rule_based_action

app = FastAPI(title="Sustainable Last-Mile Delivery Coordinator (RL)")
app.mount("/static", StaticFiles(directory="static"), name="static")
jinja_env = Environment(loader=FileSystemLoader("templates"))

env = DeliveryCoordinationEnv(episode_length=50, seed=42)
env.reset()

with open("data/physics_v1.json", "r", encoding="utf-8") as f:
    physics_cfg = json.load(f)

with open("models/benchmark_results.json", "r", encoding="utf-8") as f:
    benchmark_results = json.load(f)

agent = MaskedDQNAgent(state_dim=env.state_dim, action_dim=env.n_actions, device="cpu")
agent.load("models/dqn_model_v1.pth")


class DispatchRequest(BaseModel):
    pickup_zone: str
    delivery_zone: str
    order_type: str = "Parcel"
    order_weight_kg: float = Field(..., gt=0.0, le=30.0)
    priority: str = "Normal"
    traffic_level: str = "Medium"
    weather: str = "Clear"
    deadline_min: float = Field(default=90.0, gt=10.0, le=400.0)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    template = jinja_env.get_template("index.html")
    html_content = template.render(
        zones=physics_cfg["zones"],
        order_types=physics_cfg["order_types"],
        priorities=physics_cfg["priorities"],
        traffic_levels=physics_cfg["traffic_levels"],
        weather_conditions=physics_cfg["weather_conditions"],
        benchmarks=benchmark_results,
    )
    return HTMLResponse(content=html_content)


@app.post("/api/dispatch")
async def simulate_dispatch(payload: DispatchRequest):
    pair_key = f"{payload.pickup_zone}||{payload.delivery_zone}"
    reverse_key = f"{payload.delivery_zone}||{payload.pickup_zone}"
    if payload.pickup_zone == payload.delivery_zone:
        distance_km = 4.5
    else:
        distance_km = physics_cfg["zone_distance_km"].get(
            pair_key, physics_cfg["zone_distance_km"].get(reverse_key, 18.5)
        )

    order_row = {
        "Order_Weight_kg": payload.order_weight_kg,
        "Distance_km": distance_km,
        "Deadline_Min": payload.deadline_min,
        "State_Score": 1.68,
        "Priority": payload.priority,
        "Traffic_Level": payload.traffic_level,
        "Weather": payload.weather,
        "Service_Time_min": 9.5,
    }

    state = env._encode_order_state(order_row)
    mask = env.get_valid_action_mask(order_row)

    dqn_action = agent.select_action(state, mask, epsilon=0.0)
    dqn_res = env.simulate_order_dispatch(order_row, dqn_action, mutate_fleet=False)

    rule_action = select_rule_based_action(mask)
    rule_res = env.simulate_order_dispatch(order_row, rule_action, mutate_fleet=False)

    rng = np.random.default_rng(42)
    rand_action = select_random_valid_action(mask, rng)
    rand_res = env.simulate_order_dispatch(order_row, rand_action, mutate_fleet=False)

    return {
        "distance_km": distance_km,
        "dqn": dqn_res,
        "rule_based": rule_res,
        "random_valid": rand_res,
    }
