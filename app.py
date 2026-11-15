import json
import urllib.request
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

from src.env import DeliveryCoordinationEnv
from src.dqn_agent import MaskedDQNAgent
from src.baselines import select_random_valid_action, select_rule_based_action

app = FastAPI(title="Sustainable Last-Mile Delivery Control Tower v2.1")
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

NCR_ZONE_COORDS = {
    "Connaught Place": [28.6315, 77.2167],
    "South Delhi": [28.5245, 77.2066],
    "Saket": [28.5244, 77.2167],
    "Lajpat Nagar": [28.5677, 77.2433],
    "Karol Bagh": [28.6514, 77.1907],
    "Dwarka": [28.5921, 77.0460],
    "Janakpuri": [28.6219, 77.0878],
    "Rohini": [28.7495, 77.0565],
    "Pitampura": [28.7033, 77.1323],
    "Vasant Kunj": [28.5293, 77.1530],
    "Mayur Vihar": [28.6048, 77.2945],
    "Preet Vihar": [28.6415, 77.2950],
    "Noida Sector 18": [28.5708, 77.3261],
    "Noida Sector 62": [28.6271, 77.3649],
    "Greater Noida": [28.4744, 77.5040],
    "Cyber City": [28.4950, 77.0895],
    "Gurugram Sector 18": [28.4875, 77.0689],
    "Sohna Road": [28.4126, 77.0422],
    "Ghaziabad": [28.6692, 77.4538],
    "Faridabad": [28.4089, 77.3178],
}

ROUTE_CACHE = {}


def get_zone_latlon(zone_name: str, idx: int = 0) -> list[float]:
    if zone_name in NCR_ZONE_COORDS:
        return NCR_ZONE_COORDS[zone_name]
    angle = (idx * 18) * (np.pi / 180.0)
    return [round(28.6139 + 0.14 * np.sin(angle), 4), round(77.2090 + 0.16 * np.cos(angle), 4)]


def fetch_exact_osrm_route(start_latlon: list[float], end_latlon: list[float], fallback_km: float):
    """Fetch real road geometry, exact road distance (km), and major road names via OSRM."""
    cache_key = f"{start_latlon[0]},{start_latlon[1]}->{end_latlon[0]},{end_latlon[1]}"
    if cache_key in ROUTE_CACHE:
        return ROUTE_CACHE[cache_key]

    # OSRM expects lon,lat order
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{start_latlon[1]},{start_latlon[0]};{end_latlon[1]},{end_latlon[0]}"
        f"?overview=full&geometries=geojson&steps=true"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SustainableLastMileRL/2.1"})
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            exact_km = round(route["distance"] / 1000.0, 2)
            base_mins = round(route["duration"] / 60.0, 1)
            # Flip GeoJSON [lon, lat] -> Leaflet [lat, lon]
            coords = [[pt[1], pt[0]] for pt in route["geometry"]["coordinates"]]

            # Extract distinct major road names along the route
            roads = []
            for leg in route.get("legs", []):
                for step in leg.get("steps", []):
                    road_name = step.get("name", "").strip()
                    if road_name and road_name not in roads:
                        roads.append(road_name)

            result = {
                "distance_km": exact_km,
                "osrm_base_min": base_mins,
                "route_coords": coords,
                "major_roads": roads[:6],
                "source": "OSRM Live Road Network",
            }
            ROUTE_CACHE[cache_key] = result
            return result
    except Exception:
        pass

    return {
        "distance_km": fallback_km,
        "osrm_base_min": round(fallback_km * 2.4, 1),
        "route_coords": [start_latlon, end_latlon],
        "major_roads": ["Direct Corridor Fallback"],
        "source": "Physics Matrix Fallback",
    }


class DispatchRequest(BaseModel):
    pickup_zone: str
    delivery_zone: str
    order_type: str = "Parcel"
    order_weight_kg: float = Field(..., gt=0.0, le=30.0)
    priority: str = "Normal"
    traffic_level: str = "Medium"
    weather: str = "Clear"
    deadline_min: float = Field(default=90.0, gt=10.0, le=400.0)


class ShiftRequest(BaseModel):
    n_orders: int = Field(default=20, ge=5, le=50)
    seed: int = Field(default=2026)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    zones = physics_cfg["zones"]
    zone_coords = {z: get_zone_latlon(z, i) for i, z in enumerate(zones)}
    template = jinja_env.get_template("index.html")
    html_content = template.render(
        zones=zones,
        zone_coords_json=json.dumps(zone_coords),
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
        fallback_km = 4.5
    else:
        fallback_km = physics_cfg["zone_distance_km"].get(
            pair_key, physics_cfg["zone_distance_km"].get(reverse_key, 18.5)
        )

    zones = physics_cfg["zones"]
    p_idx = zones.index(payload.pickup_zone) if payload.pickup_zone in zones else 0
    d_idx = zones.index(payload.delivery_zone) if payload.delivery_zone in zones else 1
    pickup_latlon = get_zone_latlon(payload.pickup_zone, p_idx)
    delivery_latlon = get_zone_latlon(payload.delivery_zone, d_idx)

    # Fetch exact road distance & turn-by-turn geometry
    road_info = fetch_exact_osrm_route(pickup_latlon, delivery_latlon, fallback_km)
    distance_km = road_info["distance_km"]

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

    import torch
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_vals = agent.policy_net(state_t).squeeze(0).numpy()

    xai_slots = []
    for idx, (v_type, cap) in enumerate(env.action_catalog):
        is_valid = bool(mask[idx])
        reason = "Valid Candidate" if is_valid else ("Overweight" if cap < payload.order_weight_kg else "Low Battery/Fuel")
        xai_slots.append({
            "slot": f"{v_type}-{cap}kg",
            "vehicle_type": v_type,
            "capacity": cap,
            "valid": is_valid,
            "q_value": round(float(q_vals[idx]), 3),
            "reason": reason,
        })
    xai_slots.sort(key=lambda x: (x["valid"], x["q_value"]), reverse=True)

    dqn_action = agent.select_action(state, mask, epsilon=0.0)
    dqn_res = env.simulate_order_dispatch(order_row, dqn_action, mutate_fleet=False)

    rule_action = select_rule_based_action(mask)
    rule_res = env.simulate_order_dispatch(order_row, rule_action, mutate_fleet=False)

    rng = np.random.default_rng(42)
    rand_action = select_random_valid_action(mask, rng)
    rand_res = env.simulate_order_dispatch(order_row, rand_action, mutate_fleet=False)

    return {
        "distance_km": distance_km,
        "route_source": road_info["source"],
        "major_roads": road_info["major_roads"],
        "route_coords": road_info["route_coords"],
        "pickup_coords": pickup_latlon,
        "delivery_coords": delivery_latlon,
        "dqn": dqn_res,
        "rule_based": rule_res,
        "random_valid": rand_res,
        "xai_ranking": xai_slots,
    }


@app.post("/api/simulate-shift")
async def simulate_full_shift(payload: ShiftRequest):
    sim_env = DeliveryCoordinationEnv(episode_length=payload.n_orders, seed=payload.seed)
    state, mask = sim_env.reset()

    base_env = DeliveryCoordinationEnv(episode_length=payload.n_orders, seed=payload.seed)
    base_env.reset(order_indices=sim_env.episode_orders.index.to_numpy())
    base_env.episode_orders = sim_env.episode_orders.copy()
    base_rng = np.random.default_rng(payload.seed)

    timeline = []
    total_dqn_co2, total_base_co2 = 0.0, 0.0
    total_dqn_cost, total_base_cost = 0.0, 0.0
    dqn_on_time, base_on_time = 0, 0
    recharge_events = 0
    ev_trips = 0

    for step_i in range(payload.n_orders):
        order = sim_env.episode_orders.iloc[step_i].to_dict()
        action = agent.select_action(state, mask, epsilon=0.0)
        next_state, reward, done, next_mask, info = sim_env.step(action)

        b_mask = base_env.get_valid_action_mask()
        b_act = select_random_valid_action(b_mask, base_rng)
        _, _, _, _, b_info = base_env.step(b_act)

        if info["Vehicle_Type"] == "EV":
            ev_trips += 1
        if (info["Battery_or_Fuel_Before"] - info["Consumption"]) < 20.0:
            recharge_events += 1

        total_dqn_co2 += info["Estimated_Emissions_kgCO2"]
        total_base_co2 += b_info["Estimated_Emissions_kgCO2"]
        total_dqn_cost += info["Operating_Cost_INR"]
        total_base_cost += b_info["Operating_Cost_INR"]
        if info["Delivery_Status"] == "On Time":
            dqn_on_time += 1
        if b_info["Delivery_Status"] == "On Time":
            base_on_time += 1

        timeline.append({
            "step": step_i + 1,
            "order_id": order.get("Order_ID", f"ORD-{step_i+1}"),
            "corridor": f"{order['Pickup_Zone']} -> {order['Delivery_Zone']}",
            "weight_kg": round(float(order["Order_Weight_kg"]), 1),
            "traffic": order["Traffic_Level"],
            "vehicle": f"{info['Vehicle_ID']} ({info['Vehicle_Type']} {info['Vehicle_Capacity_kg']}kg)",
            "energy_before": info["Battery_or_Fuel_Before"],
            "co2_kg": info["Estimated_Emissions_kgCO2"],
            "cost_inr": info["Operating_Cost_INR"],
            "status": info["Delivery_Status"],
            "reward": info["RL_Reward"],
        })
        state, mask = next_state, next_mask

    co2_saved = max(0.0, round(total_base_co2 - total_dqn_co2, 2))
    cost_saved = max(0.0, round(total_base_cost - total_dqn_cost, 2))
    fleet_df = sim_env.fleet_state

    return {
        "summary": {
            "orders_processed": payload.n_orders,
            "ev_share_pct": round((ev_trips / payload.n_orders) * 100.0, 1),
            "dqn_on_time_pct": round((dqn_on_time / payload.n_orders) * 100.0, 1),
            "base_on_time_pct": round((base_on_time / payload.n_orders) * 100.0, 1),
            "co2_saved_kg": co2_saved,
            "trees_equivalent": round(co2_saved / 0.06, 1),
            "cost_saved_inr": cost_saved,
            "recharge_cycles_triggered": recharge_events,
            "avg_ev_soc_end": round(float(fleet_df[fleet_df["Vehicle_Type"] == "EV"]["Energy_Pct"].mean()), 1),
            "avg_petrol_end": round(float(fleet_df[fleet_df["Vehicle_Type"] == "Petrol"]["Energy_Pct"].mean()), 1),
            "avg_diesel_end": round(float(fleet_df[fleet_df["Vehicle_Type"] == "Diesel"]["Energy_Pct"].mean()), 1),
        },
        "timeline": timeline,
    }
