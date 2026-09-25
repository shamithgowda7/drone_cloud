"""
DroneCloud — ProAdapt Predictive-Adaptive Edge-Cloud Offloading Simulator
========================================================================
A real-time drone fleet management simulation demonstrating:
1. Static (Always Cloud) baseline
2. Heuristic (6-factor rule scoring)
3. Reactive Q-Learning (Tabular RL with current telemetry)
4. ProAdapt (Predict-then-Optimize: Tabular RL + Exponential Smoothing Battery Forecast)

Coursework Project — Cloud Computing & Edge Intelligence
"""

import asyncio
import json
import math
import random
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

from proadapt import BatteryForecaster, StateDiscretizer, TabularQLearningAgent


# ── Configuration ──────────────────────────────────────────────────
NUM_DRONES = 12
CITY_CENTER = (12.9716, 77.5946)   # Bangalore, India
CITY_RADIUS = 0.045                # ~5 km
TICK_INTERVAL = 0.6                # seconds between simulation frames
TASK_SPAWN_PROB = 0.28             # chance of new task per drone per tick


# ── Enums ──────────────────────────────────────────────────────────
class DroneStatus(str, Enum):
    IDLE = "idle"
    EN_ROUTE = "en_route"
    EXECUTING = "executing"
    RETURNING = "returning"
    LOW_BATTERY = "low_battery"
    CHARGING = "charging"


class TaskType(str, Enum):
    IMAGE_PROCESSING = "image_processing"
    PATH_PLANNING = "path_planning"
    ANOMALY_DETECTION = "anomaly_detection"
    TERRAIN_MAPPING = "terrain_mapping"
    OBJECT_TRACKING = "object_tracking"


class ProcessLoc(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


# Task profiles: compute units, data size (MB), deadline (s), priority 1-5
TASK_PROFILES = {
    TaskType.IMAGE_PROCESSING:  {"compute": 8,  "data": 50, "deadline": 3.0, "priority": 3},
    TaskType.PATH_PLANNING:     {"compute": 3,  "data": 5,  "deadline": 1.0, "priority": 5},
    TaskType.ANOMALY_DETECTION: {"compute": 6,  "data": 30, "deadline": 2.0, "priority": 4},
    TaskType.TERRAIN_MAPPING:   {"compute": 10, "data": 80, "deadline": 5.0, "priority": 2},
    TaskType.OBJECT_TRACKING:   {"compute": 5,  "data": 20, "deadline": 1.5, "priority": 5},
}


# ── Data Models ────────────────────────────────────────────────────
class Drone:
    __slots__ = (
        "id", "lat", "lng", "battery", "speed", "status",
        "heading", "altitude", "edge_cap", "cur_task",
        "tgt_lat", "tgt_lng", "total_energy_used",
    )

    def __init__(self, id: str, lat: float, lng: float):
        self.id = id
        self.lat = lat
        self.lng = lng
        self.battery: float = random.uniform(65, 100)
        self.speed: float = random.uniform(0.0003, 0.0008)
        self.status: DroneStatus = DroneStatus.IDLE
        self.heading: float = random.uniform(0, 360)
        self.altitude: float = random.uniform(50, 150)
        self.edge_cap: float = random.uniform(4, 8)
        self.cur_task: Optional[str] = None
        self.tgt_lat: Optional[float] = None
        self.tgt_lng: Optional[float] = None
        self.total_energy_used: float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "lat": round(self.lat, 6),
            "lng": round(self.lng, 6),
            "battery": round(self.battery, 1),
            "status": self.status.value,
            "heading": round(self.heading, 1),
            "altitude": round(self.altitude, 1),
            "edge_cap": round(self.edge_cap, 1),
        }


class Cloud:
    def __init__(self):
        self.queue: int = 0
        self.max_cap: int = 20
        self.load: float = 0.0
        self.power: float = 10.0          # compute-units / s
        self.base_latency: float = 0.1    # seconds
        self.scale: float = 1.0
        self.processed: int = 0

    @property
    def latency(self):
        return self.base_latency * (1 + self.load * 2) + random.uniform(0, 0.04)

    def to_dict(self):
        return {
            "queue": self.queue,
            "max_capacity": int(self.max_cap * self.scale),
            "load": round(self.load, 2),
            "power": round(self.power * self.scale, 1),
            "latency_ms": round(self.latency * 1000, 1),
            "scale": round(self.scale, 2),
            "processed": self.processed,
        }


class Metrics:
    """Aggregated strategy metrics."""
    def __init__(self, name: str = "Strategy"):
        self.name = name
        self.tasks = 0
        self.done = 0
        self.edge = 0
        self.cloud = 0
        self.sum_time = 0.0
        self.hit = 0
        self.miss = 0
        self.sum_lat = 0.0
        self.sum_batt = 0.0

    def to_dict(self):
        d = max(1, self.done)
        return {
            "name": self.name,
            "tasks": self.tasks,
            "done": self.done,
            "edge": self.edge,
            "cloud": self.cloud,
            "avg_time_ms": round(self.sum_time / d * 1000, 1),
            "hit_rate": round(self.hit / max(1, self.hit + self.miss) * 100, 1),
            "edge_pct": round(self.edge / d * 100, 1),
            "avg_lat_ms": round(self.sum_lat / d * 1000, 1),
            "batt_used": round(self.sum_batt, 1),
        }


# ── Simulation Engine ─────────────────────────────────────────────
class Engine:
    def __init__(self):
        self.drones: Dict[str, Drone] = {}
        self.cloud = Cloud()
        self.task_seq = 0
        self.tick = 0

        # 4 Strategy Metrics
        self.static = Metrics("Static (Cloud)")
        self.heuristic = Metrics("Heuristic (Rule-based)")
        self.reactive_q = Metrics("Reactive Q-Learning")
        self.proadapt = Metrics("ProAdapt (Predictive RL)")

        # Forecaster and ML Agents
        self.forecaster = BatteryForecaster(alpha=0.25, horizon=15)
        self.agent_reactive = TabularQLearningAgent(name="ReactiveQ")
        self.agent_proadapt = TabularQLearningAgent(name="ProAdapt")

        self.proactive_shifts = 0
        self.history: List[dict] = []
        self.events: List[dict] = []
        self._init_fleet()

    def _init_fleet(self):
        for i in range(NUM_DRONES):
            a = 2 * math.pi * i / NUM_DRONES
            r = random.uniform(0.008, CITY_RADIUS)
            d = Drone(
                id=f"UAV-{i+1:02d}",
                lat=CITY_CENTER[0] + r * math.cos(a),
                lng=CITY_CENTER[1] + r * math.sin(a),
            )
            self.drones[d.id] = d

    def _new_wp(self, d: Drone):
        a = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.005, CITY_RADIUS)
        d.tgt_lat = CITY_CENTER[0] + r * math.cos(a)
        d.tgt_lng = CITY_CENTER[1] + r * math.sin(a)

    def _move(self, d: Drone):
        if d.status == DroneStatus.CHARGING:
            d.battery = min(100, d.battery + 0.6)
            if d.battery >= 95:
                d.status = DroneStatus.IDLE
            return

        if d.battery < 18:
            d.status = DroneStatus.LOW_BATTERY
            d.tgt_lat, d.tgt_lng = CITY_CENTER
            dist = math.hypot(d.lat - CITY_CENTER[0], d.lng - CITY_CENTER[1])
            if dist < 0.002:
                d.status = DroneStatus.CHARGING
                return

        if d.tgt_lat is None:
            self._new_wp(d)
            d.status = DroneStatus.EN_ROUTE

        dx = d.tgt_lat - d.lat
        dy = d.tgt_lng - d.lng
        dist = math.hypot(dx, dy)

        if dist < 0.001:
            if d.status == DroneStatus.EN_ROUTE:
                d.status = DroneStatus.EXECUTING
            else:
                self._new_wp(d)
                d.status = DroneStatus.EN_ROUTE
        else:
            d.lat += (dx / dist) * d.speed
            d.lng += (dy / dist) * d.speed
            d.heading = math.degrees(math.atan2(dy, dx)) % 360

        d.battery -= random.uniform(0.04, 0.09)
        d.altitude = max(30, min(200, d.altitude + random.uniform(-2, 2)))

    def _maybe_task(self, d: Drone):
        if d.status in (DroneStatus.CHARGING, DroneStatus.LOW_BATTERY):
            return None
        if random.random() > TASK_SPAWN_PROB:
            return None

        tt = random.choice(list(TaskType))
        p = TASK_PROFILES[tt]
        self.task_seq += 1
        return {
            "id": f"T-{self.task_seq:04d}",
            "type": tt,
            "drone": d.id,
            "compute": p["compute"] * random.uniform(0.85, 1.15),
            "data": p["data"] * random.uniform(0.8, 1.2),
            "deadline": p["deadline"] * random.uniform(0.85, 1.25),
            "priority": p["priority"],
            "ts": time.time(),
        }

    def _heuristic_decide(self, t: dict, d: Drone) -> ProcessLoc:
        se, sc = 0.0, 0.0
        urg = t["priority"] / 5.0
        if t["deadline"] < 2.0:
            se += 3 * urg
        else:
            sc += 1

        if d.battery < 30:
            sc += 4
        elif d.battery < 50:
            sc += 2
        else:
            se += 1

        if d.edge_cap >= t["compute"]:
            se += 3
        else:
            sc += 3

        if self.cloud.load > 0.7:
            se += 3
        elif self.cloud.load > 0.4:
            se += 1
        else:
            sc += 2

        if t["data"] > 40:
            se += 2
        else:
            sc += 1

        if self.cloud.latency > 0.2:
            se += 2

        return ProcessLoc.EDGE if se >= sc else ProcessLoc.CLOUD

    def _simulate_processing(self, t: dict, loc: ProcessLoc, d: Drone, m: Metrics, mutate_drone: bool = False):
        if loc == ProcessLoc.EDGE:
            proc = t["compute"] / max(0.5, d.edge_cap)
            lat = random.uniform(0.005, 0.02)
            batt_cost = t["compute"] * 0.28
            if mutate_drone:
                d.battery = max(1.0, d.battery - batt_cost)
                d.edge_cap = max(0.0, d.edge_cap - t["compute"] * 0.08)
                d.total_energy_used += batt_cost
            m.edge += 1
            m.sum_batt += batt_cost
        else:
            proc = t["compute"] / (self.cloud.power * self.cloud.scale)
            lat = self.cloud.latency + (t["data"] / 100.0) * 0.1
            batt_cost = 0.0
            if mutate_drone:
                self.cloud.queue += 1
                self.cloud.processed += 1
            m.cloud += 1

        total = proc + lat
        m.tasks += 1
        m.done += 1
        m.sum_time += total
        m.sum_lat += lat
        met = (total <= t["deadline"])
        if met:
            m.hit += 1
        else:
            m.miss += 1

        return total, lat, met, batt_cost

    def _update_cloud(self):
        self.cloud.queue = max(0, self.cloud.queue - random.randint(0, 3))
        self.cloud.load = self.cloud.queue / max(1, self.cloud.max_cap * self.cloud.scale)
        if self.cloud.load > 0.8:
            self.cloud.scale = min(3.0, self.cloud.scale + 0.1)
        elif self.cloud.load < 0.3 and self.cloud.scale > 1.0:
            self.cloud.scale = max(1.0, self.cloud.scale - 0.05)

    def step(self) -> dict:
        self.tick += 1
        evts: List[dict] = []

        for d in self.drones.values():
            self._move(d)

            # Update battery forecasting model for this drone
            pred_batt = self.forecaster.update(d.id, d.battery)

            t = self._maybe_task(d)
            if t is None:
                continue

            # 1. Static (Always Cloud)
            time_static, _, _, _ = self._simulate_processing(
                t, ProcessLoc.CLOUD, d, self.static, mutate_drone=False
            )

            # 2. Heuristic (6-factor rule)
            loc_heur = self._heuristic_decide(t, d)
            time_heur, _, _, _ = self._simulate_processing(
                t, loc_heur, d, self.heuristic, mutate_drone=False
            )

            # 3. Reactive Q-Learning (discretized from current battery)
            s_reactive, _ = StateDiscretizer.discretize(
                battery=d.battery,
                deadline=t["deadline"],
                task_compute=t["compute"],
                edge_capacity=d.edge_cap,
                cloud_load=self.cloud.load,
                cloud_latency=self.cloud.latency,
            )
            act_reactive = self.agent_reactive.select_action(s_reactive)
            loc_reactive = ProcessLoc.EDGE if act_reactive == 0 else ProcessLoc.CLOUD
            time_rq, _, met_rq, batt_rq = self._simulate_processing(
                t, loc_reactive, d, self.reactive_q, mutate_drone=False
            )
            r_rq = TabularQLearningAgent.calculate_reward(
                time_rq, t["deadline"], met_rq, act_reactive, d.battery, batt_rq
            )
            self.agent_reactive.update(s_reactive, act_reactive, r_rq)

            # 4. ProAdapt (Predict-then-Optimize: uses forecasted battery!)
            s_proadapt, _ = StateDiscretizer.discretize(
                battery=pred_batt,  # PROACTIVE BATTERY ESTIMATION
                deadline=t["deadline"],
                task_compute=t["compute"],
                edge_capacity=d.edge_cap,
                cloud_load=self.cloud.load,
                cloud_latency=self.cloud.latency,
            )
            act_proadapt = self.agent_proadapt.select_action(s_proadapt)
            loc_proadapt = ProcessLoc.EDGE if act_proadapt == 0 else ProcessLoc.CLOUD

            # Detect proactive shift (Reactive chose Edge, ProAdapt saw future drain and shifted to Cloud)
            is_proactive_shift = False
            if act_reactive == 0 and act_proadapt == 1 and pred_batt < 35.0:
                self.proactive_shifts += 1
                is_proactive_shift = True

            # ProAdapt actively drives the drone & cloud state in our simulation
            time_pa, _, met_pa, batt_pa = self._simulate_processing(
                t, loc_proadapt, d, self.proadapt, mutate_drone=True
            )
            r_pa = TabularQLearningAgent.calculate_reward(
                time_pa, t["deadline"], met_pa, act_proadapt, d.battery, batt_pa
            )
            self.agent_proadapt.update(s_proadapt, act_proadapt, r_pa)

            evts.append({
                "task": t["id"],
                "type": t["type"].value,
                "drone": d.id,
                "loc": loc_proadapt.value,
                "loc_heur": loc_heur.value,
                "time_ms": round(time_pa * 1000, 1),
                "deadline_ms": round(t["deadline"] * 1000, 1),
                "met": met_pa,
                "battery": round(d.battery, 1),
                "pred_battery": round(pred_batt, 1),
                "proactive_shift": is_proactive_shift,
            })

        self._update_cloud()

        # Edge capacity slow recovery
        for d in self.drones.values():
            if d.status != DroneStatus.CHARGING:
                d.edge_cap = min(8.0, d.edge_cap + 0.08)

        # Snapshot for multi-line charts every 6 ticks
        if self.tick % 6 == 0:
            self.history.append({
                "t": self.tick,
                "static": self.static.to_dict(),
                "heuristic": self.heuristic.to_dict(),
                "reactive_q": self.reactive_q.to_dict(),
                "proadapt": self.proadapt.to_dict(),
                "cloud": self.cloud.to_dict(),
                "fleet_battery": round(
                    sum(dr.battery for dr in self.drones.values()) / NUM_DRONES, 1
                ),
            })

        if evts:
            self.events = (self.events + evts)[-15:]

        pa_stats = self.agent_proadapt.get_stats()
        pa_stats["proactive_shifts"] = self.proactive_shifts

        return {
            "tick": self.tick,
            "drones": [dr.to_dict() for dr in self.drones.values()],
            "cloud": self.cloud.to_dict(),
            "strategies": {
                "static": self.static.to_dict(),
                "heuristic": self.heuristic.to_dict(),
                "reactive_q": self.reactive_q.to_dict(),
                "proadapt": self.proadapt.to_dict(),
            },
            # Backward compatibility aliases
            "adaptive": self.proadapt.to_dict(),
            "static_alias": self.static.to_dict(),
            "proadapt_stats": pa_stats,
            "events": self.events[-8:],
            "history": self.history[-50:],
            "fleet_battery": round(
                sum(dr.battery for dr in self.drones.values()) / NUM_DRONES, 1
            ),
        }


# ── FastAPI application ───────────────────────────────────────────
app = FastAPI(title="DroneCloud — ProAdapt Offloading Simulator")
engine = Engine()


class WSManager:
    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)

    async def broadcast(self, data: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)


mgr = WSManager()


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await mgr.connect(ws)
    try:
        while True:
            state = engine.step()
            await ws.send_json(state)
            await asyncio.sleep(TICK_INTERVAL)
    except WebSocketDisconnect:
        mgr.disconnect(ws)


@app.get("/api/reset")
async def reset_sim():
    global engine
    engine = Engine()
    return {"ok": True}


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
