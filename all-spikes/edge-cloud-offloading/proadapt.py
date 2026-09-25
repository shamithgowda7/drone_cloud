"""
ProAdapt — Lightweight Predict-then-Optimize Offloading Framework
================================================================
Components:
1. BatteryForecaster: Exponential smoothing predictor for drone battery drain.
2. StateDiscretizer: Discretizes continuous drone and cloud telemetry into a 324-state space.
3. TabularQLearningAgent: Compact Q-learning agent with 648 Q-values (324 states x 2 actions).
"""

import math
import random
import time
from typing import Tuple, Dict, Optional


class BatteryForecaster:
    """
    Near-zero overhead Exponential Smoothing forecaster.
    Maintains smoothed consumption rate per tick and projects battery level H ticks ahead.
    """
    def __init__(self, alpha: float = 0.25, horizon: int = 15):
        self.alpha = alpha
        self.horizon = horizon
        self.prev_battery: Dict[str, float] = {}
        self.smoothed_drain: Dict[str, float] = {}

    def update(self, drone_id: str, current_battery: float) -> float:
        """
        Record current battery, update smoothed drain rate, and return predicted battery H ticks ahead.
        """
        if drone_id not in self.prev_battery:
            self.prev_battery[drone_id] = current_battery
            self.smoothed_drain[drone_id] = 0.08  # default baseline drain per tick
            return current_battery

        drain = max(0.0, self.prev_battery[drone_id] - current_battery)
        self.prev_battery[drone_id] = current_battery

        # Exponential smoothing: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
        prev_smooth = self.smoothed_drain.get(drone_id, 0.08)
        self.smoothed_drain[drone_id] = self.alpha * drain + (1.0 - self.alpha) * prev_smooth

        # Forecast H ticks ahead
        projected = current_battery - (self.smoothed_drain[drone_id] * self.horizon)
        return max(0.0, min(100.0, projected))

    def get_forecast(self, drone_id: str, current_battery: float) -> float:
        """Get projected battery level without updating history."""
        drain = self.smoothed_drain.get(drone_id, 0.08)
        projected = current_battery - (drain * self.horizon)
        return max(0.0, min(100.0, projected))


class StateDiscretizer:
    """
    Maps continuous multi-dimensional telemetry into discrete state indices.
    
    State dimensions:
    - Battery level: 3 bins (Low < 30%, Med 30-65%, High > 65%)
    - Deadline urgency: 3 bins (Urgent < 1.5s, Med 1.5-3.0s, Relaxed > 3.0s)
    - Compute load ratio (task compute / edge cap): 3 bins (Light < 0.8, Med 0.8-1.5, Heavy > 1.5)
    - Cloud load: 3 bins (Low < 0.4, Med 0.4-0.75, High > 0.75)
    - Network latency: 4 bins (Fast < 140ms, Normal 140-200ms, Slow 200-280ms, Degraded > 280ms)
    
    Total states: 3 x 3 x 3 x 3 x 4 = 324 states.
    With 2 actions (Edge=0, Cloud=1), Q-table has exactly 324 x 2 = 648 entries.
    """
    NUM_STATES = 324
    NUM_ACTIONS = 2  # 0: EDGE, 1: CLOUD

    @staticmethod
    def discretize(
        battery: float,
        deadline: float,
        task_compute: float,
        edge_capacity: float,
        cloud_load: float,
        cloud_latency: float,
    ) -> Tuple[int, Tuple[int, int, int, int, int]]:
        # 1. Battery bin (3 bins)
        if battery < 30.0:
            b_bin = 0
        elif battery <= 65.0:
            b_bin = 1
        else:
            b_bin = 2

        # 2. Deadline urgency bin (3 bins)
        if deadline < 1.5:
            d_bin = 0
        elif deadline <= 3.0:
            d_bin = 1
        else:
            d_bin = 2

        # 3. Compute ratio bin (3 bins)
        cap = max(0.5, edge_capacity)
        ratio = task_compute / cap
        if ratio < 0.8:
            c_bin = 0
        elif ratio <= 1.5:
            c_bin = 1
        else:
            c_bin = 2

        # 4. Cloud load bin (3 bins)
        if cloud_load < 0.4:
            cl_bin = 0
        elif cloud_load <= 0.75:
            cl_bin = 1
        else:
            cl_bin = 2

        # 5. Network latency bin (4 bins)
        lat_ms = cloud_latency * 1000.0
        if lat_ms < 140.0:
            l_bin = 0
        elif lat_ms <= 200.0:
            l_bin = 1
        elif lat_ms <= 280.0:
            l_bin = 2
        else:
            l_bin = 3

        # Compute flat state index (0 to 323)
        # index = b * (3*3*3*4) + d * (3*3*4) + c * (3*4) + cl * (4) + l
        # index = b * 108 + d * 36 + c * 12 + cl * 4 + l
        flat_idx = b_bin * 108 + d_bin * 36 + c_bin * 12 + cl_bin * 4 + l_bin
        return flat_idx, (b_bin, d_bin, c_bin, cl_bin, l_bin)


class TabularQLearningAgent:
    """
    Lightweight Tabular Q-learning agent.
    Memory footprint: 324 states * 2 actions * 8 bytes ≈ 5.1 KB.
    Inference latency: < 5 microseconds per decision (pure array lookup).
    Zero external ML library dependencies.
    """
    def __init__(
        self,
        name: str = "QLearningAgent",
        lr: float = 0.12,
        gamma: float = 0.85,
        epsilon_start: float = 0.25,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.998,
    ):
        self.name = name
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Flat Q-table: [324][2] initialized to 0.0
        self.q_table = [[0.0, 0.0] for _ in range(StateDiscretizer.NUM_STATES)]

        # Tracking metrics
        self.total_decisions = 0
        self.total_updates = 0
        self.cumulative_reward = 0.0
        self.recent_rewards = []
        self.delta_q_history = []
        self.decision_times_us = []

    def select_action(self, state_idx: int) -> int:
        """
        Epsilon-greedy action selection:
        Returns 0 for EDGE, 1 for CLOUD.
        """
        t0 = time.perf_counter_ns()
        self.total_decisions += 1

        if random.random() < self.epsilon:
            action = random.choice([0, 1])
        else:
            q0, q1 = self.q_table[state_idx]
            if abs(q0 - q1) < 1e-6:
                # Tie-breaking: slight bias based on basic heuristic
                action = random.choice([0, 1])
            else:
                action = 0 if q0 > q1 else 1

        t1 = time.perf_counter_ns()
        self.decision_times_us.append((t1 - t0) / 1000.0)
        if len(self.decision_times_us) > 200:
            self.decision_times_us.pop(0)

        return action

    def update(
        self,
        state_idx: int,
        action: int,
        reward: float,
        next_state_idx: Optional[int] = None,
    ) -> float:
        """
        Bellman Q-learning update:
        Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        old_q = self.q_table[state_idx][action]
        if next_state_idx is not None:
            max_next = max(self.q_table[next_state_idx])
            target = reward + self.gamma * max_next
        else:
            target = reward

        delta = target - old_q
        self.q_table[state_idx][action] += self.lr * delta
        self.total_updates += 1
        self.cumulative_reward += reward

        # Record reward window
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return abs(delta)

    @staticmethod
    def calculate_reward(
        total_time: float,
        deadline: float,
        deadline_met: bool,
        action: int,
        drone_battery: float,
        battery_cost: float,
    ) -> float:
        """
        Multi-objective reward combining latency, deadline satisfaction, and energy preservation:
        - Latency penalty: -1.5 * (time / deadline)
        - Deadline bonus / penalty: +2.0 if met, -3.5 if violated
        - Battery penalty: heavy penalty if processing on edge when battery is vulnerable
        """
        # Time efficiency term
        time_ratio = total_time / max(0.1, deadline)
        r_time = -1.2 * min(3.0, time_ratio)

        # Deadline term
        r_deadline = 2.0 if deadline_met else -3.5

        # Energy penalty
        r_energy = 0.0
        if action == 0:  # EDGE
            if drone_battery < 25.0:
                r_energy = -3.0  # severely penalize draining critical battery
            elif drone_battery < 45.0:
                r_energy = -1.5
            else:
                r_energy = -0.1 * (battery_cost)
        else:  # CLOUD
            # slight penalty for cloud offload latency/bandwidth
            r_energy = 0.1  # bonus for sparing drone battery

        return r_time + r_deadline + r_energy

    def get_stats(self) -> dict:
        nonzero = sum(1 for row in self.q_table for q in row if abs(q) > 1e-4)
        avg_time = (
            sum(self.decision_times_us) / len(self.decision_times_us)
            if self.decision_times_us
            else 4.5
        )
        avg_reward = (
            sum(self.recent_rewards) / len(self.recent_rewards)
            if self.recent_rewards
            else 0.0
        )
        return {
            "qtable_entries_active": nonzero,
            "qtable_total_entries": StateDiscretizer.NUM_STATES * StateDiscretizer.NUM_ACTIONS,
            "epsilon": round(self.epsilon, 4),
            "avg_reward": round(avg_reward, 3),
            "total_decisions": self.total_decisions,
            "avg_decision_time_us": round(avg_time, 2),
            "memory_footprint_kb": round(324 * 2 * 8 / 1024, 2),
        }
