# DroneCloud: ProAdapt

> **Predictive-Adaptive Edge-Cloud Task Offloading for Battery-Constrained Drone Fleets**  
> *Cloud Computing Continuous Internal Assessment (CIA 3) / Edge Intelligence & Autonomous Systems*

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Framework-FastAPI%20%2B%20WebSockets-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Reinforcement Learning](https://img.shields.io/badge/Algorithm-Tabular%20Q--Learning%20%2B%20EMA-orange)](#system-architecture)
[![Evaluation](https://img.shields.io/badge/Traces-NASA%20Battery%20%26%20Alibaba%20Cloud-green)](#real-world-datasets)

---

## 📌 Overview

Dynamic task offloading in Unmanned Aerial Vehicle (UAV) networks presents a fundamental trade-off:
- **Local Edge Computing**: Minimizes data transmission delay but rapidly depletes onboard Li-ion battery reserves, shortening drone flight time and risking mission aborts.
- **Centralized Cloud Offloading**: Preserves onboard battery life by sending compute-heavy payloads to remote servers, but introduces variable wireless transmission latency, jitter, and cloud queuing delays that threaten real-time deadlines.

While modern literature employs heavy Deep Reinforcement Learning (DRL) and deep sequence models (LSTM/Transformers), these architectures introduce severe systems-level penalties: 15–25 MB memory footprints and 10–25 ms inference latencies that exceed the hardware budgets of low-power ARM companion microcontrollers.

**ProAdapt** solves this with a lightweight **Predict-then-Optimize** framework:
- **Exponential Smoothing Battery Forecaster** ($\alpha = 0.25$, horizon $H=15$ ticks): $O(1)$ complexity, zero heap allocations.
- **Compact 5-Dimensional State Discretizer**: Maps multi-modal telemetry into a 324-state space.
- **Tabular Q-Learning Engine**: Maintains 648 Q-values, executing policy lookups in **110 nanoseconds** (< 0.2 µs) with an in-memory footprint of only **5.18 KB**.

---

## 🚀 Key Features

- 🛰️ **Real-Time 12-UAV Fleet Simulation**: Simulates a swarm of autonomous drones executing mission tasks over a 5 km urban operational radius (Bangalore coordinates).
- ⚡ **4 Decision Strategies Compared**:
  1. **Static (Always Cloud)**: Offloads all tasks to the cloud baseline.
  2. **Heuristic**: Multi-factor rule-based scoring (battery threshold, task urgency, payload).
  3. **Reactive Q-Learning**: Tabular RL driven strictly by instantaneous telemetry.
  4. **ProAdapt**: Predictive-adaptive RL combining tabular Q-learning with real-time battery forecasting.
- 📊 **Real-World Trace Validation**:
  - **NASA Ames Li-ion Battery Discharge Dataset**: Realistic non-linear battery degradation and voltage recovery dynamics.
  - **Alibaba Cloud Cluster Production Traces**: Realistic computational task distributions, payload sizes, and arrival rates.
- 🖥️ **Live Interactive Web Dashboard**: Built with **FastAPI** and **WebSockets**, featuring real-time fleet map rendering, live battery curves, offloading decision streams, latency gauges, and performance comparison charts.
- 🔬 **Extensive Benchmark Suite**: Pre-configured multi-seed ablation scripts (`evaluate_ablation.py`) to reproduce academic results across random seeds.

---

## 📐 System Architecture

```
+---------------------------------------------------------------------------------------+
|                                    SYSTEM OVERVIEW                                    |
|                                                                                       |
|   +--------------------------+                         +--------------------------+   |
|   |    UAV Fleet (N=12)      |                         |       Cloud Center       |   |
|   |  - ARM Companion MCU     |    Wireless Channel     |  - Elastic MEC Server    |   |
|   |  - Li-ion Battery B_i(t) | <=====================> |  - Dynamic Auto-Scaling  |   |
|   |  - Local Compute C_i     |   tau_net = f(L, dist)  |  - Queue Capacity Q_max  |   |
|   |  - Task T_k Arrival      |                         |  - High Power P_cloud    |   |
|   +--------------------------+                         +--------------------------+   |
|                 |                                                    |                |
|                 v                                                    v                |
|        Action a_k = 0 (Edge)                               Action a_k = 1 (Cloud)     |
|      - Zero wireless upload latency                      - Preserves UAV battery      |
|      - Incurs battery energy drain                       - Incurs network & queue lag |
|      - Limited onboard CPU cycles                        - Elastic scalable compute   |
+---------------------------------------------------------------------------------------+
```

### ProAdapt Decision Pipeline

```mermaid
flowchart LR
    A[Telemetry Stream: Battery, CPU, Queue, Latency] --> B[EMA Forecaster<br/>B_pred at t+H]
    B --> C[State Discretizer<br/>324 Discrete States]
    C --> D[Tabular Q-Agent<br/>648 Q-values]
    D --> E{Action Decision}
    E -->|a=0| F[Execute on Local Edge]
    E -->|a=1| G[Offload to Cloud MEC]
```

---

## 📊 Benchmark Results

Evaluated across 5 random seeds (5,000 operational ticks each) against real-world traces:

| Metric | Always Cloud | Heuristic | Reactive Q-Learning | **ProAdapt (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **Deadline Hit Rate** | 89.2% | 93.4% | 95.1% | **97.6%** |
| **Average Task Latency** | 1.84 s | 1.42 s | 1.18 s | **0.94 s** |
| **Fleet Survival Rate** | 100% | 83.3% | 91.7% | **100%** |
| **Decision Lookup Time** | — | ~1.2 µs | 105 ns | **110 ns** |
| **Memory Footprint** | — | ~2 KB | 5.18 KB | **5.18 KB** |
| **Proactive Offload Shifts**| 0 | 4.2 / 1k ticks | 8.6 / 1k ticks | **26.4 / 1k ticks** |

> **Key takeaway**: ProAdapt matches or exceeds Deep RL deadline compliance while consuming **>3,000× less inference time** and **~3,600× less memory**, making it directly deployable on embedded microcontrollers (e.g., STM32, ESP32, Raspberry Pi Zero).

---

## 📁 Repository Structure

```
drone_cloud/
├── PROJECT_REPORT.md                 # Complete academic technical report
├── README.md                         # Project documentation and guide
├── all-spikes/
│   └── edge-cloud-offloading/
│       ├── app.py                    # FastAPI + WebSockets real-time simulation server
│       ├── proadapt.py               # Core ProAdapt engine (Q-learning, Forecaster, Fleet model)
│       ├── data_loader.py            # NASA & Alibaba dataset parsers
│       ├── evaluate_ablation.py      # Multi-seed ablation benchmark script
│       ├── requirements.txt          # Python dependencies
│       ├── paper_draft.md            # Research paper draft
│       ├── PROJECT_REPORT.md         # In-depth technical report
│       ├── data/
│       │   ├── alibaba_task_trace.csv       # Production cloud workload trace
│       │   └── nasa_battery_discharge.csv   # Real Li-ion discharge telemetry
│       └── static/
│           ├── index.html            # Web dashboard interface
│           ├── style.css             # Glassmorphic, responsive dark theme
│           └── app.js                # Live WebSocket telemetry & visualization logic
```

---

## 🛠️ Quick Start

### 1. Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)
- `pip` package manager

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/shamithgowda7/drone_cloud.git
cd drone_cloud/all-spikes/edge-cloud-offloading
pip install -r requirements.txt
```

### 3. Run the Live Web Simulation Dashboard
Start the FastAPI server:
```bash
python app.py
```
Open your web browser and navigate to:
```
http://localhost:8000
```
- Observe real-time drone fleet movement across the map.
- Switch decision algorithms dynamically (`ProAdapt`, `Reactive Q`, `Heuristic`, `Always Cloud`).
- Inspect task queue latencies, energy drain rates, and proactive offload events live.

### 4. Run the Multi-Seed Ablation Benchmarks
To reproduce the experimental evaluation tables and ablation analysis:
```bash
python evaluate_ablation.py
```

---

## 📖 Research & Reference

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)**: Full 370+ line comprehensive technical report containing mathematical formulations, queue theory, state space discretizations, and reward functions.
- **[paper_draft.md](all-spikes/edge-cloud-offloading/paper_draft.md)**: Formal academic paper draft ready for submission.

---

## 📜 License & Acknowledgments
- Prepared for **Cloud Computing Continuous Internal Assessment (CIA 3)**.
- Telemetry datasets courtesy of **NASA Ames Prognostics Center of Excellence (PCoE)** and **Alibaba Cloud Cluster Trace Repository**.
