# ProAdapt — Edge-Cloud Task Offloading for Drone Fleets

Lightweight **Predict-then-Optimize** task offloading framework for battery-constrained UAV swarms.  
Built for **Cloud Computing CIA 3** | Domain: Mobile Edge Computing, Reinforcement Learning, Autonomous UAVs.

---

## About

ProAdapt addresses the latency-vs-battery trade-off in drone task offloading using a tabular Q-learning agent combined with an exponential smoothing battery forecaster. Unlike deep RL approaches (15–25 MB, 10–25 ms inference), ProAdapt runs in **110 ns** with a **5.18 KB** memory footprint — deployable on embedded ARM microcontrollers.

The simulation runs a 12-UAV fleet over real-world datasets and compares four offloading strategies:

| Strategy | Description |
|---|---|
| Always Cloud | Offload everything to the cloud |
| Heuristic | Rule-based multi-factor scoring |
| Reactive Q-Learning | Tabular RL on current telemetry |
| **ProAdapt** | Tabular RL + EMA battery forecasting |

---

## Project Structure

```
drone_cloud/
├── PROJECT_REPORT.md
└── all-spikes/edge-cloud-offloading/
    ├── app.py                  # FastAPI + WebSocket simulation server
    ├── proadapt.py             # Core ProAdapt engine
    ├── data_loader.py          # Dataset parsers
    ├── evaluate_ablation.py    # Multi-seed benchmark script
    ├── requirements.txt
    ├── data/
    │   ├── alibaba_task_trace.csv
    │   └── nasa_battery_discharge.csv
    └── static/
        ├── index.html
        ├── style.css
        └── app.js
```

---

## Getting Started

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/shamithgowda7/drone_cloud.git
cd drone_cloud/all-spikes/edge-cloud-offloading
pip install -r requirements.txt
```

**Run the simulation dashboard:**

```bash
python app.py
```

Open `http://localhost:8000` to view the live fleet dashboard.

**Run benchmarks:**

```bash
python evaluate_ablation.py
```

---

## Datasets

- **NASA Ames PCoE** — Li-ion battery discharge telemetry (realistic degradation curves)
- **Alibaba Cloud Cluster Trace** — Production workload distributions and task arrival patterns

---

## Results (5 Seeds × 5,000 Ticks)

| Metric | Always Cloud | Heuristic | Reactive Q | **ProAdapt** |
|---|:---:|:---:|:---:|:---:|
| Deadline Hit Rate | 89.2% | 93.4% | 95.1% | **97.6%** |
| Fleet Survival | 100% | 83.3% | 91.7% | **100%** |
| Avg Task Latency | 1.84 s | 1.42 s | 1.18 s | **0.94 s** |
| Decision Time | — | ~1.2 µs | 105 ns | **110 ns** |
| Memory Footprint | — | ~2 KB | 5.18 KB | **5.18 KB** |

---

## Tech Stack

- **Backend:** Python, FastAPI, WebSockets
- **Algorithm:** Tabular Q-Learning, Exponential Moving Average Forecasting
- **Frontend:** Vanilla JS, CSS (dark theme dashboard)

---

## License

Academic project. Datasets credited to NASA Ames PCoE and Alibaba Cloud.
