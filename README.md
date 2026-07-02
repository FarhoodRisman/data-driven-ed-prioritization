# Data-Driven Patient Prioritization and Routing in Emergency Departments

Code accompanying the paper *"Data-Driven Patient Prioritization and Routing in
Emergency Departments: A Predictive Analytics Approach"* (IISE Transactions on
Healthcare Systems Engineering).

The code implements a triage-time predictive-analytics framework for emergency
department (ED) patient flow: machine-learning models predict patient attributes
at triage; a seven-attribute code and a similarity-based heuristic route each
patient to a care pod and rank patients within pods; and a discrete-event
simulation evaluates the resulting policies. It also includes the experiments
that assess **robustness to prediction error**, **sensitivity to the acuity
weight**, a **benchmark against the accumulating priority queue (APQ)**, and
**tail-risk (percentile) waiting times**.

The simulation engine depends only on `numpy` and `pandas`; re-fitting the
predictive models additionally uses `scikit-learn`.

## Repository layout

```
.
├── src/
│   ├── ed_simulation.py       # discrete-event ED simulation engine + policies
│   ├── prediction_noise.py    # inject realistic prediction error into predicted attributes
│   ├── confusion_matrices.py  # fitted-model confusion matrices used by the noise pipeline
│   ├── run_robustness.py      # robustness-to-prediction-error experiment
│   ├── weight_sensitivity.py  # sensitivity to the acuity weight
│   ├── benchmark.py           # benchmark vs CTAS+FIFO and APQ
│   ├── tail_risk.py           # 90th/95th-percentile time-to-bed by CTAS
│   ├── fit_models.py          # fit the MLR / Random Forest predictive models
│   └── make_figures.py        # regenerate result figures (vector PDF) from the CSVs
├── results/                   # experiment outputs and fitted-model summaries
├── data/                      # data note (dataset is not redistributable)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

The experiments run on the simulation engine and do not require the patient
dataset. From `src/`:

```bash
python ed_simulation.py         # smoke test: baseline policy, per-CTAS KPIs
python run_robustness.py        # robustness to prediction error  -> results/robustness_results.csv
python weight_sensitivity.py    # acuity-weight sensitivity        -> results/weight_sensitivity_results.csv
python benchmark.py             # vs CTAS+FIFO and APQ             -> results/benchmark_results.csv
python tail_risk.py             # percentile time-to-bed by CTAS   -> results/tail_risk_results.csv
python make_figures.py          # regenerate result figures (PDF)
```

## Re-fitting the predictive models

`fit_models.py` re-fits the triage-time models and exports accuracy, per-class
precision/recall, confusion matrices, and hyper-parameters. It requires the
dataset, which is not distributed here (see `data/README.md`):

```bash
python fit_models.py --data your_data.xlsx
```

## Data availability

The patient-level dataset is not redistributable (see `data/README.md`). The
operational model parameters used by the simulation are aggregate and
non-identifying, and are defined directly in `src/ed_simulation.py`.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

Released under the MIT License (see `LICENSE`).
