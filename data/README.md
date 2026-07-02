# Data

The patient-level dataset used in the study is **not included** in this
repository. It is a de-identified administrative extract of Emergency
Department visits from the Winnipeg Regional Health Authority (2013–2018),
used under research ethics protocol H2022:128 (Shared Health), and is not
redistributable.

**To reproduce the predictive models** (`src/fit_models.py`), place the
dataset (a table with the columns described in the paper) in this folder and
pass it with `--data`. Requests for data access should be directed to the
corresponding author and are subject to the relevant data-governance
approvals.

**Operational model parameters** used by the simulation (arrival-rate,
resource, and process-time settings) are aggregate and non-identifying, and
are defined directly in `src/ed_simulation.py`.
