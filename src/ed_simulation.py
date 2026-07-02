"""
ed_simulation.py
===================================================================
Discrete-event simulation of emergency department (ED) patient flow.

A dependency-light engine (numpy/pandas only) implementing an event
scheduler, capacitated resources, and a priority dispatcher.

Model scope (two-pod abstraction):
  * Main pod   (CTAS 1-3): 15 beds, open 24/7.
  * Urgent pod (CTAS 4-5): 7 beds, open 09:00-22:00.
  * Capacitated diagnostics: 2 X-ray rooms + 1 CT scanner.
  * Non-stationary arrivals (mean patients/hour by hour-of-day).
  * Empirically parameterized service and post-treatment times.

Design:
  * TRUE attributes drive the physics (actual service time, disposition,
    diagnostic/CT use); PREDICTED attributes drive the routing/priority
    policy (see prediction_noise.py). With perfect predictions the two
    coincide, which validates the engine.
  * Within-pod allocation uses a priority dispatcher: when a bed frees,
    the waiting patient with the best policy score is admitted next.
  * KPIs: time-to-bed (TTB), length of stay (LOS), per-CTAS punctuality
    (PTP), and risk-adjusted tardiness (RATP), with warm-up + replications.

Model parameters are grouped in the PARAMS section below.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import heapq
from dataclasses import dataclass, field

# =====================================================================
# Minimal dependency-free discrete-event core (SimPy-compatible subset)
# =====================================================================
class Timeout:
    __slots__ = ("delay",)
    def __init__(self, delay): self.delay = float(delay)

class Event:
    def __init__(self, env):
        self.env = env; self.triggered = False; self.value = None; self.waiters = []
    def succeed(self, value=None):
        if self.triggered: return
        self.triggered = True; self.value = value
        for gp in self.waiters:
            self.env._schedule(0.0, gp[0], gp[1], value)
        self.waiters = []

class Process(Event):
    pass

class Resource:
    """Capacity resource with FIFO queue. request()->Event; release(req)."""
    def __init__(self, env, capacity):
        self.env = env; self.cap = capacity; self.count = 0; self.queue = []
    def request(self):
        ev = Event(self.env)
        if self.count < self.cap:
            self.count += 1; ev.succeed()
        else:
            self.queue.append(ev)
        return ev
    def release(self, req=None):
        if self.queue:
            nxt = self.queue.pop(0); nxt.succeed()   # hand the slot directly
        else:
            self.count -= 1

class Environment:
    def __init__(self):
        self.now = 0.0; self._q = []; self._c = 0
    def timeout(self, delay): return Timeout(delay)
    def event(self): return Event(self)
    def _schedule(self, delay, gen, proc, value=None):
        heapq.heappush(self._q, (self.now + delay, self._c, gen, proc, value)); self._c += 1
    def process(self, gen):
        proc = Process(self); self._resume(gen, proc, None); return proc
    def _resume(self, gen, proc, value):
        try:
            y = gen.send(value)
        except StopIteration:
            proc.succeed(None); return
        if isinstance(y, Timeout):
            self._schedule(y.delay, gen, proc, None)
        elif isinstance(y, Event):
            if y.triggered:
                self._schedule(0.0, gen, proc, y.value)
            else:
                y.waiters.append((gen, proc))
        else:
            raise RuntimeError("process yielded unsupported object: %r" % (y,))
    def run(self, until):
        while self._q and self._q[0][0] <= until:
            t, _, gen, proc, value = heapq.heappop(self._q)
            self.now = t
            self._resume(gen, proc, value)

# =====================================================================
# PARAMS
# =====================================================================
RNG_SEED = 12345
ARRIVAL_SCALE = 1.0    # arrival-rate multiplier
SERVICE_SCALE = 1.43   # service-time multiplier (calibrated to target TTB/LOS)
POST_SCALE = 1.0      # scales passive results/boarding time (-> LOS only)

# --- Arrival profile: patients/hour by hour-of-day (0..23) ---
# Aggregate arrival profile: mean patients/hour by hour-of-day (~148/day).
ARRIVAL_PER_HOUR = [3,2,2,2,2,2,3,4,6,9,9,10,10,9,9,9,9,9,8,8,8,6,5,4]

# --- Resources ---
MAIN_BEDS   = 15
URGENT_BEDS = 7
URGENT_OPEN, URGENT_CLOSE = 9, 22       # Urgent pod open 09:00-22:00
N_XRAY = 2
N_CT   = 1

# --- CTAS acuity mix (sums to 1) ---
CTAS_MIX = {1: 0.02, 2: 0.18, 3: 0.34, 4: 0.32, 5: 0.14}

# --- CTAS time-to-bed targets UB_l (min) for PTP (TUNABLE) ---
CTAS_TARGET = {1: 15, 2: 15, 3: 30, 4: 60, 5: 120}

# --- Disposition probabilities by CTAS (TUNABLE) ---
#   D code: 1 Acute-admit, 2 Non-acute-admit, 3 Transfer, 4 Home(discharge)
DISPO_BY_CTAS = {
    1: {1: 0.45, 2: 0.20, 3: 0.15, 4: 0.20},
    2: {1: 0.30, 2: 0.22, 3: 0.10, 4: 0.38},
    3: {1: 0.10, 2: 0.15, 3: 0.06, 4: 0.69},
    4: {1: 0.02, 2: 0.05, 3: 0.03, 4: 0.90},
    5: {1: 0.01, 2: 0.02, 3: 0.02, 4: 0.95},
}
# --- P(diagnostic test needed) by CTAS  (E: 1 yes / 2 no) ---
P_TEST_BY_CTAS    = {1: 0.95, 2: 0.85, 3: 0.65, 4: 0.35, 5: 0.15}
# --- P(consult) by CTAS  (G: 1 yes / 2 no) ---
P_CONSULT_BY_CTAS = {1: 0.55, 2: 0.35, 3: 0.18, 4: 0.06, 5: 0.03}
# --- Resource-use intensity among test-takers (F: 1 super,2 high,3 low) ---
P_RESOURCE = {1: 0.20, 2: 0.35, 3: 0.45}
# --- P(rad uses CT | rad) else X-ray ---
P_RAD_IS_CT = 0.35
# --- EKG / meds / procedure probabilities (TUNABLE) ---
P_EKG = 0.25; P_MED = 0.45; P_PROC = 0.15

# --- Service-time-class thresholds (min) -> B code (1 V.Long..5 V.Short) ---
B_THRESHOLDS = [(300, 1), (200, 2), (120, 3), (60, 4)]   # else 5

# --- Process-time distributions (min): TRIA = (min, mode, max) ---
TRIA = {
    "Greet": (1,2,4.5), "AmboGreet": (1,3,5), "Triage": (3,6,10),
    "RegQuick": (1,2,4), "RegFull": (6,11,17), "EKG": (1,3,7),
    "SetupUrg": (1,3,5), "SetupAcute": (1,5,12),
    "EvalRNUrg": (1,6,15), "EvalRNAcute": (5,9,18),
    "EvalPhysUrg": (3,7,12), "EvalPhysAcute": (4,8,20),
    "OrderPhys": (1,5,8),
    "Lab": (2,15,40), "LabWait": (18,36,78),
    "Rad": (10,17,45), "RadWait": (9,20,54),
    "Med": (1,4,12), "MedWait": (2,5,12),
    "Proc": (1,7,23), "ProcWait": (1,20,120),
    "DC_Phys": (1,6,10), "DC_RN": (3,8,12), "DC_Leave": (1,6,10),
    "Consult": (20,25,30),
    "AD_WaitBed": (40,60,75), "AD_Phys": (5,6.5,10), "AD_RN": (2,7.5,15),
    "AD_Leave": (60,61,62),
    "TRF_Phys": (20,20,20), "TRF_RN": (6,11,18), "TRF_Leave": (30,40,60),
    "CleanBed": (1,4,10),
    "Decide": (12,12,12),   # approx mean of disc(...)
}

# Attribute ranges R_k for Offodile normalization (n_classes - 1).
# Order: A CTAS, B service, C admission, D disposition, E test, F resource, G consult
R_K = np.array([4, 4, 1, 3, 1, 3, 1], dtype=float)


# =====================================================================
# POLICIES
# =====================================================================
# Accumulating Priority Queue (APQ) class rates b_k by CTAS: higher acuity
# accrues priority faster. Benchmark of prior dynamic prioritization
# (Stanford et al. 2014; Vanbrabant et al. 2021).
APQ_RATE = {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0, 5: 1.0}

@dataclass
class Policy:
    name: str
    q_main: tuple
    q_urgent: tuple
    weights: np.ndarray          # length-7 weights w_A..w_G
    route: str = "sim"           # pod assignment: "ctas" or "sim"
    rank: str = "sim"            # within-pod order: "fifo", "apq", or "sim"

def offodile(code, target, weights):
    code = np.asarray(code, float); target = np.asarray(target, float)
    return float(np.sum(weights * (1.0 - np.abs(code - target) / R_K)))

# Baseline weights irrelevant (FIFO), kept for shape.
POLICIES = {
    "baseline": Policy("CTAS+FIFO", (1,1,1,1,1,1,1), (5,5,2,4,2,4,2),
                       np.array([1,0,0,0,0,0,0.]), route="ctas", rank="fifo"),
    "apq": Policy("APQ", (1,1,1,1,1,1,1), (5,5,2,4,2,4,2),
                  np.array([1,0,0,0,0,0,0.]), route="ctas", rank="apq"),
    "discharge": Policy("CTAS+Discharge", (1,1,2,4,1,1,1), (5,1,2,4,1,1,1),
                        np.array([0.5,0,0.25,0.25,0,0,0.]), route="sim", rank="sim"),
    "admit": Policy("CTAS+Admit", (1,1,1,1,1,1,1), (5,1,1,1,1,1,1),
                    np.array([0.5,0,0.25,0.25,0,0,0.]), route="sim", rank="sim"),
    "service": Policy("CTAS+ServiceTime", (1,5,1,1,1,1,1), (5,5,1,1,1,1,1),
                      np.array([0.6,0.4,0,0,0,0,0.]), route="sim", rank="sim"),
    "multi": Policy("Multi-Attribute", (1,3,1,1,1,1,1), (5,5,2,4,2,4,2),
                    np.array([0.4,0.3,0.1,0.1,0.05,0.0,0.05]), route="sim", rank="sim"),
    # Our ranking with CTAS-based routing (for the prioritization-only head-to-head):
    "multi_ctasroute": Policy("Multi-Attr (CTAS routing)", (1,3,1,1,1,1,1), (5,5,2,4,2,4,2),
                    np.array([0.4,0.3,0.1,0.1,0.05,0.0,0.05]), route="ctas", rank="sim"),
}


# =====================================================================
# PATIENT
# =====================================================================
@dataclass
class Patient:
    pid: int
    arrival: float
    ctas: int
    ambulance: bool
    # true attributes (B..G)
    b_true: int; c_true: int; d_true: int; e_true: int; f_true: int; g_true: int
    # predicted attributes (default = true; corrupted by noise externally)
    b_pred: int; c_pred: int; d_pred: int; e_pred: int; f_pred: int; g_pred: int
    service_min: float = 0.0     # true in-bed service minutes (sampled)
    serving_pod: object = None
    # outcomes
    bed_time: float = None
    depart: float = None
    pod: str = None

    def code(self):
        """7-attribute code using PREDICTED B..G and TRUE (observed) CTAS."""
        return (self.ctas, self.b_pred, self.c_pred, self.d_pred,
                self.e_pred, self.f_pred, self.g_pred)


# =====================================================================
# SAMPLING HELPERS
# =====================================================================
def tri(rng, key):
    a, m, b = TRIA[key]
    if a == b:
        return float(a)
    return float(rng.triangular(a, m, b))

def svc(rng, key):
    return tri(rng, key) * SERVICE_SCALE

def pick(rng, prob_dict):
    ks = list(prob_dict); ps = np.array([prob_dict[k] for k in ks], float)
    ps = ps / ps.sum()
    return ks[rng.choice(len(ks), p=ps)]

def b_class(minutes):
    for thr, code in B_THRESHOLDS:
        if minutes >= thr:
            return code
    return 5


# =====================================================================
# PRIORITY POD (custom dispatcher)
# =====================================================================
class Pod:
    def __init__(self, env, name, capacity, policy: Policy, is_urgent=False):
        self.env = env; self.name = name; self.cap = capacity
        self.policy = policy; self.is_urgent = is_urgent
        self.in_use = 0
        self.waiting = []             # list of dict(patient, event)
        self.other = None             # other pod (for overflow pooling)
        self.open_fn = lambda: True   # availability (Urgent: 09:00-22:00)

    def _key(self, p: Patient):
        rank = self.policy.rank
        if rank == "sim":
            tgt = self.policy.q_urgent if self.is_urgent else self.policy.q_main
            return (offodile(p.code(), tgt, self.policy.weights), -p.arrival)
        if rank == "apq":
            # accumulating priority = class rate * waiting time (time-varying,
            # recomputed at each dispatch via env.now)
            return (APQ_RATE[p.ctas] * (self.env.now - p.arrival), -p.arrival)
        # fifo: higher acuity (lower CTAS) first, then first-come
        return (-p.ctas, -p.arrival)

    def request_bed(self, p: Patient):
        ev = self.env.event()
        self.waiting.append({"p": p, "ev": ev})
        self._dispatch()
        if self.other:
            self.other._dispatch()    # primary full -> other may overflow-serve
        return ev

    def release_bed(self):
        self.in_use -= 1
        self._dispatch()
        if self.other:
            self.other._dispatch()

    def _dispatch(self):
        # Serve own queue first; when empty, pull overflow from the other pod.
        while self.open_fn() and self.in_use < self.cap:
            if self.waiting:
                src = self.waiting
            elif self.other and self.other.waiting:
                src = self.other.waiting     # overflow pooled queue
            else:
                break
            best = max(range(len(src)), key=lambda i: self._key(src[i]["p"]))
            item = src.pop(best)
            self.in_use += 1
            item["p"].serving_pod = self
            if not item["ev"].triggered:
                item["ev"].succeed()


# =====================================================================
# SIMULATION
# =====================================================================
class EDModel:
    def __init__(self, policy: Policy, seed=RNG_SEED, noise=None):
        """noise: optional callable(patient, rng) that sets *_pred attributes."""
        self.policy = policy
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.env = Environment()
        self.main = Pod(self.env, "Main", MAIN_BEDS, policy, is_urgent=False)
        self.urgent = Pod(self.env, "Urgent", URGENT_BEDS, policy, is_urgent=True)
        self.main.other = self.urgent
        self.urgent.other = self.main
        env = self.env
        self.urgent.open_fn = lambda: URGENT_OPEN <= (env.now / 60.0) % 24 < URGENT_CLOSE
        self.xray = Resource(self.env, capacity=N_XRAY)
        self.ct = Resource(self.env, capacity=N_CT)
        self.records = []
        self._pid = 0

    # ---- patient generation ----
    def _make_patient(self, t):
        rng = self.rng
        ctas = pick(rng, CTAS_MIX)
        ambulance = ctas <= 2 and rng.random() < 0.6
        d = pick(rng, DISPO_BY_CTAS[ctas])
        c = 1 if d in (1, 2) else 2                       # admitted vs not
        e = 1 if rng.random() < P_TEST_BY_CTAS[ctas] else 2
        f = pick(rng, P_RESOURCE) if e == 1 else 4        # zero use if no test
        g = 1 if rng.random() < P_CONSULT_BY_CTAS[ctas] else 2
        med_flag = (e == 1 and rng.random() < P_MED)
        proc_flag = (e == 1 and rng.random() < P_PROC)
        # compose true service minutes from the care path -> sets b_true
        svc_total = self._sample_service(ctas, d, e, f, g, plan=True)
        b = b_class(svc_total)
        p = Patient(self._pid, t, ctas, ambulance,
                    b, c, d, e, f, g, b, c, d, e, f, g, service_min=svc_total)
        p._med = med_flag; p._proc = proc_flag
        self._pid += 1
        if self.noise is not None:
            self.noise(p, rng)     # overwrite *_pred with corrupted values
        return p

    def _sample_service(self, ctas, d, e, f, g, plan=False):
        """Total IN-BED service minutes (excludes diagnostic resource queueing,
        which is added dynamically). Used both to pre-compute b_true (plan=True)
        and conceptually during the run."""
        rng = self.rng
        urg = ctas >= 4
        t = 0.0
        t += svc(rng, "SetupUrg" if urg else "SetupAcute")
        t += svc(rng, "RegQuick" if urg else "RegFull")
        t += svc(rng, "EvalRNUrg" if urg else "EvalRNAcute")
        t += svc(rng, "EvalPhysUrg" if urg else "EvalPhysAcute")
        if rng.random() < P_EKG: t += svc(rng, "EKG")
        if e == 1:
            t += svc(rng, "OrderPhys")
            n_rad = {1: 2, 2: 1, 3: 1}.get(f, 0)        # super/high/low
            n_lab = {1: 2, 2: 1, 3: 1}.get(f, 0)
            for _ in range(n_lab):
                t += svc(rng, "Lab") + svc(rng, "LabWait")
            for _ in range(n_rad):
                t += svc(rng, "Rad") + svc(rng, "RadWait")
            if rng.random() < P_MED:  t += svc(rng, "Med") + svc(rng, "MedWait")
            if rng.random() < P_PROC: t += svc(rng, "Proc") + svc(rng, "ProcWait")
        if g == 1:
            t += svc(rng, "Consult")
        t += svc(rng, "Decide")
        if d == 4:        # discharge
            t += svc(rng, "DC_Phys") + svc(rng, "DC_RN") + svc(rng, "DC_Leave")
        elif d == 3:      # transfer
            t += svc(rng, "TRF_Phys") + svc(rng, "TRF_RN") + svc(rng, "TRF_Leave")
        else:             # admit (acute/non-acute)
            t += (svc(rng, "AD_WaitBed") + svc(rng, "AD_Phys")
                  + svc(rng, "AD_RN") + svc(rng, "AD_Leave"))
        return t

    # ---- processes ----
    def arrivals(self, days):
        env = self.env
        for day in range(days):
            for hr in range(24):
                rate = ARRIVAL_PER_HOUR[hr] * ARRIVAL_SCALE   # patients this hour
                # non-stationary Poisson: exp interarrivals within the hour
                t_hr = day * 1440 + hr * 60
                if rate <= 0:
                    yield env.timeout(60); continue
                mean_gap = 60.0 / rate
                clock = 0.0
                while True:
                    clock += self.rng.exponential(mean_gap)
                    if clock >= 60:
                        break
                    yield env.timeout(max(0.0, (t_hr + clock) - env.now))
                    p = self._make_patient(env.now)
                    env.process(self.patient_flow(p))
                # advance to end of hour
                if env.now < t_hr + 60:
                    yield env.timeout((t_hr + 60) - env.now)

    def _urgent_open(self):
        hr = (self.env.now / 60.0) % 24
        return URGENT_OPEN <= hr < URGENT_CLOSE

    def _assign_pod(self, p: Patient):
        """Pod assignment. Baseline (CTAS+FIFO) uses fixed CTAS-based routing.
        Similarity-based policies assign each patient to the pod whose TARGET
        CODE its (predicted) code most resembles -- so predictions drive pod
        choice, not just within-pod order. Urgent is bypassed when closed."""
        pol = self.policy
        urgent_ok = self._urgent_open()
        if pol.route == "ctas":
            if p.ctas <= 3:
                return self.main
            return self.urgent if urgent_ok else self.main
        s_main = offodile(p.code(), pol.q_main, pol.weights)
        s_urg = offodile(p.code(), pol.q_urgent, pol.weights)
        if urgent_ok and s_urg > s_main:
            return self.urgent
        return self.main

    def patient_flow(self, p: Patient):
        env = self.env
        # ---- front-end (pre-bed) ----
        yield env.timeout(tri(self.rng, "AmboGreet" if p.ambulance else "Greet"))
        yield env.timeout(tri(self.rng, "Triage"))
        # ---- pod choice ----
        pod = self._assign_pod(p)
        # ---- wait for bed (priority dispatcher; may be served via overflow) ----
        yield pod.request_bed(p)
        serving = p.serving_pod
        p.pod = serving.name
        p.bed_time = env.now
        # ---- active care (bed held); diagnostics seize X-ray/CT ----
        yield env.process(self._bed_care(p))
        serving.release_bed()
        # ---- results-pending / boarding (no bed); counts toward LOS ----
        yield env.process(self._post_bed(p))
        p.depart = env.now
        self.records.append(p)

    def _bed_care(self, p: Patient):
        """ACTIVE care while holding a treatment bed (excludes passive result
        waits and admit boarding, which are modelled as non-bed time so bed
        occupancy reflects real ED turnover)."""
        env = self.env; rng = self.rng
        urg = p.ctas >= 4
        yield env.timeout(svc(rng, "SetupUrg" if urg else "SetupAcute"))
        yield env.timeout(svc(rng, "RegQuick" if urg else "RegFull"))
        yield env.timeout(svc(rng, "EvalRNUrg" if urg else "EvalRNAcute"))
        yield env.timeout(svc(rng, "EvalPhysUrg" if urg else "EvalPhysAcute"))
        if rng.random() < P_EKG:
            yield env.timeout(svc(rng, "EKG"))
        if p.e_true == 1:
            yield env.timeout(svc(rng, "OrderPhys"))
            n_rad = {1: 2, 2: 1, 3: 1}.get(p.f_true, 0)
            for _ in range({1: 2, 2: 1, 3: 1}.get(p.f_true, 0)):
                yield env.timeout(svc(rng, "Lab"))          # draw/active
            for _ in range(n_rad):
                res = self.ct if rng.random() < P_RAD_IS_CT else self.xray
                req = res.request()
                yield req
                yield env.timeout(svc(rng, "Rad"))          # imaging (capacitated)
                res.release(req)
            if getattr(p, "_med", False):
                yield env.timeout(svc(rng, "Med"))
            if getattr(p, "_proc", False):
                yield env.timeout(svc(rng, "Proc"))
        if p.g_true == 1:
            yield env.timeout(svc(rng, "Consult"))
        yield env.timeout(svc(rng, "Decide"))
        # disposition: provider/nurse activities at bedside
        if p.d_true == 4:
            yield env.timeout(svc(rng, "DC_Phys") + svc(rng, "DC_RN"))
        elif p.d_true == 3:
            yield env.timeout(svc(rng, "TRF_Phys") + svc(rng, "TRF_RN"))
        else:
            yield env.timeout(svc(rng, "AD_Phys") + svc(rng, "AD_RN"))

    def _post_bed(self, p: Patient):
        """PASSIVE time after the treatment bed is released: results-pending,
        admit boarding, and leave delays. Counts toward LOS, not bed occupancy."""
        env = self.env; rng = self.rng
        if p.e_true == 1:
            n = {1: 2, 2: 1, 3: 1}.get(p.f_true, 0)
            for _ in range(n):
                yield env.timeout(POST_SCALE * svc(rng, "LabWait"))
            for _ in range(n):
                yield env.timeout(POST_SCALE * svc(rng, "RadWait"))
            if getattr(p, "_med", False):
                yield env.timeout(POST_SCALE * svc(rng, "MedWait"))
            if getattr(p, "_proc", False):
                yield env.timeout(POST_SCALE * svc(rng, "ProcWait"))
        if p.d_true == 4:
            yield env.timeout(POST_SCALE * svc(rng, "DC_Leave"))
        elif p.d_true == 3:
            yield env.timeout(POST_SCALE * svc(rng, "TRF_Leave"))
        else:
            yield env.timeout(POST_SCALE * svc(rng, "AD_WaitBed") + POST_SCALE * svc(rng, "AD_Leave"))

    def run(self, days=35, warmup_days=7):
        self.env.process(self.arrivals(days))
        self.env.run(until=days * 1440)
        warm = warmup_days * 1440
        rows = []
        for p in self.records:
            if p.bed_time is None or p.arrival < warm:
                continue
            rows.append(dict(pid=p.pid, ctas=p.ctas, pod=p.pod,
                             ttb=p.bed_time - p.arrival,
                             los=p.depart - p.arrival,
                             dispo=p.d_true))
        return pd.DataFrame(rows)


# =====================================================================
# KPIs
# =====================================================================
def kpis(df):
    out = {}
    df = df.copy()
    df["tardy"] = df["ttb"] > df["ctas"].map(CTAS_TARGET)
    by = df.groupby("ctas")
    out["per_ctas"] = pd.DataFrame({
        "TTB": by["ttb"].mean(),
        "LOS": by["los"].mean(),
        "TTB_p90": by["ttb"].quantile(0.90),
        "TTB_p95": by["ttb"].quantile(0.95),
        "PTP": by["tardy"].mean(),
        "n": by.size(),
    })
    ptp = out["per_ctas"]["PTP"].to_dict()
    out["RATP"] = float(sum((1.0/l) * ptp.get(l, 0.0) for l in range(1, 6)))
    out["overall_TTB"] = float(df["ttb"].mean())
    out["overall_LOS"] = float(df["los"].mean())
    return out


def run_replications(policy_key, n_reps=5, days=35, warmup_days=7, base_seed=1000,
                     noise=None):
    pol = POLICIES[policy_key]
    ttb=[]; los=[]; ratp=[]; per=[]
    for r in range(n_reps):
        m = EDModel(pol, seed=base_seed + r, noise=noise)
        df = m.run(days=days, warmup_days=warmup_days)
        k = kpis(df)
        ttb.append(k["overall_TTB"]); los.append(k["overall_LOS"]); ratp.append(k["RATP"])
        per.append(k["per_ctas"])
    per_mean = sum(p.fillna(0) for p in per) / len(per)
    return {"policy": pol.name, "TTB": np.mean(ttb), "LOS": np.mean(los),
            "RATP": np.mean(ratp), "per_ctas": per_mean,
            "TTB_sd": np.std(ttb), "RATP_sd": np.std(ratp)}


if __name__ == "__main__":
    print("Smoke test: baseline policy, 3 reps x (7d warmup + 28d)\n")
    res = run_replications("baseline", n_reps=3, days=35, warmup_days=7)
    print(f"Policy: {res['policy']}")
    print(f"Overall TTB = {res['TTB']:.1f} min   LOS = {res['LOS']:.1f} min   "
          f"RATP = {res['RATP']:.3f}")
    print("\nPer-CTAS:")
    print(res["per_ctas"].round(1).to_string())
    print("\nOK - engine ran.")
