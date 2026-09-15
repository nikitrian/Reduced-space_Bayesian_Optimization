# ============================================
# Multi-fidelity BO for Aspen HYSYS (HI) + ANN (LO)
# With enumeration + HI cooldown, and warm-start from top_*.csv
# ============================================

import os, sys, time, random
import numpy as np
import pandas as pd
import torch
from torch import Tensor

# BoTorch / GPyTorch
from botorch.models import SingleTaskMultiFidelityGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.dirname(BASE_DIR)
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)

# Aspen HYSYS COM
import hysys_python.hysys_object_persistence as hop
import hysys_gsa_util as hgu


# ----------------------
# CONFIG
# ----------------------
device = torch.device("cpu")
dtype  = torch.double

# Files
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "..", ".."))
HYSYS_FILE_DEFAULT = os.path.join(
    REPO_ROOT,
    "Data generation",
    "Dimethyl ether - Aspen HYSYS",
    "i-dme-complete-gsa-equil.hsc",
)
HYSYS_FILE = os.environ.get("FLOWRATE_HYSYS_FILE", HYSYS_FILE_DEFAULT)
ANN_PKL_PATH     = os.path.join(BASE_DIR, "ann_ACC_0.13_44720_0.0002_418_1.pkl")
INPUTS_CSV       = os.path.join(BASE_DIR, "inputs.csv")
OUTPUTS_CSV      = os.path.join(BASE_DIR, "outputs.csv")
TOP_INPUTS_CSV   = os.path.join(BASE_DIR, "top_inputs.csv")     # optional
TOP_OUTPUTS_CSV  = os.path.join(BASE_DIR, "top_outputs.csv")    # optional
METRIC           = "DME flowrate"

# Costs / BO knobs
cost_hi, cost_lo = 5.0, 1.0
budget           = 200.0
beta             = 20.0
alpha            = 0.1
HI_BIAS = 1  # try 1.05-1.25

# --- Periodic LO->HI promotion knobs ---
PROMOTE_EVERY_ITERS = 10   # promote every N iterations
K_PROMOTE_INTERVAL  = 3    # promote top-K LO points each time
CHARGE_PROMOTION    = False # count these HI evals against budget
DEDUP_ATOL          = 1e-6 # treat x's equal if all dims within this abs tol



# Seeds / iters
num_seeds  = 10
max_iters  = 1000
N_hi_init, N_lo_init = 3, 2

# HI cooldown (force an HI if too many LO in a row)
HI_COOLDOWN_STEPS = 10

# Output root
out_root = os.path.join(BASE_DIR, "results_flowrate")
os.makedirs(out_root, exist_ok=True)

# ----------------------
# Utilities
# ----------------------
def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

def to_2d(x: Tensor):
    return x.view(-1, x.shape[-1])

# ----------------------
# Derived constants for the flowrate case
# ----------------------
def _compute_meoh_nominal_volume() -> float:
    ref_co2_flow = 88_000.0  # kg/h CO2
    co2_flow = 28_333.3620500565  # kg/h in simulation
    co2_ratio = co2_flow / ref_co2_flow
    ref_cat_mass = 44_500.0  # kg catalyst
    cat_mass = ref_cat_mass * co2_ratio
    void = 0.5
    density = 1_775.0  # kg/m3
    return cat_mass * (1.0 / density) * (1.0 / (1.0 - void))


MEOH_NOMINAL_VOL = _compute_meoh_nominal_volume()

# ----------------------
# SPACE (6D), ANN norm space (~10%)
# ----------------------
SPACE = [
    {"name": "h2_ratio",      "type": "continuous", "domain": (2.4, 3.6)},
    {"name": "Pmeoh",         "type": "continuous", "domain": (5000.0, 10000.0)},
    {"name": "Tmeoh",         "type": "continuous", "domain": (210.0, 270.0)},
    {"name": "reactor_mode",  "type": "continuous", "domain": (0.0, 1.0)},
    {"name": "recycle_ratio", "type": "continuous", "domain": (0.95, 0.991)},
    {"name": "Vmeoh",         "type": "continuous", "domain": (0.8 * MEOH_NOMINAL_VOL, 1.2 * MEOH_NOMINAL_VOL)},
]
D = len(SPACE)
SPACE_COLS = [v['name'] for v in SPACE]

ANN_NORM_SPACE = None  # overwritten once the ANN checkpoint is loaded

# ----------------------
# Normalization helpers
# ----------------------
from botorch.utils.transforms import unnormalize as _unnormalize

def _bounds_tensor(space, like: torch.Tensor):
    return torch.tensor([v['domain'] for v in space], dtype=like.dtype, device=like.device).t()

def unnormalize_x(X01: torch.Tensor, space):
    bounds = _bounds_tensor(space, X01)
    return _unnormalize(X01, bounds)

def real_to_unit01_np(arr: np.ndarray, space) -> np.ndarray:
    xmax = np.array([v['domain'][1] for v in space], dtype=np.float64)
    xmin = np.array([v['domain'][0] for v in space], dtype=np.float64)
    return (arr - xmin) / (xmax - xmin)

def mm_norm_np(arr: np.ndarray, space):
    xmax = np.array([v['domain'][1] for v in space], dtype=np.float64)
    xmin = np.array([v['domain'][0] for v in space], dtype=np.float64)
    return (arr - xmin) / (xmax - xmin)

def mm_norm_np_clipped(arr: np.ndarray, space):
    return np.clip(mm_norm_np(arr, space), 0.0, 1.0)

# ----------------------
# ANN model (6 inputs), scaler from outputs.csv
# ----------------------
import pickle
from torch import nn
from sklearn.preprocessing import StandardScaler

class ReLUNet(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, num_layers):
        super().__init__()
        self.hidden_layers = torch.nn.ModuleList(
            [nn.Linear(n_input, n_hidden)] +
            [nn.Linear(n_hidden, n_hidden) for _ in range(num_layers-1)]
        )
        self.output = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.output(x)

def load_pkl(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)

def save_pkl(fp, obj):
    with open(fp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _standard_scaler_state(scaler: StandardScaler) -> dict:
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "n_samples": int(getattr(scaler, "n_samples_seen_", mean.shape[0]) or mean.shape[0]),
    }

def _restore_standard_scaler(state: dict) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]
    scaler.n_samples_seen_ = int(state.get("n_samples", scaler.n_features_in_)) or scaler.n_features_in_
    return scaler

_TARGET_SCALER_KEY = "target_scaler"

def _scaler_states_close(old: dict, new: dict, atol: float = 1e-12) -> bool:
    if old is None or new is None:
        return False
    old_mean = np.asarray(old.get("mean", []), dtype=np.float64)
    new_mean = np.asarray(new.get("mean", []), dtype=np.float64)
    if old_mean.shape != new_mean.shape:
        return False
    old_scale = np.asarray(old.get("scale", []), dtype=np.float64)
    new_scale = np.asarray(new.get("scale", []), dtype=np.float64)
    if old_scale.shape != new_scale.shape:
        return False
    return np.allclose(old_mean, new_mean, atol=atol, rtol=0.0) and np.allclose(old_scale, new_scale, atol=atol, rtol=0.0)

def _fit_target_scaler_from_outputs() -> StandardScaler:
    out_df = pd.read_csv(OUTPUTS_CSV)
    if METRIC in out_df.columns:
        out_df = out_df[[METRIC]]
    elif out_df.shape[1] == 1:
        out_df.columns = [METRIC]
    else:
        raise ValueError(f"{OUTPUTS_CSV} must have '{METRIC}' or a single column.")
    y = out_df.to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(y)
    return scaler

def load_ann_and_scaler():
    P = load_pkl(ANN_PKL_PATH)
    SD = P['state_dict']
    structure = P['structure']  # [*, *, hidden, num_layers]
    normP = P.get('normP')

    y_scaler = _fit_target_scaler_from_outputs()
    new_state = _standard_scaler_state(y_scaler)
    stored_state = P.get(_TARGET_SCALER_KEY)
    if stored_state is None or not _scaler_states_close(stored_state, new_state):
        P[_TARGET_SCALER_KEY] = new_state
        save_pkl(ANN_PKL_PATH, P)

    if normP is not None:
        max_arr = np.asarray(normP[0], dtype=np.float64)
        min_arr = np.asarray(normP[1], dtype=np.float64)
        if max_arr.shape != min_arr.shape:
            raise ValueError("ANN checkpoint normP has mismatched shapes.")
        if max_arr.shape[0] != D:
            raise ValueError(f"ANN checkpoint expects {max_arr.shape[0]} inputs but SPACE defines {D}.")
        global ANN_NORM_SPACE
        ANN_NORM_SPACE = [
            {"name": SPACE_COLS[i], "type": SPACE[i]['type'], "domain": (float(min_arr[i]), float(max_arr[i]))}
            for i in range(D)
        ]

    n_input = D
    n_output = 1
    n_hidden = structure[2]; num_layers = structure[3]
    ann = ReLUNet(n_input, n_hidden, n_output, num_layers).to(dtype=torch.float64)
    ann.load_state_dict(SD); ann.eval()
    return ann, y_scaler

ANN_MODEL, Y_SCALER = load_ann_and_scaler()
# ----------------------
# HI: Aspen HYSYS flowrate (DME production)
# ----------------------

FLOWRATE_INPUT_TEMPLATE = [
    None,             # h2_ratio
    None,             # Pmeoh
    None,             # Tmeoh
    None,             # reactor_mode
    None,             # recycle_ratio
    None,             # Vmeoh
    275.0,            # Tdme (fixed)
    1500.0,           # Pdme (fixed)
    0.5,              # xmeoh (fixed)
    0.5,              # xdme (fixed)
    57.0,             # n_trays1 (fixed)
    17.0,             # n_trays2 (fixed)
    44.0 / 57.0,      # feed_loc1 (fixed)
    10.0 / 17.0,      # feed_loc2 (fixed)
]

HYSYS_FAILURE_VALUE = 0.0
_HYSYS_SIM = None
_HYSYS_FLOWSHEET = None
_HYSYS_SOLVER = None

def _reset_hysys() -> None:
    global _HYSYS_SIM, _HYSYS_FLOWSHEET, _HYSYS_SOLVER
    _HYSYS_SIM = None
    _HYSYS_FLOWSHEET = None
    _HYSYS_SOLVER = None

def _ensure_hysys():
    global _HYSYS_SIM, _HYSYS_FLOWSHEET, _HYSYS_SOLVER
    if _HYSYS_SIM is not None:
        return _HYSYS_SIM, _HYSYS_FLOWSHEET, _HYSYS_SOLVER
    if not HYSYS_FILE:
        raise RuntimeError('HYSYS file path not set. Provide FLOWRATE_HYSYS_FILE environment variable.')
    if not os.path.exists(HYSYS_FILE):
        raise FileNotFoundError(f'HYSYS file not found: {HYSYS_FILE}')
    try:
        _HYSYS_SIM = hop.hysys_connection(HYSYS_FILE, active=1)
    except Exception:
        _HYSYS_SIM = hop.hysys_connection(HYSYS_FILE, active=0)
    _HYSYS_FLOWSHEET = _HYSYS_SIM.Flowsheet
    _HYSYS_SOLVER = _HYSYS_SIM.Solver
    print(f"[HI] Connected to HYSYS model at '{HYSYS_FILE}'.")
    return _HYSYS_SIM, _HYSYS_FLOWSHEET, _HYSYS_SOLVER

def _evaluate_hysys_point(x_real: np.ndarray) -> float:
    try:
        sim, fsheet, solver = _ensure_hysys()
    except Exception as err:
        print(f"[HI] Connection error: {err}")
        return HYSYS_FAILURE_VALUE

    if x_real.shape[-1] != D:
        raise ValueError(f"Expected {D} decision variables, received {x_real.shape[-1]}.")

    inputs = []
    values_iter = iter(map(float, x_real.tolist()))
    for entry in FLOWRATE_INPUT_TEMPLATE:
        if entry is None:
            try:
                inputs.append(next(values_iter))
            except StopIteration:
                raise RuntimeError("Insufficient decision variables when constructing HYSYS inputs.") from None
        else:
            inputs.append(float(entry))
    try:
        next(values_iter)
        raise RuntimeError("Received extra decision variables when constructing HYSYS inputs.")
    except StopIteration:
        pass
    try:
        sim.Visible = False
        outputs = hgu.solve_calc_flowsheet(fsheet, solver, inputs)
    except Exception as err:
        print(f"[HI] solve_calc_flowsheet failed: {err}. Resetting connection.")
        _reset_hysys()
        return HYSYS_FAILURE_VALUE

    if outputs is None:
        return HYSYS_FAILURE_VALUE

    outputs_arr = np.asarray(outputs, dtype=object)
    if outputs_arr.size < 1:
        return HYSYS_FAILURE_VALUE
    value = outputs_arr.flat[0]
    if value is None:
        return HYSYS_FAILURE_VALUE
    try:
        value = float(value)
    except (TypeError, ValueError):
        return HYSYS_FAILURE_VALUE
    if not np.isfinite(value):
        return HYSYS_FAILURE_VALUE
    return value

def flowrate_sim_hi(X01: torch.Tensor) -> torch.Tensor:
    if X01.dim() == 1:
        X01 = X01.unsqueeze(0)
    X_real = unnormalize_x(X01, SPACE).cpu().numpy()
    out = np.zeros((X_real.shape[0],), dtype=np.float64)
    for i, row in enumerate(X_real):
        out[i] = _evaluate_hysys_point(row)
    return torch.from_numpy(out).unsqueeze(-1).to(dtype=dtype)
# ----------------------
# LO: ANN surrogate (uses training-time min/max from checkpoint)
# ----------------------
@torch.no_grad()
def flowrate_ann_lo(X01: torch.Tensor) -> torch.Tensor:
    # X01: (n, D) in [0,1]^D
    if X01.dim() == 1:
        X01 = X01.unsqueeze(0)
    if ANN_NORM_SPACE is None:
        raise RuntimeError("ANN normalization bounds not loaded; ensure ANN checkpoint includes normP.")

    # unnormalize to real ranges (simulation bounds)
    real = unnormalize_x(X01, SPACE).cpu().numpy()              # (n, D) original ranges

    # normalize inputs for the ANN using the Â±10% bounds it was trained on
    x_norm = mm_norm_np_clipped(real, ANN_NORM_SPACE)           # (n, D) in ANN's [0,1] box
    xt = torch.from_numpy(x_norm).to(dtype=torch.float64)

    # ANN predicts *standardized* y (because it was trained that way)
    y_std = ANN_MODEL(xt).cpu().numpy().reshape(-1, 1)

    # CRUCIAL: convert back to original units with the same scaler you fit on outputs.csv
    y_orig = Y_SCALER.inverse_transform(y_std)                  # (n, 1) original units

    return torch.from_numpy(y_orig).to(dtype=torch.double)      # feed GP in original units


f_hi, f_lo = flowrate_sim_hi, flowrate_ann_lo

# ----------------------
# Model builders (MF + SF)
# ----------------------
def build_mf_model(train_x: Tensor, train_y: Tensor):
    train_x = train_x.to(dtype=dtype).contiguous()
    train_y = train_y.to(dtype=dtype).contiguous()
    model = SingleTaskMultiFidelityGP(
        train_x, train_y,
        data_fidelities=[train_x.shape[-1]-1],
        linear_truncated=True, nu=5/2,
        input_transform=Normalize(train_x.shape[-1]),
        outcome_transform=Standardize(1),
    ).to(dtype=dtype)
    mll = ExactMarginalLogLikelihood(model.likelihood, model); fit_gpytorch_mll(mll)
    return model

def build_sf_model(train_x: Tensor, train_y: Tensor):
    train_x = train_x.to(dtype=dtype).contiguous()
    train_y = train_y.to(dtype=dtype).contiguous()
    y_std = float(train_y.std())
    outcome_tf = Standardize(1) if y_std > 1e-6 else None
    model = SingleTaskGP(
        train_X=train_x, train_Y=train_y,
        input_transform=Normalize(train_x.shape[-1]),
        outcome_transform=outcome_tf,
    ).to(dtype=dtype)
    try:
        init_noise = max(1e-6, 0.01 * float(train_y.var().item()))
        model.likelihood.noise_covar.initialize(noise=init_noise)
    except Exception:
        pass
    mll = ExactMarginalLogLikelihood(model.likelihood, model); fit_gpytorch_mll(mll)
    return model

# ----------------------
# Acquisition helpers
# ----------------------
def optimize_ucb_sf(model, bounds_x, beta, num_restarts=15, raw_samples=128):
    acq = UpperConfidenceBound(model, beta=beta)
    x, val = optimize_acqf(acq, bounds=bounds_x, q=1,
                           num_restarts=num_restarts, raw_samples=raw_samples)
    return x.squeeze(0), float(val.item())

def optimize_ucb_at_fidelity_mf(model, bounds_x, fidelity, beta, num_restarts=15, raw_samples=128):
    acq = UpperConfidenceBound(model, beta=beta)
    def wrapped(X):
        if X.dim() == 2: X = X.unsqueeze(1)
        b, q, d = X.shape
        fcol = torch.full((b,q,1), float(fidelity), dtype=X.dtype, device=X.device)
        return acq(torch.cat([X, fcol], dim=-1))
    x, val = optimize_acqf(wrapped, bounds=bounds_x, q=1,
                           num_restarts=num_restarts, raw_samples=raw_samples)
    return x.squeeze(0), float(val.item())

def select_xf_by_enumeration(model, bounds_x, beta, alpha, cost_lo, cost_hi):
    x_lo, u_lo = optimize_ucb_at_fidelity_mf(model, bounds_x, fidelity=0.0, beta=beta)
    x_hi, u_hi = optimize_ucb_at_fidelity_mf(model, bounds_x, fidelity=1.0, beta=beta)
    adj_lo = u_lo / (cost_lo ** alpha)
    adj_hi = (u_hi * HI_BIAS) / (cost_hi ** alpha)
    if adj_hi >= adj_lo:  # ties -> HI
        return x_hi, torch.tensor([[1.0]], dtype=dtype), cost_hi, "HI", u_hi, u_lo, adj_hi, adj_lo
    else:
        return x_lo, torch.tensor([[0.0]], dtype=dtype), cost_lo, "LO", u_hi, u_lo, adj_hi, adj_lo

# ----------------------
# Warm-start from prior HI (top_*.csv)
# ----------------------
def load_top_hi_data():
    if (not os.path.exists(TOP_INPUTS_CSV)) or (not os.path.exists(TOP_OUTPUTS_CSV)):
        print('[warm] No top_inputs/top_outputs found. Starting without prior HI data.')
        return None, None
    X_df = pd.read_csv(TOP_INPUTS_CSV)
    # Expect exactly the D SPACE columns (either named as SPACE or same order)
    if X_df.shape[1] != D:
        raise ValueError(f"{TOP_INPUTS_CSV} must have exactly {D} columns.")
    # If named, reorder to SPACE; if unnamed, assume correct order
    if set(X_df.columns) == set(SPACE_COLS):
        X_df = X_df.loc[:, SPACE_COLS]
    X_real = X_df.to_numpy(dtype=np.float64)

    Y_df = pd.read_csv(TOP_OUTPUTS_CSV)
    if METRIC in Y_df.columns:
        Y_df = Y_df[[METRIC]]
    elif Y_df.shape[1] == 1:
        Y_df.columns = [METRIC]
    else:
        raise ValueError(f"{TOP_OUTPUTS_CSV} must have '{METRIC}' or a single column.")
    Y = Y_df.to_numpy(dtype=np.float64).reshape(-1, 1)

    X01 = real_to_unit01_np(X_real, SPACE)
    X01 = torch.from_numpy(X01).to(dtype=dtype)
    Yt = torch.from_numpy(Y).to(dtype=dtype)
    F1 = torch.ones(X01.shape[0], 1, dtype=dtype)
    X_aug = torch.cat([X01, F1], dim=-1)
    print(f"[warm] Loaded {X_aug.shape[0]} prior HI rows from top_inputs/top_outputs.")
    return X_aug, Yt

def _dedup_mask(new_X, existing_X, atol=1e-6):
    """
    Returns a boolean mask of new_X rows that are NOT within atol of any existing_X row.
    Shapes: new_X: (k, D), existing_X: (n, D). All in [0,1]^D.
    """
    if existing_X is None or existing_X.numel() == 0:
        return torch.ones(new_X.shape[0], dtype=torch.bool, device=new_X.device)
    # pairwise |x_i - y_j|_âˆž (max abs per row)
    diffs = (new_X.unsqueeze(1) - existing_X.unsqueeze(0)).abs().amax(dim=-1)
    is_new = (diffs > atol).all(dim=1)
    return is_new

def _periodic_promote_lo_to_hi(train_x, train_y, D, f_hi,
                               k_promote, charge, cost_hi, total_cost,
                               promoted_bank=None, atol=1e-6):
    """
    Picks top-k LO archive points (by LO value stored in train_y at LO rows),
    dedups against previously promoted points, evaluates them at HI once,
    appends to archive (X with fidelity=1), and returns updated (train_x, train_y, total_cost, promoted_bank, num_promoted).
    """
    device = train_x.device
    lo_mask = (train_x[:, -1] < 0.5)
    if not torch.any(lo_mask):
        return train_x, train_y, total_cost, promoted_bank, 0

    lo_vals = train_y[lo_mask].view(-1)
    lo_X = train_x[lo_mask][:, :D]

    k = min(k_promote, lo_vals.numel())
    if k == 0:
        return train_x, train_y, total_cost, promoted_bank, 0
    top_idx = torch.topk(lo_vals, k=k).indices
    X_cands = lo_X[top_idx]

    hi_mask = (train_x[:, -1] > 0.5)
    X_hi_archive = train_x[hi_mask][:, :D]
    if promoted_bank is not None and promoted_bank.numel() > 0:
        dedup_bank = torch.cat([X_hi_archive, promoted_bank], dim=0)
    else:
        dedup_bank = X_hi_archive

    keep_mask = _dedup_mask(X_cands, dedup_bank, atol=atol)
    X_promote = X_cands[keep_mask]

    if X_promote.shape[0] == 0:
        return train_x, train_y, total_cost, promoted_bank, 0

    with torch.no_grad():
        Y_promote_hi = f_hi(X_promote).view(-1, 1).to(dtype=train_y.dtype, device=device)

    F_hi = torch.ones(X_promote.shape[0], 1, dtype=train_x.dtype, device=device)
    train_x = torch.cat([train_x, torch.cat([X_promote, F_hi], dim=-1)], dim=0)
    train_y = torch.cat([train_y, Y_promote_hi], dim=0)

    if charge:
        total_cost += X_promote.shape[0] * float(cost_hi)

    promoted_bank = X_promote if promoted_bank is None else torch.cat([promoted_bank, X_promote], dim=0)
    return train_x, train_y, total_cost, promoted_bank, int(X_promote.shape[0])

# ----------------------
# Runners
# ----------------------
def run_mf_enum(bounds_x, f_hi, f_lo, f_star=None, tag="superpro"):
    all_results = []
    out_dir = os.path.join(out_root, f"mf_enum_{tag}_a{alpha}_chi{cost_hi}")
    os.makedirs(out_dir, exist_ok=True)

    for seed in range(4,6):
        set_seed(seed); t0 = time.time()

        # Warm-start from top files
        X_top_aug, Y_top = load_top_hi_data()

        # Initial designs
        x_hi0 = draw_sobol_samples(bounds=bounds_x, n=N_hi_init, q=1).squeeze(-2).to(dtype=dtype)
        x_lo0 = draw_sobol_samples(bounds=bounds_x, n=N_lo_init, q=1).squeeze(-2).to(dtype=dtype)
        y_hi0, y_lo0 = f_hi(x_hi0), f_lo(x_lo0)

        X_hi = torch.cat([x_hi0, torch.ones(x_hi0.shape[0], 1, dtype=dtype)], dim=-1)
        X_lo = torch.cat([x_lo0, torch.zeros(x_lo0.shape[0], 1, dtype=dtype)], dim=-1)
        train_x = torch.cat([X_hi, X_lo], dim=0)
        train_y = torch.cat([y_hi0, y_lo0], dim=0).view(-1,1)

        if (X_top_aug is not None) and (Y_top is not None):
            train_x = torch.cat([train_x, X_top_aug], dim=0)
            train_y = torch.cat([train_y, Y_top], dim=0)

        # Initial cost (don't charge for historical)
        total_cost = N_hi_init*cost_hi + N_lo_init*cost_lo
        # Best HI so far
        hi_mask = train_x[:, -1] > 0.5
        best_hi_so_far = float(train_y[hi_mask].max().item()) if torch.any(hi_mask) else -float("inf")

        results = []; it = 0
        steps_since_hi = 0

        promoted_bank = None  # stores all LO points promoted so far (in [0,1]^D), for dedup


        while total_cost < budget and it < max_iters:
            it += 1
            model = build_mf_model(train_x, train_y)

            force_hi = steps_since_hi >= HI_COOLDOWN_STEPS
            if force_hi:
                x_next, u_hi = optimize_ucb_at_fidelity_mf(model, bounds_x, fidelity=1.0, beta=beta)
                f_next = torch.tensor([[1.0]], dtype=dtype)
                step_cost, tagF = cost_hi, "HI"
                # Optional: compute LO UCB for logging
                _, u_lo = optimize_ucb_at_fidelity_mf(model, bounds_x, fidelity=0.0, beta=beta)
                adj_hi = u_hi / (cost_hi ** alpha)
                adj_lo = u_lo / (cost_lo ** alpha)
            else:
                x_next, f_next, step_cost, tagF, u_hi, u_lo, adj_hi, adj_lo = select_xf_by_enumeration(
                    model, bounds_x, beta, alpha, cost_lo, cost_hi
                )

            y_next = f_hi(x_next) if tagF=="HI" else f_lo(x_next)
            if tagF=="HI":
                best_hi_so_far = max(best_hi_so_far, float(y_next.item()))
                steps_since_hi = 0
            else:
                steps_since_hi += 1

            regret = np.nan if f_star is None else (f_star - best_hi_so_far)
            print(f"[MF][Seed {seed}][Iter {it:02d}] tag={tagF} | "
                  f"UCB_hi={u_hi:.3f} UCB_lo={u_lo:.3f} | adj_hi={adj_hi:.3f} adj_lo={adj_lo:.3f} | "
                  f"y={float(y_next.item()):+.4f} | cost={step_cost} | total={total_cost:.1f}")

            results.append(dict(seed=seed, iteration=it,
                                x=x_next.detach().cpu().numpy().tolist(),
                                fidelity=int(tagF=="HI"), y=float(y_next.item()),
                                cost=float(step_cost), total_cost=float(total_cost),
                                best_y_hi_so_far=float(best_hi_so_far),
                                regret=float(regret) if not np.isnan(regret) else np.nan,
                                method="MF-ENUM+COOLDOWN"))

            X_new = torch.cat([to_2d(x_next), to_2d(f_next)], dim=-1)
            train_x = torch.cat([train_x, X_new], dim=0)
            train_y = torch.cat([train_y, y_next.view(1,1)], dim=0)
            total_cost += step_cost

            # --- periodic LO->HI promotion ---
            if it % PROMOTE_EVERY_ITERS == 0:
                (train_x, train_y, total_cost, promoted_bank, n_prom) = _periodic_promote_lo_to_hi(
                    train_x=train_x, train_y=train_y, D=D, f_hi=f_hi,
                    k_promote=K_PROMOTE_INTERVAL, charge=CHARGE_PROMOTION,
                    cost_hi=cost_hi, total_cost=total_cost,
                    promoted_bank=promoted_bank, atol=DEDUP_ATOL
                )
                if n_prom > 0:
                    print(f"ðŸ” Periodic promotion: evaluated {n_prom} LO points at HI (budget now {total_cost:.1f}).")
                    # also refresh best HI if any were better
                    hi_mask = (train_x[:, -1] > 0.5)
                    if torch.any(hi_mask):
                        best_hi_so_far = float(train_y[hi_mask].max().item())
                        steps_since_hi = 0  # just had HI evals


        # Final report: best HI observed so far
        hi_mask = train_x[:, -1] > 0.5
        best_idx = torch.argmax(train_y[hi_mask].squeeze(-1))
        x_hi_all = train_x[hi_mask][:, :D]
        y_hi_all = train_y[hi_mask]
        x_star = x_hi_all[best_idx:best_idx+1]
        y_star = float(y_hi_all[best_idx].item())

        final_regret = np.nan if f_star is None else (f_star - y_star)
        run_time = time.time() - t0

        results.append(dict(seed=seed, iteration=it+1,
                            x=x_star.view(-1).detach().cpu().numpy().tolist(),
                            fidelity=1, y=y_star, cost=0.0, total_cost=float(total_cost),
                            best_y_hi_so_far=float(best_hi_so_far),
                            regret=float(final_regret) if not np.isnan(final_regret) else np.nan,
                            method="MF-ENUM+COOLDOWN",
                            final_eval=y_star, run_time=float(run_time)))

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(out_dir, f"mf_enum_seed{seed}.csv"), index=False)
        print(f"[MF] Seed {seed} saved.")

        # (collect across seeds if needed)
        # You can aggregate later from the CSVs in out_dir.

    print("[MF] Done ->", out_dir)

# ----------------------
# Single-fidelity runners (optional)
# ----------------------
def run_sf_ucb(which: str, bounds_x, f_hi, f_lo, f_star=None, tag="superpro"):
    assert which in ("HI","LO")
    name = f"SF-{which}"
    out_dir = os.path.join(out_root, f"{name}_{tag}_beta{beta}")
    os.makedirs(out_dir, exist_ok=True)

    for seed in range(num_seeds):
        set_seed(seed); t0 = time.time()

        if which=="HI":
            x0 = draw_sobol_samples(bounds=bounds_x, n=N_hi_init, q=1).squeeze(-2).to(dtype=dtype)
            y0 = f_hi(x0); per_cost = cost_hi
            best_hi_so_far = float(y0.max().item()) if y0.numel() else -float("inf")
        else:
            x0 = draw_sobol_samples(bounds=bounds_x, n=N_lo_init, q=1).squeeze(-2).to(dtype=dtype)
            y0 = f_lo(x0); per_cost = cost_lo
            best_hi_so_far = -float("inf")

        train_x = x0.to(dtype=dtype)
        train_y = y0.view(-1,1).to(dtype=dtype)
        total_cost = (N_hi_init if which=="HI" else N_lo_init) * per_cost
        results = []; it = 0

        while total_cost < budget and it < max_iters:
            it += 1
            model = build_sf_model(train_x, train_y)
            x_next, _ = optimize_ucb_sf(model, bounds_x, beta=beta)
            if which=="HI":
                y_next = f_hi(x_next); tagF="HI"; bit=1
                best_hi_so_far = max(best_hi_so_far, float(y_next.item()))
            else:
                y_next = f_lo(x_next); tagF="LO"; bit=0

            regret = np.nan if f_star is None else (f_star - best_hi_so_far)
            print(f"[{which}][Seed {seed}][Iter {it:02d}] tag={tagF} | "
                  f"y={float(y_next.item()):+.4f} | cost={per_cost:.1f} | total={total_cost:.1f}")

            results.append(dict(seed=seed, iteration=it,
                                x=x_next.detach().cpu().numpy().tolist(),
                                fidelity=bit, y=float(y_next.item()),
                                cost=float(per_cost), total_cost=float(total_cost),
                                best_y_hi_so_far=float(best_hi_so_far),
                                regret=float(regret) if not np.isnan(regret) else np.nan,
                                method=name))
            train_x = torch.cat([train_x, to_2d(x_next)], dim=0)
            train_y = torch.cat([train_y, y_next.view(1,1)], dim=0)
            total_cost += per_cost

        # Final selection
        if which == "HI":
            best_idx = torch.argmax(train_y.squeeze(-1))
            x_star = train_x[best_idx:best_idx+1, :D]
            y_star_hi = float(train_y[best_idx].item())
        else:
            best_idx = torch.argmax(train_y.squeeze(-1))
            x_star = train_x[best_idx:best_idx+1, :D]
            y_star_hi = float(f_hi(x_star).item())
            best_hi_so_far = max(best_hi_so_far, y_star_hi)

        final_regret = np.nan if f_star is None else (f_star - y_star_hi)
        run_time = time.time() - t0
        results.append(dict(seed=seed, iteration=it+1,
                            x=x_star.view(-1).detach().cpu().numpy().tolist(),
                            fidelity=1 if which=="HI" else 0, y=y_star_hi,
                            cost=0.0, total_cost=float(total_cost),
                            best_y_hi_so_far=float(best_hi_so_far),
                            regret=float(final_regret) if not np.isnan(final_regret) else np.nan,
                            method=name, final_eval=y_star_hi, run_time=float(run_time)))
        pd.DataFrame(results).to_csv(os.path.join(out_dir, f"{name}_seed{seed}.csv"), index=False)
        print(f"[{name}] Seed {seed} saved.")

    print(f"[{name}] Done -> {out_dir}")


def sanity_check_once(x01: torch.Tensor):
    with torch.no_grad():
        y_lo = float(flowrate_ann_lo(x01.unsqueeze(0)).item())
        y_hi = float(flowrate_sim_hi(x01.unsqueeze(0)).item())
    print(f"[sanity] same x -> LO={y_lo:.6f}, HI={y_hi:.6f} (original units)")


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":

    # quick check using a random point (or convert a row from top_inputs to [0,1] first)
    x01 = torch.rand(D, dtype=dtype)
    sanity_check_once(x01)

    # BO bounds in [0,1]^D from SPACE only (we never search outside true sim domain)
    bounds_x = torch.stack([torch.zeros(D, dtype=dtype), torch.ones(D, dtype=dtype)])

    print(">>> Running MF-BO (enumeration + HI cooldown) ...")
    run_mf_enum(bounds_x, f_hi, f_lo, f_star=None, tag="superpro")

    # Optional: run SF baselines too
    # print(">>> Running SF-HI UCB ..."); run_sf_ucb("HI", bounds_x, f_hi, f_lo, tag="superpro")
    # print(">>> Running SF-LO UCB ..."); run_sf_ucb("LO", bounds_x, f_hi, f_lo, tag="superpro")

    print("Done.")




