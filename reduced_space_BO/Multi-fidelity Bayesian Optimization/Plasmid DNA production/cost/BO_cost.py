# ============================================
# Multi-fidelity BO for SuperPro (HI) + ANN (LO)
# With enumeration + HI cooldown, and warm-start from top_*.csv
# ============================================

import os, time, random
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

# Excel COM for SuperPro
import win32com.client
import pywintypes


# ----------------------
# CONFIG
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cpu")
dtype  = torch.double

# Files
EXCEL_MACRO_PATH = os.path.join(BASE_DIR, "pDNA.xlsm")
ANN_PKL_PATH     = os.path.join(BASE_DIR, "ann_ACC_0.18_11940_0.0001_512_1.pkl")
INPUTS_CSV       = os.path.join(BASE_DIR, "inputs.csv")
OUTPUTS_CSV      = os.path.join(BASE_DIR, "outputs.csv")
TOP_INPUTS_CSV   = os.path.join(BASE_DIR, "top_inputs.csv")     # optional
TOP_OUTPUTS_CSV  = os.path.join(BASE_DIR, "top_outputs.csv")    # optional
METRIC           = "cost"

# Costs / BO knobs
cost_hi, cost_lo = 5.0, 1.0
budget           = 150.0
beta             = 15.0
alpha            = 0.1
HI_BIAS = 1  # try 1.05â€“1.25

# --- Periodic LOâ†’HI promotion knobs ---
PROMOTE_EVERY_ITERS = 10   # promote every N iterations
K_PROMOTE_INTERVAL  = 1    # promote top-K LO points each time
CHARGE_PROMOTION    = True # count these HI evals against budget
DEDUP_ATOL          = 1e-6 # treat xâ€™s equal if all dims within this abs tol



# Seeds / iters
num_seeds  = 10
max_iters  = 1000
N_hi_init, N_lo_init = 2, 2

# HI cooldown (force an HI if too many LO in a row)
HI_COOLDOWN_STEPS = 10

# Early stop on repeated objective
STOP_K_SAME = 20  # stop if y unchanged for K iterations (<=0 disables)
STOP_Y_TOL = 1e-6

# Output root
out_root = os.path.join(BASE_DIR, "results_superpro")
os.makedirs(out_root, exist_ok=True)

# ----------------------
# Utilities
# ----------------------
def set_seed(seed: int):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)

def to_2d(x: Tensor):
    return x.view(-1, x.shape[-1])

# ----------------------
# SPACE (7D), ANN norm space (Â±10%)
# ----------------------
SPACE = [
    #{'name': 'flask_time', 'type': 'continuous', 'domain': (21600, 108000)},
    {'name': 'seed_time', 'type': 'continuous', 'domain': (43200, 108000)},
    #{'name': 'fed_batch_seed', 'type': 'continuous', 'domain': (0.001636, 0.004908)},
    {'name': 'batch_seed', 'type': 'continuous', 'domain': (0.010225325, 0.030675975)},
    #{'name': 'conversion_seed', 'type': 'continuous', 'domain': (0.9, 0.98)},
    {'name': 'main_time', 'type': 'continuous', 'domain': (64800, 144000)},
    {'name': 'fed_batch', 'type': 'continuous', 'domain': (0.0190875, 0.0572625)},
    {'name': 'batch_med', 'type': 'continuous', 'domain': (0.102253255, 0.306759765)},
    {'name': 'conversion_main', 'type': 'continuous', 'domain': (0.9, 0.98)},
    #{'name': 'solid_conc', 'type': 'continuous', 'domain': (50, 300)},
    {'name': 'res_vol', 'type': 'continuous', 'domain': (0.28903585,  0.86710755)},
    #{'name': 'equi1', 'type': 'continuous', 'domain': (0.01, 0.05)},
    {'name': 'dia1', 'type': 'discrete', 'domain': (10, 50)},
    #{'name': 'flush1', 'type': 'continuous', 'domain': (0.0015,  0.0045)},
    #{'name': 'equi2', 'type': 'continuous', 'domain': (0.01, 0.05)},
    {'name': 'dia2', 'type': 'discrete', 'domain': (10, 50)},
    {'name': 'flush2', 'type': 'continuous', 'domain': (0.00067,  0.002)},
    #{'name': 'failure', 'type': 'continuous', 'domain': (0,  0.1)},
]

D = len(SPACE)
SPACE_COLS = [v['name'] for v in SPACE]

def expand_bounds(bounds, lo_factor=0.9, hi_factor=1.1):
    lo, hi = bounds
    return (lo * lo_factor, hi * hi_factor)

ANN_NORM_SPACE = []
for v in SPACE:
    lo, hi = float(v['domain'][0]), float(v['domain'][1])
    lo2, hi2 = expand_bounds((lo, hi), 0.9, 1.1)
    ANN_NORM_SPACE.append({'name': v['name'], 'type': v['type'], 'domain': (lo2, hi2)})

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
# ANN model (7 inputs), scaler from outputs.csv
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

    y_scaler = _fit_target_scaler_from_outputs()
    new_state = _standard_scaler_state(y_scaler)
    stored_state = P.get(_TARGET_SCALER_KEY)
    if stored_state is None or not _scaler_states_close(stored_state, new_state):
        P[_TARGET_SCALER_KEY] = new_state
        save_pkl(ANN_PKL_PATH, P)

    n_input = D
    n_output = 1
    n_hidden = structure[2]; num_layers = structure[3]
    ann = ReLUNet(n_input, n_hidden, n_output, num_layers).to(dtype=torch.float64)
    ann.load_state_dict(SD); ann.eval()
    return ann, y_scaler

ANN_MODEL, Y_SCALER = load_ann_and_scaler()
# ----------------------
# HI: SuperPro (Excel macro)
# ----------------------


def _open_excel_workbook(max_retries: int = 3, retry_delay: float = 1.0):
    """Launch a fresh Excel instance and open the SuperPro workbook with retries."""
    last_exc = None
    for _ in range(max_retries):
        excel = None
        try:
            excel = win32com.client.DispatchEx("Excel.Application")
            excel.Visible = False
            wb = excel.Workbooks.Open(EXCEL_MACRO_PATH)
            return excel, wb
        except pywintypes.com_error as exc:
            last_exc = exc
            if excel is not None:
                try:
                    excel.Application.Quit()
                except Exception:
                    pass
            time.sleep(retry_delay)
    raise RuntimeError(
        "Failed to launch Excel for SuperPro simulation. "
        "Check that pDNA.xlsm is available and macros are enabled."
    ) from last_exc



def superpro_sim_hi(X01: torch.Tensor, return_x: bool = False):
    if X01.dim() == 1:
        X01 = X01.unsqueeze(0)
    X_real = unnormalize_x(X01, SPACE).cpu().numpy()

    excel = wb = None
    try:
        excel, wb = _open_excel_workbook()
        excel.Application.Run('BeforeSuperExcelMatlab')

        out = np.zeros((X_real.shape[0],), dtype=np.float64)
        for i in range(X_real.shape[0]):
            p = X_real[i]
            temp4 = excel.Application.Run('SuperExcelMatlab', 21600, p[0], 0.0033, 
                                            p[1], 0.95, p[2], p[3], p[4], 
                                            p[5], 150, p[6], 0.03, np.round(p[7]), 
                                            0.003, 0.02, np.round(p[8]), p[9], 0.05)

            # Choose ONE of these; comment the others:
            #out[i] = float(temp4[1])               # productivity (kg/batch)
            # out[i] = float(temp4[2])  # CAPEX ($)
            # out[i] = float(temp4[3]) / float(temp4[4])   # OPEX per batch
            out[i] = float(temp4[3]) / (float(temp4[4]) * (float(temp4[1]) / 0.001))  # cost per gram ($/g)
            # out[i] = float(temp4[5])  # batch time (s)
            # out[i] = float(temp4[6])  # cycle time (s)
        try:
            excel.Application.Run('AfterSuperExcelMatlab')
        except pywintypes.com_error:
            pass
    finally:
        if wb is not None:
            try:
                wb.Close(False)
            except pywintypes.com_error:
                pass
        if excel is not None:
            try:
                excel.Application.Quit()
            except pywintypes.com_error:
                pass
    y_tensor = torch.from_numpy(out).unsqueeze(-1).to(dtype=dtype)
    x_tensor = torch.from_numpy(X_real).to(dtype=dtype)
    return (y_tensor, x_tensor) if return_x else y_tensor




# ----------------------
# LO: ANN (inputs normalized with Â±10% training bounds)
# ----------------------
@torch.no_grad()
def superpro_ann_lo(X01: torch.Tensor) -> torch.Tensor:
    # X01: (n, D) in [0,1]^D
    if X01.dim() == 1:
        X01 = X01.unsqueeze(0)

    # unnormalize to real ranges (simulation bounds)
    real = unnormalize_x(X01, SPACE).cpu().numpy()              # (n, D) original ranges

    # normalize inputs for the ANN using the Â±10% bounds it was trained on
    x_norm = mm_norm_np_clipped(real, ANN_NORM_SPACE)           # (n, D) in ANNâ€™s [0,1] box
    xt = torch.from_numpy(x_norm).to(dtype=torch.float64)

    # ANN predicts *standardized* y (because it was trained that way)
    y_std = ANN_MODEL(xt).cpu().numpy().reshape(-1, 1)

    # CRUCIAL: convert back to original units with the same scaler you fit on outputs.csv
    y_orig = Y_SCALER.inverse_transform(y_std)                  # (n, 1) original units

    return torch.from_numpy(y_orig).to(dtype=torch.double)      # feed GP in original units


f_hi, f_lo = superpro_sim_hi, superpro_ann_lo

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
    acq = UpperConfidenceBound(model, beta=beta, maximize=False)
    x, val = optimize_acqf(acq, bounds=bounds_x, q=1,
                           num_restarts=num_restarts, raw_samples=raw_samples)
    return x.squeeze(0), float(val.item())

def optimize_ucb_at_fidelity_mf(model, bounds_x, fidelity, beta, num_restarts=15, raw_samples=128):
    acq = UpperConfidenceBound(model, beta=beta, maximize=False)
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
    adj_lo = u_lo * (cost_lo ** alpha)
    adj_hi = (u_hi * HI_BIAS) * (cost_hi ** alpha)
    if adj_hi >= adj_lo:  # ties â†’ HI
        return x_hi, torch.tensor([[1.0]], dtype=dtype), cost_hi, "HI", u_hi, u_lo, adj_hi, adj_lo
    else:
        return x_lo, torch.tensor([[0.0]], dtype=dtype), cost_lo, "LO", u_hi, u_lo, adj_hi, adj_lo

# ----------------------
# Warm-start from prior HI (top_*.csv)
# ----------------------
def load_top_hi_data():
    if (not os.path.exists(TOP_INPUTS_CSV)) or (not os.path.exists(TOP_OUTPUTS_CSV)):
        print("â„¹ï¸ No top_inputs/top_outputs found. Starting without prior HI data.")
        return None, None
    X_df = pd.read_csv(TOP_INPUTS_CSV)
    # Expect exactly the 7 SPACE columns (either named as SPACE or same order)
    if X_df.shape[1] != D:
        raise ValueError(f"{TOP_INPUTS_CSV} must have exactly {D} columns.")
    # If named, reorder to SPACE; if unnamed, assume correct order
    if set(X_df.columns) == set(SPACE_COLS):
        X_df = X_df.loc[:, SPACE_COLS]
    X_real = X_df.to_numpy(dtype=np.float64)

    Y_df = pd.read_csv(TOP_OUTPUTS_CSV)
    if METRIC in Y_df.columns: Y_df = Y_df[[METRIC]]
    elif Y_df.shape[1]==1: Y_df.columns=[METRIC]
    else: raise ValueError(f"{TOP_OUTPUTS_CSV} must have '{METRIC}' or a single column.")
    Y = Y_df.to_numpy(dtype=np.float64).reshape(-1,1)

    X01 = real_to_unit01_np(X_real, SPACE)
    X01 = torch.from_numpy(X01).to(dtype=dtype)
    Yt  = torch.from_numpy(Y).to(dtype=dtype)
    F1  = torch.ones(X01.shape[0], 1, dtype=dtype)
    X_aug = torch.cat([X01, F1], dim=-1)
    print(f"âœ… Loaded {X_aug.shape[0]} prior HI rows from top_inputs/top_outputs.")
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
    top_idx = torch.topk(lo_vals, k=k, largest = False).indices
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

    for seed in range(num_seeds):
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

        # Initial cost (donâ€™t charge for historical)
        total_cost = N_hi_init*cost_hi + N_lo_init*cost_lo
        # Best HI so far
        hi_mask = train_x[:, -1] > 0.5
        best_hi_so_far = float(train_y[hi_mask].min().item()) if torch.any(hi_mask) else +float("inf")

        results = []; it = 0
        steps_since_hi = 0

        promoted_bank = None  # stores all LO points promoted so far (in [0,1]^D), for dedup
        recent_y = []
        stop_reason = None

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
                adj_hi = u_hi * (cost_hi ** alpha)
                adj_lo = u_lo * (cost_lo ** alpha)
            else:
                x_next, f_next, step_cost, tagF, u_hi, u_lo, adj_hi, adj_lo = select_xf_by_enumeration(
                    model, bounds_x, beta, alpha, cost_lo, cost_hi
                )

            if tagF == "HI":
                y_next, x_real_tensor = superpro_sim_hi(x_next, return_x=True)
                x_real_logged = x_real_tensor.view(-1).cpu().numpy().tolist()
                y_val = float(y_next.item())
                best_hi_so_far = min(best_hi_so_far, y_val)
                steps_since_hi = 0
            else:
                y_next = f_lo(x_next)
                x_real_logged = unnormalize_x(to_2d(x_next), SPACE).view(-1).cpu().numpy().tolist()
                y_val = float(y_next.item())
                steps_since_hi += 1

            regret = np.nan if f_star is None else (best_hi_so_far - f_star)
            print(f"[MF][Seed {seed}][Iter {it:02d}] tag={tagF} | "
                  f"UCB_hi={u_hi:.3f} UCB_lo={u_lo:.3f} | adj_hi={adj_hi:.3f} adj_lo={adj_lo:.3f} | "
                  f"y={y_val:+.4f} | cost={step_cost} | total={total_cost:.1f}")

            results.append(dict(seed=seed, iteration=it,
                                x=x_next.detach().cpu().numpy().tolist(),
                                x_real=x_real_logged,
                                fidelity=int(tagF=="HI"), y=y_val,
                                cost=float(step_cost), total_cost=float(total_cost),
                                best_y_hi_so_far=float(best_hi_so_far),
                                regret=float(regret) if not np.isnan(regret) else np.nan,
                                method="MF-ENUM+COOLDOWN"))

            X_new = torch.cat([to_2d(x_next), to_2d(f_next)], dim=-1)
            train_x = torch.cat([train_x, X_new], dim=0)
            train_y = torch.cat([train_y, y_next.view(1,1)], dim=0)
            total_cost += step_cost

            if STOP_K_SAME and STOP_K_SAME > 0:
                recent_y.append(y_val)
                if len(recent_y) > STOP_K_SAME:
                    recent_y.pop(0)
                if len(recent_y) == STOP_K_SAME:
                    delta = max(recent_y) - min(recent_y)
                    if delta <= STOP_Y_TOL:
                        print(f"ðŸ›‘ Stable y detected for {STOP_K_SAME} consecutive iterations (Î”={delta:.3e}). Early stopping.")
                        stop_reason = f"stable_y_{STOP_K_SAME}"
                        results[-1]["stop_reason"] = stop_reason
                        break


            # --- periodic LOâ†’HI promotion ---
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
                        best_hi_so_far = float(train_y[hi_mask].min().item())
                        steps_since_hi = 0  # just had HI evals


        # Final report: best HI observed so far
        hi_mask = train_x[:, -1] > 0.5
        best_idx = torch.argmin(train_y[hi_mask].squeeze(-1))
        x_hi_all = train_x[hi_mask][:, :D]
        y_hi_all = train_y[hi_mask]
        x_star = x_hi_all[best_idx:best_idx+1]
        y_star = float(y_hi_all[best_idx].item())

        final_regret = np.nan if f_star is None else (y_star - f_star)
        run_time = time.time() - t0

        results.append(dict(seed=seed, iteration=it+1,
                            x=x_star.view(-1).detach().cpu().numpy().tolist(),
                            x_real=unnormalize_x(x_star, SPACE).view(-1).cpu().numpy().tolist(),
                            fidelity=1, y=y_star, cost=0.0, total_cost=float(total_cost),
                            best_y_hi_so_far=float(best_hi_so_far),
                            regret=float(final_regret) if not np.isnan(final_regret) else np.nan,
                            method="MF-ENUM+COOLDOWN",
                            final_eval=y_star, run_time=float(run_time),
                            stop_reason=stop_reason))

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(out_dir, f"mf_enum_seed{seed}.csv"), index=False)
        print(f"âœ… [MF] Seed {seed} saved.")

        # (collect across seeds if needed)
        # You can aggregate later from the CSVs in out_dir.

    print("âœ… [MF] Done â†’", out_dir)

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
            best_hi_so_far = float(y0.min().item()) if y0.numel() else +float("inf")
        else:
            x0 = draw_sobol_samples(bounds=bounds_x, n=N_lo_init, q=1).squeeze(-2).to(dtype=dtype)
            y0 = f_lo(x0); per_cost = cost_lo
            best_hi_so_far = +float("inf")

        train_x = x0.to(dtype=dtype)
        train_y = y0.view(-1,1).to(dtype=dtype)
        total_cost = (N_hi_init if which=="HI" else N_lo_init) * per_cost
        results = []; it = 0
        recent_y = []
        stop_reason = None

        while total_cost < budget and it < max_iters:
            it += 1
            model = build_sf_model(train_x, train_y)
            x_next, _ = optimize_ucb_sf(model, bounds_x, beta=beta)
            if which=="HI":
                y_next, x_real_tensor = superpro_sim_hi(x_next, return_x=True); tagF="HI"; bit=1
                x_real_logged = x_real_tensor.view(-1).cpu().numpy().tolist()
                y_val = float(y_next.item())
                best_hi_so_far = min(best_hi_so_far, y_val)
            else:
                y_next = f_lo(x_next); tagF="LO"; bit=0
                x_real_logged = unnormalize_x(to_2d(x_next), SPACE).view(-1).cpu().numpy().tolist()
                y_val = float(y_next.item())

            regret = np.nan if f_star is None else (best_hi_so_far - f_star)
            print(f"[{which}][Seed {seed}][Iter {it:02d}] tag={tagF} | "
                  f"y={y_val:+.4f} | cost={per_cost:.1f} | total={total_cost:.1f}")

            results.append(dict(seed=seed, iteration=it,
                                x=x_next.detach().cpu().numpy().tolist(),
                                x_real=x_real_logged,
                                fidelity=bit, y=y_val,
                                cost=float(per_cost), total_cost=float(total_cost),
                                best_y_hi_so_far=float(best_hi_so_far),
                                regret=float(regret) if not np.isnan(regret) else np.nan,
                                method=name))
            train_x = torch.cat([train_x, to_2d(x_next)], dim=0)
            train_y = torch.cat([train_y, y_next.view(1,1)], dim=0)
            total_cost += per_cost

            if STOP_K_SAME and STOP_K_SAME > 0:
                recent_y.append(y_val)
                if len(recent_y) > STOP_K_SAME:
                    recent_y.pop(0)
                if len(recent_y) == STOP_K_SAME:
                    delta = max(recent_y) - min(recent_y)
                    if delta <= STOP_Y_TOL:
                        print(f"ðŸ›‘ Stable y detected for {STOP_K_SAME} consecutive iterations (Î”={delta:.3e}). Early stopping.")
                        stop_reason = f"stable_y_{STOP_K_SAME}"
                        results[-1]["stop_reason"] = stop_reason
                        break

        # Final selection
        if which == "HI":
            best_idx = torch.argmin(train_y.squeeze(-1))
            x_star = train_x[best_idx:best_idx+1, :D]
            y_star_hi = float(train_y[best_idx].item())
        else:
            best_idx = torch.argmin(train_y.squeeze(-1))
            x_star = train_x[best_idx:best_idx+1, :D]
            y_star_hi = float(f_hi(x_star).item())
            best_hi_so_far = min(best_hi_so_far, y_star_hi)

        final_regret = np.nan if f_star is None else (best_hi_so_far - f_star) 
        run_time = time.time() - t0
        results.append(dict(seed=seed, iteration=it+1,
                            x=x_star.view(-1).detach().cpu().numpy().tolist(),
                            x_real=unnormalize_x(x_star, SPACE).view(-1).cpu().numpy().tolist(),
                            fidelity=1 if which=="HI" else 0, y=y_star_hi,
                            cost=0.0, total_cost=float(total_cost),
                            best_y_hi_so_far=float(best_hi_so_far),
                            regret=float(final_regret) if not np.isnan(final_regret) else np.nan,
                            method=name, final_eval=y_star_hi, run_time=float(run_time),
                            stop_reason=stop_reason))
        pd.DataFrame(results).to_csv(os.path.join(out_dir, f"{name}_seed{seed}.csv"), index=False)
        print(f"âœ… [{name}] Seed {seed} saved.")

    print(f"âœ… [{name}] Done â†’", out_dir)


def sanity_check_once(x01: torch.Tensor):
    with torch.no_grad():
        y_lo = float(superpro_ann_lo(x01.unsqueeze(0)).item())
        y_hi = float(superpro_sim_hi(x01.unsqueeze(0)).item())
    print(f"[sanity] same x â†’ LO={y_lo:.6f}, HI={y_hi:.6f} (original units)")


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

    print("âœ… Done.")



