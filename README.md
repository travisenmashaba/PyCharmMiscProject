# PyCharmMiscProject
"""
Differential Privacy Analysis: South African Demographic Dataset
Two-Dataset Strategy: Real (N=52) + Synthetic Scaled (N=1,000,000)
Complete Fixed Implementation with All Epsilon Values Displayed
"""

# ============================================================================
# REPRODUCIBILITY — Set all random seeds
# ============================================================================

import numpy as np
import random
import torch
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import math
import time
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, shapiro
from typing import List, Dict, Tuple, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = "C:/Users/travi/Downloads/dhs-quickstats_national_zaf.csv"

EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
NUM_TRIALS = 50
DELTA = 1e-5
QUERY_TYPES = ['count', 'mean', 'median', 'sum']
CLIP_PERCENTILES = (1, 99)

SYNTHETIC_TARGET_SIZE = 1_000_000
SYNTHETIC_NOISE_SIGMA = 0.05
TOST_MARGIN = 0.10

OUTPUT_DIR = 'C:/Users/travi/Downloads'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("DIFFERENTIAL PRIVACY ANALYSIS — SOUTH AFRICAN DATASET")
print("Two-Dataset Strategy: Real (N=52) + Synthetic Scaled (N=1,000,000)")
print("=" * 80)
print(f"Dataset path:             {DATASET_PATH}")
print(f"Privacy budgets (ε):      {EPSILON_VALUES}")
print(f"Number of trials per ε:   {NUM_TRIALS}")
print(f"Total per (query, ε):     {NUM_TRIALS} × {len(EPSILON_VALUES)} = "
      f"{NUM_TRIALS * len(EPSILON_VALUES)} runs")
print(f"Synthetic target size:    {SYNTHETIC_TARGET_SIZE:,}")
print(f"TOST equivalence margin:  {TOST_MARGIN*100:.0f}% MRE")
print("=" * 80)
print()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(file_path: str) -> pd.DataFrame:
    print("\n📂 LOADING DATASET")
    print("-" * 60)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    print(f"✓ Dataset loaded: {len(df):,} records × {len(df.columns)} columns")

    print("\n🔄 Preprocessing...")
    original_len = len(df)
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip().str.lower()

    # HXL tag removal
    hxl_mask = None
    for col in df.columns:
        if df[col].dtype == object:
            mask = df[col].astype(str).str.startswith('#')
            hxl_mask = mask if hxl_mask is None else (hxl_mask | mask)
    if hxl_mask is not None and hxl_mask.any():
        print(f"  ⚠️ Removed {int(hxl_mask.sum())} HXL tag row(s)")
        df = df[~hxl_mask]

    df.dropna(how="all", inplace=True)
    df = df.fillna(0)

    print(f"  Records after cleaning: {len(df):,} (removed {original_len - len(df):,})")
    print("-" * 60)
    return df


def identify_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\n📊 COLUMN IDENTIFICATION")
    print("-" * 60)
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
    print("-" * 60)
    return numeric_cols, cat_cols


def compute_clipping_bounds(df: pd.DataFrame,
                            numeric_cols: List[str]) -> Dict[str, Tuple[float, float]]:
    clip_bounds = {}
    for col in numeric_cols:
        col_series = df[col].dropna().astype(float)
        if not col_series.empty:
            low, high = np.percentile(col_series, CLIP_PERCENTILES)
            if low == high:
                low = float(col_series.min())
                high = float(col_series.max())
            clip_bounds[col] = (float(low), float(high))

    print(f"\n📐 CLIPPING BOUNDS COMPUTED")
    print("-" * 60)
    for col, (low, high) in list(clip_bounds.items())[:6]:
        print(f"  {col}: [{low:.2f}, {high:.2f}]")
    if len(clip_bounds) > 6:
        print(f"  ... and {len(clip_bounds) - 6} more")
    print("-" * 60)
    return clip_bounds


def apply_clipping(df: pd.DataFrame,
                   clip_bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    df_clipped = df.copy()
    for col, (low, high) in clip_bounds.items():
        if col in df_clipped.columns:
            df_clipped[col] = df_clipped[col].astype(float).clip(low, high)
    return df_clipped


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_sanitized = df.copy()
    for col in df_sanitized.select_dtypes(include=[np.number]).columns:
        df_sanitized[col] = pd.to_numeric(df_sanitized[col], errors='coerce')
        df_sanitized[col] = df_sanitized[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df_sanitized


def prepare_data_vectors(df: pd.DataFrame,
                        numeric_cols: List[str]) -> Dict[str, List[float]]:
    METADATA_LIKE = {
        'indicatororder', 'characteristicid', 'characteristicorder',
        'regionid', 'levelrank', 'surveyyearlabel', 'surveyyear'
    }
    substantive_cols = [c for c in numeric_cols if c not in METADATA_LIKE]
    preferred_order = ['denominatorweighted', 'denominatorunweighted',
                       'value', 'cihigh', 'cilow']

    primary_col = None
    for pref in preferred_order:
        if pref in substantive_cols:
            primary_col = pref
            break
    if primary_col is None and substantive_cols:
        primary_col = substantive_cols[0]

    if primary_col is None:
        rng = np.random.default_rng(seed=42)
        data = rng.normal(100, 25, 1000).tolist()
        primary_col = "synthetic"
    else:
        data = df[primary_col].astype(float).tolist()
        print(f"\n📈 DATA VECTORS PREPARED")
        print("-" * 60)
        print(f"Using column: '{primary_col}' (substantive numeric variable)")
        print(f"Data points: {len(data):,}")
        print(f"Min: {min(data):.2f}, Max: {max(data):.2f}, Mean: {np.mean(data):.2f}")
        print("-" * 60)

    return {
        'count': data,
        'mean': data,
        'median': sorted(data),
        'sum': data
    }


def generate_synthetic_dataset(real_data: List[float],
                               target_size: int = 1_000_000,
                               noise_sigma: float = 0.05,
                               seed: int = 42) -> List[float]:
    rng = np.random.default_rng(seed=seed)
    base = np.array(real_data, dtype=float)
    base_std = float(np.std(base))
    resampled = rng.choice(base, size=target_size, replace=True)
    noise = rng.normal(0.0, noise_sigma * base_std, size=target_size)
    return (resampled + noise).tolist()


# ============================================================================
# DP MECHANISMS
# ============================================================================

def laplace_noise(scale: float, size: int = 1) -> np.ndarray:
    return np.random.laplace(loc=0.0, scale=scale, size=size)


def gaussian_noise(sigma: float, size: int = 1) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=sigma, size=size)


def gaussian_sigma_for_eps_delta(sensitivity: float, eps: float,
                                 delta: float) -> float:
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / eps


def apply_laplace_count(data: List[float], epsilon: float) -> float:
    return len(data) + laplace_noise(1.0 / epsilon, 1)[0]


def apply_laplace_mean(data: List[float], epsilon: float) -> float:
    if not data:
        return 0.0
    sensitivity = ((max(data) - min(data)) / len(data)
                   if max(data) != min(data) else 1.0)
    return np.mean(data) + laplace_noise(sensitivity / epsilon, 1)[0]


def apply_laplace_sum(data: List[float], epsilon: float) -> float:
    if not data:
        return 0.0
    sensitivity = (max(data) - min(data) if max(data) != min(data) else 1.0)
    return np.sum(data) + laplace_noise(sensitivity / epsilon, 1)[0]


def apply_gaussian_sum(data: List[float], epsilon: float,
                       delta: float = 1e-5) -> float:
    if not data:
        return 0.0
    sensitivity = (max(data) - min(data) if max(data) != min(data) else 1.0)
    sigma = gaussian_sigma_for_eps_delta(sensitivity, epsilon, delta)
    return np.sum(data) + gaussian_noise(sigma, 1)[0]


def apply_laplace_median(data: List[float], epsilon: float) -> float:
    if not data:
        return 0.0
    n = len(data)
    sensitivity = 1.0 if n % 2 == 1 else 0.5
    return np.median(data) + laplace_noise(sensitivity / epsilon, 1)[0]


# ============================================================================
# DP OPERATIONS ON DATAFRAMES
# ============================================================================

def dp_counts_custom(df: pd.DataFrame, cat_col: str,
                     epsilon: float = 1.0) -> pd.DataFrame:
    true_counts = df[cat_col].value_counts().sort_index()
    scale = 1.0 / epsilon
    noisy_counts = true_counts + laplace_noise(scale, true_counts.shape[0])
    return pd.DataFrame({
        'category': true_counts.index,
        'true_count': true_counts.values,
        'noisy_count': noisy_counts
    })


def dp_sums_means_custom(df: pd.DataFrame,
                         numeric_cols: List[str],
                         clip_bounds: Dict[str, Tuple[float, float]],
                         epsilon: float = 1.0,
                         delta: float = 1e-5) -> pd.DataFrame:
    n = len(df)
    rows = []
    for col in numeric_cols:
        col_values = df[col].fillna(0.0).astype(float)
        a, b = clip_bounds.get(col, (0.0, 1.0))
        sensitivity_sum = b - a
        sigma = gaussian_sigma_for_eps_delta(sensitivity_sum, epsilon, delta)
        true_sum = col_values.sum()
        noisy_sum = true_sum + gaussian_noise(sigma, 1)[0]
        noisy_mean = noisy_sum / n if n > 0 else np.nan
        rows.append({
            'column': col,
            'true_sum': float(true_sum),
            'noisy_sum': float(noisy_sum),
            'true_mean': float(col_values.mean()),
            'noisy_mean': float(noisy_mean),
            'clip_low': a, 'clip_high': b,
            'sensitivity_sum': sensitivity_sum,
            'epsilon': epsilon, 'delta': delta,
            'gaussian_sigma': sigma
        })
    return sanitize_dataframe(pd.DataFrame(rows))


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(true_value: float,
                    noisy_values: List[float]) -> Dict[str, float]:
    errors = [abs(true_value - nv) for nv in noisy_values]
    mae = np.mean(errors)
    mre = (np.mean([abs(true_value - nv) / abs(true_value)
                    for nv in noisy_values])
           if true_value != 0 else float('nan'))
    return {'MAE': mae, 'MRE': mre, 'std': np.std(noisy_values)}


def compute_true_value(data: List[float], query_type: str) -> float:
    if query_type == 'count':
        return float(len(data))
    if query_type == 'mean':
        return float(np.mean(data))
    if query_type == 'median':
        return float(np.median(data))
    if query_type == 'sum':
        return float(np.sum(data))
    raise ValueError(f"Unknown query type: {query_type}")


# ============================================================================
# FORMAL STATISTICAL HYPOTHESIS TESTING
# ============================================================================

def formal_hypothesis_testing(true_value: float,
                              noisy_values: List[float],
                              threshold_mre: float = 0.10,
                              n_bootstrap: int = 10000,
                              seed: int = 42) -> Dict:
    noisy_values = np.asarray(noisy_values, dtype=float)
    n = noisy_values.size
    differences = noisy_values - true_value

    results = {
        'true_value': float(true_value),
        'n_trials': int(n),
        'mean_noisy': float(np.mean(noisy_values)),
        'std_noisy': float(np.std(noisy_values, ddof=1)) if n > 1 else 0.0,
        'mean_difference': float(np.mean(differences)),
    }

    abs_errors = np.abs(differences)
    results['MAE'] = float(np.mean(abs_errors))
    results['MRE'] = (float(np.mean(abs_errors / np.abs(true_value)))
                      if true_value != 0 else float('nan'))

    # Shapiro-Wilk
    if n >= 3:
        shapiro_stat, shapiro_p = shapiro(differences)
        results['shapiro_W'] = float(shapiro_stat)
        results['shapiro_p'] = float(shapiro_p)
        results['normality_rejected'] = bool(shapiro_p < 0.05)
    else:
        results['normality_rejected'] = True

    # Wilcoxon
    try:
        if np.allclose(differences, 0):
            results['wilcoxon_W'] = 0.0
            results['wilcoxon_p'] = 1.0
            results['wilcoxon_reject_H0'] = False
        else:
            w_stat, w_p = wilcoxon(differences, alternative='two-sided')
            results['wilcoxon_W'] = float(w_stat)
            results['wilcoxon_p'] = float(w_p)
            results['wilcoxon_reject_H0'] = bool(w_p < 0.05)
    except Exception as e:
        results['wilcoxon_error'] = str(e)

    # Paired t-test if normal
    if not results.get('normality_rejected', True):
        try:
            t_stat, t_p = ttest_rel(noisy_values, np.full(n, true_value))
            results['t_stat'] = float(t_stat)
            results['t_p'] = float(t_p)
            results['t_reject_H0'] = bool(t_p < 0.05)
        except Exception as e:
            results['t_error'] = str(e)

    # Bootstrap CI
    rng = np.random.default_rng(seed=seed)
    boot_mae = np.empty(n_bootstrap)
    boot_mre = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(abs_errors, size=n, replace=True)
        boot_mae[i] = np.mean(resample)
        if true_value != 0:
            boot_mre[i] = np.mean(resample / np.abs(true_value))

    results['MAE_CI_lower'] = float(np.percentile(boot_mae, 2.5))
    results['MAE_CI_upper'] = float(np.percentile(boot_mae, 97.5))
    if true_value != 0:
        results['MRE_CI_lower'] = float(np.percentile(boot_mre, 2.5))
        results['MRE_CI_upper'] = float(np.percentile(boot_mre, 97.5))

    # Cliff's delta
    n_greater = int(np.sum(differences > 0))
    n_less = int(np.sum(differences < 0))
    cliff_delta = (n_greater - n_less) / n
    results['cliff_delta'] = float(cliff_delta)

    abs_delta = abs(cliff_delta)
    if abs_delta < 0.147:
        results['effect_size'] = 'negligible'
    elif abs_delta < 0.33:
        results['effect_size'] = 'small'
    elif abs_delta < 0.474:
        results['effect_size'] = 'medium'
    else:
        results['effect_size'] = 'large'

    # TOST
    if true_value != 0:
        mre_values = abs_errors / np.abs(true_value)
        try:
            _, t1_p = stats.ttest_1samp(mre_values, threshold_mre,
                                         alternative='less')
            _, t2_p = stats.ttest_1samp(mre_values, -threshold_mre,
                                         alternative='greater')
            results['TOST_t1_p'] = float(t1_p)
            results['TOST_t2_p'] = float(t2_p)
            results['TOST_equivalence'] = bool((t1_p < 0.05) and (t2_p < 0.05))
            results['TOST_threshold'] = float(threshold_mre)
        except Exception as e:
            results['TOST_error'] = str(e)

    return results


# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_experiments(data_vectors: Dict[str, List[float]],
                    dataset_label: str = "Dataset",
                    verbose: bool = True) -> Dict:
    if verbose:
        print(f"\n🧪 RUNNING EXPERIMENTS ON {dataset_label.upper()}")
        print("-" * 70)

    results = {}

    for query_type in QUERY_TYPES:
        data = data_vectors[query_type]
        true_value = compute_true_value(data, query_type)

        if verbose:
            print(f"\nQuery: {query_type.upper()} (True: {true_value:,.4f})")
            print(f"  {'ε':<6} {'MAE':<18} {'MRE':<14} {'Std Dev':<18}")
            print(f"  {'-'*6} {'-'*18} {'-'*14} {'-'*18}")

        results[query_type] = {
            'true_value': true_value,
            'laplace': {},
            'gaussian': {}
        }

        for epsilon in EPSILON_VALUES:
            noisy_results = []
            for _ in range(NUM_TRIALS):
                if query_type == 'count':
                    noisy = apply_laplace_count(data, epsilon)
                elif query_type == 'mean':
                    noisy = apply_laplace_mean(data, epsilon)
                elif query_type == 'sum':
                    noisy = apply_laplace_sum(data, epsilon)
                else:
                    noisy = apply_laplace_median(data, epsilon)
                noisy_results.append(noisy)

            metrics = compute_metrics(true_value, noisy_results)
            metrics['noisy_values'] = noisy_results
            results[query_type]['laplace'][epsilon] = metrics

            if verbose:
                mre_str = (f"{metrics['MRE']*100:.4f}%"
                           if not np.isnan(metrics['MRE']) else "N/A")
                print(f"  {epsilon:<6.1f} {metrics['MAE']:<18.4f} "
                      f"{mre_str:<14} {metrics['std']:<18.4f}")

        # Gaussian for sum queries
        if query_type == 'sum':
            if verbose:
                print(f"\n  Gaussian mechanism for SUM queries:")
                print(f"  {'ε':<6} {'MAE':<18} {'MRE':<14} {'Std Dev':<18}")
                print(f"  {'-'*6} {'-'*18} {'-'*14} {'-'*18}")

            for epsilon in EPSILON_VALUES:
                noisy_results = []
                for _ in range(NUM_TRIALS):
                    noisy = apply_gaussian_sum(data, epsilon, DELTA)
                    noisy_results.append(noisy)

                metrics = compute_metrics(true_value, noisy_results)
                metrics['noisy_values'] = noisy_results
                results[query_type]['gaussian'][epsilon] = metrics

                if verbose:
                    mre_str = (f"{metrics['MRE']*100:.4f}%"
                               if not np.isnan(metrics['MRE']) else "N/A")
                    print(f"  {epsilon:<6.1f} {metrics['MAE']:<18.4f} "
                          f"{mre_str:<14} {metrics['std']:<18.4f}")

    if verbose:
        print("-" * 70)
        print(f"✓ {dataset_label} experiments complete!")

    return results


# ============================================================================
# SUBGROUP ANALYSIS
# ============================================================================

def analyze_subgroups(data: List[float],
                      subgroup_sizes: List[int],
                      dataset_label: str = "Dataset") -> Dict:
    print(f"\n📊 SUBGROUP ANALYSIS — {dataset_label.upper()}")
    print("-" * 80)
    print(f"Requested sizes: {subgroup_sizes}")
    print(f"Available records: {len(data):,}")

    feasible = [s for s in subgroup_sizes if s <= len(data)]
    infeasible = [s for s in subgroup_sizes if s > len(data)]
    if infeasible:
        print(f"⚠️ Sizes exceeding dataset length will be skipped: {infeasible}")

    subgroup_results = {}

    header = f"  {'Size':>9} |"
    for eps in EPSILON_VALUES:
        header += f" ε={eps:<4.1f} (MRE) |"
    print(header)
    print("  " + "-" * (len(header) - 3))

    for size in feasible:
        subgroup_data = data[:size] if size < len(data) else list(data)
        true_count = len(subgroup_data)
        subgroup_results[size] = {}

        for epsilon in EPSILON_VALUES:
            noisy_results = [apply_laplace_count(subgroup_data, epsilon)
                             for _ in range(NUM_TRIALS)]
            subgroup_results[size][epsilon] = compute_metrics(
                true_count, noisy_results)

        row = f"  {size:>9,} |"
        for epsilon in EPSILON_VALUES:
            mre = subgroup_results[size][epsilon]['MRE']
            mre_str = "N/A" if np.isnan(mre) else f"{mre*100:.4f}%"
            row += f" {mre_str:>14} |"
        print(row)

    print("-" * 80)
    print(f"✓ {dataset_label} subgroup analysis complete!")
    return subgroup_results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: Dict,
                    subgroup_results: Dict,
                    formal_results: Optional[Dict],
                    dataset_info: Dict,
                    dataset_label: str = "Dataset") -> str:
    report = []
    report.append("=" * 90)
    report.append(f"DIFFERENTIAL PRIVACY ANALYSIS REPORT — {dataset_label}")
    report.append("South African Demographic Dataset")
    report.append("=" * 90)
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Epsilon values tested: {EPSILON_VALUES}")
    report.append(f"Trials per epsilon: {NUM_TRIALS}")
    report.append("")

    report.append("DATASET SUMMARY:")
    report.append("-" * 60)
    report.append(f"Total records: {dataset_info.get('total_records', 0):,}")
    report.append(f"Numeric columns: {len(dataset_info.get('numeric_cols', []))}")
    report.append(f"Categorical columns: {len(dataset_info.get('cat_cols', []))}")
    report.append("")

    report.append("TRUE STATISTICS:")
    report.append("-" * 60)
    for qtype, qres in results.items():
        report.append(f"{qtype.capitalize()}: {qres['true_value']:,.4f}")
    report.append("")

    report.append("LAPLACE MECHANISM RESULTS:")
    report.append("=" * 90)
    for qtype, qres in results.items():
        if qres['laplace']:
            report.append(f"\nQuery: {qtype.capitalize()}")
            header = "  Metric    |"
            for eps in EPSILON_VALUES:
                header += f"  ε={eps:<6.1f} |"
            report.append(header)
            report.append("  " + "-" * (len(header) - 3))

            row = "  MAE       |"
            for eps in EPSILON_VALUES:
                m = qres['laplace'][eps]
                row += f"  {m['MAE']:>10.4f} |"
            report.append(row)

            row = "  MRE       |"
            for eps in EPSILON_VALUES:
                m = qres['laplace'][eps]
                mre_str = (f"{m['MRE']*100:.4f}%"
                           if not np.isnan(m['MRE']) else "N/A")
                row += f"  {mre_str:>10} |"
            report.append(row)

            row = "  Std Dev   |"
            for eps in EPSILON_VALUES:
                m = qres['laplace'][eps]
                row += f"  {m['std']:>10.4f} |"
            report.append(row)
    report.append("")

    report.append("GAUSSIAN MECHANISM RESULTS (Sum Queries):")
    report.append("=" * 90)
    if 'sum' in results and results['sum']['gaussian']:
        header = "  Metric    |"
        for eps in EPSILON_VALUES:
            header += f"  ε={eps:<6.1f} |"
        report.append(header)
        report.append("  " + "-" * (len(header) - 3))

        row = "  MAE       |"
        for eps in EPSILON_VALUES:
            m = results['sum']['gaussian'][eps]
            row += f"  {m['MAE']:>10.4f} |"
        report.append(row)

        row = "  MRE       |"
        for eps in EPSILON_VALUES:
            m = results['sum']['gaussian'][eps]
            mre_str = (f"{m['MRE']*100:.4f}%"
                       if not np.isnan(m['MRE']) else "N/A")
            row += f"  {mre_str:>10} |"
        report.append(row)

        row = "  Std Dev   |"
        for eps in EPSILON_VALUES:
            m = results['sum']['gaussian'][eps]
            row += f"  {m['std']:>10.4f} |"
        report.append(row)
    report.append("")

    if formal_results:
        report.append("FORMAL STATISTICAL HYPOTHESIS TESTING:")
        report.append("=" * 90)
        report.append(f"Wilcoxon signed-rank (H0: median diff = 0)")
        report.append(f"TOST equivalence (H1: MRE < {TOST_MARGIN*100:.0f}%)")
        report.append(f"Cliff's delta effect size")
        report.append("")
        report.append(
            "  Query | Mech    | ε   | Wilcoxon p | Cliff delta | Effect     | "
            "TOST | H0 Rej | H1 Sup"
        )
        report.append(
            "  ------|---------|-----|------------|-------------|------------|"
            "------|--------|-------"
        )
        for qtype, mechs in formal_results.items():
            for mech, eps_dict in mechs.items():
                for eps_val in sorted(eps_dict.keys()):
                    r = eps_dict[eps_val]
                    wp = r.get('wilcoxon_p', float('nan'))
                    cd = r.get('cliff_delta', float('nan'))
                    es = r.get('effect_size', 'N/A')
                    tost = "Pass" if r.get('TOST_equivalence', False) else "Fail"
                    h0 = "Yes" if r.get('wilcoxon_reject_H0', False) else "No"
                    h1 = "Yes" if r.get('TOST_equivalence', False) else "No"
                    report.append(
                        f"  {qtype:<5} | {mech:<7} | {eps_val:<3.1f} | "
                        f"{wp:<10.4f} | {cd:<11.4f} | {es:<10} | "
                        f"{tost:<4} | {h0:<6} | {h1:<6}"
                    )
        report.append("")

    report.append("SUBGROUP ANALYSIS:")
    report.append("=" * 90)
    if subgroup_results:
        header = f"  {'Size':>9} |"
        for eps in EPSILON_VALUES:
            header += f" ε={eps:<4.1f} (MRE) |"
        report.append(header)
        report.append("  " + "-" * (len(header) - 3))
        for size in sorted(subgroup_results.keys()):
            row = f"  {size:>9,} |"
            for eps in EPSILON_VALUES:
                mre = subgroup_results[size][eps]['MRE']
                mre_str = "N/A" if np.isnan(mre) else f"{mre*100:.4f}%"
                row += f" {mre_str:>14} |"
            report.append(row)
    report.append("")

    report.append("=" * 90)
    report.append("END OF REPORT")
    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("🚀 STARTING DIFFERENTIAL PRIVACY ANALYSIS")
    print("=" * 80)

    try:
        # 1. Load data
        df = load_data(DATASET_PATH)

        # 2. Identify columns
        numeric_cols, cat_cols = identify_columns(df)

        if not numeric_cols:
            print("\n⚠️ No numeric columns found. Creating synthetic placeholder.")
            rng = np.random.default_rng(seed=42)
            df = pd.DataFrame({
                'value': rng.normal(100, 25, 5000),
                'category': rng.choice(['A', 'B', 'C', 'D'], 5000),
            })
            numeric_cols = ['value']
            cat_cols = ['category']

        # 3. Clipping
        clip_bounds = compute_clipping_bounds(df, numeric_cols)
        df = apply_clipping(df, clip_bounds)

        # 4. Real data vectors (Dataset A)
        data_vectors_A = prepare_data_vectors(df, numeric_cols)

        # 5. Synthetic dataset (Dataset B)
        print(f"\n🔧 GENERATING SYNTHETIC SCALED DATASET (N={SYNTHETIC_TARGET_SIZE:,})")
        print("-" * 60)
        data_vectors_B = {}
        for qtype, vec in data_vectors_A.items():
            data_vectors_B[qtype] = generate_synthetic_dataset(
                vec, target_size=SYNTHETIC_TARGET_SIZE,
                noise_sigma=SYNTHETIC_NOISE_SIGMA, seed=42)
        print(f"✓ Synthetic dataset generated: "
              f"{len(data_vectors_B['count']):,} records per vector")

        # 6. DP counts on categorical
        if cat_cols:
            cat_col = cat_cols[0]
            print(f"\n📋 DP COUNTS (Laplace) for: {cat_col}")
            dp_counts = dp_counts_custom(df, cat_col, epsilon=1.0)
            dp_counts_df = sanitize_dataframe(dp_counts)
            print(dp_counts_df.head(10))

        # 7. Run experiments on both datasets
        results_A = run_experiments(data_vectors_A,
                                    dataset_label="Dataset A (Real, N=52)")
        results_B = run_experiments(data_vectors_B,
                                    dataset_label=f"Dataset B (Synthetic, N={SYNTHETIC_TARGET_SIZE:,})")

        # 8. Formal hypothesis tests
        print("\n📐 RUNNING FORMAL STATISTICAL HYPOTHESIS TESTS")
        print("-" * 70)
        formal_results_A = {}
        formal_results_B = {}

        for label, results_set, store in [
            ("Dataset A", results_A, formal_results_A),
            ("Dataset B", results_B, formal_results_B),
        ]:
            print(f"\n  {label}:")
            for qtype, qres in results_set.items():
                store[qtype] = {}
                for mech in ['laplace', 'gaussian']:
                    if mech in qres and qres[mech]:
                        store[qtype][mech] = {}
                        for eps_val, m in qres[mech].items():
                            if 'noisy_values' in m:
                                store[qtype][mech][eps_val] = formal_hypothesis_testing(
                                    qres['true_value'], m['noisy_values'],
                                    threshold_mre=TOST_MARGIN)
                        print(f"    {qtype.upper():<7} | {mech:<8} | "
                              f"ε values tested: {sorted(store[qtype][mech].keys())}")

        print("\n✓ Formal hypothesis tests complete.")

        # 9. Subgroup analysis
        subgroup_sizes_real = [50]
        subgroup_sizes_syn = [50, 100, 500, 1000, 5000, 10000, 100000, 1000000]

        subgroup_results_A = analyze_subgroups(
            data_vectors_A['count'], subgroup_sizes_real,
            dataset_label="Dataset A (Real)")
        subgroup_results_B = analyze_subgroups(
            data_vectors_B['count'], subgroup_sizes_syn,
            dataset_label="Dataset B (Synthetic)")

        # 10. Consolidated ε-by-ε summary matrix
        print("\n" + "=" * 80)
        print("📊 CONSOLIDATED ε-BY-ε SUMMARY MATRIX")
        print("=" * 80)

        for label, results_set in [("Dataset A (Real, N=52)", results_A),
                                   (f"Dataset B (Synthetic, N={SYNTHETIC_TARGET_SIZE:,})", results_B)]:
            print(f"\n{label}")
            print("-" * 80)

            for qtype in QUERY_TYPES:
                qres = results_set[qtype]
                print(f"\n  Query: {qtype.upper()} (True: {qres['true_value']:,.4f})")

                header = f"    {'Mechanism':<10} |"
                for eps in EPSILON_VALUES:
                    header += f"  ε={eps:<4.1f} (MRE)  |"
                print(header)
                print("    " + "-" * (len(header) - 4))

                row = f"    {'Laplace':<10} |"
                for eps in EPSILON_VALUES:
                    if eps in qres['laplace']:
                        m = qres['laplace'][eps]
                        mre = m['MRE']
                        mre_str = "N/A" if np.isnan(mre) else f"{mre*100:>9.4f}%"
                    else:
                        mre_str = "   N/A   "
                    row += f"  {mre_str:>13} |"
                print(row)

                if qtype == 'sum' and qres.get('gaussian'):
                    row = f"    {'Gaussian':<10} |"
                    for eps in EPSILON_VALUES:
                        if eps in qres['gaussian']:
                            m = qres['gaussian'][eps]
                            mre = m['MRE']
                            mre_str = "N/A" if np.isnan(mre) else f"{mre*100:>9.4f}%"
                        else:
                            mre_str = "   N/A   "
                        row += f"  {mre_str:>13} |"
                    print(row)

        # 11. Formal test summary across all ε
        print("\n" + "=" * 80)
        print("📐 FORMAL TEST SUMMARY — ALL ε VALUES")
        print("=" * 80)

        for label, formal_set in [("Dataset A (Real, N=52)", formal_results_A),
                                  (f"Dataset B (Synthetic, N={SYNTHETIC_TARGET_SIZE:,})", formal_results_B)]:
            print(f"\n{label}")
            print("-" * 100)
            print(f"  {'Query':<6} | {'Mech':<8} | {'ε':<5} | "
                  f"{'Wilcoxon p':<11} | {'Cliff δ':<9} | {'Effect':<11} | "
                  f"{'TOST':<6} | {'H0 Rej':<7} | {'H1 Sup':<7}")
            print("  " + "-" * 96)

            for qtype, mechs in formal_set.items():
                for mech, eps_dict in mechs.items():
                    for eps_val in sorted(eps_dict.keys()):
                        r = eps_dict[eps_val]
                        wp = r.get('wilcoxon_p', float('nan'))
                        cd = r.get('cliff_delta', float('nan'))
                        es = r.get('effect_size', 'N/A')
                        tost = "Pass" if r.get('TOST_equivalence', False) else "Fail"
                        h0 = "Yes" if r.get('wilcoxon_reject_H0', False) else "No"
                        h1 = "Yes" if r.get('TOST_equivalence', False) else "No"

                        print(f"  {qtype:<6} | {mech:<8} | {eps_val:<5.1f} | "
                              f"{wp:<11.4f} | {cd:<9.4f} | {es:<11} | "
                              f"{tost:<6} | {h0:<7} | {h1:<7}")

        # 12. Save reports (UTF-8 encoding)
        dataset_info_A = {
            'total_records': len(df),
            'numeric_cols': numeric_cols,
            'cat_cols': cat_cols
        }
        dataset_info_B = {
            'total_records': len(data_vectors_B['count']),
            'numeric_cols': numeric_cols,
            'cat_cols': cat_cols
        }

        report_A = generate_report(results_A, subgroup_results_A,
                                   formal_results_A, dataset_info_A,
                                   dataset_label="Dataset A (Real)")
        report_B = generate_report(results_B, subgroup_results_B,
                                   formal_results_B, dataset_info_B,
                                   dataset_label="Dataset B (Synthetic)")

        report_A_path = os.path.join(OUTPUT_DIR, 'dp_report_dataset_A.txt')
        report_B_path = os.path.join(OUTPUT_DIR, 'dp_report_dataset_B.txt')

        with open(report_A_path, 'w', encoding='utf-8') as f:
            f.write(report_A)
        with open(report_B_path, 'w', encoding='utf-8') as f:
            f.write(report_B)

        print(f"\n✓ Report A saved: {report_A_path}")
        print(f"✓ Report B saved: {report_B_path}")

        # 13. Consolidated CSV summary
        results_df = []
        for label, results_set in [("A_Real", results_A),
                                   ("B_Synthetic", results_B)]:
            for qtype, qres in results_set.items():
                for mech in ['laplace', 'gaussian']:
                    for eps_val, m in qres.get(mech, {}).items():
                        results_df.append({
                            'dataset': label,
                            'query_type': qtype,
                            'mechanism': mech,
                            'epsilon': eps_val,
                            'true_value': qres['true_value'],
                            'MAE': m['MAE'],
                            'MRE': m['MRE'],
                            'std_dev': m['std']
                        })

        results_summary_df = pd.DataFrame(results_df)
        summary_path = os.path.join(OUTPUT_DIR, 'dp_results_summary.csv')
        results_summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary CSV saved: {summary_path}")

        # 14. Formal test CSV
        formal_rows = []
        for label, formal_set in [("A_Real", formal_results_A),
                                  ("B_Synthetic", formal_results_B)]:
            for qtype, mechs in formal_set.items():
                for mech, eps_dict in mechs.items():
                    for eps_val, r in eps_dict.items():
                        formal_rows.append({
                            'dataset': label,
                            'query_type': qtype,
                            'mechanism': mech,
                            'epsilon': eps_val,
                            'MAE': r.get('MAE', np.nan),
                            'MRE': r.get('MRE', np.nan),
                            'MAE_CI_lower': r.get('MAE_CI_lower', np.nan),
                            'MAE_CI_upper': r.get('MAE_CI_upper', np.nan),
                            'MRE_CI_lower': r.get('MRE_CI_lower', np.nan),
                            'MRE_CI_upper': r.get('MRE_CI_upper', np.nan),
                            'wilcoxon_p': r.get('wilcoxon_p', np.nan),
                            'cliff_delta': r.get('cliff_delta', np.nan),
                            'effect_size': r.get('effect_size', 'N/A'),
                            'TOST_equivalence': r.get('TOST_equivalence', False),
                            'wilcoxon_reject_H0': r.get('wilcoxon_reject_H0', False)
                        })

        formal_df = pd.DataFrame(formal_rows)
        formal_path = os.path.join(OUTPUT_DIR, 'dp_formal_tests.csv')
        formal_df.to_csv(formal_path, index=False)
        print(f"✓ Formal tests CSV saved: {formal_path}")

        # 15. Files generated
        print("\n" + "=" * 80)
        print("✅ EXECUTION COMPLETE")
        print("=" * 80)
        print("\n📁 Files generated:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp):
                print(f"  - {f} ({os.path.getsize(fp):,} bytes)")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
