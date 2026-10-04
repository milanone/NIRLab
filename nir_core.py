"""
Core logic for parsing multi-block JCAMP-DX (.jdx) NIR spectra and running
the analysis pipeline (SNV, duplicate check, PCA, Ward clustering).

Kept UI-independent on purpose so it can be reused by the Streamlit app
(app_streamlit.py) and, later, by a desktop app, without touching the
underlying logic.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Data lives next to the app, separate from the code, one subfolder per
# analysis project (e.g. "salicornia-critmo"): data/<project>/raw holds the
# source JDX files, data/<project>/<group> holds each group's working
# CSVs/figures, data/<project>/ itself holds files shared across groups
# (e.g. LAB_values.csv) and data/<project>/reports/ holds write-ups. This
# keeps unrelated analyses (different instruments, different sample sets)
# from mixing in one flat folder as the tool is reused over time.
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

RESERVED_GROUP_NAMES = {"raw", "reports"}


def list_projects() -> list[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))


def project_dir(project: str) -> Path:
    d = DATA_DIR / project
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_dir(project: str) -> Path:
    d = project_dir(project) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir(project: str) -> Path:
    d = project_dir(project) / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_raw_files(project: str) -> list[Path]:
    return sorted(raw_dir(project).glob("*.jdx"))


def group_dir(project: str, name: str) -> Path:
    d = project_dir(project) / name
    d.mkdir(parents=True, exist_ok=True)
    return d
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

TITLE_RE = re.compile(r"^##TITLE=(.*)$", re.MULTILINE)
XY_BLOCK_RE = re.compile(r"##XYPOINTS=\(XY\.\.XY\)\s*(.*?)##END=", re.DOTALL)
CONC_RE = re.compile(r"##CONCENTRATIONS=.*?\n(.*?)##XYPOINTS", re.DOTALL)
LAB_LINE_RE = re.compile(r"\(([Lab]),\s*(-?\d+\.?\d*),")
PREFIX_RE = re.compile(r"^([A-Za-z]+)(\d+)(bis)?$", re.IGNORECASE)


@dataclass
class Spectrum:
    sample_name: str
    wavelengths: np.ndarray
    absorbance: np.ndarray
    lab: dict = field(default_factory=dict)


def parse_jdx(text: str) -> list[Spectrum]:
    """Split a (possibly multi-block) JCAMP-DX file into individual spectra."""
    starts = [m.start() for m in TITLE_RE.finditer(text)]
    starts.append(len(text))
    spectra = []
    for start, end in zip(starts, starts[1:]):
        chunk = text[start:end]
        title_match = TITLE_RE.search(chunk)
        xy_match = XY_BLOCK_RE.search(chunk)
        if not title_match or not xy_match:
            continue
        sample_name = title_match.group(1).strip().split("/")[-1].strip()

        pairs = []
        for token in xy_match.group(1).replace("\n", "").split(";"):
            token = token.strip()
            if not token:
                continue
            x_str, y_str = token.split(",")
            pairs.append((float(x_str), float(y_str)))
        wavelengths = np.array([p[0] for p in pairs])
        absorbance = np.array([p[1] for p in pairs])

        lab = {}
        conc_match = CONC_RE.search(chunk)
        if conc_match:
            lab = {k: float(v) for k, v in LAB_LINE_RE.findall(conc_match.group(1))}

        spectra.append(Spectrum(sample_name, wavelengths, absorbance, lab))
    return spectra


def detect_groups(sample_names: list[str]) -> dict[str, list[str]]:
    """Group sample names by their alphabetic prefix (e.g. 'Sal12' -> 'Sal')."""
    groups: dict[str, list[str]] = {}
    for name in sample_names:
        m = PREFIX_RE.match(name)
        prefix = m.group(1).lower() if m else "altro"
        groups.setdefault(prefix, []).append(name)
    return groups


def spectra_to_dataframe(spectra: list[Spectrum]) -> pd.DataFrame:
    """Build a wavelength x sample matrix. Assumes all spectra share the same grid
    (true for spectra from the same instrument/method); truncates to the shortest
    if lengths differ, rather than silently misaligning wavelengths."""
    if not spectra:
        return pd.DataFrame({"wavelength_nm": []})
    min_len = min(len(s.wavelengths) for s in spectra)
    wavelengths = spectra[0].wavelengths[:min_len]
    data = {"wavelength_nm": wavelengths}
    for s in spectra:
        data[s.sample_name] = s.absorbance[:min_len]
    return pd.DataFrame(data)


def lab_dataframe(spectra: list[Spectrum]) -> pd.DataFrame:
    rows = [
        {"sample": s.sample_name, "L": s.lab.get("L"), "a": s.lab.get("a"), "b": s.lab.get("b")}
        for s in spectra
        if s.lab
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.iloc[df["sample"].map(natural_sample_key).argsort(kind="stable")].reset_index(drop=True)
    return df


def natural_sample_key(name: str) -> tuple:
    """Sort key for sample names like 'Sal2', 'Sal12', 'Sal7bis': groups by
    prefix, then numerically (not lexicographically, so Sal2 < Sal12), with
    a 'bis' duplicate placed right after its original."""
    m = PREFIX_RE.match(name)
    if not m:
        return (name.lower(), 0, 0)
    prefix, num, bis = m.groups()
    return (prefix.lower(), int(num), 1 if bis else 0)


def compute_snv(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in sample_cols:
        y = df[col]
        out[col] = (y - y.mean()) / y.std(ddof=0)
    return out


def find_duplicate_pairs(sample_cols: list[str]) -> list[tuple[str, str]]:
    """Match e.g. 'Sal7' with 'Sal7bis' (case-insensitive)."""
    pairs = []
    for col in sample_cols:
        if col.lower().endswith("bis"):
            base = col[: -len("bis")]
            candidates = [c for c in sample_cols if c.lower() == base.lower()]
            if candidates:
                pairs.append((candidates[0], col))
    return pairs


def duplicate_metrics(df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for orig, bis in pairs:
        y1, y2 = df[orig].values, df[bis].values
        diff = y1 - y2
        rows.append({
            "coppia": f"{orig}/{bis}",
            "r": float(np.corrcoef(y1, y2)[0, 1]),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "max_abs_diff": float(np.max(np.abs(diff))),
        })
    return pd.DataFrame(rows)


def average_duplicates(df: pd.DataFrame, sample_cols: list[str], pairs: list[tuple[str, str]]) -> pd.DataFrame:
    out = df[["wavelength_nm"]].copy()
    paired = {c for pair in pairs for c in pair}
    for orig, bis in pairs:
        out[orig] = df[[orig, bis]].mean(axis=1)
    for col in sample_cols:
        if col not in paired:
            out[col] = df[col]
    return out


def average_lab_duplicates(lab_df: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    if lab_df.empty:
        return lab_df
    paired = {c for pair in pairs for c in pair}
    rows = []
    for orig, bis in pairs:
        sub = lab_df[lab_df["sample"].isin([orig, bis])]
        if sub.empty:
            continue
        rows.append({"sample": orig, "L": sub["L"].mean(), "a": sub["a"].mean(), "b": sub["b"].mean()})
    for _, row in lab_df[~lab_df["sample"].isin(paired)].iterrows():
        rows.append({"sample": row["sample"], "L": row["L"], "a": row["a"], "b": row["b"]})
    return pd.DataFrame(rows)


def outlier_scores(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    """RMSE of each spectrum from the mean spectrum, sorted descending."""
    mean_spec = df[sample_cols].mean(axis=1)
    rows = [
        {"campione": c, "rmse_da_media": float(np.sqrt(np.mean((df[c] - mean_spec) ** 2)))}
        for c in sample_cols
    ]
    return pd.DataFrame(rows).sort_values("rmse_da_media", ascending=False).reset_index(drop=True)


def run_pca(df: pd.DataFrame, sample_cols: list[str], n_components: int = 5):
    X = df[sample_cols].values.T  # samples x wavelengths
    n_components = min(n_components, len(sample_cols) - 1, X.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    scores_df = pd.DataFrame(
        scores, columns=[f"PC{i+1}" for i in range(n_components)], index=sample_cols
    )
    explained = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(
        pca.components_.T, columns=[f"PC{i+1}" for i in range(n_components)], index=df["wavelength_nm"]
    )
    return scores_df, loadings, explained


def pls_cross_validate(X: np.ndarray, y: np.ndarray, max_components: int = 10) -> tuple[dict, int]:
    """Leave-one-out CV to pick the number of PLS latent variables (RMSECV-minimizing).
    LOO is used instead of k-fold because sample counts here are typically ~10-15."""
    n_samples = X.shape[0]
    max_components = max(1, min(max_components, n_samples - 2, X.shape[1]))
    rmsecv = {}
    for n_comp in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_comp)
        y_pred_cv = np.ravel(cross_val_predict(pls, X, y, cv=LeaveOneOut()))
        rmsecv[n_comp] = float(np.sqrt(mean_squared_error(y, y_pred_cv)))
    best_n = min(rmsecv, key=rmsecv.get)
    return rmsecv, best_n


def fit_pls(X: np.ndarray, y: np.ndarray, n_components: int):
    """Fit PLS on all data (calibration) and also compute leave-one-out CV predictions,
    so both an optimistic (calibration) and a more honest (CV) performance estimate are
    available - with n~10-15 samples the gap between the two is usually the whole story."""
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    y_pred_cal = np.ravel(pls.predict(X))
    y_pred_cv = np.ravel(cross_val_predict(pls, X, y, cv=LeaveOneOut()))

    metrics = {
        "r2_cal": r2_score(y, y_pred_cal),
        "r2_cv": r2_score(y, y_pred_cv),
        "rmse_cal": float(np.sqrt(mean_squared_error(y, y_pred_cal))),
        "rmse_cv": float(np.sqrt(mean_squared_error(y, y_pred_cv))),
    }
    return pls, y_pred_cal, y_pred_cv, metrics


def run_ward_clustering(df: pd.DataFrame, sample_cols: list[str], k_max: int | None = None):
    X = df[sample_cols].values.T
    n = len(sample_cols)
    k_max = k_max or (n - 1)
    Z = linkage(X, method="ward", metric="euclidean")

    sil_scores = {}
    for k in range(2, min(k_max, n - 1) + 1):
        clusters = fcluster(Z, k, criterion="maxclust")
        sil_scores[k] = silhouette_score(X, clusters, metric="euclidean")

    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else 1
    clusters = fcluster(Z, best_k, criterion="maxclust") if sil_scores else np.ones(n, dtype=int)
    assignment = pd.Series(clusters, index=sample_cols, name="cluster")
    return Z, sil_scores, best_k, assignment
