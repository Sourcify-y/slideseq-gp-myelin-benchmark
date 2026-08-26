import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import pearsonr, combine_pvalues

import scanpy as sc
import squidpy as sq

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_GENES = {
    "Mbp":    "myelin marker (primary target)",
    "Plp1":   "myelin marker (replication check)",
    "Mobp":   "myelin marker (replication check)",
    "Mog":    "myelin marker (replication check)",
    "Cnp":    "oligodendrocyte-lineage marker",
    "Sox10":  "oligodendrocyte-lineage transcription factor",
    "Snap25": "diffuse neuronal marker (negative/weak-spatial control)",
    "Actb":   "housekeeping gene (negative/weak-spatial control)",
    "Malat1": "broadly/ubiquitously expressed (negative/weak-spatial control)",
}

N_SUBSAMPLE = 2000
N_REPEATS = 10
TEST_SIZE = 0.2
VAL_SIZE = 0.2
GRID_RES = 50
OUT_PREFIX = "gp_results"

N_SPLITS = 5
PERM_PER_SPLIT = 150
SVGP_PERM = 2000
N_BOOTSTRAP = 1000
ALPHA_FDR = 0.05
WEAK_AGREEMENT_R = 0.3

N_BLOCKS_A = 40
N_BLOCKS_B = 150

N_INDUCING = 400
EPOCHS = 150
BATCH_SIZE = 1024
LR = 0.01
CONVERGENCE_WINDOW = 20
CONVERGENCE_REL_TOL = 0.01


def load_data():
    print("Loading SlideSeq v2 dataset via squidpy...")
    adata = sq.datasets.slideseqv2()
    coords = adata.obsm["spatial"].astype(float)
    print(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    present = [g for g in TARGET_GENES if g in adata.var_names]
    missing = [g for g in TARGET_GENES if g not in adata.var_names]
    if missing:
        print(f"Note: {len(missing)} gene(s) not in this dataset, skipping: {missing}")
    print(f"Proceeding with {len(present)} gene(s): {present}")

    ensure_normalized(adata)
    return adata, coords


def ensure_normalized(adata, sample_size=2000, seed=SEED):
    # sample randomly instead of just grabbing the first N rows, in case
    # cells are ordered by capture batch or something similar
    X = adata.X
    X_dense = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    if X_dense.shape[0] > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X_dense.shape[0], size=sample_size, replace=False)
        sample = X_dense[idx]
    else:
        sample = X_dense

    looks_like_counts = np.allclose(sample, np.round(sample), atol=1e-6) and sample.max() > 30
    if looks_like_counts:
        print("Detected raw-count-like expression matrix -> applying normalize_total + log1p.")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        print("Expression matrix already normalized/log-scale -> leaving as-is.")


def get_gene_expression(adata, gene):
    if gene not in adata.var_names:
        raise ValueError(f"Gene '{gene}' not found in dataset.")
    expr = adata[:, gene].X
    return np.asarray(expr.todense()).flatten() if hasattr(expr, "todense") else np.asarray(expr).flatten()


def spatial_block_split(coords, test_size, val_size=None, n_blocks=40, seed=SEED):
    """
    KMeans the coordinates into blocks and assign whole blocks to
    train/val/test, rather than splitting individual points. Random
    point-level splits let a model cheat by exploiting local spatial
    autocorrelation between neighboring train/test points.
    """
    n_total = len(coords)
    km = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit(coords)
    block_labels = km.labels_
    unique_blocks = np.unique(block_labels)

    rng = np.random.RandomState(seed)
    shuffled_blocks = unique_blocks.copy()
    rng.shuffle(shuffled_blocks)
    block_sizes = {b: int(np.sum(block_labels == b)) for b in shuffled_blocks}

    target_test = test_size * n_total
    test_blocks, cum = [], 0
    for b in shuffled_blocks:
        if cum >= target_test:
            break
        test_blocks.append(b)
        cum += block_sizes[b]
    test_blocks = set(test_blocks)
    remaining_blocks = [b for b in shuffled_blocks if b not in test_blocks]

    if val_size is None:
        train_idx = np.where(~np.isin(block_labels, list(test_blocks)))[0]
        test_idx = np.where(np.isin(block_labels, list(test_blocks)))[0]
        return train_idx, test_idx

    n_nontest = n_total - cum
    target_val = val_size * n_nontest
    val_blocks, cum2 = [], 0
    for b in remaining_blocks:
        if cum2 >= target_val:
            break
        val_blocks.append(b)
        cum2 += block_sizes[b]
    val_blocks = set(val_blocks)
    train_blocks = set(remaining_blocks) - val_blocks

    train_idx = np.where(np.isin(block_labels, list(train_blocks)))[0]
    val_idx = np.where(np.isin(block_labels, list(val_blocks)))[0]
    test_idx = np.where(np.isin(block_labels, list(test_blocks)))[0]
    return train_idx, val_idx, test_idx


def idw_predict(train_coords, train_vals, query_coords, power=2, eps=1e-9):
    preds = np.zeros(len(query_coords))
    for i, q in enumerate(query_coords):
        d = np.linalg.norm(train_coords - q, axis=1)
        d = np.where(d < eps, eps, d)
        w = 1.0 / (d ** power)
        preds[i] = np.sum(w * train_vals) / np.sum(w)
    return preds


def moran_baseline(adata, genes):
    print("\nComputing Moran's I spatial autocorrelation baseline (squidpy, full dataset)...")
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
    sq.gr.spatial_autocorr(adata, mode="moran", genes=genes, n_perms=999, n_jobs=1, seed=SEED)
    moran_df = adata.uns["moranI"].loc[genes].rename_axis("gene").reset_index()
    print(moran_df[["gene", "I", "pval_norm"]].to_string(index=False))
    return moran_df


def in_hull(points, hull_points):
    hull = Delaunay(hull_points[ConvexHull(hull_points).vertices])
    return hull.find_simplex(points) >= 0


def make_grid(coords, res=GRID_RES):
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = in_hull(grid_pts, coords)
    return xx, yy, grid_pts, mask


def _safe_pearsonr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan, np.nan
    return pearsonr(a, b)


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


def _rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


def _pearson_r_metric(a, b):
    return _safe_pearsonr(a, b)[0]


def benjamini_hochberg(pvals, alpha=ALPHA_FDR):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals_sorted = ranked * n / np.arange(1, n + 1)
    qvals_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals = np.empty(n)
    qvals[order] = np.clip(qvals_sorted, 0, 1)
    return qvals, qvals <= alpha


def robustness_check(coords_scaled, expr, gene_name, n_repeats=N_REPEATS, n_sub=N_SUBSAMPLE):
    print(f"\n[{gene_name}] Hyperparameter robustness across {n_repeats} random subsamples...")
    length_scales, noise_levels = [], []
    for r in range(n_repeats):
        rng = np.random.RandomState(SEED + r)
        idx = rng.choice(len(expr), size=n_sub, replace=False)
        X, y = coords_scaled[idx], expr[idx]
        kernel = RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=1.0)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                       normalize_y=True, random_state=SEED)
        gp.fit(X, y)
        params = gp.kernel_.get_params()
        length_scales.append(np.mean(params["k1__length_scale"]))
        noise_levels.append(params["k2__noise_level"])

    ls_mean, ls_std = np.mean(length_scales), np.std(length_scales)
    nl_mean, nl_std = np.mean(noise_levels), np.std(noise_levels)
    print(f"  length_scale (standardized units, mean over x,y): {ls_mean:.3f} +/- {ls_std:.3f}")
    print(f"  noise_level:  {nl_mean:.3f} +/- {nl_std:.3f}")
    return {"length_scale": (ls_mean, ls_std), "noise_level": (nl_mean, nl_std)}


def _kernel_ctor(name):
    if name == "RBF":
        return RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=1.0)
    elif name == "Matern(nu=1.5)":
        return Matern(length_scale=[1.0, 1.0], nu=1.5) + WhiteKernel(noise_level=1.0)
    elif name == "Matern(nu=2.5)":
        return Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel(noise_level=1.0)
    raise ValueError(name)


def select_kernel_on_val(X_train, y_train, X_val, y_val):
    candidates = ["RBF", "Matern(nu=1.5)", "Matern(nu=2.5)"]
    results = {}
    for name in candidates:
        gp = GaussianProcessRegressor(kernel=_kernel_ctor(name), n_restarts_optimizer=3,
                                       normalize_y=True, random_state=SEED)
        gp.fit(X_train, y_train)
        rmse_val = np.sqrt(mean_squared_error(y_val, gp.predict(X_val)))
        results[name] = rmse_val
        print(f"    {name:<16} VAL RMSE={rmse_val:.4f}")
    best_name = min(results, key=results.get)
    print(f"    -> selected on validation set: {best_name}")
    return best_name


def permutation_null_test_single_split(X_train, y_train, X_test, y_test, kernel_name,
                                        n_perm=PERM_PER_SPLIT, seed=SEED):
    gp_real = GaussianProcessRegressor(kernel=_kernel_ctor(kernel_name), n_restarts_optimizer=3,
                                        normalize_y=True, random_state=seed)
    gp_real.fit(X_train, y_train)
    observed_r2 = r2_score(y_test, gp_real.predict(X_test))

    # hold the fitted kernel fixed and just refit the mean under permuted
    # labels -- refitting hyperparameters for every permutation isn't
    # feasible at this scale
    fixed_kernel = gp_real.kernel_.clone_with_theta(gp_real.kernel_.theta)

    rng = np.random.RandomState(seed)
    null_r2 = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y_train)
        gp_null = GaussianProcessRegressor(kernel=fixed_kernel, optimizer=None, normalize_y=True)
        gp_null.fit(X_train, y_perm)
        null_r2[i] = r2_score(y_test, gp_null.predict(X_test))

    p_value = (np.sum(null_r2 >= observed_r2) + 1) / (n_perm + 1)
    return {"observed_r2": observed_r2, "null_r2_mean": float(np.mean(null_r2)),
            "null_r2_std": float(np.std(null_r2)), "p_value": p_value}


def run_exact_gp(adata, coords, gene, description):
    print("\n" + "=" * 70)
    print(f"PART A (exact GP, n={N_SUBSAMPLE}) — GENE: {gene} ({description})")
    print("=" * 70)

    expr = get_gene_expression(adata, gene)
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(expr), size=N_SUBSAMPLE, replace=False)
    X_all_raw, y_all = coords[idx], expr[idx]

    coord_scaler = StandardScaler().fit(X_all_raw)
    X_all = coord_scaler.transform(X_all_raw)

    robustness = robustness_check(X_all, y_all, gene)

    split_metrics = {"rmse": [], "r2": [], "pearson_r": [], "rmse_knn": [], "rmse_idw": [],
                      "calib_1sigma": [], "calib_2sigma": [], "kernel_chosen": [],
                      "degenerate_test_fold": []}
    split_artifacts = []
    split_perm_results = []

    for split_i in range(N_SPLITS):
        split_seed = SEED + split_i

        trainval_idx, test_idx = spatial_block_split(X_all_raw, test_size=TEST_SIZE,
                                                       n_blocks=N_BLOCKS_A, seed=split_seed)
        trainval_coords = X_all_raw[trainval_idx]
        train_sub_idx, val_sub_idx = spatial_block_split(
            trainval_coords, test_size=VAL_SIZE,
            n_blocks=max(5, int(N_BLOCKS_A * (1 - TEST_SIZE))), seed=split_seed + 1000)
        train_idx = trainval_idx[train_sub_idx]
        val_idx = trainval_idx[val_sub_idx]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        X_trainval, y_trainval = X_all[trainval_idx], y_all[trainval_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        print(f"\n  [split {split_i+1}/{N_SPLITS}] "
              f"(spatial blocks: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}) "
              f"Kernel selection on validation set...")
        best_kernel_name = select_kernel_on_val(X_train, y_train, X_val, y_val)

        gp_final_split = GaussianProcessRegressor(kernel=_kernel_ctor(best_kernel_name),
                                                    n_restarts_optimizer=3, normalize_y=True,
                                                    random_state=split_seed)
        gp_final_split.fit(X_trainval, y_trainval)
        y_pred, y_std = gp_final_split.predict(X_test, return_std=True)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        r, _ = _safe_pearsonr(y_test, y_pred)
        degenerate_fold = bool(np.isnan(r))

        knn = KNeighborsRegressor(n_neighbors=10).fit(X_trainval, y_trainval)
        rmse_knn = np.sqrt(mean_squared_error(y_test, knn.predict(X_test)))
        rmse_idw = np.sqrt(mean_squared_error(y_test, idw_predict(X_trainval, y_trainval, X_test)))

        errors = np.abs(y_test - y_pred)
        within_1sigma = np.mean(errors <= y_std)
        within_2sigma = np.mean(errors <= 2 * y_std)

        r_str = f"{r:.3f}" if not degenerate_fold else "NaN (constant test fold)"
        print(f"    TEST: RMSE={rmse:.4f}  R^2={r2:.3f}  r={r_str}  "
              f"(kNN={rmse_knn:.4f}, IDW={rmse_idw:.4f})  "
              f"calib 1sig={within_1sigma:.3f} 2sig={within_2sigma:.3f}")

        split_metrics["rmse"].append(rmse)
        split_metrics["r2"].append(r2)
        split_metrics["pearson_r"].append(r)
        split_metrics["rmse_knn"].append(rmse_knn)
        split_metrics["rmse_idw"].append(rmse_idw)
        split_metrics["calib_1sigma"].append(within_1sigma)
        split_metrics["calib_2sigma"].append(within_2sigma)
        split_metrics["kernel_chosen"].append(best_kernel_name)
        split_metrics["degenerate_test_fold"].append(degenerate_fold)

        split_artifacts.append(dict(y_test=y_test, y_pred=y_pred, rmse=rmse))

        print(f"    Permutation null test for this split ({PERM_PER_SPLIT} perms, kernel={best_kernel_name})...")
        perm_result = permutation_null_test_single_split(X_trainval, y_trainval, X_test, y_test,
                                                           best_kernel_name, n_perm=PERM_PER_SPLIT,
                                                           seed=split_seed)
        print(f"      Observed R^2={perm_result['observed_r2']:.4f}  "
              f"Null R^2={perm_result['null_r2_mean']:.4f} +/- {perm_result['null_r2_std']:.4f}  "
              f"p={perm_result['p_value']:.4g}")
        split_perm_results.append(perm_result)

    modal_kernel = pd.Series(split_metrics["kernel_chosen"]).mode()[0]
    n_degenerate = int(np.sum(split_metrics["degenerate_test_fold"]))
    print(f"\n  Modal kernel choice across {N_SPLITS} splits: {modal_kernel}")
    print(f"  RMSE:  {np.mean(split_metrics['rmse']):.4f} +/- {np.std(split_metrics['rmse']):.4f}  (across {N_SPLITS} splits)")
    print(f"  R^2:   {np.mean(split_metrics['r2']):.4f} +/- {np.std(split_metrics['r2']):.4f}")
    print(f"  r:     {np.nanmean(split_metrics['pearson_r']):.4f} +/- {np.nanstd(split_metrics['pearson_r']):.4f}  "
          f"({n_degenerate}/{N_SPLITS} splits had a zero-variance test fold, excluded from r)")

    # combine the per-split permutation p-values into one gene-level p-value
    per_split_pvals = [p["p_value"] for p in split_perm_results]
    _, combined_p = combine_pvalues(per_split_pvals, method="fisher")
    combined_observed_r2 = float(np.mean([p["observed_r2"] for p in split_perm_results]))
    combined_null_r2_mean = float(np.mean([p["null_r2_mean"] for p in split_perm_results]))
    print(f"\n  Combined permutation test across {N_SPLITS} splits "
          f"(Fisher's method, {PERM_PER_SPLIT} perms/split, {PERM_PER_SPLIT * N_SPLITS} total):")
    print(f"    per-split p-values: {[f'{p:.4g}' for p in per_split_pvals]}")
    print(f"    Combined p={combined_p:.4g}  "
          f"(mean observed R^2={combined_observed_r2:.4f}, mean null R^2={combined_null_r2_mean:.4f})")

    rmses = [a["rmse"] for a in split_artifacts]
    median_rmse = np.median(rmses)
    rep_idx = int(np.argmin(np.abs(np.array(rmses) - median_rmse)))
    rep = split_artifacts[rep_idx]
    print(f"\n  Bootstrap CIs computed on split {rep_idx + 1}/{N_SPLITS} "
          f"(closest to median RMSE of {median_rmse:.4f} across splits)")

    _, rmse_lo, rmse_hi = bootstrap_ci(rep["y_test"], rep["y_pred"], _rmse)
    _, r2_lo, r2_hi = bootstrap_ci(rep["y_test"], rep["y_pred"], r2_score)
    _, r_lo, r_hi = bootstrap_ci(rep["y_test"], rep["y_pred"], _pearson_r_metric)
    print(f"  Bootstrap 95% CI (representative split): "
          f"RMSE [{rmse_lo:.4f}, {rmse_hi:.4f}]  R^2 [{r2_lo:.4f}, {r2_hi:.4f}]  "
          f"r [{r_lo:.4f}, {r_hi:.4f}]  (n_boot={N_BOOTSTRAP})")

    gp_final = GaussianProcessRegressor(kernel=_kernel_ctor(modal_kernel), n_restarts_optimizer=3,
                                         normalize_y=True, random_state=SEED)
    gp_final.fit(X_all, y_all)

    xx, yy, grid_pts, mask = make_grid(X_all_raw, res=GRID_RES)
    grid_pts_scaled = coord_scaler.transform(grid_pts)
    mean_pred, std_pred = gp_final.predict(grid_pts_scaled, return_std=True)
    mean_grid = np.full(grid_pts.shape[0], np.nan)
    mean_grid[mask] = mean_pred[mask]
    mean_grid = mean_grid.reshape(xx.shape)
    std_grid = np.full(grid_pts.shape[0], np.nan)
    std_grid[mask] = std_pred[mask]
    std_grid = std_grid.reshape(xx.shape)

    return {
        "gene": gene, "best_kernel": modal_kernel,
        "rmse_gp": np.mean(split_metrics["rmse"]), "rmse_gp_std": np.std(split_metrics["rmse"]),
        "rmse_knn": np.mean(split_metrics["rmse_knn"]), "rmse_idw": np.mean(split_metrics["rmse_idw"]),
        "r2": np.mean(split_metrics["r2"]), "r2_std": np.std(split_metrics["r2"]),
        "pearson_r": np.nanmean(split_metrics["pearson_r"]), "pearson_r_std": np.nanstd(split_metrics["pearson_r"]),
        "n_degenerate_folds": n_degenerate,
        "calib_1sigma": np.mean(split_metrics["calib_1sigma"]), "calib_2sigma": np.mean(split_metrics["calib_2sigma"]),
        "rmse_ci_lo": rmse_lo, "rmse_ci_hi": rmse_hi,
        "r2_ci_lo": r2_lo, "r2_ci_hi": r2_hi,
        "pearson_r_ci_lo": r_lo, "pearson_r_ci_hi": r_hi,
        "perm_observed_r2": combined_observed_r2, "perm_null_r2_mean": combined_null_r2_mean,
        "perm_p_value": combined_p, "perm_p_values_per_split": per_split_pvals,
        "robustness": robustness,
        "grid": (xx, yy, mask), "mean_grid": mean_grid, "std_grid": std_grid,
        "X_all": X_all_raw, "y_all": y_all,
    }


class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                     learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_svgp(X_train, y_train, n_inducing=N_INDUCING, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    km = KMeans(n_clusters=n_inducing, random_state=SEED, n_init=10).fit(X_train)
    inducing_points = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(DEVICE)

    model = SVGPModel(inducing_points).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_t.size(0))

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_t, y_t),
                                          batch_size=batch_size, shuffle=True)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        mean_epoch_loss = epoch_loss / len(loader)
        loss_history.append(mean_epoch_loss)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={mean_epoch_loss:.4f}")

    converged, rel_change = check_convergence(loss_history)
    status = "converged" if converged else "NOT clearly converged"
    print(f"    Convergence check: {status} "
          f"(relative change over last {CONVERGENCE_WINDOW} epochs = {rel_change:.4f}, tolerance = {CONVERGENCE_REL_TOL})")
    if not converged:
        print(f"    -> consider increasing EPOCHS (currently {epochs}) for this gene.")

    return model, likelihood, loss_history, converged


def check_convergence(loss_history, window=CONVERGENCE_WINDOW, rel_tol=CONVERGENCE_REL_TOL):
    if len(loss_history) < window + 1:
        return False, np.nan
    recent = np.array(loss_history[-window:])
    rel_change = abs(recent[0] - recent[-1]) / (abs(recent[0]) + 1e-8)
    return rel_change <= rel_tol, rel_change


def predict_svgp(model, likelihood, X_query, batch_size=4096):
    model.eval()
    likelihood.eval()
    X_t = torch.tensor(X_query, dtype=torch.float32).to(DEVICE)
    means, stds = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X_t), batch_size):
            pred = likelihood(model(X_t[i:i + batch_size]))
            means.append(pred.mean.cpu().numpy())
            stds.append(pred.stddev.cpu().numpy())
    return np.concatenate(means), np.concatenate(stds)


def svgp_permutation_test(y_test_orig, y_pred, n_perm=SVGP_PERM, seed=SEED):
    # retraining the SVGP for every permutation isn't tractable at full
    # dataset size, so this just permutes the pairing between predictions
    # and labels instead -- weaker than the Part A test, but runs at the
    # same n as Moran's I
    rng = np.random.RandomState(seed)
    observed_r2 = r2_score(y_test_orig, y_pred)
    n = len(y_test_orig)
    null_r2 = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null_r2[i] = r2_score(y_test_orig, y_pred[perm])
    p_value = (np.sum(null_r2 >= observed_r2) + 1) / (n_perm + 1)
    return {"observed_r2": observed_r2, "null_r2_mean": float(np.mean(null_r2)),
            "null_r2_std": float(np.std(null_r2)), "p_value": p_value}


def run_svgp(adata, coords, gene, description):
    print("\n" + "=" * 70)
    print(f"PART B (sparse variational GP, full n={coords.shape[0]}) — GENE: {gene}")
    print("=" * 70)

    expr = get_gene_expression(adata, gene)
    coord_scaler = StandardScaler().fit(coords)
    expr_scaler = StandardScaler().fit(expr.reshape(-1, 1))
    X_scaled = coord_scaler.transform(coords)
    y_scaled = expr_scaler.transform(expr.reshape(-1, 1)).flatten()

    train_idx, test_idx = spatial_block_split(coords, test_size=TEST_SIZE, n_blocks=N_BLOCKS_B, seed=SEED)
    X_train, y_train = X_scaled[train_idx], y_scaled[train_idx]
    X_test, y_test = X_scaled[test_idx], y_scaled[test_idx]
    print(f"  (spatial blocks: train={len(train_idx)}, test={len(test_idx)})")

    print(f"[{gene}] Training SVGP ({N_INDUCING} inducing points, {EPOCHS} epochs, device={DEVICE})...")
    model, likelihood, loss_history, converged = train_svgp(X_train, y_train)

    y_pred_s, y_std_s = predict_svgp(model, likelihood, X_test)
    y_pred = expr_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_std = y_std_s * expr_scaler.scale_[0]
    y_test_orig = expr_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    r, p = _safe_pearsonr(y_test_orig, y_pred)
    p_str = f"{p:.2e}" if not np.isnan(p) else "NaN"
    print(f"[{gene}] SVGP held-out: RMSE={rmse:.4f}  R^2={r2:.3f}  r={r:.3f} (p={p_str})")

    errors = np.abs(y_test_orig - y_pred)
    within_1sigma = np.mean(errors <= y_std)
    within_2sigma = np.mean(errors <= 2 * y_std)
    print(f"  Calibration: 1sigma={within_1sigma:.3f} (exp ~0.68), 2sigma={within_2sigma:.3f} (exp ~0.95)")

    _, rmse_lo, rmse_hi = bootstrap_ci(y_test_orig, y_pred, _rmse)
    _, r2_lo, r2_hi = bootstrap_ci(y_test_orig, y_pred, r2_score)
    _, r_lo, r_hi = bootstrap_ci(y_test_orig, y_pred, _pearson_r_metric)
    print(f"  Bootstrap 95% CI: RMSE [{rmse_lo:.4f}, {rmse_hi:.4f}]  R^2 [{r2_lo:.4f}, {r2_hi:.4f}]  r [{r_lo:.4f}, {r_hi:.4f}]")

    print(f"  Post-hoc permutation test at full n ({SVGP_PERM} permutations, no retraining)...")
    perm_result = svgp_permutation_test(y_test_orig, y_pred, n_perm=SVGP_PERM, seed=SEED)
    print(f"    Observed R^2={perm_result['observed_r2']:.4f}  "
          f"Null R^2={perm_result['null_r2_mean']:.4f} +/- {perm_result['null_r2_std']:.4f}  "
          f"p={perm_result['p_value']:.4g}")

    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO (mean over batches)")
    plt.title(f"{gene}: SVGP training loss ({'converged' if converged else 'NOT converged'})")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_{gene}_svgp_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    xx, yy, grid_pts, mask = make_grid(coords, res=GRID_RES)
    grid_scaled = coord_scaler.transform(grid_pts)
    mean_s, std_s = predict_svgp(model, likelihood, grid_scaled)
    mean_pred = expr_scaler.inverse_transform(mean_s.reshape(-1, 1)).flatten()
    std_pred = std_s * expr_scaler.scale_[0]

    mean_grid = np.full(grid_pts.shape[0], np.nan)
    mean_grid[mask] = mean_pred[mask]
    mean_grid = mean_grid.reshape(xx.shape)
    std_grid = np.full(grid_pts.shape[0], np.nan)
    std_grid[mask] = std_pred[mask]
    std_grid = std_grid.reshape(xx.shape)

    return {
        "gene": gene, "rmse_svgp": rmse, "r2_svgp": r2, "pearson_r_svgp": r,
        "calib_1sigma_svgp": within_1sigma, "calib_2sigma_svgp": within_2sigma,
        "rmse_svgp_ci_lo": rmse_lo, "rmse_svgp_ci_hi": rmse_hi,
        "r2_svgp_ci_lo": r2_lo, "r2_svgp_ci_hi": r2_hi,
        "pearson_r_svgp_ci_lo": r_lo, "pearson_r_svgp_ci_hi": r_hi,
        "svgp_converged": converged,
        "svgp_perm_observed_r2": perm_result["observed_r2"],
        "svgp_perm_null_r2_mean": perm_result["null_r2_mean"],
        "svgp_perm_p_value": perm_result["p_value"],
        "n_total": coords.shape[0],
        "grid": (xx, yy, mask), "mean_grid": mean_grid, "std_grid": std_grid,
    }


def compare_methods(gene, exact_res, svgp_res):
    print(f"\n[{gene}] (Part C) Cross-method consistency — exact GP (n=2000) vs SVGP (full data)")
    mean_a = exact_res["mean_grid"].flatten()
    mean_b = svgp_res["mean_grid"].flatten()
    valid = ~np.isnan(mean_a) & ~np.isnan(mean_b)

    std_a, std_b = np.std(mean_a[valid]), np.std(mean_b[valid])
    degenerate = bool(std_a < 1e-8 or std_b < 1e-8)
    weak_agreement = False

    if degenerate:
        print(f"  WARNING: one of the predicted surfaces is (near-)constant "
              f"(exact GP std={std_a:.2e}, SVGP std={std_b:.2e}) -> cross-method correlation undefined.")
        r, p = np.nan, np.nan
    else:
        r, p = pearsonr(mean_a[valid], mean_b[valid])
        print(f"  Spatial map correlation (exact GP vs SVGP): r={r:.3f} (p={p:.2e}, n={valid.sum()} grid cells)")
        if abs(r) < WEAK_AGREEMENT_R:
            weak_agreement = True
            print(f"  NOTE: |r|={abs(r):.3f} is below the weak-agreement threshold ({WEAK_AGREEMENT_R}) -- "
                  f"the two methods aren't agreeing well on this gene's spatial pattern.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    xx, yy, _ = exact_res["grid"]
    im0 = axes[0].imshow(exact_res["mean_grid"], origin="lower", cmap="viridis",
                          extent=[xx.min(), xx.max(), yy.min(), yy.max()])
    axes[0].set_title("Exact GP (n=2000)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(svgp_res["mean_grid"], origin="lower", cmap="viridis",
                          extent=[xx.min(), xx.max(), yy.min(), yy.max()])
    axes[1].set_title(f"SVGP (full n={svgp_res['n_total']:,})")
    plt.colorbar(im1, ax=axes[1])

    axes[2].scatter(mean_a[valid], mean_b[valid], alpha=0.4, s=10)
    axes[2].set_xlabel("Exact GP prediction")
    axes[2].set_ylabel("SVGP prediction")
    axes[2].set_title(f"Agreement: r={r:.3f}" if not degenerate else "Agreement: degenerate")
    plt.suptitle(f"{gene}: Exact GP vs SVGP consistency")
    plt.tight_layout()
    plt.savefig(f"{OUT_PREFIX}_{gene}_crossmethod.png", dpi=300, bbox_inches="tight")
    plt.close()

    return {"gene": gene, "crossmethod_r": r, "crossmethod_p": p,
            "crossmethod_degenerate": degenerate, "crossmethod_weak_agreement": weak_agreement}


def main():
    adata, coords = load_data()
    genes_present = [g for g in TARGET_GENES if g in adata.var_names]
    moran_df = moran_baseline(adata, genes_present)

    exact_summary, svgp_summary, cross_summary = [], [], []

    for gene in genes_present:
        desc = TARGET_GENES[gene]
        exact_res = run_exact_gp(adata, coords, gene, desc)
        svgp_res = run_svgp(adata, coords, gene, desc)
        cross_res = compare_methods(gene, exact_res, svgp_res)

        exact_summary.append({k: v for k, v in exact_res.items()
                               if k not in ("grid", "mean_grid", "std_grid", "X_all", "y_all",
                                            "robustness", "perm_p_values_per_split")})
        svgp_summary.append({k: v for k, v in svgp_res.items() if k not in ("grid", "mean_grid", "std_grid")})
        cross_summary.append(cross_res)

    exact_df = pd.DataFrame(exact_summary)
    svgp_df = pd.DataFrame(svgp_summary)
    cross_df = pd.DataFrame(cross_summary)

    # FDR correction is applied separately per family since Part A and
    # Part B operate at different sample sizes / test different nulls
    qvals_a, sig_a = benjamini_hochberg(exact_df["perm_p_value"].values, alpha=ALPHA_FDR)
    exact_df["perm_q_value"] = qvals_a
    exact_df["spatially_significant_fdr"] = sig_a

    qvals_b, sig_b = benjamini_hochberg(svgp_df["svgp_perm_p_value"].values, alpha=ALPHA_FDR)
    svgp_df["svgp_perm_q_value"] = qvals_b
    svgp_df["svgp_spatially_significant_fdr"] = sig_b

    combined = exact_df.merge(moran_df, on="gene", how="left").merge(
        svgp_df[["gene", "svgp_perm_p_value", "svgp_perm_q_value", "svgp_spatially_significant_fdr"]],
        on="gene", how="left")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nPart A (exact GP, n=2000, spatially blocked repeated splits, Fisher-combined permutation p across splits):")
    print(exact_df[["gene", "best_kernel", "rmse_gp", "rmse_gp_std", "r2", "r2_std", "pearson_r",
                     "n_degenerate_folds", "rmse_knn", "rmse_idw", "perm_observed_r2", "perm_p_value",
                     "perm_q_value", "spatially_significant_fdr"]].to_string(index=False))

    print("\nPart B (SVGP, full dataset, spatially blocked split, + full-n post-hoc permutation test):")
    print(svgp_df[["gene", "rmse_svgp", "r2_svgp", "pearson_r_svgp", "calib_1sigma_svgp", "calib_2sigma_svgp",
                    "svgp_converged", "svgp_perm_p_value", "svgp_perm_q_value",
                    "svgp_spatially_significant_fdr"]].to_string(index=False))

    print("\nPart C (cross-method agreement):")
    print(cross_df.to_string(index=False))

    print(f"\nMoran's I baseline (BH alpha={ALPHA_FDR}) alongside both permutation-test families:")
    print(combined[["gene", "I", "pval_norm", "perm_observed_r2", "perm_p_value", "perm_q_value",
                     "spatially_significant_fdr", "svgp_perm_p_value", "svgp_perm_q_value",
                     "svgp_spatially_significant_fdr"]].to_string(index=False))

    exact_df.to_csv(f"{OUT_PREFIX}_exactGP_summary.csv", index=False)
    svgp_df.to_csv(f"{OUT_PREFIX}_SVGP_summary.csv", index=False)
    cross_df.to_csv(f"{OUT_PREFIX}_crossmethod_summary.csv", index=False)
    combined.to_csv(f"{OUT_PREFIX}_combined_significance_summary.csv", index=False)

    print("\nSaved 4 summary CSVs and per-gene PNG figures.")
    print("Full pipeline complete.")


if __name__ == "__main__":
    main()
