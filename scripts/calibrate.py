"""Calibration and abstention: make the confidence numbers mean something.

The model ranks well (AUC 0.989) but decides badly (accuracy 0.856) because the
0.5 threshold was never right for this distribution. Two fixes, in order:

  temperature scaling  rescales logits so p(AIGC) is numerically honest
  selective prediction two thresholds instead of one; refuse the middle

Split conformal turns the second into a guarantee: on the images we answer,
error <= alpha, distribution-free and finite-sample.

Thresholds are fitted on the VAL split and applied to the held-out benchmark.
Fitting them on the benchmark would be tuning on the number we report.

    python scripts/calibrate.py --checkpoint checkpoints/svd_r32_film.pt \\
        --calib-manifest data/extracted/manifest.csv \\
        --test-manifest data/heldout/manifest.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.components.peft import apply_peft          # noqa: E402
from src.models.detector import Detector            # noqa: E402
from src.utils import dataset as data               # noqa: E402
from src.utils.dataset import CellDataset           # noqa: E402
from src.utils.features import pick_device          # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def logits_for(model, df, preprocess, root, device, bs, workers, pre):
    loader = DataLoader(CellDataset(df, preprocess, "clean", root, pre),
                        batch_size=bs, num_workers=workers)
    zs, ys = [], []
    for x, y in loader:
        zs.append(model(x.to(device))["logit"].float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single scalar T minimising NLL. Cannot change the ranking, only the
    spread -- so AUC is untouched while the probabilities become honest."""
    z = torch.tensor(logits, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z / log_T.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp())


def conformal_threshold(conf: np.ndarray, correct: np.ndarray, alpha: float) -> float:
    """Confidence cutoff such that error among answered samples is <= alpha.

    Sort by confidence, walk down, stop where the cumulative error rate would
    exceed alpha. The (n+1) correction makes the guarantee hold on unseen data
    rather than only on this calibration sample.
    """
    order = np.argsort(-conf)
    err = (~correct[order]).astype(float)
    cum_err_rate = np.cumsum(err) / np.arange(1, len(err) + 1)

    ok = np.where(cum_err_rate <= alpha)[0]
    if len(ok) == 0:
        return 1.01                                  # abstain on everything
    n = len(conf)
    k = min(int(np.ceil((n + 1) * (1 - alpha))) - 1, ok[-1])
    return float(conf[order][max(k, 0)])


def risk_coverage(conf: np.ndarray, correct: np.ndarray, n_points: int = 40):
    """Sweep the abstention threshold; report (coverage, risk) at each."""
    rows = []
    for q in np.linspace(0.0, 0.95, n_points):
        thr = np.quantile(conf, q)
        answered = conf >= thr
        if answered.sum() < 20:
            continue
        rows.append({"threshold": float(thr),
                     "coverage": float(answered.mean()),
                     "risk": float((~correct[answered]).mean())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--calib-manifest", required=True, type=Path)
    ap.add_argument("--test-manifest", required=True, type=Path)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="cap images per split, for smoke tests")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or pick_device()
    root = data.data_root()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = ckpt["args"]

    bb = timm.create_model(targs["backbone"], pretrained=True, num_classes=0)
    bb = apply_peft(bb, mode=targs["arm"], r=targs["rank"])
    model = Detector(bb, dim=bb.num_features, conditioner=targs["conditioner"])
    model.load_state_dict(ckpt["state_dict"])
    model = model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model.backbone)
    preprocess = timm.data.create_transform(**cfg, is_training=False)

    calib_df = pd.read_csv(args.calib_manifest)
    calib_df = calib_df[calib_df.split == data.SPLIT_VAL]
    test_df = pd.read_csv(args.test_manifest)
    test_df = test_df[test_df.split == data.SPLIT_HELDOUT]
    if args.limit:
        # Stratified: the manifests are class-ordered, so head() would take one
        # class and every metric would be degenerate.
        def sample(df, n):
            return (df.groupby("y", group_keys=False)
                      .apply(lambda g: g.head(max(1, n // 2))).reset_index(drop=True))
        calib_df, test_df = sample(calib_df, args.limit), sample(test_df, args.limit)
    # Rows addressing a parquet row or zip member still need bias alignment;
    # pre-extracted files already have it baked in.
    def is_pre(df):
        return not df.path.iloc[0].count("#") and "::" not in df.path.iloc[0]
    pre_cal, pre_test = is_pre(calib_df), is_pre(test_df)
    logger.info("calibrate on %d val images (pre_extracted=%s), report on %d "
                "benchmark images (pre_extracted=%s)",
                len(calib_df), pre_cal, len(test_df), pre_test)

    z_cal, y_cal = logits_for(model, calib_df, preprocess, root, device,
                              args.batch_size, args.workers, pre_cal)
    z_test, y_test = logits_for(model, test_df, preprocess, root, device,
                                args.batch_size, args.workers, pre_test)

    T = fit_temperature(z_cal, y_cal)
    logger.info("temperature T = %.4f", T)

    def probs(z):
        return 1.0 / (1.0 + np.exp(-z / T))

    p_cal, p_test = probs(z_cal), probs(z_test)

    # Threshold that maximises accuracy on calibration data, applied to test.
    ths = np.quantile(p_cal, np.linspace(0.01, 0.99, 199))
    best_thr = float(ths[np.argmax([((p_cal > t) == y_cal).mean() for t in ths])])

    rows = [{
        "setting": "uncalibrated @0.5",
        "coverage": 1.0,
        "accuracy": float(((p_test > 0.5) == y_test).mean()),
        "risk": float(((p_test > 0.5) != y_test).mean()),
        "auc": roc_auc_score(y_test, p_test),
    }, {
        "setting": f"calibrated @{best_thr:.3f}",
        "coverage": 1.0,
        "accuracy": float(((p_test > best_thr) == y_test).mean()),
        "risk": float(((p_test > best_thr) != y_test).mean()),
        "auc": roc_auc_score(y_test, p_test),
    }]

    # Selective prediction: confidence is distance from the decision boundary.
    conf_cal = np.abs(p_cal - best_thr)
    conf_test = np.abs(p_test - best_thr)
    correct_cal = (p_cal > best_thr) == y_cal
    correct_test = (p_test > best_thr) == y_test

    for alpha in args.alphas:
        thr = conformal_threshold(conf_cal, correct_cal, alpha)
        answered = conf_test >= thr
        if answered.sum() == 0:
            continue
        rows.append({
            "setting": f"abstain @ target risk {alpha:.0%}",
            "coverage": float(answered.mean()),
            "accuracy": float(correct_test[answered].mean()),
            "risk": float((~correct_test[answered]).mean()),
            "auc": roc_auc_score(y_test[answered], p_test[answered])
            if len(np.unique(y_test[answered])) > 1 else float("nan"),
        })

    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    name = args.checkpoint.stem
    out.to_csv(f"results/tables/calibration_{name}.csv", index=False)
    risk_coverage(conf_test, correct_test).to_csv(
        f"results/tables/risk_coverage_{name}.csv", index=False)
    np.savez_compressed(root / "cache" / f"scores_{name}.npz",
                        p_cal=p_cal, y_cal=y_cal, p_test=p_test, y_test=y_test,
                        temperature=T, threshold=best_thr)
    logger.info("wrote results/tables/calibration_%s.csv and risk_coverage_%s.csv",
                name, name)


if __name__ == "__main__":
    main()
