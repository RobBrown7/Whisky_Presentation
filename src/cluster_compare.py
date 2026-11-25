"""Compare k-means, EM (Gaussian mixture) and fuzzy c-means on whisky flavour data.

Version: 0.1.0
Usage:   python src/cluster_compare.py   (run from repo root)
Outputs: reports/clustering/*.csv, *.png, summary.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as PolyPath
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

VERSION = "0.1.0"
SEED = 42
K_RANGE = range(2, 9)          # fixed bounds (silhouette needs k >= 2)
FCM_MAX_ITER = 300             # fixed bound for fuzzy c-means loop
FCM_M = 1.2                    # fuzzifier; m=2 collapses to uniform 0.2 in 12-D (tested)
GMM_PCS = 3                    # GMM fit in PCA space: 86 rows cannot support 12-D covariances
GMM_COV = "tied"
FLAVOURS = ["Body", "Sweetness", "Smoky", "Medicinal", "Tobacco", "Honey",
            "Spicy", "Winey", "Nutty", "Malty", "Fruity", "Floral"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "clustering"


def load_data() -> pd.DataFrame:
    """Raw 0-4 flavour scores joined to decimal lat/lon from Whisky_DF9."""
    raw = pd.read_csv(ROOT / "whisky.csv")
    geo = pd.read_csv(ROOT / "Whisky_DF9.csv", encoding="utf-8-sig")[["RowID", "Name", "Latitude", "Longitude"]]
    df = raw[["RowID"] + FLAVOURS].merge(geo, on="RowID", how="inner")
    assert len(df) == 86, f"expected 86 distilleries, got {len(df)}"
    assert df[FLAVOURS].isna().sum().sum() == 0, "missing flavour scores"
    return df


def assign_region(df: pd.DataFrame) -> pd.Series:
    """Point-in-polygon lookup; nearest polygon vertex as fallback for coastal points."""
    poly = pd.read_csv(ROOT / "whisky_regions_polygons_2.csv")
    assert {"Region", "PolygonID", "Latitude", "Longitude"} <= set(poly.columns)
    pts = df[["Longitude", "Latitude"]].to_numpy()
    region = np.array([None] * len(df), dtype=object)
    for (reg, _pid), grp in poly.sort_values("PointOrder").groupby(["Region", "PolygonID"]):
        inside = PolyPath(grp[["Longitude", "Latitude"]].to_numpy()).contains_points(pts)
        region[inside & (region == None)] = reg  # noqa: E711
    verts = poly[["Longitude", "Latitude"]].to_numpy()
    for i in np.where(region == None)[0]:  # noqa: E711
        region[i] = poly.Region.iloc[np.argmin(((verts - pts[i]) ** 2).sum(axis=1))]
    assert (region != None).all()  # noqa: E711
    return pd.Series(region, index=df.index, name="Region")


def fuzzy_cmeans(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Bezdek fuzzy c-means. Returns (centres k x d, membership n x k)."""
    assert x.ndim == 2 and 1 < k < len(x)
    rng = np.random.default_rng(SEED)
    u = rng.dirichlet(np.ones(k), size=len(x))
    centres = np.zeros((k, x.shape[1]))
    for _ in range(FCM_MAX_ITER):
        um = u ** FCM_M
        centres = um.T @ x / um.sum(axis=0)[:, None]
        dist = np.linalg.norm(x[:, None, :] - centres[None], axis=2) + 1e-12
        new_u = 1.0 / (dist ** (2 / (FCM_M - 1)) * (1.0 / dist ** (2 / (FCM_M - 1))).sum(axis=1, keepdims=True))
        if np.abs(new_u - u).max() < 1e-6:
            u = new_u
            break
        u = new_u
    assert np.allclose(u.sum(axis=1), 1.0)
    return centres, u


def fit_gmm(z: np.ndarray, k: int) -> GaussianMixture:
    """EM-fitted Gaussian mixture on PCA-reduced data."""
    assert z.shape[1] == GMM_PCS and k >= 1
    gm = GaussianMixture(k, covariance_type=GMM_COV, n_init=10, random_state=SEED).fit(z)
    assert gm.converged_, f"GMM k={k} did not converge"
    return gm


def scan_k(x: np.ndarray, z: np.ndarray) -> pd.DataFrame:
    """k-means silhouette (full 12-D) and GMM BIC (PCA space) across K_RANGE, plus k=1 BIC baseline."""
    rows = [{"k": 1, "kmeans_inertia": float(((x - x.mean(0)) ** 2).sum()), "kmeans_silhouette": np.nan,
             "gmm_bic": fit_gmm(z, 1).bic(z)}]
    for k in K_RANGE:
        km = KMeans(k, n_init=20, random_state=SEED).fit(x)
        rows.append({"k": k, "kmeans_inertia": km.inertia_,
                     "kmeans_silhouette": silhouette_score(x, km.labels_), "gmm_bic": fit_gmm(z, k).bic(z)})
    out = pd.DataFrame(rows)
    assert len(out) == len(K_RANGE) + 1
    return out


def entropy(p: np.ndarray) -> np.ndarray:
    """Normalised membership entropy: 0 = certain, 1 = evenly split."""
    assert p.ndim == 2
    h = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1) / np.log(p.shape[1])
    return h


def plot_results(x2: np.ndarray, df: pd.DataFrame, scan: pd.DataFrame) -> None:
    """Three-panel cluster map in PCA space plus the k-scan chart."""
    assert x2.shape == (len(df), 2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    panels = [("kmeans", "k-means (hard)", None), ("gmm", "EM / GMM (soft)", "gmm_conf"),
              ("fcm", "Fuzzy c-means (soft)", "fcm_conf")]
    for ax, (col, title, conf) in zip(axes, panels):
        size = 30 + 150 * (df[conf] if conf else 1)
        ax.scatter(x2[:, 0], x2[:, 1], c=df[col], cmap="tab10", s=size, alpha=0.75, edgecolor="k", lw=0.4)
        if conf:
            amb = df[conf] < 0.6
            ax.scatter(x2[amb, 0], x2[amb, 1], facecolor="none", edgecolor="red", s=260, lw=1.5)
        ax.set_title(title + ("\nred ring = max membership < 0.6" if conf else ""))
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    for i, name in enumerate(df.Name):
        if name in {"Ardbeg", "Laphroig", "Lagavulin", "Macallan", "Glenfiddich", "Talisker", "Highland Park"}:
            axes[0].annotate(name, x2[i], fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "cluster_comparison_pca.png", dpi=150)
    fig2, a1 = plt.subplots(figsize=(7, 4))
    a1.plot(scan.k, scan.gmm_bic, "o-", label="GMM BIC (lower = better)")
    a2 = a1.twinx()
    a2.plot(scan.k, scan.kmeans_silhouette, "s--", color="C1", label="k-means silhouette")
    a1.set_xlabel("k"); a1.set_ylabel("BIC"); a2.set_ylabel("silhouette")
    fig2.legend(loc="upper right"); fig2.tight_layout()
    fig2.savefig(OUT / "k_selection.png", dpi=150)
    plt.close("all")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    df["Region"] = assign_region(df)
    x = df[FLAVOURS].to_numpy(float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    pca = PCA(GMM_PCS, random_state=SEED).fit(x)
    z = pca.transform(x)
    scan = scan_k(x, z)
    k = 5  # match the original paper's elbow choice for a like-for-like comparison
    df["kmeans"] = KMeans(k, n_init=20, random_state=SEED).fit_predict(x)
    pg = fit_gmm(z, k).predict_proba(z)
    k_bic = int(scan.loc[scan.gmm_bic.idxmin(), "k"])
    p2 = fit_gmm(z, k_bic).predict_proba(z)
    df[f"gmm_bestk{k_bic}"], df[f"gmm_bestk{k_bic}_conf"] = p2.argmax(1), p2.max(1)
    _, pf = fuzzy_cmeans(x, k)
    df["gmm"], df["gmm_conf"], df["gmm_entropy"] = pg.argmax(1), pg.max(1), entropy(pg)
    df["fcm"], df["fcm_conf"], df["fcm_entropy"] = pf.argmax(1), pf.max(1), entropy(pf)
    metrics = pd.DataFrame({
        "method": ["kmeans", "gmm", "fcm"],
        "silhouette": [silhouette_score(x, df[m]) for m in ["kmeans", "gmm", "fcm"]],
        "ARI_vs_kmeans": [1.0, adjusted_rand_score(df.kmeans, df.gmm), adjusted_rand_score(df.kmeans, df.fcm)],
        "ARI_vs_region": [adjusted_rand_score(df.Region, df[m]) for m in ["kmeans", "gmm", "fcm"]],
        "n_ambiguous_lt0.6": [0, int((df.gmm_conf < 0.6).sum()), int((df.fcm_conf < 0.6).sum())],
    })
    scan.to_csv(OUT / "k_scan.csv", index=False)
    metrics.to_csv(OUT / "method_metrics.csv", index=False)
    df.drop(columns=FLAVOURS).to_csv(OUT / "distillery_memberships.csv", index=False)
    plot_results(PCA(2, random_state=SEED).fit_transform(x), df, scan)
    print(f"cluster_compare v{VERSION}\n", scan.round(3).to_string(index=False), "\n")
    print(metrics.round(3).to_string(index=False), "\n")
    print("Region counts:", df.Region.value_counts().to_dict())
    print(f"BIC-optimal k = {k_bic}; PCA var explained by {GMM_PCS} PCs = {pca.explained_variance_ratio_.sum():.2f}")
    prof = df.groupby(f"gmm_bestk{k_bic}")[FLAVOURS].mean().round(2)
    prof["n"] = df[f"gmm_bestk{k_bic}"].value_counts().sort_index()
    prof.to_csv(OUT / "bestk_profiles.csv")
    print(prof.to_string())
    print(pd.crosstab(df.Region, df[f"gmm_bestk{k_bic}"]).to_string())
    amb = df.sort_values("fcm_conf")[["Name", "Region", "kmeans", "gmm", "gmm_conf", "fcm", "fcm_conf"]].head(12)
    print(amb.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
