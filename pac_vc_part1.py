"""Assignment 4, Part I: PAC learning and VC dimension. Run top to bottom."""
import itertools, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq, linprog
from scipy.spatial import ConvexHull
from numpy.polynomial.legendre import legvander
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
SEED = 0
ETA = 0.05

# ---------------------------------------------------------------- P1
def f_m(h, m, eta=ETA):
    """VC confidence term, eq. (1). Works with scalar or array h."""
    h = np.asarray(h, dtype=float)
    return np.sqrt((h * (np.log(2 * m / h) + 1) - np.log(eta / 4)) / m)

print("=== P1 ===")
plt.figure(figsize=(6, 4))
for m in (10**3, 10**4, 10**5):
    hs = np.unique(np.round(np.linspace(1 / m, 1, 200) * m).astype(int))
    hs = hs[hs >= 1]
    plt.plot(hs / m, f_m(hs, m), label=f"m = {m}")
plt.axhline(1, color="gray", ls="--", lw=0.8)
plt.xlabel("h / m"); plt.ylabel(r"$f_m(h)$")
plt.title(r"VC confidence term ($\eta=0.05$)"); plt.legend(); plt.tight_layout()
plt.savefig("p1a_vc_confidence.png", dpi=150); plt.close()

for m in (10**3, 10**4, 10**5):
    g = lambda r: f_m(r * m, m) - 1
    r_star = brentq(g, 1e-6, 1.0)
    print(f"m={m:>6}: f_m=1 at h/m = {r_star:.4f}")

m = 10**4
for remp in (0.0, 0.05):
    ok = [h for h in range(1, m + 1) if remp + f_m(h, m) <= 0.1]
    print(f"m=1e4, Remp={remp}: largest h with bound <= 0.1 is {max(ok) if ok else None}")

# ---------------------------------------------------------------- P2
print("\n=== P2 ===")
def separable(X, y):
    m, n = X.shape
    A_ub = -y[:, None] * np.hstack([X, np.ones((m, 1))])
    res = linprog(c=np.zeros(n + 1), A_ub=A_ub, b_ub=-np.ones(m),
                  bounds=[(None, None)] * (n + 1), method="highs")
    return res.status == 0

def bad_labelings(X):
    """All non-separable labelings (both y and -y listed)."""
    bad = []
    for y in itertools.product([-1, 1], repeat=len(X)):
        if not separable(X, np.array(y, dtype=float)):
            bad.append(y)
    return bad

def can_shatter(X):
    # y and -y are equally (non)separable, so fix y_0 = +1 and stop at first failure
    for rest in itertools.product([-1, 1], repeat=len(X) - 1):
        y = np.array((1,) + rest, dtype=float)
        if not separable(X, y):
            return False
    return True

print("(a) triangle:", can_shatter(np.array([[0, 0], [1, 0], [0, 1.]])))
print("(a) collinear:", can_shatter(np.array([[0, 0], [1, 0], [2, 0.]])))

rng = np.random.default_rng(SEED)
print("(b) fraction shattered (1000 sets each)")
print(f"{'n':>2} | {'n+1 points':>10} | {'n+2 points':>10}")
for n in (2, 3, 4, 5):
    f1 = np.mean([can_shatter(rng.standard_normal((n + 1, n))) for _ in range(1000)])
    f2 = np.mean([can_shatter(rng.standard_normal((n + 2, n))) for _ in range(1000)])
    print(f"{n:>2} | {f1:>10.3f} | {f2:>10.3f}")

# (c) four random points in R^2
rng = np.random.default_rng(1)
X4 = rng.standard_normal((4, 2))
bad = bad_labelings(X4)
hull = ConvexHull(X4)
case = "triangle + interior point" if len(hull.vertices) == 3 else "convex quadrilateral"
print("(c) case:", case, "| hull vertex order:", list(hull.vertices))
print("    non-separable labelings:", bad)
y = np.array(bad[0])
plt.figure(figsize=(4, 4))
plt.scatter(*X4[y == 1].T, c="tab:red", s=80, label="+1")
plt.scatter(*X4[y == -1].T, c="tab:blue", s=80, label="-1")
for i, p in enumerate(X4):
    plt.annotate(str(i), p, textcoords="offset points", xytext=(6, 6))
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.title(f"Non-separable labeling {tuple(y)}"); plt.legend(); plt.tight_layout()
plt.savefig("p2c_nonseparable.png", dpi=150); plt.close()

# (d) rectangles
def rect_realizable(X, y):
    pos = X[y == 1]
    if len(pos) == 0:
        return True
    lo, hi = pos.min(0), pos.max(0)
    neg = X[y == -1]
    inside = np.all((neg >= lo) & (neg <= hi), axis=1)
    return not inside.any()

def rect_can_shatter(X):
    return all(rect_realizable(X, np.array(y))
               for y in itertools.product([-1, 1], repeat=len(X)))

diamond = np.array([[1, 0], [-1, 0], [0, 1], [0, -1.]])
print("(d) diamond shattered by rectangles:", rect_can_shatter(diamond))
rng = np.random.default_rng(2)
cnt = sum(rect_can_shatter(rng.standard_normal((5, 2))) for _ in range(1000))
print("(d) random 5-point sets shattered:", cnt, "/ 1000")

# ---------------------------------------------------------------- P3
print("\n=== P3 ===")
LO, HI = np.array([0.2, 0.3]), np.array([0.7, 0.6])
AREA = np.prod(HI - LO)  # 0.15

def trial_errors(m, T, rng):
    X = rng.random((T, m, 2))
    pos = np.all((X >= LO) & (X <= HI), axis=2)           # (T, m)
    anyp = pos.any(1)
    big = np.where(pos[..., None], X, np.inf).min(1)       # (T, 2) lower corner
    sml = np.where(pos[..., None], X, -np.inf).max(1)      # (T, 2) upper corner
    area = np.where(anyp, np.prod(sml - big, axis=1), 0.0)
    return AREA - area

rng = np.random.default_rng(SEED)
ms = [10, 20, 50, 100, 200, 500, 1000]
eps = delta = 0.05
mean_err, p_bad = [], []
for m in ms:
    R = trial_errors(m, 2000, rng)
    mean_err.append(R.mean()); p_bad.append(np.mean(R > eps))
    print(f"m={m:>4}: mean R={R.mean():.5f}  P(R>0.05)={p_bad[-1]:.4f}")

m_theory = int(np.ceil(4 / eps * np.log(4 / delta)))
print("theoretical m:", m_theory)

scan = list(range(10, 400, 10)); pscan = []
for m in scan:
    pscan.append(np.mean(trial_errors(m, 5000, rng) > eps))
m_emp = next(m for m, p in zip(scan, pscan) if p <= delta)
print(f"smallest empirical m (step 10) with P<=0.05: {m_emp}; "
      f"pessimism factor ~ {m_theory / m_emp:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(ms, mean_err, "o-")
ax[0].set_xlabel("m"); ax[0].set_ylabel("mean true error R")
ax[0].set_title("Rectangle learner: mean error")
ax[1].semilogx(ms, p_bad, "o-", label="empirical")
ax[1].axvline(m_theory, color="r", ls="--", label=f"PAC bound m={m_theory}")
ax[1].axhline(delta, color="gray", ls=":", label=r"$\delta=0.05$")
ax[1].set_xlabel("m"); ax[1].set_ylabel(r"$\hat P(R>\varepsilon)$, $\varepsilon=0.05$")
ax[1].set_title("Failure probability"); ax[1].legend()
plt.tight_layout(); plt.savefig("p3_rectangles.png", dpi=150); plt.close()

slope, _ = np.polyfit(np.log(ms), np.log(mean_err), 1)
print(f"fitted slope: {slope:.3f}  (R ~ m^{-abs(slope):.2f})")

# ---------------------------------------------------------------- P4
print("\n=== P4 ===")
rng = np.random.default_rng(SEED)
def make(m):
    x = rng.uniform(-1, 1, m)
    y = np.where(np.sin(3 * np.pi * x) > 0, 1, -1)
    flip = rng.random(m) < 0.1
    return x, np.where(flip, -y, y)
x_tr, y_tr = make(200); x_te, y_te = make(10**4)
feats = lambda x, d: legvander(x, d)[:, 1:]

ds = list(range(1, 16))
emp, bound, test, cv = [], [], [], []
for d in ds:
    clf = LinearSVC(C=1e4, max_iter=200000)
    clf.fit(feats(x_tr, d), y_tr)
    tr = np.mean(np.where(clf.decision_function(feats(x_tr, d)) > 0, 1, -1) != y_tr)
    te = np.mean(np.where(clf.decision_function(feats(x_te, d)) > 0, 1, -1) != y_te)
    cvs = 1 - cross_val_score(LinearSVC(C=1e4, max_iter=200000),
                              feats(x_tr, d), y_tr, cv=5).mean()
    emp.append(tr); test.append(te); cv.append(cvs)
    bound.append(tr + float(f_m(d + 1, 200)))
    print(f"d={d:>2}: train={tr:.3f} bound={bound[-1]:.3f} cv={cvs:.3f} test={te:.3f}")

print("SRM picks d =", ds[int(np.argmin(bound))],
      "| CV picks d =", ds[int(np.argmin(cv))],
      "| test-optimal d =", ds[int(np.argmin(test))])
print("min bound value:", min(bound))

plt.figure(figsize=(6.5, 4.5))
plt.plot(ds, emp, "o-", label=r"training error $R_{emp}$")
plt.plot(ds, bound, "s-", label=r"VC bound $R_{emp}+f_m(d+1)$")
plt.plot(ds, test, "^-", label="test error")
plt.plot(ds, cv, "d--", label="5-fold CV error", alpha=0.7)
plt.axhline(0.5, color="gray", ls=":", label="random guessing")
plt.axhline(0.1, color="green", ls=":", label="Bayes error 0.1")
plt.xlabel("polynomial degree d"); plt.ylabel("error")
plt.title("SRM vs cross-validation"); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("p4_srm_cv.png", dpi=150); plt.close()

knn = KNeighborsClassifier(n_neighbors=1).fit(x_tr[:, None], y_tr)
print(f"1-NN: train={np.mean(knn.predict(x_tr[:, None]) != y_tr):.3f} "
      f"test={np.mean(knn.predict(x_te[:, None]) != y_te):.3f}")
