"""
Graph 6 — Application to Real CMS Collision Data
=================================================
Applies the trained BDT to real collision data.
Produces the key plot: mT distribution of W candidates
selected by the classifier from real data.

This is the MOST IMPORTANT plot for your thesis —
it shows your model working on real physics data.

Run: python3 graph6_real_data.py
"""

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
REAL_COLOR = "#66BB6A"   # green for real data
MC_COLOR   = "#4FC3F7"   # blue for MC prediction
ACCENT     = "#FFD54F"

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Load trained model
# ═══════════════════════════════════════════════════════════════════
print("Loading trained model...")
try:
    model_data = joblib.load("w_boson_classifier2.pkl")
except FileNotFoundError:
    model_data = joblib.load("w_classifier_balanced.pkl")

model    = model_data["model"]
features = model_data["features"]
print(f"Model loaded. Features: {features}")
print(f"Model AUC: {model_data['auc']:.4f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Load and process real data ROOT file
# ═══════════════════════════════════════════════════════════════════
# Change this to your real collision data ROOT file path
REAL_DATA_FILE = "real_collision_data.root"
TREE_NAME = "Events"

def lead(arr):
    return ak.to_numpy(arr[:, 0])

def calc_delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi >  np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return dphi

def process_real_data(filepath):
    """Process a real data ROOT file into a DataFrame."""
    print(f"\nOpening: {filepath}")
    f    = uproot.open(filepath)
    keys = list(f.keys())
    tree_name = None
    for k in keys:
        if k.split(";")[0] in ["Events","events","tree"]:
            tree_name = k
            break
    if tree_name is None:
        tree_name = keys[0]

    tree = f[tree_name]
    print(f"Tree: {tree_name}  Events: {tree.num_entries:,}")

    # Detect format
    tree_keys = tree.keys()
    btag_br   = ("Jet_btagDeepB" if "Jet_btagDeepB" in tree_keys
                  else "Jet_btag" if "Jet_btag" in tree_keys
                  else None)
    has_sig   = "MET_significance" in tree_keys

    branches = [
        "Muon_pt", "Muon_eta", "Muon_phi",
        "Muon_pfRelIso04_all", "Muon_tightId",
        "Muon_dxy", "Muon_dz",
        "MET_pt", "MET_phi",
    ]
    if has_sig:   branches.append("MET_significance")
    if btag_br:   branches.append(btag_br)
    branches += ["Jet_pt"]

    arrays = tree.arrays(branches, library="ak")
    mask   = ak.num(arrays["Muon_pt"]) >= 1
    arrays = arrays[mask]
    n_evts = ak.num(arrays["Muon_pt"], axis=0)
    print(f"Events with ≥1 muon: {n_evts:,}")

    muon_pt  = lead(arrays["Muon_pt"])
    muon_phi = lead(arrays["Muon_phi"])
    met_pt   = ak.to_numpy(arrays["MET_pt"])
    met_phi  = ak.to_numpy(arrays["MET_phi"])
    dphi     = calc_delta_phi(muon_phi, met_phi)
    mT       = np.sqrt(2 * muon_pt * met_pt * (1 - np.cos(dphi)))

    n_jets   = ak.to_numpy(ak.sum(arrays["Jet_pt"] > 30, axis=1))

    if btag_br:
        has_j    = ak.num(arrays[btag_br]) > 0
        max_btag = np.where(
            ak.to_numpy(has_j),
            ak.to_numpy(ak.max(arrays[btag_br],
                               axis=1, mask_identity=True)), 0.0)
        max_btag = np.nan_to_num(max_btag, nan=0.0).astype(float)
    else:
        max_btag = np.zeros(n_evts)

    met_sig = (ak.to_numpy(arrays["MET_significance"])
               if has_sig else np.ones(n_evts) * 5.0)

    df = pd.DataFrame({
        "muon_pt"          : muon_pt,
        "muon_eta"         : lead(arrays["Muon_eta"]),
        "muon_iso"         : lead(arrays["Muon_pfRelIso04_all"]),
        "muon_dxy"         : lead(arrays["Muon_dxy"]),
        "muon_dz"          : lead(arrays["Muon_dz"]),
        "met_pt"           : met_pt,
        "met_significance" : met_sig,
        "delta_phi"        : dphi,
        "mT"               : mT,
        "n_jets"           : n_jets,
        "max_btag"         : max_btag,
    })

    # Apply preselection cuts
    cut = (
        (df["muon_pt"]        > 26)   &
        (df["muon_eta"].abs() < 2.4)  &
        (df["muon_iso"]       < 0.15) &
        (df["met_pt"]         > 25)   &
        (df["mT"]             > 40)   &
        (df["mT"]             < 130)
    )
    df = df[cut].reset_index(drop=True)
    print(f"After preselection: {len(df):,} events")
    return df

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Apply classifier
# ═══════════════════════════════════════════════════════════════════
if not os.path.exists(REAL_DATA_FILE):
    print(f"\n⚠️  Real data file not found: {REAL_DATA_FILE}")
    print("Creating DEMO with MC data to show plot structure...")
    # Demo mode: use MC signal as proxy for real data
    df_mc = pd.read_csv("w_boson_combined.csv")
    df_real = df_mc[features].fillna(0).copy()
    df_real["mT"] = df_mc["mT"].values
    demo_mode = True
else:
    df_real   = process_real_data(REAL_DATA_FILE)
    demo_mode = False

print("\nApplying classifier to data...")
X_real = df_real[features].fillna(0).values
df_real["bdt_score"] = model.predict_proba(X_real)[:, 1]
df_real["is_W"]      = (df_real["bdt_score"] > 0.5).astype(int)

n_total = len(df_real)
n_W     = df_real["is_W"].sum()
frac    = 100 * n_W / n_total

print(f"\nResults on {'DEMO (MC)' if demo_mode else 'REAL'} data:")
print(f"  Total events after preselection : {n_total:,}")
print(f"  W candidates (BDT > 0.5)        : {n_W:,} ({frac:.1f}%)")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Plots
# ═══════════════════════════════════════════════════════════════════
label = "DEMO (MC as proxy)" if demo_mode else "Real CMS Data"

fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle(f"BDT Classifier Applied to {label}\n"
             f"W→μν Candidate Selection",
             fontsize=15)

# ── Plot 1: BDT score distribution ──────────────────────────────
ax = axes[0, 0]
ax.hist(df_real["bdt_score"], bins=50, range=(0, 1),
        color=REAL_COLOR, alpha=0.8,
        histtype="stepfilled",
        label=f"All events ({n_total:,})")
ax.axvline(0.5, color=ACCENT, lw=2.5, linestyle="--",
           label="Default threshold (0.5)")
ax.axvspan(0.5, 1.0, alpha=0.1, color=ACCENT,
           label=f"W candidates ({n_W:,})")
ax.set_xlabel("BDT score  P(W event)", fontsize=12)
ax.set_ylabel("Events", fontsize=12)
ax.set_title(f"BDT Score Distribution — {label}", fontsize=11)
ax.legend(fontsize=10)
ax.text(0.55, 0.6,
        f"{frac:.1f}% of events\nselected as W",
        transform=ax.transAxes, fontsize=11,
        color=ACCENT, va="center")

# ── Plot 2: mT of ALL preselected events ────────────────────────
ax = axes[0, 1]
ax.hist(df_real["mT"], bins=50, range=(0, 130),
        color=REAL_COLOR, alpha=0.8,
        histtype="stepfilled",
        label=f"All preselected ({n_total:,})")
ax.axvspan(60, 100, alpha=0.15, color=ACCENT,
           label="W mass window 60–100 GeV")
ax.set_xlabel(r"$m_T$ [GeV]", fontsize=12)
ax.set_ylabel("Events", fontsize=12)
ax.set_title("mT — All Preselected Events\n"
             "(before BDT cut)", fontsize=11)
ax.legend(fontsize=10)

# ── Plot 3: mT of BDT-SELECTED W candidates ──────────────────────
ax = axes[1, 0]
w_candidates = df_real[df_real["is_W"] == 1]
ax.hist(w_candidates["mT"], bins=50, range=(0, 130),
        color=REAL_COLOR, alpha=0.8,
        histtype="stepfilled",
        label=f"W candidates ({n_W:,})")
ax.axvspan(60, 100, alpha=0.15, color=ACCENT,
           label="W mass window")
ax.axvline(80.4, color="#EF5350", lw=2, linestyle=":",
           label=r"$m_W = 80.4$ GeV (PDG)")
ax.set_xlabel(r"$m_T$ [GeV]", fontsize=12)
ax.set_ylabel("Events", fontsize=12)
ax.set_title("mT — BDT-Selected W Candidates\n"
             "(BDT score > 0.5)", fontsize=11)
ax.legend(fontsize=10)

# Annotate the Jacobian peak
peak_bin = np.histogram(w_candidates["mT"],
                         bins=50, range=(0, 130))[0]
peak_val  = np.argmax(peak_bin) * (130/50)
ax.annotate("Jacobian peak\n(W boson signature)",
            xy=(peak_val, max(peak_bin)),
            xytext=(peak_val - 30, max(peak_bin) * 0.8),
            fontsize=9, color=ACCENT,
            arrowprops=dict(arrowstyle="->",
                           color=ACCENT))

# ── Plot 4: mT at different thresholds ──────────────────────────
ax = axes[1, 1]
for thresh, color, lw in [(0.3, "#EF5350", 1.5),
                            (0.5, REAL_COLOR, 2.5),
                            (0.7, ACCENT, 1.5)]:
    sel  = df_real[df_real["bdt_score"] > thresh]
    n    = len(sel)
    ax.hist(sel["mT"], bins=50, range=(0, 130),
            density=True, histtype="step",
            color=color, lw=lw,
            label=f"BDT > {thresh}  ({n:,} events)")

ax.axvline(80.4, color="white", lw=1, linestyle=":",
           label=r"$m_W$ = 80.4 GeV")
ax.set_xlabel(r"$m_T$ [GeV]", fontsize=12)
ax.set_ylabel("Normalised events", fontsize=12)
ax.set_title("mT at Different BDT Thresholds\n"
             "(Normalised)", fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("graph6_real_data_results.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("\nSaved: graph6_real_data_results.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Print summary table for thesis
# ═══════════════════════════════════════════════════════════════════
print(f"\nTable for thesis — W candidates at different thresholds:")
print(f"{'Threshold':<12} {'Events passing':>16} "
      f"{'% of total':>12} {'mT peak (GeV)':>15}")
print("─" * 56)
for thr in [0.3, 0.5, 0.7, 0.9]:
    sel     = df_real[df_real["bdt_score"] > thr]
    n_sel   = len(sel)
    pct     = 100 * n_sel / n_total
    if len(sel) > 0:
        hist, edges = np.histogram(sel["mT"], bins=50,
                                   range=(40, 130))
        peak_mT = edges[np.argmax(hist)] + (edges[1]-edges[0])/2
    else:
        peak_mT = 0
    print(f"{thr:<12.1f} {n_sel:>16,} {pct:>12.1f}% "
          f"{peak_mT:>15.1f}")