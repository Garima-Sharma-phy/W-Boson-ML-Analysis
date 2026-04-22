"""
W Boson Analysis — ROOT to CSV Converter (v3)
==============================================
Extracts relevant branches for W → μν analysis
from a CMS NanoAOD ROOT file and saves as a flat CSV.

LABELING (v3 fix):
  - label        = EXACT + TAU VETO
                   Traces GenPart mother chain → pdgId=24 (W boson)
                   BUT excludes W → τ → μ events (tau in chain)
                   This is the CORRECT label for W mass measurement ✅

  - label_simple = SIMPLE (genPartFlav == 1, ~80-90% pure, for comparison only)
  - label_exact  = EXACT without tau veto (v2 version, kept for reference)

WHY TAU VETO:
  W → τ → μ has 3 neutrinos in final state (not 1)
  This smears MET and biases the mT distribution → bad for W mass fit
  So we exclude these events from signal.

Requirements:
    pip install uproot awkward numpy pandas

Usage:
    python root_to_csv_v3.py
    Change INPUT_FILE below to your ROOT file path.
"""

import uproot
import awkward as ak
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
INPUT_FILE  = "W2.root"        # ← change to your ROOT file path
OUTPUT_FILE = "w2.csv"
TREE_NAME   = "Events"

# ═══════════════════════════════════════════════════════
# STEP 1 — Open file
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("  W Boson ROOT → CSV Converter (v3 — Exact Labels + Tau Veto)")
print("=" * 60)

print(f"\n[1/7] Opening: {INPUT_FILE}")
file = uproot.open(INPUT_FILE)
tree = file[TREE_NAME]
print(f"      Total events in file: {tree.num_entries:,}")

# ═══════════════════════════════════════════════════════
# STEP 2 — Define branches to load
# ═══════════════════════════════════════════════════════
BRANCHES = [
    # Muon
    "nMuon",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_charge",
    "Muon_mass",
    "Muon_pfRelIso04_all",
    "Muon_tightId",
    "Muon_isGlobal",
    "Muon_isPFcand",
    "Muon_dxy",
    "Muon_dz",
    "Muon_genPartFlav",            # for simple label
    "Muon_genPartIdx",             # for exact label (links reco → gen)

    # MET
    "MET_pt",
    "MET_phi",
    "MET_significance",

    # Jets
    "nJet",
    "Jet_pt",
    "Jet_eta",
    "Jet_btagDeepB",

    # Generator truth (for exact label)
    "GenPart_pdgId",
    "GenPart_genPartIdxMother",

    # Event quality flags
    "Flag_goodVertices",
    "Flag_METFilters",
    "Flag_HBHENoiseFilter",
    "Flag_BadPFMuonFilter",

    # Triggers
    "HLT_IsoMu24",
    "HLT_IsoMu27",

    # Generator weight
    "genWeight",
]

# ═══════════════════════════════════════════════════════
# STEP 3 — Load arrays and select ≥1 muon events
# ═══════════════════════════════════════════════════════
print(f"\n[2/7] Loading branches...")
arrays = tree.arrays(BRANCHES, library="ak")

# Keep only events that have at least 1 muon
mask_1mu = ak.num(arrays["Muon_pt"]) >= 1
arrays   = arrays[mask_1mu]
n_events = ak.num(arrays["Muon_pt"], axis=0)
print(f"      Events with ≥1 muon: {n_events:,}")

# ═══════════════════════════════════════════════════════
# HELPER — extract leading (first) muon per event
# ═══════════════════════════════════════════════════════
def lead(arr):
    """Take the first element (leading muon) from each event."""
    return ak.to_numpy(arr[:, 0])

# ═══════════════════════════════════════════════════════
# STEP 4 — Labeling functions
# ═══════════════════════════════════════════════════════

def trace_mother_chain(gen_pdgId, gen_mother_idx, muon_gen_idx,
                       exclude_tau=True):
    """
    Trace the GenPart mother chain for each event's leading muon.

    Parameters:
        gen_pdgId      : GenPart_pdgId array
        gen_mother_idx : GenPart_genPartIdxMother array
        muon_gen_idx   : Muon_genPartIdx array
        exclude_tau    : If True, events with tau (pdgId=15) in the
                         decay chain are labeled 0 even if W is found.
                         This removes W → τ → μ events.

    Returns:
        numpy array of 0/1 labels

    Decay chain walk example (W → μ direct):
        muon → W(24) ✅ found W, no tau → label=1

    Decay chain walk example (W → τ → μ):
        muon → tau(15) → W(24)
        exclude_tau=True  → label=0 (tau in chain, bad for mT)
        exclude_tau=False → label=1 (W found, tau ignored)
    """
    results = []

    for ev in range(len(muon_gen_idx)):

        # No muons in this event
        if len(muon_gen_idx[ev]) == 0:
            results.append(0)
            continue

        # Gen particle index matched to the leading reco muon
        mu_idx = int(muon_gen_idx[ev][0])

        # Negative index = no gen match = fake muon
        if mu_idx < 0:
            results.append(0)
            continue

        # Walk up the decay chain
        found_W  = False
        has_tau  = False
        current  = mu_idx

        for _ in range(15):          # max 15 steps up the chain
            if current < 0:
                break

            pdgid   = abs(int(gen_pdgId[ev][current]))
            mom_idx = int(gen_mother_idx[ev][current])

            # Check for tau in the chain
            if pdgid == 15:
                has_tau = True

            # W boson found!
            if pdgid == 24:
                found_W = True
                break

            # Reached top of decay chain (proton or gluon)
            if pdgid in [2212, 21]:
                break

            # Self-referencing protection
            if mom_idx == current:
                break

            current = mom_idx

        # Apply tau veto if requested
        if exclude_tau:
            is_signal = found_W and not has_tau   # W found, no tau
        else:
            is_signal = found_W                   # W found, tau ignored

        results.append(1 if is_signal else 0)

    return np.array(results, dtype=int)


# ── Compute all three label versions ────────────────────
print(f"\n[3/7] Computing labels...")
print(f"      (a) Simple label  — genPartFlav == 1")
print(f"      (b) Exact label   — mother chain to W, no tau veto (v2)")
print(f"      (c) Final label   — mother chain to W + tau veto    (v3) ✅")
print(f"      This may take a moment...")

label_simple = (lead(arrays["Muon_genPartFlav"]) == 1).astype(int)

label_exact  = trace_mother_chain(                 # v2 style, no tau veto
    arrays["GenPart_pdgId"],
    arrays["GenPart_genPartIdxMother"],
    arrays["Muon_genPartIdx"],
    exclude_tau=False
)

label_final  = trace_mother_chain(                 # v3: WITH tau veto ✅
    arrays["GenPart_pdgId"],
    arrays["GenPart_genPartIdxMother"],
    arrays["Muon_genPartIdx"],
    exclude_tau=True
)

print(f"      Done.")

# ═══════════════════════════════════════════════════════
# STEP 5 — Compute derived variables
# ═══════════════════════════════════════════════════════
print(f"\n[4/7] Computing derived variables...")

muon_phi = lead(arrays["Muon_phi"])
met_phi  = ak.to_numpy(arrays["MET_phi"])
met_pt   = ak.to_numpy(arrays["MET_pt"])
muon_pt  = lead(arrays["Muon_pt"])

# Delta phi — angle between muon and MET, wrapped to [-π, +π]
dphi = muon_phi - met_phi
dphi = np.where(dphi >  np.pi, dphi - 2*np.pi, dphi)
dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)

# Transverse mass — peaks at W mass (~80 GeV) for true W events
# mT = sqrt(2 * pT_mu * MET * (1 - cos(Δφ)))
mT = np.sqrt(2 * muon_pt * met_pt * (1 - np.cos(dphi)))

# Jet count with pT > 30 GeV
n_good_jets = ak.to_numpy(
    ak.sum(arrays["Jet_pt"] > 30.0, axis=1)
)

# Max b-tag score (high value → likely ttbar contamination)
has_jets = ak.num(arrays["Jet_btagDeepB"]) > 0
max_btag = np.where(
    ak.to_numpy(has_jets),
    ak.to_numpy(ak.max(
        arrays["Jet_btagDeepB"], axis=1, mask_identity=True
    )),
    0.0
)
max_btag = np.nan_to_num(max_btag, nan=0.0).astype(float)

# ═══════════════════════════════════════════════════════
# STEP 6 — Build flat DataFrame
# ═══════════════════════════════════════════════════════
print(f"\n[5/7] Building DataFrame...")

df = pd.DataFrame({

    # ── Muon (from ROOT) ────────────────────────────────
    "muon_pt"        : muon_pt,
    "muon_eta"       : lead(arrays["Muon_eta"]),
    "muon_phi"       : muon_phi,
    "muon_charge"    : lead(arrays["Muon_charge"]),
    "muon_mass"      : lead(arrays["Muon_mass"]),
    "muon_iso"       : lead(arrays["Muon_pfRelIso04_all"]),
    "muon_tightId"   : lead(arrays["Muon_tightId"]).astype(int),
    "muon_isGlobal"  : lead(arrays["Muon_isGlobal"]).astype(int),
    "muon_isPFcand"  : lead(arrays["Muon_isPFcand"]).astype(int),
    "muon_dxy"       : lead(arrays["Muon_dxy"]),
    "muon_dz"        : lead(arrays["Muon_dz"]),

    # ── MET (from ROOT) ─────────────────────────────────
    "met_pt"           : met_pt,
    "met_phi"          : met_phi,
    "met_significance" : ak.to_numpy(arrays["MET_significance"]),

    # ── Derived variables (computed by us) ─────────────
    "delta_phi"      : dphi,
    "mT"             : mT,

    # ── Jets (computed by us) ───────────────────────────
    "n_jets"         : n_good_jets,
    "max_btag"       : max_btag,

    # ── Event quality flags (from ROOT) ─────────────────
    "flag_goodVtx"   : ak.to_numpy(arrays["Flag_goodVertices"]).astype(int),
    "flag_METfilt"   : ak.to_numpy(arrays["Flag_METFilters"]).astype(int),
    "flag_HBHE"      : ak.to_numpy(arrays["Flag_HBHENoiseFilter"]).astype(int),
    "flag_badMuon"   : ak.to_numpy(arrays["Flag_BadPFMuonFilter"]).astype(int),

    # ── Triggers (from ROOT) ────────────────────────────
    "HLT_IsoMu24"    : ak.to_numpy(arrays["HLT_IsoMu24"]).astype(int),
    "HLT_IsoMu27"    : ak.to_numpy(arrays["HLT_IsoMu27"]).astype(int),

    # ── Generator weight (from ROOT) ────────────────────
    "genWeight"      : ak.to_numpy(arrays["genWeight"]),

    # ── Labels ──────────────────────────────────────────
    # label         → FINAL (v3): W mother chain + tau veto ✅ USE THIS
    # label_exact   → v2 style  : W mother chain, no tau veto
    # label_simple  → simple    : genPartFlav==1 (includes Z/γ*)
    # genPartFlav   → raw value : 1=prompt, 3=tau, 4=b, 0=fake
    "label"          : label_final,
    "label_exact"    : label_exact,
    "label_simple"   : label_simple,
    "genPartFlav"    : lead(arrays["Muon_genPartFlav"]).astype(int),
})

# ═══════════════════════════════════════════════════════
# STEP 7 — Event selection cuts
# ═══════════════════════════════════════════════════════
print(f"\n[6/7] Applying event selection cuts...")
print(f"      Events before cuts: {len(df):,}")

selection = (
    (df["muon_pt"]           > 26)   &   # muon pT > 26 GeV
    (df["muon_eta"].abs()    < 2.4)  &   # within detector acceptance
    (df["muon_iso"]          < 0.15) &   # isolated muon (W muons are isolated)
    (df["muon_tightId"]      == 1)   &   # tight muon quality
    (df["muon_isGlobal"]     == 1)   &   # global muon (tracker + muon chambers)
    (df["met_pt"]            > 25)   &   # MET > 25 GeV (neutrino signature)
    (df["mT"]                > 40)   &   # mT lower cut (removes QCD)
    (df["mT"]                < 130)  &   # mT upper cut (removes tails)
    (df["flag_goodVtx"]      == 1)   &   # good primary vertex
    (df["flag_METfilt"]      == 1)   &   # MET filters passed
    (df["flag_HBHE"]         == 1)   &   # no HCAL noise
    (df["flag_badMuon"]      == 1)       # no bad PF muon
)

df_selected = df[selection].reset_index(drop=True)
print(f"      Events after  cuts: {len(df_selected):,}")

# ═══════════════════════════════════════════════════════
# LABEL COMPARISON REPORT
# ═══════════════════════════════════════════════════════
n_total  = len(df_selected)
n_final  = int(df_selected["label"].sum())
n_exact  = int(df_selected["label_exact"].sum())
n_simple = int(df_selected["label_simple"].sum())

# Count W→τ→μ events (in exact but not in final)
n_tau    = n_exact - n_final

# Count Z/γ* events (in exact or final but not in simple ... actually other way)
n_zgamma = n_simple - n_final if n_simple > n_final else 0

print(f"\n{'═'*60}")
print(f"  LABEL COMPARISON REPORT")
print(f"{'═'*60}")
print(f"  Total selected events               : {n_total:,}")
print(f"")
print(f"  ✅ FINAL label (v3 — USE THIS):")
print(f"     Signal W→μ direct    label=1     : {n_final:,}  ({100*n_final/n_total:.1f}%)")
print(f"     Background           label=0     : {n_total-n_final:,}  ({100*(n_total-n_final)/n_total:.1f}%)")
print(f"")
print(f"  ⚠️  EXACT label (v2 — no tau veto):")
print(f"     Signal (W→μ+W→τ→μ)  label=1     : {n_exact:,}  ({100*n_exact/n_total:.1f}%)")
print(f"     Background           label=0     : {n_total-n_exact:,}  ({100*(n_total-n_exact)/n_total:.1f}%)")
print(f"")
print(f"  ⚠️  SIMPLE label (genPartFlav==1):")
print(f"     Signal (prompt μ)    label=1     : {n_simple:,}  ({100*n_simple/n_total:.1f}%)")
print(f"     Background           label=0     : {n_total-n_simple:,}  ({100*(n_total-n_simple)/n_total:.1f}%)")
print(f"")
print(f"  ── Contamination breakdown ──────────")
print(f"  W→τ→μ removed by tau veto           : {n_tau:,} events")
print(f"  Z/γ* removed by exact labeling      : {n_zgamma:,} events")
print(f"{'═'*60}")

# ═══════════════════════════════════════════════════════
# SAVE TO CSV
# ═══════════════════════════════════════════════════════
print(f"\n[7/7] Saving CSV...")
df_selected.to_csv(OUTPUT_FILE, index=False)

print(f"\n{'═'*60}")
print(f"  ✅ Saved: {OUTPUT_FILE}")
print(f"  Shape  : {df_selected.shape[0]:,} rows × {df_selected.shape[1]} columns")
print(f"{'═'*60}")

print(f"\nColumn reference:")
col_origin = {
    "delta_phi"   : "computed — angle between muon and MET",
    "mT"          : "computed — transverse mass (KEY variable)",
    "n_jets"      : "computed — jets with pT > 30 GeV",
    "max_btag"    : "computed — max b-tag score in event",
    "label"       : "computed — FINAL label (W direct, tau vetoed) ✅",
    "label_exact" : "computed — exact label, no tau veto",
    "label_simple": "computed — simple label (genPartFlav==1)",
}
for col in df_selected.columns:
    if col in col_origin:
        print(f"  {col:<20} ← {col_origin[col]}")
    else:
        print(f"  {col:<20} ← from ROOT file")
