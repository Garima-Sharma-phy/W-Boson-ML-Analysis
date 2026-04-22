import uproot
import awkward as ak
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
INPUT_FILE  = "TTbar.root"  
OUTPUT_FILE = "ttbar_ready.csv"
TREE_NAME   = "Events"

print(f"Opening Background File: {INPUT_FILE}")
file = uproot.open(INPUT_FILE)
tree = file[TREE_NAME]

# ═══════════════════════════════════════════════════════
# EXACT BRANCHES (Matched to your specific file)
# ═══════════════════════════════════════════════════════
BRANCHES = [
    "Muon_pt", "Muon_eta", "Muon_phi", "Muon_pfRelIso04_all",
    "Muon_dxy", "Muon_dz", "Muon_tightId",
    "MET_pt", "MET_phi", "MET_significance",
    "nJet", "Jet_btag"
]

print("Loading data...")
arrays = tree.arrays(BRANCHES, library="ak")

# Keep only events with at least 1 muon
mask_1mu = ak.num(arrays["Muon_pt"]) >= 1
arrays   = arrays[mask_1mu]

# Helper to grab the leading muon
def lead(arr): return ak.to_numpy(arr[:, 0])

# ═══════════════════════════════════════════════════════
# CALCULATE PHYSICS GEOMETRIES
# ═══════════════════════════════════════════════════════
print("Calculating Delta Phi and Transverse Mass...")
muon_phi = lead(arrays["Muon_phi"])
met_phi  = ak.to_numpy(arrays["MET_phi"])
muon_pt  = lead(arrays["Muon_pt"])
met_pt   = ak.to_numpy(arrays["MET_pt"])

# Delta Phi
dphi = np.abs(muon_phi - met_phi)
dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)

# Transverse Mass (mT)
mT = np.sqrt(2 * muon_pt * met_pt * (1 - np.cos(dphi)))

# Max B-Tag (Using your specific 'Jet_btag' column)
has_jets = ak.num(arrays["Jet_btag"]) > 0
max_btag = np.where(
    ak.to_numpy(has_jets),
    ak.to_numpy(ak.max(arrays["Jet_btag"], axis=1, mask_identity=True)),
    -1.0 # -1.0 for events with 0 jets
)
max_btag = np.nan_to_num(max_btag, nan=-1.0).astype(float)

# ═══════════════════════════════════════════════════════
# BUILD DATAFRAME
# ═══════════════════════════════════════════════════════
print("Building DataFrame...")
df = pd.DataFrame({
    "muon_pt"          : muon_pt,
    "muon_eta"         : lead(arrays["Muon_eta"]),
    "muon_iso"         : lead(arrays["Muon_pfRelIso04_all"]),
    "muon_dxy"         : lead(arrays["Muon_dxy"]),
    "muon_dz"          : lead(arrays["Muon_dz"]),
    "muon_tightId"     : lead(arrays["Muon_tightId"]).astype(int),
    "met_pt"           : met_pt,
    "met_significance" : ak.to_numpy(arrays["MET_significance"]),
    "delta_phi"        : dphi,
    "mT"               : mT,
    "n_jets"           : ak.to_numpy(arrays["nJet"]),
    "max_btag"         : max_btag,
    "training_label"   : 0   # THIS IS CRITICAL! 0 means Background
})

# ═══════════════════════════════════════════════════════
# APPLY PHYSICS CUTS
# ═══════════════════════════════════════════════════════
print("Applying cuts...")
selection = (
    (df["muon_pt"]      > 26)   &
    (df["muon_eta"].abs() < 2.4)&
    (df["muon_iso"]     < 0.15) &
    (df["muon_tightId"] == 1)   &
    (df["met_pt"]       > 25)   &
    (df["mT"]           > 40)   &
    (df["mT"]           < 130)
)

df_selected = df[selection].reset_index(drop=True)

# Drop tightId as it's not in your ML feature list, just used for cuts
df_selected = df_selected.drop(columns=["muon_tightId"])

print(f"Saving {len(df_selected)} pure background events to CSV...")
df_selected.to_csv(OUTPUT_FILE, index=False)
print(f"Success! Saved to {OUTPUT_FILE}.")
