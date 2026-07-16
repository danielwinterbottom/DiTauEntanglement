import ROOT
ROOT.gROOT.SetBatch(True)
from numpy import arange
import numpy as np
import pandas as pd
from array import array
from taupolaris.utils.kinematic_helpers import EntanglementVariables
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=int, help="Determines which model to assume for the signal. 1 = CP-even Higgs, 2 = spin0 no entanglement, 3 = uncorrelated", default=1)
parser.add_argument("--sig-file", default="outputs_model_LHC_TransformerFlow_Hadronic_100e_June22_TRIAL2/output_results_UnCorr.parquet")
parser.add_argument("--bkg-file", default="outputs_model_LHC_TransformerFlow_Hadronic_100e_June22_TRIAL2/output_results_ZToTauTau.parquet")
parser.add_argument("--no-replace", action="store_true", help="Sample toys without replacement (non-overlapping chunks). Limits N_toys to pool_size/N_sig_events.")
parser.add_argument("--n-toys", type=int, default=1000, help="Number of toys to generate. If --no-replace is set, this will be limited to pool_size/N_sig_events.")
parser.add_argument("--outdir", default="nll_fits_fast_incdm2_v2", help="Directory to write output files to.")
parser.add_argument("--measure-B", action="store_true", help="Also measure B vector (tau polarization) elements.")
args = parser.parse_args()


_ij_map = {'nn':(0,0),'rr':(1,1),'nr':(0,1),'rn':(1,0),'kn':(2,0),'kr':(2,1),'nk':(0,2),'rk':(1,2),'kk':(2,2)}
_i_map = {'n': 0, 'r': 1, 'k': 2}

def precompute_model_hists(df_sig, prod_vals, C_base, ii, jj, rho_vals, nbins, B_p_base=None, B_m_base=None):
    """
    For each value in rho_vals, set C_base[ii,jj]=rho, reweight df_sig, histogram prod_vals.
    B_p_base and B_m_base are held fixed during the Cij scan.
    Returns array of shape (len(rho_vals), nbins), normalised to sum=1.
    """
    model_hists = np.zeros((len(rho_vals), nbins), dtype=np.float64)
    C = C_base.copy()
    for k, rho in enumerate(rho_vals):
        C[ii, jj] = rho
        wt = compute_event_weights(df_sig, C, B_p_base, B_m_base)
        h, _ = np.histogram(prod_vals, bins=nbins, range=(-1, 1), weights=wt)
        h = h.astype(np.float64)
        if h.sum() > 0:
            h /= h.sum()
        model_hists[k] = h
    return model_hists


def precompute_model_hists_B(df_sig, var_vals, which, idx, rho_vals, nbins, C_base, B_p_base, B_m_base):
    """
    For B vector scanning: which='p' (tau+) or 'm' (tau-), idx=0/1/2 (n/r/k).
    Uses 1D histogram of cos_i+ or cos_j- weighted by the full spin weight.
    Returns array of shape (len(rho_vals), nbins), normalised to sum=1.
    """
    model_hists = np.zeros((len(rho_vals), nbins), dtype=np.float64)
    B_p = B_p_base.copy()
    B_m = B_m_base.copy()
    for k, rho in enumerate(rho_vals):
        if which == 'p':
            B_p[idx] = rho
        else:
            B_m[idx] = rho
        wt = compute_event_weights(df_sig, C_base, B_p, B_m)
        h, _ = np.histogram(var_vals, bins=nbins, range=(-1, 1), weights=wt)
        h = h.astype(np.float64)
        if h.sum() > 0:
            h /= h.sum()
        model_hists[k] = h
    return model_hists


def nll_scan_np(data_counts, model_hists, rho_vals, N_sig_exp=None, N_bkg_exp=None, bkg_counts=None):
    """NLL scan using precomputed per-rho model histograms."""
    if bkg_counts is not None and (N_sig_exp is None or N_bkg_exp is None):
        raise ValueError("If bkg_counts is provided, N_sig_exp and N_bkg_exp must be provided.")

    Nexp = (N_sig_exp + N_bkg_exp) if bkg_counts is not None else (N_sig_exp if N_sig_exp is not None else data_counts.sum())

    nll_values = np.full(len(rho_vals), np.inf)
    for k, (rho, pred) in enumerate(zip(rho_vals, model_hists)):
        if np.any(pred < 0):
            continue

        if bkg_counts is not None:
            combined = pred * N_sig_exp
            bkg_norm = bkg_counts * (N_bkg_exp / bkg_counts.sum() if bkg_counts.sum() > 0 else 0)
            combined = combined + bkg_norm
            total = combined.sum()
            if total > 0:
                combined /= total
            pred = combined

        mu = Nexp * pred
        n = data_counts
        mask = mu > 0
        nll_values[k] = 2.0 * np.sum(mu[mask] - n[mask] * np.log(mu[mask]))

    best_nll = nll_values.min()
    # When the NLL is flat over a range of rho values (degenerate model),
    # np.argmin returns the first occurrence (left edge). Instead take the midpoint.
    flat_mask = np.isclose(nll_values, best_nll, rtol=0, atol=1e-6)
    flat_indices = np.where(flat_mask)[0]
    i_best = flat_indices[len(flat_indices) // 2]
    best_rho = rho_vals[i_best]
    delta_nll = nll_values - best_nll

    left = rho_vals < best_rho
    right = rho_vals > best_rho

    rho_low = np.interp(1.0, delta_nll[left][::-1], rho_vals[left][::-1]) if left.any() else best_rho
    rho_high = np.interp(1.0, delta_nll[right], rho_vals[right]) if right.any() else best_rho

    if rho_low > rho_high:
        rho_low, rho_high = rho_high, rho_low

    return best_rho, rho_low, rho_high


def GetSpinCorrelationMatrix(phiCP):
    # return the expected spin correlation matrix for Higgs bosons production with different CP-mixing angles
    delta = np.radians(phiCP)

    C = np.array(
        [[np.cos(2*delta),np.sin(2*delta),0],
        [-np.sin(2*delta),np.cos(2*delta),0],
        [0,0,-1]], dtype=float
    )

    return C

def GetMatrixCoefficient(C, element):
    # element "nn" or "rr" or "nr" or "rn", "kn" etc...
    if element == "nn":
        return C[0,0]
    elif element == "rr":
        return C[1,1]
    elif element == "nr":
        return C[0,1]
    elif element == "rn":
        return C[1,0]
    elif element == "kn":
        return C[2,0]
    elif element == "kr":
        return C[2,1]
    elif element == "nk":
        return C[0,2]
    elif element == "rk":
        return C[1,2]
    elif element == "kk":
        return C[2,2]
    else:
        raise ValueError(f"Invalid element: {element}. Allowed elements are: nn, rr, nr, rn, kn, kr, nk, rk, kk")


_axes = ['n', 'r', 'k']
_wt_cols = [f'wt_hp_{a}_hm_{b}' for a in _axes for b in _axes]

def compute_event_weights(df, C, B_p=None, B_m=None):
    """Compute per-event spin weights from C matrix and optional B vectors."""
    wt = np.ones(len(df), dtype=np.float64)
    for i, a in enumerate(_axes):
        for j, b in enumerate(_axes):
            if C[i, j] != 0:
                wt += C[i, j] * df[f'wt_hp_{a}_hm_{b}'].values
    if B_p is not None:
        for i, a in enumerate(_axes):
            if B_p[i] != 0:
                wt += B_p[i] * df[f'wt_hp_{a}'].values
    if B_m is not None:
        for j, b in enumerate(_axes):
            if B_m[j] != 0:
                wt += B_m[j] * df[f'wt_hm_{b}'].values
    return np.clip(wt, 0, None)

### dm 1, 1
##gen_cuts =  'true_taup_npizero==1&&true_taun_npizero==1&&true_taup_is3prong==0&&true_taun_is3prong==0'
##cuts = 'reco_taup_npizero==1&&reco_taun_npizero==1&&reco_taup_is3prong==0&&reco_taun_is3prong==0' 

#dm 0 or 1:

#gen_cuts =  'true_taup_npizero<2&&true_taun_npizero<2&&true_taup_is3prong==0&&true_taun_is3prong==0'
#cuts = 'reco_taup_npizero<2&&reco_taun_npizero<2&&reco_taup_is3prong==0&&reco_taun_is3prong==0' 

#dm 0 or 1 or 2 in gen; sel 0 or 1 in reco:
gen_cuts =  'true_taup_npizero<=2&&true_taun_npizero<=2&&true_taup_is3prong==0&&true_taun_is3prong==0'
cuts = 'reco_taup_npizero<2&&reco_taun_npizero<2&&reco_taup_is3prong==0&&reco_taun_is3prong==0' 

###dm 0, 1, or 10
##
##gen_cuts =  '((true_taup_npizero<2 && true_taup_is3prong==0) || (true_taup_is3prong==1 && true_taup_npizero==0)) && ((true_taun_npizero<2 && true_taun_is3prong==0) || (true_taun_is3prong==1 && true_taun_npizero==0))'
##cuts = '((reco_taup_npizero<2 && reco_taup_is3prong==0) || (reco_taup_is3prong==1 && reco_taup_npizero==0)) && ((reco_taun_npizero<2 && reco_taun_is3prong==0) || (reco_taun_is3prong==1 && reco_taun_npizero==0))'
##
###gen dm 0, 1, 10, or 11
###reco dm 0,1 or 10
##
##gen_cuts =  '((true_taup_npizero<2 && true_taup_is3prong==0) || (true_taup_is3prong==1)) && ((true_taun_npizero<2 && true_taun_is3prong==0) || (true_taun_is3prong==1))'
##cuts = '((reco_taup_npizero<2 && reco_taup_is3prong==0) || (reco_taup_is3prong==1 && reco_taup_npizero==0)) && ((reco_taun_npizero<2 && reco_taun_is3prong==0) || (reco_taun_is3prong==1 && reco_taun_npizero==0))'

var_prefix = 'map_pred'
#var_prefix = 'true'

print("Loading parquet files...")
_B_wt_cols = [f'wt_hp_{a}' for a in _axes] + [f'wt_hm_{b}' for b in _axes]
_needed_cols = (
    [f'{var_prefix}_cos{a}_{s}' for a in _axes for s in ('plus', 'minus')] +
    _wt_cols +
    ((_B_wt_cols) if args.measure_B else []) +
    ['true_taup_npizero', 'true_taun_npizero', 'true_taup_is3prong', 'true_taun_is3prong',
     'reco_taup_npizero', 'reco_taun_npizero', 'reco_taup_is3prong', 'reco_taun_is3prong']
)
df_sig_raw = pd.read_parquet(args.sig_file, columns=_needed_cols)
_bkg_cols = [c for c in _needed_cols if c not in _wt_cols and c not in _B_wt_cols]
df_bkg_raw = pd.read_parquet(args.bkg_file, columns=_bkg_cols)
print(f"Loaded {len(df_sig_raw)} sig events, {len(df_bkg_raw)} bkg events")

def apply_cuts(df):
    gen_mask = (df['true_taup_npizero'] < 2) & (df['true_taun_npizero'] < 2) & \
               (df['true_taup_is3prong'] == 0) & (df['true_taun_is3prong'] == 0)
    reco_mask = (df['reco_taup_npizero'] < 2) & (df['reco_taun_npizero'] < 2) & \
                (df['reco_taup_is3prong'] == 0) & (df['reco_taun_is3prong'] == 0)
    return df[gen_mask & reco_mask].reset_index(drop=True)

df_sig = apply_cuts(df_sig_raw)
df_bkg = apply_cuts(df_bkg_raw)
del df_sig_raw, df_bkg_raw
print(f"After cuts: {len(df_sig)} sig, {len(df_bkg)} bkg")


n_bins=10
# details on signal/background estimates:
# rhorho: sig = 13.203052, ZTT = 23.640888, bkg = 43.041533
# pirho: sig = 8.3268526, ZTT 16.686166, bkg = 26.272433
# get numbers for rhorho and pirho from HIG-25-012 hepdata
# estimate pipi from: 1/4*(N_pirho)**2/N_rhorho
# pipi: sig = 1.31289, ZTT = 2.9443, bkg = 4.0091551556695775 
# total: sig = 22.8427946, ZTT = 43.271354, bkg = 73.3231
# lum scale: sig = 1098.2, ZTT = 2080.4, bkg = 3525.2
# round: sig = 1100, ZTT = 2100, bkg = 3500
# for now we are only including ZTT background
N_sig_events = 1100
N_bkg_events = 2100
N_events = N_sig_events + N_bkg_events
step=0.01
vals = np.linspace(-1, 1, int(2/step) + 1)

import os
os.makedirs(args.outdir, exist_ok=True)


if args.model == 1:
    print("Using CP-even Higgs model for signal")
    phiCP = 0
    C_rwt = GetSpinCorrelationMatrix(phiCP)
elif args.model == 2:
    print("Using spin-0 no entanglement model for signal")
    C_rwt = np.array(
            [[0,0,0],
            [0,0,0],
            [0,0,-1]], dtype=float
        )
elif args.model == 3:
    print("Using uncorrelated model for signal")
    C_rwt = np.array(
            [[0,0,0],
            [0,0,0],
            [0,0,0]], dtype=float
        )
else:
    raise ValueError(f"Invalid model: {args.model}. Allowed models are: 1 = CP-even Higgs, 2 = spin0 no entanglement, 3 = uncorrelated")


# B vectors (tau polarization) — zero for all current models
B_rwt_p = np.zeros(3, dtype=float)
B_rwt_m = np.zeros(3, dtype=float)

# pre-compute event weights for signal model
sig_model_wt = compute_event_weights(df_sig, C_rwt, B_rwt_p, B_rwt_m)

# pre-cache product values keyed by (x_var, y_var)
sig_prod_vals = {}
bkg_prod_vals = {}

Cij_elements = ["nn", "rr", "nr", "rn", "kk", "kn", "kr", "nk", "rk", "kk"]
C_exp = C_rwt
measured_rho_values = {}
measured_Cij_values = {}

# precompute model histograms for each Cij (reused in toy loop)
print("Precomputing model histograms for each Cij...")
model_hists_per_Cij = {}
for Cij in Cij_elements:
    ii, jj = _ij_map[Cij]
    x_var = f'{var_prefix}_cos{Cij[0]}_plus'
    y_var = f'{var_prefix}_cos{Cij[1]}_minus'
    prod = df_sig[x_var].values * df_sig[y_var].values
    sig_prod_vals[(x_var, y_var)] = prod
    bkg_prod_vals[(x_var, y_var)] = df_bkg[x_var].values * df_bkg[y_var].values
    # scan with only the Cij element varied; other elements fixed at C_rwt values
    C_base = C_rwt.copy()
    model_hists_per_Cij[Cij] = precompute_model_hists(df_sig, prod, C_base, ii, jj, vals, n_bins, B_rwt_p, B_rwt_m)
    print(f"  C{Cij} done")

for Cij in Cij_elements:
    x_var = f'{var_prefix}_cos{Cij[0]}_plus'
    y_var = f'{var_prefix}_cos{Cij[1]}_minus'

    sig_prod = sig_prod_vals[(x_var, y_var)]
    h_azimov_sig_np, _ = np.histogram(sig_prod, bins=n_bins, range=(-1,1), weights=sig_model_wt)
    h_azimov_sig_np = h_azimov_sig_np.astype(np.float64)
    if h_azimov_sig_np.sum() > 0:
        h_azimov_sig_np *= N_sig_events / h_azimov_sig_np.sum()

    bkg_prod = bkg_prod_vals[(x_var, y_var)]
    h_azimov_bkg_np, _ = np.histogram(bkg_prod, bins=n_bins, range=(-1,1))
    h_azimov_bkg_np = h_azimov_bkg_np.astype(np.float64)
    if h_azimov_bkg_np.sum() > 0:
        h_azimov_bkg_np *= N_bkg_events / h_azimov_bkg_np.sum()

    h_azimov_np = h_azimov_sig_np + h_azimov_bkg_np

    best_rho, rho_low, rho_high = nll_scan_np(
        h_azimov_np, model_hists_per_Cij[Cij], vals,
        N_sig_exp=N_sig_events, N_bkg_exp=N_bkg_events, bkg_counts=h_azimov_bkg_np,
    )
    print(f"C{Cij}: rho = {best_rho:.3f} (+{rho_high-best_rho:.3f}/-{best_rho-rho_low:.3f})")
    measured_rho_values[Cij] = (best_rho, rho_low, rho_high)
    measured_Cij_values[Cij] = (best_rho, rho_low, rho_high)  # rho = Cij directly

    # plotting
    def _np_to_th1(arr, name):
        h = ROOT.TH1D(name, "", n_bins, -1, 1)
        h.Sumw2()
        for i, v in enumerate(arr):
            h.SetBinContent(i+1, v)
            h.SetBinError(i+1, np.sqrt(abs(v)))
        return h

    i_best = np.argmin([nll for nll in np.abs(vals - best_rho)])  # index of best rho in vals
    best_model_hist = model_hists_per_Cij[Cij][np.argmin(np.abs(vals - best_rho))]
    pred_corr_np = best_model_hist * N_sig_events

    h_data_root = _np_to_th1(h_azimov_np, "h_data")
    h_azimov_bkg_root = _np_to_th1(h_azimov_bkg_np, "h_azimov_bkg")
    h_pred_corr_root = _np_to_th1(pred_corr_np, "h_pred_corr")

    canv = ROOT.TCanvas("canv", "canv", 800, 600)
    h_data_root.SetTitle("")
    h_data_root.SetStats(0)
    h_data_root.SetLineColor(ROOT.kBlack)
    h_data_root.SetLineWidth(2)
    h_data_root.SetMarkerStyle(20)
    h_data_root.SetMarkerColor(ROOT.kBlack)
    h_data_root.GetXaxis().SetTitle(f"cos#theta^{{+}}_{{{Cij[0]}}}cos#theta^{{-}}_{{{Cij[1]}}}")
    h_data_root.GetYaxis().SetTitle("Events")
    h_data_root.Draw("pE1")
    hs = ROOT.THStack("hs", "")
    h_pred_corr_root.SetLineColor(ROOT.kBlack)
    h_pred_corr_root.SetFillColor(ROOT.kRed)
    h_pred_corr_root.SetLineWidth(1)
    h_azimov_bkg_root.SetLineColor(ROOT.kBlack)
    h_azimov_bkg_root.SetFillColor(ROOT.kBlue)
    h_azimov_bkg_root.SetLineWidth(1)
    hs.Add(h_azimov_bkg_root)
    hs.Add(h_pred_corr_root)
    hs.Draw("hist same")
    h_data_root.Draw("pE1 same")
    leg = ROOT.TLegend(0.58, 0.53, 0.9, 0.9)
    leg.AddEntry(h_data_root, "Asimov Data", "pe")
    leg.AddEntry(h_azimov_bkg_root, "Background", "f")
    leg.AddEntry(h_pred_corr_root, "Best-fit signal", "f")
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    dummy_hist = ROOT.TH1D("dummy_hist", "", 1, 0, 1)
    dummy_hist.SetLineColor(ROOT.kWhite)
    leg.AddEntry(dummy_hist, f"Fitted C{Cij} = {best_rho:.2f}^{{+{rho_high-best_rho:.2f}}}_{{-{best_rho-rho_low:.2f}}}", "l")
    leg.AddEntry(dummy_hist, f"True C{Cij} = {GetMatrixCoefficient(C_exp, Cij):.2f}", "l")
    leg.Draw()
    if args.model == 1:
        canv.Print(f"{args.outdir}/C{Cij}_fitted_phiCP{phiCP:.0f}.pdf")
    elif args.model == 2:
        canv.Print(f"{args.outdir}/C{Cij}_fitted_spin0_noentanglement.pdf")
    elif args.model == 3:
        canv.Print(f"{args.outdir}/C{Cij}_fitted_uncorrelated.pdf")

    del h_data_root, h_pred_corr_root, h_azimov_bkg_root, canv, dummy_hist

print("\nMeasured Cij values (direct reweighting fit):")
for Cij, (Cij_val, Cij_low, Cij_high) in measured_Cij_values.items():
    print(f"C{Cij}: C{Cij} = {Cij_val:.3f} (+{Cij_high-Cij_val:.3f}/-{Cij_val-Cij_low:.3f})")
print("\nTrue Cij values:")
for Cij in Cij_elements:
    print(f"C{Cij}: C{Cij} = {GetMatrixCoefficient(C_exp, Cij):.3f}")

# --- B vector measurement ---
B_elements = [('p','n'), ('p','r'), ('p','k'), ('m','n'), ('m','r'), ('m','k')]
measured_B_values = {}
model_hists_per_B = {}
sig_B_vals = {}
bkg_B_vals = {}

if args.measure_B:
    print("\nPrecomputing model histograms for B elements...")
    for which, ax in B_elements:
        key = (which, ax)
        idx = _i_map[ax]
        if which == 'p':
            var = f'{var_prefix}_cos{ax}_plus'
        else:
            var = f'{var_prefix}_cos{ax}_minus'
        sig_B_vals[key] = df_sig[var].values
        bkg_B_vals[key] = df_bkg[var].values
        model_hists_per_B[key] = precompute_model_hists_B(
            df_sig, sig_B_vals[key], which, idx, vals, n_bins, C_rwt.copy(), B_rwt_p, B_rwt_m
        )
        print(f"  B{which}_{ax} done")

    for which, ax in B_elements:
        key = (which, ax)
        idx = _i_map[ax]
        var_vals = sig_B_vals[key]

        h_azimov_sig_B, _ = np.histogram(var_vals, bins=n_bins, range=(-1,1), weights=sig_model_wt)
        h_azimov_sig_B = h_azimov_sig_B.astype(np.float64)
        if h_azimov_sig_B.sum() > 0:
            h_azimov_sig_B *= N_sig_events / h_azimov_sig_B.sum()

        bkg_var_vals = bkg_B_vals[key]
        h_azimov_bkg_B, _ = np.histogram(bkg_var_vals, bins=n_bins, range=(-1,1))
        h_azimov_bkg_B = h_azimov_bkg_B.astype(np.float64)
        if h_azimov_bkg_B.sum() > 0:
            h_azimov_bkg_B *= N_bkg_events / h_azimov_bkg_B.sum()

        h_azimov_B = h_azimov_sig_B + h_azimov_bkg_B

        best_rho, rho_low, rho_high = nll_scan_np(
            h_azimov_B, model_hists_per_B[key], vals,
            N_sig_exp=N_sig_events, N_bkg_exp=N_bkg_events, bkg_counts=h_azimov_bkg_B,
        )
        measured_B_values[key] = (best_rho, rho_low, rho_high)
        label = f"B{'tau+' if which=='p' else 'tau-'}_{ax}"
        print(f"{label}: {best_rho:.3f} (+{rho_high-best_rho:.3f}/-{best_rho-rho_low:.3f})  [true=0]")



C = np.array([[measured_Cij_values["nn"][0], measured_Cij_values["nr"][0], measured_Cij_values["nk"][0]],
              [measured_Cij_values["rn"][0], measured_Cij_values["rr"][0], measured_Cij_values["rk"][0]],
              [measured_Cij_values["kn"][0], measured_Cij_values["kr"][0], measured_Cij_values["kk"][0]]])
if args.measure_B:
    _Bp = np.array([[measured_B_values[('p',a)][0]] for a in _axes])
    _Bm = np.array([[measured_B_values[('m',a)][0]] for a in _axes])
    con, m12 = EntanglementVariables(C, Bplus=_Bp, Bminus=_Bm)
else:
    con, m12 = EntanglementVariables(C)

print("C:", C)
print("Con:", con)
print("m12:", m12)

#now to get the errors:
# we will loop over t1 and take chunks of events (sampling a poisson with mean = N_events), then fit each toy, compute concurrence, m12 etc and use these toys to get the errors on the measured quantities
N_toys = args.n_toys
con_toys = []
m12_toys = []
Cnn_toys = []
B_toys = {(w, a): [] for w, a in B_elements} if args.measure_B else {}

print_every = 1

# pre-compute fixed background templates (from full bkg sample) for use in toy NLL
fixed_bkg_templates = {}
for Cij in Cij_elements:
    x_var = f'{var_prefix}_cos{Cij[0]}_plus'
    y_var = f'{var_prefix}_cos{Cij[1]}_minus'
    key = (x_var, y_var)
    if key not in fixed_bkg_templates:
        tmpl, _ = np.histogram(bkg_prod_vals[key], bins=n_bins, range=(-1, 1))
        tmpl = tmpl.astype(np.float64)
        if tmpl.sum() > 0:
            tmpl *= N_bkg_events / tmpl.sum()
        fixed_bkg_templates[key] = tmpl

n_sig_total = len(df_sig)
n_bkg_total = len(df_bkg)

if args.no_replace:
    max_toys_sig = int((sig_model_wt > 0).sum()) // N_sig_events
    max_toys_bkg = n_bkg_total // N_bkg_events if N_bkg_events > 0 else N_toys
    max_toys = min(max_toys_sig, max_toys_bkg, N_toys)
    if max_toys < N_toys:
        print(f"Warning: --no-replace limits toys to {max_toys} (pool: {n_sig_total} sig, {n_bkg_total} bkg)")
        N_toys = max_toys
    # pre-shuffle pool indices weighted by sig_model_wt for signal
    # only sample from events with non-zero weight
    nonzero_mask = sig_model_wt > 0
    nonzero_idx = np.where(nonzero_mask)[0]
    nonzero_wt = sig_model_wt[nonzero_idx]
    n_needed = N_toys * N_sig_events
    sig_pool = nonzero_idx[np.random.choice(len(nonzero_idx), size=min(n_needed, len(nonzero_idx)), replace=False, p=nonzero_wt/nonzero_wt.sum())]
    bkg_pool = np.random.permutation(n_bkg_total)

# create output file with only toy_tree
fout = ROOT.TFile.Open(f"{args.outdir}/toys_tree.root", "RECREATE")
fout.cd()
toy_tree_out = ROOT.TTree("toy_tree", "toy_tree")
branches = [
  'iToy', 'con', 'm12',
  'Cnn', 'Cnr', 'Cnk',
  'Crn', 'Crr', 'Crk',
  'Ckn', 'Ckr', 'Ckk'
]
if args.measure_B:
    branches += ['Bpn', 'Bpr', 'Bpk', 'Bmn', 'Bmr', 'Bmk']
branch_vals = {}
for b in branches:
    branch_vals[b] = array('f', [0])
    toy_tree_out.Branch(b, branch_vals[b], '%s/F' % b)

for toy in range(N_toys):
    N_sig_toy_events = np.random.poisson(N_sig_events)
    N_bkg_toy_events = np.random.poisson(N_bkg_events)
    N_toy_events = N_sig_toy_events + N_bkg_toy_events

    if args.no_replace:
        sig_idx = sig_pool[toy * N_sig_events : toy * N_sig_events + N_sig_toy_events]
        bkg_idx = bkg_pool[toy * N_bkg_events : toy * N_bkg_events + N_bkg_toy_events]
        toy_sig_wt = sig_model_wt[sig_idx]
    else:
        sig_idx = np.random.choice(n_sig_total, size=N_sig_toy_events, replace=True, p=sig_model_wt/sig_model_wt.sum())
        bkg_idx = np.random.choice(n_bkg_total, size=N_bkg_toy_events, replace=True)
        toy_sig_wt = None

    measured_B_values_toy = {}
    if args.measure_B:
        for which, ax in B_elements:
            key = (which, ax)
            var_vals = sig_B_vals[key]
            toy_sig_B, _ = np.histogram(var_vals[sig_idx], bins=n_bins, range=(-1, 1), weights=toy_sig_wt)
            toy_sig_B = toy_sig_B.astype(np.float64)
            if toy_sig_B.sum() > 0:
                toy_sig_B *= N_sig_toy_events / toy_sig_B.sum()
            bkg_var_vals = bkg_B_vals[key]
            bkg_key = (f'{var_prefix}_cos{ax}_{"plus" if which=="p" else "minus"}',)
            if bkg_key not in fixed_bkg_templates:
                tmpl, _ = np.histogram(bkg_var_vals, bins=n_bins, range=(-1, 1))
                tmpl = tmpl.astype(np.float64)
                if tmpl.sum() > 0:
                    tmpl *= N_bkg_events / tmpl.sum()
                fixed_bkg_templates[bkg_key] = tmpl
            if N_bkg_toy_events > 0:
                toy_bkg_B, _ = np.histogram(bkg_var_vals[bkg_idx], bins=n_bins, range=(-1, 1))
                toy_bkg_B = toy_bkg_B.astype(np.float64)
                if toy_bkg_B.sum() > 0:
                    toy_bkg_B_scaled = toy_bkg_B * (N_bkg_toy_events / toy_bkg_B.sum())
                else:
                    toy_bkg_B_scaled = np.zeros(n_bins, dtype=np.float64)
            else:
                toy_bkg_B_scaled = np.zeros(n_bins, dtype=np.float64)
            toy_counts_B = toy_sig_B + toy_bkg_B_scaled
            best_rho, _, _ = nll_scan_np(
                toy_counts_B, model_hists_per_B[key], vals,
                N_sig_exp=N_sig_toy_events, N_bkg_exp=N_bkg_toy_events,
                bkg_counts=fixed_bkg_templates[bkg_key],
            )
            measured_B_values_toy[key] = best_rho

    measured_Cij_values_toy = {}
    for Cij in Cij_elements:
        x_var = f'{var_prefix}_cos{Cij[0]}_plus'
        y_var = f'{var_prefix}_cos{Cij[1]}_minus'
        key = (x_var, y_var)

        sig_prod_all = sig_prod_vals[key]
        toy_sig_counts, _ = np.histogram(sig_prod_all[sig_idx], bins=n_bins, range=(-1, 1), weights=toy_sig_wt)
        toy_sig_counts = toy_sig_counts.astype(np.float64)
        if toy_sig_counts.sum() > 0:
            toy_sig_counts *= N_sig_toy_events / toy_sig_counts.sum()

        if N_bkg_toy_events > 0:
            bkg_prod_all = bkg_prod_vals[key]
            toy_bkg_counts, _ = np.histogram(bkg_prod_all[bkg_idx], bins=n_bins, range=(-1, 1))
            toy_bkg_counts = toy_bkg_counts.astype(np.float64)
            if toy_bkg_counts.sum() > 0:
                toy_bkg_counts *= N_bkg_toy_events / toy_bkg_counts.sum()
        else:
            toy_bkg_counts = np.zeros(n_bins, dtype=np.float64)

        toy_counts = toy_sig_counts + toy_bkg_counts

        best_rho, rho_low, rho_high = nll_scan_np(
            toy_counts, model_hists_per_Cij[Cij], vals,
            N_sig_exp=N_sig_toy_events, N_bkg_exp=N_bkg_toy_events,
            bkg_counts=fixed_bkg_templates[key],
        )
        measured_Cij_values_toy[Cij] = best_rho  # rho = Cij directly

    C_toy = np.array([[measured_Cij_values_toy["nn"], measured_Cij_values_toy["nr"], measured_Cij_values_toy["nk"]],
                      [measured_Cij_values_toy["rn"], measured_Cij_values_toy["rr"], measured_Cij_values_toy["rk"]],
                      [measured_Cij_values_toy["kn"], measured_Cij_values_toy["kr"], measured_Cij_values_toy["kk"]]])
    if args.measure_B:
        _Bp_toy = np.array([[measured_B_values_toy.get(('p',a), 0.)] for a in _axes])
        _Bm_toy = np.array([[measured_B_values_toy.get(('m',a), 0.)] for a in _axes])
        con_toy, m12_toy = EntanglementVariables(C_toy, Bplus=_Bp_toy, Bminus=_Bm_toy)
    else:
        con_toy, m12_toy = EntanglementVariables(C_toy)
    con_toys.append(con_toy)
    m12_toys.append(m12_toy)
    Cnn_toys.append(C_toy[0,0])
    if args.measure_B:
        for w, a in B_elements:
            B_toys[(w, a)].append(measured_B_values_toy.get((w, a), 0.))

    if toy % print_every == 0 or toy == N_toys - 1:
        print(f"------------------------------------------------")
        print(f"Toy {toy+1}/{N_toys}")
        print(f"N_sig_toy_events = {N_sig_toy_events}, N_bkg_toy_events = {N_bkg_toy_events}, N_toy_events = {N_toy_events}")
        print(f"Con = {con_toy:.3f}, M12 = {m12_toy:.3f}")
        print(f"C = {C_toy}")

        # get uncertainties on con and m12 using toys (asymetric errors)
        
        con_toys_array = np.array(con_toys)
        m12_toys_array = np.array(m12_toys)
        
        con_err_low = np.percentile(con_toys_array, 16)
        con_err_high = np.percentile(con_toys_array, 84)
        
        m12_err_low = np.percentile(m12_toys_array, 16)
        m12_err_high = np.percentile(m12_toys_array, 84)
        
        con_mean = np.mean(con_toys_array)
        m12_mean = np.mean(m12_toys_array)
        
        con_median = np.median(con_toys_array)
        m12_median = np.median(m12_toys_array)

        Cnn_toys_array = np.array(Cnn_toys)

        Cnn_err_low = np.percentile(Cnn_toys_array, 16)
        Cnn_err_high = np.percentile(Cnn_toys_array, 84)
        Cnn_median = np.median(Cnn_toys_array)

        print(f"Mean Con from toys: {con_mean:.3f}")
        print(f"Mean M12 from toys: {m12_mean:.3f}")
        print(f"Median Con from toys: {con_median:.3f}")
        print(f"Median M12 from toys: {m12_median:.3f}")
        print(f"Concurrence 68% interval: {con_err_low:.3f} - {con_err_high:.3f}")
        print(f"M12 68% interval: {m12_err_low:.3f} - {m12_err_high:.3f}\n")

        azimov_Cnn = measured_Cij_values["nn"]
        print(f"Cnn  | Asimov: {azimov_Cnn[0]:.3f} (+{azimov_Cnn[2]-azimov_Cnn[0]:.3f}/-{azimov_Cnn[0]-azimov_Cnn[1]:.3f})  |  Toy median: {Cnn_median:.3f} ({Cnn_err_low-Cnn_median:.3f}/+{Cnn_err_high-Cnn_median:.3f})\n")

        if args.measure_B:
            print("B components (Asimov vs toy median):")
            B_labels = {'p': 'tau+', 'm': 'tau-'}
            for w, a in B_elements:
                az = measured_B_values[(w, a)]
                b_arr = np.array(B_toys[(w, a)])
                b_med = np.median(b_arr)
                b_lo  = np.percentile(b_arr, 16)
                b_hi  = np.percentile(b_arr, 84)
                print(f"  B{B_labels[w]}_{a} | Asimov: {az[0]:.3f} (+{az[2]-az[0]:.3f}/-{az[0]-az[1]:.3f})  |  Toy median: {b_med:.3f} ({b_lo-b_med:.3f}/+{b_hi-b_med:.3f})")
            print()

        branch_vals['iToy'][0] = toy
        branch_vals['con'][0] = con_toy
        branch_vals['m12'][0] = m12_toy
        branch_vals['Cnn'][0] = C_toy[0,0]
        branch_vals['Cnr'][0] = C_toy[0,1]
        branch_vals['Cnk'][0] = C_toy[0,2]
        branch_vals['Crn'][0] = C_toy[1,0]
        branch_vals['Crr'][0] = C_toy[1,1]
        branch_vals['Crk'][0] = C_toy[1,2]
        branch_vals['Ckn'][0] = C_toy[2,0]
        branch_vals['Ckr'][0] = C_toy[2,1]
        branch_vals['Ckk'][0] = C_toy[2,2]
        if args.measure_B:
            branch_vals['Bpn'][0] = measured_B_values_toy.get(('p','n'), 0.)
            branch_vals['Bpr'][0] = measured_B_values_toy.get(('p','r'), 0.)
            branch_vals['Bpk'][0] = measured_B_values_toy.get(('p','k'), 0.)
            branch_vals['Bmn'][0] = measured_B_values_toy.get(('m','n'), 0.)
            branch_vals['Bmr'][0] = measured_B_values_toy.get(('m','r'), 0.)
            branch_vals['Bmk'][0] = measured_B_values_toy.get(('m','k'), 0.)
        toy_tree_out.Fill()

        fout.cd()
        toy_tree_out.Write("", ROOT.TObject.kOverwrite)
        fout.Flush()

fout.Close()

    
    
    

