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
parser.add_argument("--run_calibration", action="store_true", help="Run calibration step to get calibration functions for each Cij element. If not set, will use precomputed calibration functions from calibration_funcs_rewt.root")
parser.add_argument("--sig-file", default="../outputs_model_LHC_TransformerFlow_Hadronic_100e_June22_TRIAL2/output_results_UnCorr.parquet")
parser.add_argument("--bkg-file", default="../outputs_model_LHC_TransformerFlow_Hadronic_100e_June22_TRIAL2/output_results_ZToTauTau.parquet")
parser.add_argument("--no-replace", action="store_true", help="Sample toys without replacement (non-overlapping chunks). Limits N_toys to pool_size/N_sig_events.")
parser.add_argument("--n-toys", type=int, default=1000, help="Number of toys to generate. If --no-replace is set, this will be limited to pool_size/N_sig_events.")
args = parser.parse_args()


def _build_product_hist_cache(hx_vals, hy_vals, nbins):
    """Pre-compute bin-index and outer-product arrays for product_hist_np (call once per Cij)."""
    nx = len(hx_vals)
    ny = len(hy_vals)
    x_centers = np.linspace(-1 + 1/nx, 1 - 1/nx, nx)
    y_centers = np.linspace(-1 + 1/ny, 1 - 1/ny, ny)
    z_edges = np.linspace(-1, 1, nbins + 1)

    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')  # (nx, ny)
    zz = xx * yy
    iz = np.clip(np.digitize(zz, z_edges) - 1, 0, nbins - 1)   # (nx, ny)
    ww = np.outer(hx_vals, hy_vals)                              # (nx, ny)
    xy = xx * yy                                                  # same as zz, kept for clarity
    return iz, ww, xx, yy

_product_cache = {}

def product_hist_np(hx_vals, hy_vals, rho, nbins=10):
    """Vectorized version of product_hist_correlated."""
    key = (id(hx_vals), id(hy_vals), nbins)  # arrays are fixed, id is stable
    if key not in _product_cache:
        _product_cache[key] = _build_product_hist_cache(hx_vals, hy_vals, nbins)
    iz, ww, xx, yy = _product_cache[key]

    corr = 1.0 + rho * xx * yy   # (nx, ny)
    weights = ww * corr           # (nx, ny)

    hz = np.zeros(nbins)
    np.add.at(hz, iz.ravel(), weights.ravel())

    total = hz.sum()
    if total > 0:
        hz /= total
    return hz

def nll_scan_np(data_counts, hx_vals, hy_vals, vals, nbins=10, N_sig_exp=None, N_bkg_exp=None, bkg_counts=None):
    """Pure numpy version of nll_scan — no ROOT objects in hot loop."""
    if bkg_counts is not None and (N_sig_exp is None or N_bkg_exp is None):
        raise ValueError("If bkg_counts is provided, N_sig_exp and N_bkg_exp must be provided.")

    Nexp = (N_sig_exp + N_bkg_exp) if bkg_counts is not None else (N_sig_exp if N_sig_exp is not None else data_counts.sum())

    rho_values = []
    nll_values = []

    for rho in vals:
        pred = product_hist_np(hx_vals, hy_vals, rho, nbins=nbins)

        if np.any(pred < 0):
            continue

        if bkg_counts is not None:
            pred = pred * N_sig_exp
            bkg_norm = bkg_counts * (N_bkg_exp / bkg_counts.sum() if bkg_counts.sum() > 0 else 0)
            pred = pred + bkg_norm
            total = pred.sum()
            if total > 0:
                pred /= total

        mu = Nexp * pred
        n = data_counts

        mask = mu > 0
        nll = 2.0 * np.sum(mu[mask] - n[mask] * np.log(mu[mask]))

        rho_values.append(rho)
        nll_values.append(nll)

    nll_values = np.array(nll_values)
    rho_values = np.array(rho_values)

    i_best = np.argmin(nll_values)
    best_rho = rho_values[i_best]
    best_nll = nll_values[i_best]
    delta_nll = nll_values - best_nll

    left = rho_values < best_rho
    right = rho_values > best_rho

    rho_low = np.interp(1.0, delta_nll[left][::-1], rho_values[left][::-1]) if left.any() else best_rho
    rho_high = np.interp(1.0, delta_nll[right], rho_values[right]) if right.any() else best_rho

    if rho_low > rho_high:
        rho_low, rho_high = rho_high, rho_low

    return best_rho, rho_low, rho_high


def ComputeCPScalingFractions(phiCP):
    phiCP_rad = np.radians(phiCP)
    cos_phiCP = np.cos(phiCP_rad)
    sin_phiCP = np.sin(phiCP_rad)
    cpeven_scaling = cos_phiCP**2 - cos_phiCP * sin_phiCP
    cpodd_scaling = sin_phiCP**2 - cos_phiCP * sin_phiCP
    cpmix_scaling = 2 * cos_phiCP * sin_phiCP

    return cpeven_scaling, cpodd_scaling, cpmix_scaling

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

def PlotCalibrationGraph(x_points, y_points, Cij, postfix=''):

    #sort x_point and y_points by x_points (lowest to highest)
    x_points, y_points = zip(*sorted(zip(x_points, y_points)))

    gr = ROOT.TGraph(len(x_points), array('d', x_points), array('d', y_points))
    canv = ROOT.TCanvas("canv", "canv", 800, 600)
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(1.5)
    gr.SetTitle("")
    gr.GetXaxis().SetTitle("rho (reco)")
    gr.GetYaxis().SetTitle(f"C{Cij[0]}{Cij[1]} (gen)")
    if Cij == "kk":
        func = ROOT.TF1(f"func_{Cij}{postfix}", "pol2", -1, 1)
    else: 
        func = ROOT.TF1(f"func_{Cij}{postfix}", "pol1", -1, 1)
    gr.Fit(func)
    gr.Draw("AP")
    # display the function parameters on the plot
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.04)
    text.DrawLatex(0.2, 0.93, f"y = {func.GetParameter(0):.3f} + {func.GetParameter(1):.3f}*x" + (f"{func.GetParameter(2):.3f}*x^2" if func.GetNpar() > 2 else "") )
    canv.Print(f"calibration_graph_C{Cij}{postfix}.pdf")
    print(f"Calibration for C{Cij}: C{Cij} = {func.GetParameter(0):.3f} + {func.GetParameter(1):.3f}*rho (reco)")
    # if kk then store the graph as a root file for later use (useful for selecting best functions)
    if Cij == "kk":
        fout = ROOT.TFile.Open(f"calibration_graph_C{Cij}{postfix}.root", "RECREATE")
        gr.Write()
        func.Write()
        fout.Close()
    return func

def GetWeightFromCMatrix(C):
    wt = f"1 + {C[0,0]}*wt_hp_n_hm_n + {C[0,1]}*wt_hp_n_hm_r + {C[0,2]}*wt_hp_n_hm_k \
    + {C[1,0]}*wt_hp_r_hm_n + {C[1,1]}*wt_hp_r_hm_r + {C[1,2]}*wt_hp_r_hm_k \
    + {C[2,0]}*wt_hp_k_hm_n + {C[2,1]}*wt_hp_k_hm_r + {C[2,2]}*wt_hp_k_hm_k"
    return wt

_axes = ['n', 'r', 'k']
_wt_cols = [f'wt_hp_{a}_hm_{b}' for a in _axes for b in _axes]

def compute_event_weights(df, C):
    """Compute per-event spin weights from C matrix using parquet wt_hp_*_hm_* columns."""
    wt = np.ones(len(df), dtype=np.float64)
    for i, a in enumerate(_axes):
        for j, b in enumerate(_axes):
            if C[i, j] != 0:
                wt += C[i, j] * df[f'wt_hp_{a}_hm_{b}'].values
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
_needed_cols = (
    [f'{var_prefix}_cos{a}_{s}' for a in _axes for s in ('plus', 'minus')] +
    _wt_cols +
    ['true_taup_npizero', 'true_taun_npizero', 'true_taup_is3prong', 'true_taun_is3prong',
     'reco_taup_npizero', 'reco_taun_npizero', 'reco_taup_is3prong', 'reco_taun_is3prong']
)
df_sig_raw = pd.read_parquet(args.sig_file, columns=_needed_cols)
_bkg_cols = [c for c in _needed_cols if c not in _wt_cols]
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
vals = arange(-1.2, 1.2 + step, step) #TODO try extending range?
if 1 not in vals: 
    vals = np.append(vals, 1) 
if -1 not in vals:
    vals = np.append(vals, -1)
vals = np.sort(vals)

if args.run_calibration:

    calibration_funcs_rewt = {}
    Cij_elements = ["nn", "rr", "nr", "rn", "kk", "kn", "kr", "nk", "rk", "kk"]
    for Cij in Cij_elements:
        print(f"\nCoefficient {Cij} from reweighting")
        x_var = f'{var_prefix}_cos{Cij[0]}_plus'
        y_var = f'{var_prefix}_cos{Cij[1]}_minus'

        hx_calib, _ = np.histogram(df_sig[x_var].values, bins=n_bins*10, range=(-1,1))
        hy_calib, _ = np.histogram(df_sig[y_var].values, bins=n_bins*10, range=(-1,1))

        C_zeros = np.zeros((3,3))
        x_points = []
        y_points = []

        vals_list = [-1, -0.5, 0, 0.5, 1]
        if Cij == "kk":
            vals_list = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

        _ij = {'nn':(0,0),'rr':(1,1),'nr':(0,1),'rn':(1,0),'kn':(2,0),'kr':(2,1),'nk':(0,2),'rk':(1,2),'kk':(2,2)}
        ii, jj = _ij[Cij]

        for val in vals_list:
            C_rewt = C_zeros.copy()
            C_rewt[ii, jj] = val
            ev_wt = compute_event_weights(df_sig, C_rewt)
            prod = df_sig[x_var].values * df_sig[y_var].values
            h_azimov, _ = np.histogram(prod, bins=n_bins, range=(-1,1), weights=ev_wt)
            h_azimov = h_azimov.astype(np.float64)
            best_rho, _, _ = nll_scan_np(h_azimov, hx_calib, hy_calib, vals, nbins=n_bins, N_sig_exp=h_azimov.sum())
            x_points.append(best_rho)
            y_points.append(val)

        calibration_funcs_rewt[Cij] = PlotCalibrationGraph(x_points, y_points, Cij, postfix='_reweighting')

    print("\nCalibration coefficients from reweighting:")
    fout = ROOT.TFile.Open("calibration_funcs_rewt.root", "RECREATE")
    fout.cd()
    for Cij, func in calibration_funcs_rewt.items():
        print(f"C{Cij}: {func.GetParameter(0):.3f} + {func.GetParameter(1):.3f}*rho (reco)")
        func.Write()
    fout.Close()

f_calib = ROOT.TFile.Open("calibration_funcs_rewt.root")

if args.model == 1:
    print("Using CP-even Higgs model for signal")
    phiCP = 0
    C_rwt = GetSpinCorrelationMatrix(phiCP)
elif args.model == 2:
    print("Using spin-0 no entanglement model for signal")
    C_rwt = C = np.array(
            [[0,0,0],
            [0,0,0],
            [0,0,-1]], dtype=float
        )
elif args.model == 3:
    print("Using uncorrelated model for signal")
    C_rwt = C = np.array(
            [[0,0,0],
            [0,0,0],
            [0,0,0]], dtype=float
        )
else:
    raise ValueError(f"Invalid model: {args.model}. Allowed models are: 1 = CP-even Higgs, 2 = spin0 no entanglement, 3 = uncorrelated")


wt = GetWeightFromCMatrix(C_rwt)

# pre-compute event weights for signal model
sig_model_wt = compute_event_weights(df_sig, C_rwt)

# pre-cache marginal histograms (fine-binned) keyed by variable name
hists_1d_np = {}
for a in _axes:
    for s in ('plus', 'minus'):
        v = f'{var_prefix}_cos{a}_{s}'
        hists_1d_np[v], _ = np.histogram(df_sig[v].values, bins=n_bins*10, range=(-1,1))

Cij_elements = ["nn", "rr", "nr", "rn", "kk", "kn", "kr", "nk", "rk", "kk"]
C_exp = C_rwt
measured_rho_values = {}
measured_Cij_values = {}

for Cij in Cij_elements:
    calib_func = f_calib.Get(f"func_{Cij}_reweighting")
    x_var = f'{var_prefix}_cos{Cij[0]}_plus'
    y_var = f'{var_prefix}_cos{Cij[1]}_minus'

    hx_np = hists_1d_np[x_var]
    hy_np = hists_1d_np[y_var]

    sig_prod = df_sig[x_var].values * df_sig[y_var].values
    h_azimov_sig_np, _ = np.histogram(sig_prod, bins=n_bins, range=(-1,1), weights=sig_model_wt)
    h_azimov_sig_np = h_azimov_sig_np.astype(np.float64)
    if h_azimov_sig_np.sum() > 0:
        h_azimov_sig_np *= N_sig_events / h_azimov_sig_np.sum()

    bkg_prod = df_bkg[x_var].values * df_bkg[y_var].values
    h_azimov_bkg_np, _ = np.histogram(bkg_prod, bins=n_bins, range=(-1,1))
    h_azimov_bkg_np = h_azimov_bkg_np.astype(np.float64)
    if h_azimov_bkg_np.sum() > 0:
        h_azimov_bkg_np *= N_bkg_events / h_azimov_bkg_np.sum()

    h_azimov_np = h_azimov_sig_np + h_azimov_bkg_np

    best_rho, rho_low, rho_high = nll_scan_np(
        h_azimov_np, hx_np, hy_np, vals, nbins=n_bins,
        N_sig_exp=N_sig_events, N_bkg_exp=N_bkg_events, bkg_counts=h_azimov_bkg_np,
    )
    print(f"Fitted rho for C{Cij}: {best_rho:.3f} (+{rho_high-best_rho:.3f}/-{best_rho-rho_low:.3f})")
    Cij_fitted = calib_func.Eval(best_rho)
    Cij_low = calib_func.Eval(rho_low)
    Cij_high = calib_func.Eval(rho_high)
    if Cij_low > Cij_high:
        Cij_low, Cij_high = Cij_high, Cij_low

    measured_rho_values[Cij] = (best_rho, rho_low, rho_high)
    measured_Cij_values[Cij] = (Cij_fitted, Cij_low, Cij_high)

    # plotting — convert numpy arrays to ROOT histograms just for the canvas
    def _np_to_th1(arr, name):
        h = ROOT.TH1D(name, "", n_bins, -1, 1)
        h.Sumw2()
        for i, v in enumerate(arr):
            h.SetBinContent(i+1, v)
            h.SetBinError(i+1, np.sqrt(abs(v)))
        return h

    h_data_root = _np_to_th1(h_azimov_np, "h_data")
    h_azimov_bkg_root = _np_to_th1(h_azimov_bkg_np, "h_azimov_bkg")
    pred_corr_np = product_hist_np(hx_np, hy_np, best_rho, nbins=n_bins) * N_sig_events
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
    leg.AddEntry(dummy_hist, f"Fitted #rho = {best_rho:.2f}^{{+{rho_high-best_rho:.2f}}}_{{-{best_rho-rho_low:.2f}}}", "l")
    leg.AddEntry(dummy_hist, f"Fitted C{Cij} = {Cij_fitted:.2f}^{{+{Cij_high-Cij_fitted:.2f}}}_{{-{Cij_fitted-Cij_low:.2f}}}", "l")
    leg.AddEntry(dummy_hist, f"True C{Cij} = {GetMatrixCoefficient(C_exp, Cij):.2f}", "l")
    leg.Draw()
    if args.model == 1:
        canv.Print(f"C{Cij}_fitted_phiCP{phiCP:.0f}.pdf")
    elif args.model == 2:
        canv.Print(f"C{Cij}_fitted_spin0_noentanglement.pdf")
    elif args.model == 3:
        canv.Print(f"C{Cij}_fitted_uncorrelated.pdf")

    del h_data_root, h_pred_corr_root, h_azimov_bkg_root, canv, dummy_hist

#print measured rho values and Cij values
print("\nMeasured rho values:")
for Cij, (best_rho, rho_low, rho_high) in measured_rho_values.items():
    print(f"C{Cij}: rho = {best_rho:.3f} (+{rho_high-best_rho:.3f}/-{best_rho-rho_low:.3f})")
print("\nMeasured Cij values:")
for Cij, (Cij_val, Cij_low, Cij_high) in measured_Cij_values.items():
    print(f"C{Cij}: C{Cij} = {Cij_val:.3f} (+{Cij_high-Cij_val:.3f}/-{Cij_val-Cij_low:.3f})")
print ("\nTrue Cij values:")
for Cij in Cij_elements:
    print(f"C{Cij}: C{Cij} = {GetMatrixCoefficient(C_exp, Cij):.3f}")


C = np.array([[measured_Cij_values["nn"][0], measured_Cij_values["nr"][0], measured_Cij_values["nk"][0]],
              [measured_Cij_values["rn"][0], measured_Cij_values["rr"][0], measured_Cij_values["rk"][0]],
              [measured_Cij_values["kn"][0], measured_Cij_values["kr"][0], measured_Cij_values["kk"][0]]])
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

print_every = 1

# extract numpy arrays from the already-cut dataframes (no ROOT needed)
sig_cos = {v: df_sig[v].values for v in hists_1d_np}
bkg_cos = {v: df_bkg[v].values for v in hists_1d_np if v in df_bkg.columns}

# pre-compute fixed background templates (from full bkg sample) for use in toy NLL
# using the same template in both data and model cancels bkg fluctuations artificially
fixed_bkg_templates = {}
for Cij in Cij_elements:
    x_var = f'{var_prefix}_cos{Cij[0]}_plus'
    y_var = f'{var_prefix}_cos{Cij[1]}_minus'
    key = (x_var, y_var)
    if key not in fixed_bkg_templates:
        bkg_prod = df_bkg[x_var].values * df_bkg[y_var].values
        tmpl, _ = np.histogram(bkg_prod, bins=n_bins, range=(-1, 1))
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
fout = ROOT.TFile.Open("toys_tree.root", "RECREATE")
fout.cd()
toy_tree_out = ROOT.TTree("toy_tree", "toy_tree")
branches = [
  'iToy', 'con', 'm12',
  'Cnn', 'Cnr', 'Cnk',
  'Crn', 'Crr', 'Crk',
  'Ckn', 'Ckr', 'Ckk'
]
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
    else:
        sig_idx = np.random.choice(n_sig_total, size=N_sig_toy_events, replace=True, p=sig_model_wt/sig_model_wt.sum())
        bkg_idx = np.random.choice(n_bkg_total, size=N_bkg_toy_events, replace=True)

    measured_Cij_values_toy = {}
    for Cij in Cij_elements:
        calib_func = f_calib.Get(f"func_{Cij}_reweighting")
        x_var = f'{var_prefix}_cos{Cij[0]}_plus'
        y_var = f'{var_prefix}_cos{Cij[1]}_minus'

        sig_prod = sig_cos[x_var][sig_idx] * sig_cos[y_var][sig_idx]
        toy_sig_counts, _ = np.histogram(sig_prod, bins=n_bins, range=(-1, 1))
        toy_sig_counts = toy_sig_counts.astype(np.float64)
        if toy_sig_counts.sum() > 0:
            toy_sig_counts *= N_sig_toy_events / toy_sig_counts.sum()

        if N_bkg_toy_events > 0:
            bkg_prod = bkg_cos[x_var][bkg_idx] * bkg_cos[y_var][bkg_idx]
            toy_bkg_counts, _ = np.histogram(bkg_prod, bins=n_bins, range=(-1, 1))
            toy_bkg_counts = toy_bkg_counts.astype(np.float64)
            if toy_bkg_counts.sum() > 0:
                toy_bkg_counts *= N_bkg_toy_events / toy_bkg_counts.sum()
        else:
            toy_bkg_counts = np.zeros(n_bins, dtype=np.float64)

        toy_counts = toy_sig_counts + toy_bkg_counts

        best_rho, rho_low, rho_high = nll_scan_np(
            toy_counts, hists_1d_np[x_var], hists_1d_np[y_var], vals,
            nbins=n_bins, N_sig_exp=N_sig_toy_events, N_bkg_exp=N_bkg_toy_events,
            bkg_counts=fixed_bkg_templates[(x_var, y_var)],
        )
        measured_Cij_values_toy[Cij] = calib_func.Eval(best_rho)

    C_toy = np.array([[measured_Cij_values_toy["nn"], measured_Cij_values_toy["nr"], measured_Cij_values_toy["nk"]],
                      [measured_Cij_values_toy["rn"], measured_Cij_values_toy["rr"], measured_Cij_values_toy["rk"]],
                      [measured_Cij_values_toy["kn"], measured_Cij_values_toy["kr"], measured_Cij_values_toy["kk"]]])
    con_toy, m12_toy = EntanglementVariables(C_toy)
    con_toys.append(con_toy)
    m12_toys.append(m12_toy)
    Cnn_toys.append(C_toy[0,0])

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

        print(f"Median Cnn from toys: {Cnn_median:.3f}")
        print(f"Cnn 68% interval: {Cnn_err_low:.3f} - {Cnn_err_high:.3f}\n")

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
        toy_tree_out.Fill()

        fout.cd()
        toy_tree_out.Write("", ROOT.TObject.kOverwrite)
        fout.Flush()

fout.Close()

    
    
    

