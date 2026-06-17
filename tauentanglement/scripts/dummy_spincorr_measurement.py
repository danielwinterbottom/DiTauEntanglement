import ROOT
ROOT.gROOT.SetBatch(True)
from numpy import arange
import numpy as np

def product_hist_correlated(
    hx,
    hy,
    name="hxy_corr",
    nbins=100,
    rho=0.0,
    normalize=True,
):
    """
    Build histogram of z = x*y from 1D ROOT histograms hx, hy,
    with an approximate correlation weight w = 1 + rho*x*y.

    Valid for x,y in [-1,1].
    rho > 0: favors x,y same sign
    rho < 0: favors x,y opposite sign
    """

    #if abs(rho) > 1:
    #    raise ValueError("For w = 1 + rho*x*y and x,y in [-1,1], use -1 <= rho <= 1")

    hz = ROOT.TH1D(name, ";x#times y;Probability", nbins, -1.0, 1.0)
    hz.Sumw2()

    for ix in range(1, hx.GetNbinsX() + 1):
        x = hx.GetBinCenter(ix)
        wx = hx.GetBinContent(ix)

        if wx == 0:
            continue

        for iy in range(1, hy.GetNbinsX() + 1):
            y = hy.GetBinCenter(iy)
            wy = hy.GetBinContent(iy)

            if wy == 0:
                continue

            corr_weight = 1.0 + rho * x * y

            z = x * y
            w = wx * wy * corr_weight

            hz.Fill(z, w)

    if normalize and hz.Integral() > 0:
        hz.Scale(1.0 / hz.Integral())

    #check if any bins are negative
    for ibin in range(1, hz.GetNbinsX() + 1):
        if hz.GetBinContent(ibin) < 0:
            print("Warning: bin", ibin, "has negative content:", hz.GetBinContent(ibin))

    return hz

x_var = 'true_cosr_plus'
y_var = 'true_cosn_minus'
cuts = 'true_taup_npizero==1&&true_taun_npizero==1&&true_taup_is3prong==0&&true_taun_is3prong==0'

#x_var = 'map_pred_cosk_plus'
#y_var = 'map_pred_cosk_minus'
#cuts = 'reco_taup_npizero==1&&reco_taun_npizero==1&&reco_taup_is3prong==0&&reco_taun_is3prong==0'


n_bins=10

f1 = ROOT.TFile.Open("outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_100e_June5/evalJun12_output_results_CPEven.root")
t1 = f1.Get("tree")
f2 = ROOT.TFile.Open("outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_100e_June5/evalJun12_output_results_CPMix.root")
t2 = f2.Get("tree")

#first get fine binned histograms for x and y (10*n_bins)
h_x = t1.Draw(x_var + '>>h_x(' + str(n_bins*10) + ',-1,1)', cuts)
h_x = t1.GetHistogram()
h_y = t1.Draw(y_var + '>>h_y(' + str(n_bins*10) + ',-1,1)', cuts)
h_y = t1.GetHistogram()

#now get true product histogram
h_prod = t2.Draw(x_var + '*' + y_var + '>>h_prod(' + str(n_bins) + ',-1,1)', cuts)
h_prod = t2.GetHistogram()
h_prod.Scale(1.0 / h_prod.Integral())

chi2_values = []
rho_values = []

step=0.01

vals = arange(-1.2, 1.2 + step, step)
if 1 not in vals: 
    vals = np.append(vals, 1) 
if -1 not in vals:
    vals = np.append(vals, -1)

vals = np.sort(vals)

for rho in vals:

    h_pred = product_hist_correlated(
        h_x, h_y,
        name=f"h_pred_{rho:.3f}",
        nbins=n_bins,
        rho=rho,
        normalize=True,
    )

    chi2 = h_prod.Chi2Test(h_pred, "WW CHI2")
    chi2_values.append(chi2)
    rho_values.append(rho)

    del h_pred

rho_values = np.array(rho_values)
chi2_values = np.array(chi2_values)

i_best = np.argmin(chi2_values)
best_rho = rho_values[i_best]
best_chi2 = chi2_values[i_best]

# 1 sigma interval for one fitted parameter
target = best_chi2 + 1.0

mask_left = rho_values < best_rho
mask_right = rho_values > best_rho

if len(chi2_values[mask_left]) == 0:
    rho_low = best_rho
else:
    rho_low = np.interp(
        target,
        chi2_values[mask_left][::-1],
        rho_values[mask_left][::-1],
    )

if len(chi2_values[mask_right]) == 0:
    rho_high = best_rho
else:
    rho_high = np.interp(
        target,
        chi2_values[mask_right],
        rho_values[mask_right],
    )

rho_high = np.interp(
    target,
    chi2_values[mask_right],
    rho_values[mask_right],
)

print("Best rho:", best_rho)
print("1 sigma interval:", rho_low, rho_high)
print("rho uncertainty: -", best_rho - rho_low, "+", rho_high - best_rho)

###

h_pred_corr = product_hist_correlated(h_x, h_y, name="h_pred_corr", nbins=n_bins, rho=best_rho, normalize=True)

canv = ROOT.TCanvas("canv", "canv", 800, 600)
h_prod.SetTitle("")
h_prod.SetStats(0)
h_prod.SetLineColor(ROOT.kBlack)
h_prod.SetLineWidth(2)
h_prod.Draw("hist")
h_pred_corr.SetLineColor(ROOT.kRed)
h_pred_corr.SetLineWidth(2)
h_pred_corr.Draw("hist same")
leg = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
leg.AddEntry(h_prod, "True product", "l")
leg.AddEntry(h_pred_corr, "Predicted product", "l")
leg.Draw()
canv.Print("product_hist_correlated.pdf")


