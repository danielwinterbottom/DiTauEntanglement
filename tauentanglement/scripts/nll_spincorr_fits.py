import ROOT
ROOT.gROOT.SetBatch(True)
from numpy import arange
import numpy as np
import matplotlib.pyplot as plt
from array import array

def product_hist_correlated(
    hx,
    hy,
    name="hxy_corr",
    nbins=10,
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

    hz = ROOT.TH1D(name, "", nbins, -1.0, 1.0)
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

def GetDataTree(tree,N=1000,replace=False):
    # get a tree with N events sampled at random from the tree
    N_entries = tree.GetEntries()
    if N > N_entries:
        N = N_entries
    indices = np.random.choice(N_entries, N, replace=replace)
    new_tree = tree.CloneTree(0)
    for i in indices:
        tree.GetEntry(int(i))
        new_tree.Fill()
    return new_tree

def nll_scan(h_data, h_x, h_y, vals, verbose=False):

    rho_values = []
    nll_values = []

    Nobs = h_data.Integral()

    for rho in vals:

        h_pred = product_hist_correlated(
            h_x,
            h_y,
            rho=rho,
            nbins=h_data.GetNbinsX(),
            normalize=True,
        )

        nll = 0

        for i in range(1, h_data.GetNbinsX()+1):

            n = h_data.GetBinContent(i)
            mu = Nobs * h_pred.GetBinContent(i)

            if mu < 0:
                raise ValueError(f"Expected count mu={mu} is non-positive for bin {i} with observed count n={n}.")
            if n < 0:
                raise ValueError(f"Observed count n={n} is negative for bin {i}.")

            nll += 2.0 * (mu - n * np.log(mu))

        rho_values.append(rho)
        nll_values.append(nll)

        del h_pred

    nll_values = np.array(nll_values)
    rho_values = np.array(rho_values)

    i_best = np.argmin(nll_values)
    best_rho = rho_values[i_best]
    best_nll = nll_values[i_best]

    delta_nll = nll_values - best_nll

    target = 1.0

    left = rho_values < best_rho
    right = rho_values > best_rho

    if len(delta_nll[left]) == 0:
        rho_low = best_rho
    else:
        rho_low = np.interp(
            target,
            delta_nll[left][::-1],
            rho_values[left][::-1],
        )

    if len(delta_nll[right]) == 0:
        rho_high = best_rho
    else:
        rho_high = np.interp(
            target,
            delta_nll[right],
            rho_values[right],
        )

    if verbose:
        print("Best rho:", best_rho)
        print("Best -2DeltaLogL:", best_nll)
        print("rho interval:", rho_low, rho_high)
        print("rho uncertainty: -", best_rho - rho_low, "+", rho_high - best_rho)

    return best_rho, rho_low, rho_high

def SampleToyFromAzimov(h_azimov):
    # sample a toy dataset from the azimov histogram
    # return a histogram with the same binning as h_azimov
    h_toy = ROOT.TH1D("h_toy", "", h_azimov.GetNbinsX(), -1.0, 1.0)
    h_toy.Sumw2()
    for i in range(1, h_azimov.GetNbinsX()+1):
        mu = h_azimov.GetBinContent(i)
        n = np.random.poisson(mu)
        h_toy.SetBinContent(i, n)
    return h_toy

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
        [0,0,-1]]
    )

    return C

def GetMarixCoefficient(C, element):
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

def PlotCalibrationGraph(x_points, y_points, Cij):

    #sort x_point and y_points by x_points (lowest to highest)
    x_points, y_points = zip(*sorted(zip(x_points, y_points)))

    gr = ROOT.TGraph(len(x_points), array('d', x_points), array('d', y_points))
    canv = ROOT.TCanvas("canv", "canv", 800, 600)
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(1.5)
    gr.SetTitle("")
    gr.GetXaxis().SetTitle("rho (reco)")
    gr.GetYaxis().SetTitle(f"C{Cij[0]}{Cij[1]} (gen)")
    gr.Fit("pol1")
    func = gr.GetFunction("pol1")
    gr.Draw("AP")
    # display the function parameters on the plot
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.04)
    text.DrawLatex(0.2, 0.93, f"y = {func.GetParameter(0):.3f} + {func.GetParameter(1):.3f}*x")
    canv.Print(f"calibration_graph_C{Cij}.pdf")
    # now fit a new function where intercept is [0]
    func = ROOT.TF1("func", "[0]*x", -1, 1)
    gr.Fit(func, "R")
    coeff = func.GetParameter(0)
    print(f"Calibration for C{Cij}: C{Cij} = {coeff:.3f} * rho (reco)")
    return coeff

####fout = ROOT.TFile.Open("skimmed.root", "RECREATE")
####fout.cd()   # important: make output file the current directory
####print("Applying cuts to data tree")
####t2 = t2.CopyTree(cuts)
####print("Finished applying cuts to data tree")

#####now get true product histogram
####data_tree = GetDataTree(t2, N=N_events, replace=False)
####
####h_data = data_tree.Draw(x_var + '*' + y_var + '>>h_data(' + str(n_bins) + ',-1,1)', cuts)
####h_data = data_tree.GetHistogram()
####
####print("Data histogram integral:", h_data.Integral())

coeff_check = True

#x_var = 'true_cosn_plus'
#y_var = 'true_cosn_minus'

## dm 1, 1
#gen_cuts =  'true_taup_npizero==1&&true_taun_npizero==1&&true_taup_is3prong==0&&true_taun_is3prong==0'
#cuts = 'reco_taup_npizero==1&&reco_taun_npizero==1&&reco_taup_is3prong==0&&reco_taun_is3prong==0' 

##dm 0, 0
#gen_cuts =  'true_taup_npizero==0&&true_taun_npizero==0&&true_taup_is3prong==0&&true_taun_is3prong==0'
#cuts = 'reco_taup_npizero==0&&reco_taun_npizero==0&&reco_taup_is3prong==0&&reco_taun_is3prong==0' 

##dm 1, 0
#gen_cuts =  'true_taup_npizero==1&&true_taun_npizero==0&&true_taup_is3prong==0&&true_taun_is3prong==0'
#cuts = 'reco_taup_npizero==1&&reco_taun_npizero==0&&reco_taup_is3prong==0&&reco_taun_is3prong==0' 

#dm 0, 1
gen_cuts =  'true_taup_npizero==0&&true_taun_npizero==1&&true_taup_is3prong==0&&true_taun_is3prong==0'
cuts = 'reco_taup_npizero==0&&reco_taun_npizero==1&&reco_taup_is3prong==0&&reco_taun_is3prong==0'

x_var = 'map_pred_cosn_plus'
y_var = 'map_pred_cosn_minus'

if coeff_check:
    cuts += ' && ' + gen_cuts

Cij_elements = ["nn", "rr", "nr", "rn", "kk"] #, "kn", "kr", "nk", "rk", "kk"]

var_prefix = 'map_pred'
#var_prefix = 'true'

f1 = ROOT.TFile.Open("outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_100e_June5/evalJun12_output_results_CPEven.root")
t1 = f1.Get("tree")
f2 = ROOT.TFile.Open("outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_100e_June5/evalJun12_output_results_CPOdd.root")
t2 = f2.Get("tree")
f3 = ROOT.TFile.Open("outputs_model_LHC_TransformerFlow_Hadronic_AllDMs_100e_June5/evalJun12_output_results_CPMix.root")
t3 = f3.Get("tree")

n_bins=10
N_events = 2000
step=0.01
vals = arange(-1, 1 + step, step)
if 1 not in vals: 
    vals = np.append(vals, 1) 
if -1 not in vals:
    vals = np.append(vals, -1)
vals = np.sort(vals)

calibration_coeffs = {}

for Cij in Cij_elements:
    print(f"\nCoefficient {Cij}")

    x_var = f'{var_prefix}_cos{Cij[0]}_plus'
    y_var = f'{var_prefix}_cos{Cij[1]}_minus'

    #first get fine binned histograms for x and y (10*n_bins) - we will use these to construct the predicted product histogram for a given rho
    h_x = t1.Draw(x_var + '>>h_x(' + str(n_bins*10) + ',-1,1)', cuts)
    h_x = t1.GetHistogram()
    h_y = t1.Draw(y_var + '>>h_y(' + str(n_bins*10) + ',-1,1)', cuts)
    h_y = t1.GetHistogram()

    # get CP-even, CP-odd, and CP-mixed histograms, which will be used to make azimov histogram for a given phiCP
    t1.Draw(x_var + '*' + y_var + '>>h_even(' + str(n_bins) + ',-1,1)', cuts)
    h_even = t1.GetHistogram()
    t2.Draw(x_var + '*' + y_var + '>>h_odd(' + str(n_bins) + ',-1,1)', cuts)
    h_odd = t2.GetHistogram()
    t3.Draw(x_var + '*' + y_var + '>>h_mix(' + str(n_bins) + ',-1,1)', cuts)
    h_mix = t3.GetHistogram()

    # first do calibrations
    # for n,r elements this is based on the spin correlation matrix for phiCP=0 (SM Higgs)
    if "k" not in Cij:
        phiCP_vals = [-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90]
    elif Cij == "kk":
        phiCP_vals = [0]
    else:
        continue

    x_points = []
    y_points = []

    for phiCP in phiCP_vals:
        C = GetSpinCorrelationMatrix(phiCP)
        Cij_true = GetMarixCoefficient(C, Cij)

       
        cpeven_scaling, cpodd_scaling, cpmix_scaling = ComputeCPScalingFractions(phiCP)
        h_azimov = h_even.Clone("h_azimov")
        h_azimov.Scale(cpeven_scaling)
        h_azimov.Add(h_odd, cpodd_scaling)
        h_azimov.Add(h_mix, cpmix_scaling)

        best_rho, _, _ = nll_scan(h_azimov, h_x, h_y, vals, verbose=False)

        x_points.append(best_rho)
        y_points.append(Cij_true)

    if Cij == "kk":
        # add point at 0,0
        x_points.append(0.0)
        y_points.append(0.0)

    calibration_coeffs[Cij] = PlotCalibrationGraph(x_points, y_points, Cij)

# For Cij where i or j = k but i!=j, we use the coefficients for Ckk Cnn, and Crr to estimate the coefficients for Ckn, Ckr, Cnk, and Crk
# We assume the response to a spin-correlation coefficient factorizes into an analyzing power on the + side and an analyzing power on the − side
# This is an assumption that needs to be checked!!
Ckk_coeff = calibration_coeffs["kk"]
Cnn_coeff = calibration_coeffs["nn"]
Crr_coeff = calibration_coeffs["rr"]

calibration_coeffs["kn"] = np.sqrt(np.abs(Ckk_coeff * Cnn_coeff))*np.sign(Ckk_coeff)
calibration_coeffs["kr"] = np.sqrt(np.abs(Ckk_coeff * Crr_coeff))*np.sign(Ckk_coeff)
calibration_coeffs["nk"] = np.sqrt(np.abs(Cnn_coeff * Ckk_coeff))*np.sign(Ckk_coeff)
calibration_coeffs["rk"] = np.sqrt(np.abs(Crr_coeff * Ckk_coeff))*np.sign(Ckk_coeff)
    
print("\nCalibration coefficients:")
for Cij, coeff in calibration_coeffs.items():
    print(f"C{Cij}: {coeff:.3f}")

if coeff_check:
    print("\nChecking calibration coefficients by reweighting events with Ckn=Cnk=Ckr=Crk=1 and fitting for rho")
    #to check the above, lets make a reweighted histogram for Ckn = Cnk = Crk = Ckr = 1
    Ckr_rewt = 1
    Cnk_rewt = 1
    Ckn_rewt = 1
    Crk_rewt = 1
    wt = f"1+ {Cnk_rewt}*true_cosn_plus*true_cosk_minus + {Ckn_rewt}*true_cosk_plus*true_cosn_minus + {Ckr_rewt}*true_cosk_plus*true_cosr_minus + {Crk_rewt}*true_cosr_plus*true_cosk_minus"


    x_var = 'map_pred_cosn_plus'
    y_var = 'map_pred_cosk_minus'

    t3.Draw(f"{x_var}*{y_var}>>h_rewt({n_bins},-1,1)", f"({cuts})*({gen_cuts})*({wt})")
    h_rewt = t3.GetHistogram()

    canv = ROOT.TCanvas("canv", "canv", 800, 600)
    h_rewt.SetTitle("")
    h_rewt.SetStats(0)
    h_rewt.SetLineColor(ROOT.kBlack)
    h_rewt.SetLineWidth(2)
    h_rewt.Draw("hist")
    canv.Print("reweighted_histogram.pdf")

    h_x = t1.Draw(x_var + '>>h_x(' + str(n_bins*10) + ',-1,1)', cuts)
    h_x = t1.GetHistogram()
    h_y = t1.Draw(y_var + '>>h_y(' + str(n_bins*10) + ',-1,1)', cuts)
    h_y = t1.GetHistogram()

    best_rho, rho_low, rho_high = nll_scan(h_rewt, h_x, h_y, vals, verbose=True)
    print(f"Best rho for Ckn=Cnk=Ckr=Crk=1: {best_rho:.3f}")

exit()

phiCP=45 # phiCP in degrees

#first get fine binned histograms for x and y (10*n_bins) - we will use these to construct the predicted product histogram for a given rho
h_x = t1.Draw(x_var + '>>h_x(' + str(n_bins*10) + ',-1,1)', cuts)
h_x = t1.GetHistogram()
h_y = t1.Draw(y_var + '>>h_y(' + str(n_bins*10) + ',-1,1)', cuts)
h_y = t1.GetHistogram()

# get CP-even, CP-odd, and CP-mixed histograms, which will be used to make azimov histogram for a given phiCP
t1.Draw(x_var + '*' + y_var + '>>h_even(' + str(n_bins) + ',-1,1)', cuts)
h_even = t1.GetHistogram()
t2.Draw(x_var + '*' + y_var + '>>h_odd(' + str(n_bins) + ',-1,1)', cuts)
h_odd = t2.GetHistogram()
t3.Draw(x_var + '*' + y_var + '>>h_mix(' + str(n_bins) + ',-1,1)', cuts)
h_mix = t3.GetHistogram()


cpeven_scaling, cpodd_scaling, cpmix_scaling = ComputeCPScalingFractions(phiCP)
h_even.Scale(cpeven_scaling)
h_odd.Scale(cpodd_scaling)
h_mix.Scale(cpmix_scaling)
h_azimov = h_even.Clone("h_azimov")
h_azimov.Add(h_odd)
h_azimov.Add(h_mix)

#t2.Draw(x_var + '*' + y_var + '>>h_azimov(' + str(n_bins) + ',-1,1)', cuts)
#h_azimov = t2.GetHistogram()
#h_azimov.Scale(N_events / h_azimov.Integral())

print(f"phiCP: {phiCP} degrees")
#fit azimov
best_rho, rho_low, rho_high = nll_scan(h_azimov, h_x, h_y, vals, verbose=True)

# for plotting:
h_data = h_azimov.Clone("h_data")

h_pred_corr = product_hist_correlated(h_x, h_y, name="h_pred_corr", nbins=n_bins, rho=best_rho, normalize=True)
h_pred_corr.Scale(h_data.Integral())  # scale to same integral as h_data

canv = ROOT.TCanvas("canv", "canv", 800, 600)
h_data.SetTitle("")
h_data.SetStats(0)
h_data.SetLineColor(ROOT.kBlack)
h_data.SetLineWidth(2)
h_data.Draw("pE1")
h_pred_corr.SetLineColor(ROOT.kRed)
h_pred_corr.SetLineWidth(2)
h_pred_corr.Draw("hist same")
leg = ROOT.TLegend(0.6, 0.7, 0.9, 0.9)
leg.AddEntry(h_data, "True product", "pe")
leg.AddEntry(h_pred_corr, "Predicted product", "l")
leg.Draw()
canv.Print("product_hist_correlated.pdf")


