import ROOT
ROOT.gROOT.SetBatch(1)

#lhc_pp_to_z_plusj_10M_master_external2_pythia8p310

#Run-2 Pythia setup
#f1 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_external0_dm_0_1_2_10_mu_pythia8p240/pythia_events.root')
#f2 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_external1_dm_0_1_2_10_mu_pythia8p240/pythia_events.root')
#f3 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_external2_dm_0_1_2_10_mu_pythia8p240/pythia_events.root')

#Run-3 Pythia setup
#f1 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_external0_dm_0_1_2_10_mu_pythia8p310/pythia_events.root')
#f2 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_external1_dm_0_1_2_10_mu_pythia8p310/pythia_events.root')
#f3 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_external2_dm_0_1_2_10_mu_pythia8p310/pythia_events.root')

# Run-3 Pythia setup with modified GF to change the weinberg angle
#f1 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_master_modGF_external0_pythia8p310/pythia_events.root')
#f2 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_master_modGF_external1_pythia8p310/pythia_events.root')
#f3 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_10M_master_modGF_external2_pythia8p310/pythia_events.root')

#Run-2 setup for a DY+j sample
f1 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_plusj_10M_master_external0_pythia8p310/pythia_events.root')
f2 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_plusj_10M_master_external1_pythia8p310/pythia_events.root')
f3 = ROOT.TFile('batch_job_outputs/lhc_pp_to_z_plusj_10M_master_external2_pythia8p310/pythia_events.root')

t1 = f1.Get('tree')
t2 = f2.Get('tree')
t3 = f3.Get('tree')


cuts = {
    'mupi_pospi': '(taup_vis_pt>40&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'mupi_negpi': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==0)',
    'a1a1_lowpT': 'taup_vis_pt>20&&taup_npi==3&&taup_npizero==0&&taun_vis_pt>20&&taun_npi==3&&taun_npizero==0',
    'pipi_lowpT': 'taup_vis_pt>20&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>20&&taun_npi==1&&taun_npizero==0',
    'pirho_lowpT': '(taup_vis_pt>20&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>20&&taun_npi==1&&taun_npizero==1) || (taup_vis_pt>20&&taup_npi==1&&taup_npizero==1&&taun_vis_pt>20&&taun_npi==1&&taun_npizero==0)',
    'rhorho_lowpT': 'taup_vis_pt>20&&taup_npi==1&&taup_npizero==1&&taun_vis_pt>20&&taun_npi==1&&taun_npizero==1',


    'pipi': 'taup_vis_pt>40&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==0',
    'pirho': '(taup_vis_pt>40&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==1) || (taup_vis_pt>40&&taup_npi==1&&taup_npizero==1&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==0)',
    'rhorho': 'taup_vis_pt>40&&taup_npi==1&&taup_npizero==1&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==1',
    'a1a1': 'taup_vis_pt>40&&taup_npi==3&&taup_npizero==0&&taun_vis_pt>40&&taun_npi==3&&taun_npizero==0',
    'hadhad': 'taup_vis_pt>40&&taup_npi>=1&&taup_npizero>=0&&taun_vis_pt>40&&taun_npi>=1&&taun_npizero>=0',
    'mupi': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==0) || (taup_vis_pt>40&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'murho': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>40&&taun_npi==1&&taun_npizero==1) || (taup_vis_pt>40&&taup_npi==1&&taup_npizero==1&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'mua1': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>40&&taun_npi==3&&taun_npizero==0) || (taup_vis_pt>40&&taup_npi==3&&taup_npizero==0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'muhad': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>40&&taun_npi>=1&&taun_npizero>=0) || (taup_vis_pt>40&&taup_npi>=1&&taup_npizero>=0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',

    'mupi_lowpT': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>20&&taun_npi==1&&taun_npizero==0) || (taup_vis_pt>20&&taup_npi==1&&taup_npizero==0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'murho_lowpT': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>20&&taun_npi==1&&taun_npizero==1) || (taup_vis_pt>20&&taup_npi==1&&taup_npizero==1&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'mua1_lowpT': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>20&&taun_npi==3&&taun_npizero==0) || (taup_vis_pt>20&&taup_npi==3&&taup_npizero==0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0)',
    'muhad_lowpT': '(taup_vis_pt>25&&taup_nmu==1&&taup_npi==0&&taun_vis_pt>20&&taun_npi>=1&&taun_npizero>=0) || (taup_vis_pt>20&&taup_npi>=1&&taup_npizero>=0&&taun_vis_pt>25&&taun_nmu==1&&taun_npi==0&&taun_npi==0)',
}

for name, cuts in cuts.items():
    
    t1.Draw('m_vis>>h1_%s(40,0,160)' % name, '('+cuts+')*1.001')
    h1 = t1.GetHistogram()
    t2.Draw('m_vis>>h2_%s(40,0,160)' % name, '('+cuts+')*1.001')
    h2 = t2.GetHistogram()
    t3.Draw('m_vis>>h3_%s(40,0,160)' % name, '('+cuts+')*1.001')
    h3 = t3.GetHistogram()

    h1.SetLineColor(ROOT.kRed)
    h3.SetLineColor(ROOT.kGreen+2)
    c1 = ROOT.TCanvas()


    # draw hists using stack
    hs = ROOT.THStack('hs', '')
    hs.Add(h1)
    hs.Add(h2)
    hs.Add(h3)
    hs.Draw('nostack')
    hs.GetXaxis().SetTitle('m_{vis} [GeV]')
    hs.GetYaxis().SetTitle('Events')

    N1 = h1.Integral(-1,-1)
    N2 = h2.Integral(-1,-1)
    N3 = h3.Integral(-1,-1)

    r = N1/ N2
    r_error = r * ((1/N1) + (1/N2))**0.5
    r2 = N1/ N3
    r2_error = r2 * ((1/N1) + (1/N3))**0.5

    leg = ROOT.TLegend(0.1, 0.95, 0.9, 0.9)
    leg.SetNColumns(2)
    leg.AddEntry(h1, 'externalMode = 0: N0=%.0f' % N1, 'l')
    leg.AddEntry(h2, 'externalMode = 1: N1=%.0f' % N2, 'l')
    leg.AddEntry(h3, 'externalMode = 2: N2=%.0f' % N3, 'l')
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.Draw()

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.05)
    text.DrawLatex(0.15, 0.85, name)
    text.DrawLatex(0.15, 0.80, 'N0/N1 = %.2f +/- %.2f' % (r,r_error))
    text.DrawLatex(0.15, 0.75, 'N0/N2 = %.2f +/- %.2f' % (r2,r2_error))

    m1= h1.GetMean()
    m2= h2.GetMean()
    print(f'{name}: {m1:.2f} GeV (externalMode=0), {m2:.2f} GeV (externalMode=1)')

    c1.Print('m_vis_comps_%s.pdf' % name)


for tau in ['taup', 'taun']:

    cuts = f'({tau}_npi==1&&{tau}_npizero==0)'
    t1.Draw(f'({tau}_e-{tau}_nu_e)/{tau}_e>>h1_Efrac_{tau}(40,0,1)', '('+cuts+')*1.001')
    h1 = t1.GetHistogram()
    t2.Draw(f'({tau}_e-{tau}_nu_e)/{tau}_e>>h2_Efrac_{tau}(40,0,1)', '('+cuts+')*1.001')
    h2 = t2.GetHistogram()
    t3.Draw(f'({tau}_e-{tau}_nu_e)/{tau}_e>>h3_Efrac_{tau}(40,0,1)', '('+cuts+')*1.001')
    h3 = t3.GetHistogram()

    h1.SetLineColor(ROOT.kRed)
    h3.SetLineColor(ROOT.kGreen+2)
    c1 = ROOT.TCanvas()


    # draw hists using stack
    hs = ROOT.THStack('hs', '')
    hs.Add(h1)
    hs.Add(h2)
    hs.Add(h3)
    hs.Draw('nostack')
    hs.GetXaxis().SetTitle('E_{#pi^{#pm}}/E_{#tau}')
    hs.GetYaxis().SetTitle('Events')

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextSize(0.05)
    if tau == 'taup': text.DrawLatex(0.15, 0.15, '#tau^{+}#rightarrow#pi^{+}#nu')
    else:             text.DrawLatex(0.15, 0.15, '#tau^{-}#rightarrow#pi^{-}#nu')

    leg = ROOT.TLegend(0.1, 0.95, 0.9, 0.9)
    leg.SetNColumns(2)
    leg.AddEntry(h1, 'externalMode = 0', 'l')
    leg.AddEntry(h2, 'externalMode = 1', 'l')
    leg.AddEntry(h3, 'externalMode = 2', 'l')
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.Draw()    

    c1.Print('E_frac_comps_%s.pdf' % tau)