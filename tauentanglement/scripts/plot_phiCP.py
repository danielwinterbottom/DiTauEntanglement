import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep
import numpy as np
import awkward as ak
import argparse
import os

import math
from tauentanglement.utils.calculate_hh import Particle, _prepare_kinematic_for_hh, calculateHH, getHHVectors

from tauentanglement.utils.acoplanarity_tools import (
    compute_aco_polarimetric,
    get_R_P_vectors_all,
    compute_aco_classic,
    compute_aco_classic_a1a1,
    get_ditau_polarimetric,
)

plt.style.use(hep.style.CMS)
plt.rcParams.update({"font.size": 16})

options = {
    'files':{  # set files here (ones from eval have all info we need)
 'even': '/Users/dw515/ClaudeCode/testfiles/evalJun12_output_results_CPEven_FromLHE.parquet',
 'odd': '/Users/dw515/ClaudeCode/testfiles/evalJun12_output_results_CPOdd_FromLHE.parquet',
'sl_even': '/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/outputs_model_LHC_TransformerFlow_Semileptonic_AllDMs_25e_June8/output_results_CPEven.parquet',
'sl_odd': '/vols/cms/lcr119/offline/HiggsCP/DiTauEntanglement/outputs_model_LHC_TransformerFlow_Semileptonic_AllDMs_25e_June8/output_results_CPOdd.parquet',
'mix': None
},
    'gen': {
        'label': 'Generator Neutrino',
        'tag':   'POL_GEN',
    },
    'gen_ts': {
        'label': 'Generator (TauSpinner)',
        'tag':   'POL_GEN_TS',
    },
    'recoRun3': {
        'label': 'Run 3 Reconstruction',
        'tag':   'RecoRun3',
    },
    'recoNu': {
        'label': 'Regressed Neutrino',
        'tag':   'RecoNu_Smeared',
    },
    'recoNu_ts': {
        'label': 'Regressed Neutrino (TauSpinner)',
        'tag':   'RecoNu_Smeared_TS',
    },
    'recoNu_hybrid': {
        'label': 'Regressed Neutrino (Hybrid)',
        'tag':   'RecoNu_Smeared_Hybrid',
    },
}

def _build_daughters_for_hh(row, side, tau_prefix='true', pion_prefix='true'):
    """Build Particle list for calculateHH from a dataframe row."""
    prefix = f'{pion_prefix}_{side}'
    decay_prefix = f'true_{side}'  # decay mode flags always from true
    is_had     = row[f'{decay_prefix}_ishadronic']
    is3prong   = row[f'{decay_prefix}_is3prong']
    npizero    = int(row[f'{decay_prefix}_npizero'])
    ismuon     = row[f'{decay_prefix}_ismuon']
    iselectron = row[f'{decay_prefix}_iselectron']

    nu_key    = f'{tau_prefix}_nubar' if side == 'taup' else f'{tau_prefix}_nu'
    nu_pdgid  = -16 if side == 'taup' else  16
    # For 1-prong: pi_pdgid is the single charged pion (same sign as tau)
    pi_pdgid  =  211 if side == 'taup' else -211
    pip_pdgid = -211 if side == 'taup' else  211
    # For 3-prong: parquet stores pi1=OS (opposite sign to tau), pi2=SS1, pi3=SS2.
    # TAUOLA damppk expects [nu, PIM1(SS), PIM2(SS), PIM3(OS)] — OS pion must be last.
    piOS_pdgid = -211 if side == 'taup' else  211   # pi1 in parquet
    piSS_pdgid =  211 if side == 'taup' else -211   # pi2, pi3 in parquet

    nu = Particle(row[f'{nu_key}_px'], row[f'{nu_key}_py'],
                  row[f'{nu_key}_pz'], row[f'{nu_key}_E'], nu_pdgid)

    def pi(n):
        return Particle(row[f'{prefix}_pi{n}_px'], row[f'{prefix}_pi{n}_py'],
                        row[f'{prefix}_pi{n}_pz'], row[f'{prefix}_pi{n}_E'], pi_pdgid)
    def pip():
        return Particle(row[f'{prefix}_pi2_px'], row[f'{prefix}_pi2_py'],
                        row[f'{prefix}_pi2_pz'], row[f'{prefix}_pi2_E'], pip_pdgid)
    def pi0(n=1):
        return Particle(row[f'{prefix}_pizero{n}_px'], row[f'{prefix}_pizero{n}_py'],
                        row[f'{prefix}_pizero{n}_pz'], row[f'{prefix}_pizero{n}_E'], 111)

    if is_had:
        if is3prong:
            p_os  = Particle(row[f'{prefix}_pi1_px'], row[f'{prefix}_pi1_py'],
                             row[f'{prefix}_pi1_pz'], row[f'{prefix}_pi1_E'], piOS_pdgid)
            p_ss1 = Particle(row[f'{prefix}_pi2_px'], row[f'{prefix}_pi2_py'],
                             row[f'{prefix}_pi2_pz'], row[f'{prefix}_pi2_E'], piSS_pdgid)
            p_ss2 = Particle(row[f'{prefix}_pi3_px'], row[f'{prefix}_pi3_py'],
                             row[f'{prefix}_pi3_pz'], row[f'{prefix}_pi3_E'], piSS_pdgid)
            daughters = [nu, p_ss1, p_ss2, p_os]
            if npizero >= 1:
                daughters.append(pi0())
            return daughters
        else:
            if npizero == 0: return [nu, pi(1)]
            if npizero == 1 or npizero == 2: return [nu, pi(1), pi0()]
            return None
    elif ismuon:
        mu_pdgid  = -13 if side == 'taup' else 13
        nul_pdgid =  14 if side == 'taup' else -14
        return [nu,
                Particle(row[f'{prefix}_pi1_px'], row[f'{prefix}_pi1_py'],
                         row[f'{prefix}_pi1_pz'], row[f'{prefix}_pi1_E'], mu_pdgid),
                Particle(row[f'{prefix}_pizero1_px'], row[f'{prefix}_pizero1_py'],
                         row[f'{prefix}_pizero1_pz'], row[f'{prefix}_pizero1_E'], nul_pdgid)]
    elif iselectron:
        e_pdgid   = -11 if side == 'taup' else 11
        nul_pdgid =  12 if side == 'taup' else -12
        return [nu,
                Particle(row[f'{prefix}_pi1_px'], row[f'{prefix}_pi1_py'],
                         row[f'{prefix}_pi1_pz'], row[f'{prefix}_pi1_E'], e_pdgid),
                Particle(row[f'{prefix}_pizero1_px'], row[f'{prefix}_pizero1_py'],
                         row[f'{prefix}_pizero1_pz'], row[f'{prefix}_pizero1_E'], nul_pdgid)]
    return None


def _hh_to_higgs_rf(hh3, tau_part, boson_part):
    """
    Convert a TauSpinner HH polarimetric vector to Higgs-rest-frame Cartesian
    coordinates, matching the frame used by get_ditau_polarimetric.

    TauSpinner computes HH in a rotated frame where the tau points to -Z.
    This function undoes that rotation by recovering the tau's polar angles
    (phi, theta) in the Higgs RF and applying the inverse rotations.
    """
    tau_c = tau_part.copy()
    bos_c = boson_part.copy()
    tau_c.boostToRestFrame(bos_c)          # boost tau to Higgs RF
    phi   = tau_c.getAnglePhi()
    tau_c.rotateXY(-phi)
    theta = tau_c.getAngleTheta()          # polar angle of tau in Higgs RF
    # Undo the rotation: inverse of [rotateXY(-phi), rotateXZ(pi-theta)]
    h = Particle(hh3[0], hh3[1], hh3[2], 0.0, 0)
    h.rotateXZ(theta - math.pi)
    h.rotateXY(phi)
    return np.array([h.px(), h.py(), h.pz()])


def _run_hh_loop(df, n_events=None, tau_prefix='true', pion_prefix='true', fix_tau_mass=False, event_mask=None):
    """
    Run calculateHH for each event and return Higgs-RF polarimetric vectors.

    tau_prefix:  prefix for tau/neutrino 4-momenta ('true', 'map_pred', 'pred')
    pion_prefix: prefix for pion 4-momenta ('true' or 'reco')
    event_mask:  optional boolean array of length len(df); if given, only events where
                 event_mask is True are processed. hh_p/hh_m are still returned with
                 length len(df), with NaN for unprocessed events.

    Returns (hh_p, hh_m, dm_p_arr, dm_m_arr) where hh_p/hh_m are (n, 3) float arrays
    with NaN for events where calculateHH failed or the decay mode is unsupported.
    """
    df_full = df.iloc[:n_events].reset_index(drop=True) if n_events is not None else df.reset_index(drop=True)
    n_full = len(df_full)
    dm_p_arr = np.array(df_full['taup_DM'])
    dm_m_arr = np.array(df_full['taun_DM'])

    hh_p_full = np.full((n_full, 3), np.nan)
    hh_m_full = np.full((n_full, 3), np.nan)

    if event_mask is not None:
        event_mask = np.asarray(event_mask, dtype=bool)
        df_sub = df_full[event_mask].reset_index(drop=True)
        subset_indices = np.where(event_mask)[0]
    else:
        df_sub = df_full
        subset_indices = np.arange(n_full)

    n = len(df_sub)

    tau_p_pdg = -15
    tau_m_pdg =  15

    _MTAU = 1.77682  # physical tau mass in GeV

    def _enforce_tau_mass(tau_part, pdgid):
        """Return tau_part with energy set to sqrt(|p|^2 + m_tau^2), keeping 3-momentum fixed."""
        pmag2 = tau_part.px()**2 + tau_part.py()**2 + tau_part.pz()**2
        E_phys = math.sqrt(pmag2 + _MTAU**2)
        return Particle(tau_part.px(), tau_part.py(), tau_part.pz(), E_phys, pdgid)

    print(f"  Running calculateHH loop over {n} events (tau_prefix={tau_prefix}, pion_prefix={pion_prefix})...", flush=True)
    for i, (_, row) in enumerate(df_sub.iterrows()):
        if n > 5000 and i % 10000 == 0:
            print(f"  {i}/{n}", flush=True)
        dau_p = _build_daughters_for_hh(row, 'taup', tau_prefix=tau_prefix, pion_prefix=pion_prefix)
        dau_m = _build_daughters_for_hh(row, 'taun', tau_prefix=tau_prefix, pion_prefix=pion_prefix)
        if dau_p is None or dau_m is None:
            continue
        tau_p_part = Particle(row[f'{tau_prefix}_tau_plus_px'],  row[f'{tau_prefix}_tau_plus_py'],
                              row[f'{tau_prefix}_tau_plus_pz'],  row[f'{tau_prefix}_tau_plus_E'],  tau_p_pdg)
        tau_m_part = Particle(row[f'{tau_prefix}_tau_minus_px'], row[f'{tau_prefix}_tau_minus_py'],
                              row[f'{tau_prefix}_tau_minus_pz'], row[f'{tau_prefix}_tau_minus_E'], tau_m_pdg)
        if fix_tau_mass:
            tau_p_part = _enforce_tau_mass(tau_p_part, tau_p_pdg)
            tau_m_part = _enforce_tau_mass(tau_m_part, tau_m_pdg)
        boson = Particle(row[f'{tau_prefix}_tau_plus_px']  + row[f'{tau_prefix}_tau_minus_px'],
                         row[f'{tau_prefix}_tau_plus_py']  + row[f'{tau_prefix}_tau_minus_py'],
                         row[f'{tau_prefix}_tau_plus_pz']  + row[f'{tau_prefix}_tau_minus_pz'],
                         row[f'{tau_prefix}_tau_plus_E']   + row[f'{tau_prefix}_tau_minus_E'], 25)
        try:
            HHp_vec, _, HHm_vec, _ = getHHVectors(boson, tau_p_part, tau_m_part, dau_p, dau_m)
            full_i = subset_indices[i]
            hh_p_full[full_i] = _hh_to_higgs_rf(HHp_vec[:3], tau_p_part, boson)
            hh_m_full[full_i] = _hh_to_higgs_rf(HHm_vec[:3], tau_m_part, boson)
        except Exception:
            pass

    return hh_p_full, hh_m_full, dm_p_arr, dm_m_arr


def compute_phicp_all(df, option, use_map=True, output_dir='.'):
    # Compute phiCP for all events in the df (splitting of methods by DM done automatically, vectorised)
    df = df.copy()
    if option == 'gen':
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix='true', reco_pions=True)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'gen_ts':
        # Use true tau directions for P1/P2 (same as gen), polarimetric vectors from TauSpinner
        _, P1, _, P2 = get_ditau_polarimetric(df, tau_prefix='true', reco_pions=False)
        hh_p, hh_m, _, _ = _run_hh_loop(df)
        R1 = ak.zip({"x": hh_p[:, 0], "y": hh_p[:, 1], "z": hh_p[:, 2]}, with_name="Vector3D")
        R2 = ak.zip({"x": hh_m[:, 0], "y": hh_m[:, 1], "z": hh_m[:, 2]}, with_name="Vector3D")
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu_ts':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1_old, P1, R2_old, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        hh_p, hh_m, _, _ = _run_hh_loop(df, tau_prefix=tau_prefix, pion_prefix='reco', fix_tau_mass=False) # fix_tau_mass=True - needs to be studies more 
        nan_p = np.isnan(hh_p[:, 0])
        nan_m = np.isnan(hh_m[:, 0])
        if nan_p.any() or nan_m.any():
            print(f"  recoNu_ts: falling back to recoNu for {nan_p.sum()} tau+ and {nan_m.sum()} tau- events where calculateHH failed")
        hh_p[nan_p] = np.stack([np.array(R1_old.x)[nan_p], np.array(R1_old.y)[nan_p], np.array(R1_old.z)[nan_p]], axis=1)
        hh_m[nan_m] = np.stack([np.array(R2_old.x)[nan_m], np.array(R2_old.y)[nan_m], np.array(R2_old.z)[nan_m]], axis=1)
        R1 = ak.zip({"x": hh_p[:, 0], "y": hh_p[:, 1], "z": hh_p[:, 2]}, with_name="Vector3D")
        R2 = ak.zip({"x": hh_m[:, 0], "y": hh_m[:, 1], "z": hh_m[:, 2]}, with_name="Vector3D")
        #TODO need to study how to deal with the nans properly, for now we just use the old R1/R2 for those events
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoNu_hybrid':
        tau_prefix = 'map_pred' if use_map else 'pred'
        R1, P1, R2, P2 = get_ditau_polarimetric(df, tau_prefix=tau_prefix, reco_pions=True)
        dm_p_arr = np.array(df['taup_DM'])
        dm_m_arr = np.array(df['taun_DM'])
        needs_ts = (dm_p_arr == 11) | (dm_m_arr == 11)
        hh_p, hh_m, _, _ = _run_hh_loop(df, tau_prefix=tau_prefix, pion_prefix='reco', fix_tau_mass=False, event_mask=needs_ts)
        r1_arr = np.stack([np.array(R1.x), np.array(R1.y), np.array(R1.z)], axis=1)
        r2_arr = np.stack([np.array(R2.x), np.array(R2.y), np.array(R2.z)], axis=1)
        nan_p = np.isnan(hh_p[:, 0])
        nan_m = np.isnan(hh_m[:, 0])
        # Replace each leg independently: only swap if that leg is DM=11
        use_ts_p = (dm_p_arr == 11) & ~nan_p
        use_ts_m = (dm_m_arr == 11) & ~nan_m
        r1_arr[use_ts_p] = hh_p[use_ts_p]
        r2_arr[use_ts_m] = hh_m[use_ts_m]
        R1 = ak.zip({"x": r1_arr[:, 0], "y": r1_arr[:, 1], "z": r1_arr[:, 2]}, with_name="Vector3D")
        R2 = ak.zip({"x": r2_arr[:, 0], "y": r2_arr[:, 1], "z": r2_arr[:, 2]}, with_name="Vector3D")
        phiCP = compute_aco_polarimetric(R1, P1, R2, P2)
    elif option == 'recoRun3':
        R1, P1, leg1_is_dp = get_R_P_vectors_all(df, tau_prefix='taup')
        R2, P2, leg2_is_dp = get_R_P_vectors_all(df, tau_prefix='taun')
        phiCPmain = compute_aco_classic(R1, P1, R2, P2, leg1_is_dp, leg2_is_dp)
        phiCPa1a1 = compute_aco_classic_a1a1(df)
        phiCP = np.where((df['taup_DM'] == 10) & (df['taun_DM'] == 10), phiCPa1a1, phiCPmain)
    df['phiCP'] = np.array(phiCP)
    return df

def add_DM(df, dm_prefix='reco'):
    for tau in ['taup', 'taun']:
        tau_is_lep = df[f'{dm_prefix}_{tau}_ishadronic'].values == 0
        tau_is_dm0 = (df[f"{dm_prefix}_{tau}_npizero"].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm1 = (df[f"{dm_prefix}_{tau}_npizero"].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm2 = ((df[f"{dm_prefix}_{tau}_npizero"].values == 1) | (df[f"{dm_prefix}_{tau}_npizero"].values == 2)) & (df[f'{dm_prefix}_{tau}_is3prong'] == 0) & (~tau_is_lep)
        tau_is_dm10 = (df[f"{dm_prefix}_{tau}_npizero"].values == 0) & (df[f'{dm_prefix}_{tau}_is3prong'] == 1) & (~tau_is_lep)
        tau_is_dm11 = (df[f"{dm_prefix}_{tau}_npizero"].values == 1) & (df[f'{dm_prefix}_{tau}_is3prong'] == 1) & (~tau_is_lep)
        df[f'{tau}_DM'] = np.where(tau_is_dm0, 0,
                             np.where(tau_is_dm1, 1,
                                      np.where(tau_is_dm2, 2,
                                               np.where(tau_is_dm10, 10, 
                                                    np.where(tau_is_dm11, 11,
                                                        np.where(tau_is_lep, 100, -1))))))
    return df

def plot_phicp_histogram(ax, data, bin_edges, variable, label, color, hide_errors=False):
    bin_width = bin_edges[1] - bin_edges[0]
    step_x = np.repeat(bin_edges, 2)[1:-1]
    raw, _ = np.histogram(data[variable], bins=bin_edges)
    counts = raw / (raw.sum() * bin_width)
    ax.hist(data[variable], bins=bin_edges, histtype='step', label=label,
            density=True, linewidth=2, color=color)
    if not hide_errors:
        err = np.sqrt(raw) / (raw.sum() * bin_width)
        ax.fill_between(step_x, np.repeat(counts - err, 2),
                        np.repeat(counts + err, 2), alpha=0.25, color=color)
    return counts


def load_data(prefix='',extra_pt_cut=-1):
    cfg = options['files']
    read = pd.read_parquet
    mix_df = read(cfg['mix']) if cfg.get('mix') is not None else None
    even_df = read(cfg[f'{prefix}even'])
    print(f'EVEN File: {cfg[f"{prefix}even"]}')
    odd_df = read(cfg[f'{prefix}odd'])
    print(f'ODD File: {cfg[f"{prefix}even"]}')
    # estimate visible pT from sum of true_taun_charged_px true_taun_pizero1_px, etc and apply cut if extra_pt_cut>0
    if extra_pt_cut > 0:
        def compute_vis_pt(df, prefix):
            taup_px = df[f'reco_taup_charged_px'] + df[f'reco_taup_pizero1_px']
            taun_px = df[f'reco_taun_charged_px'] + df[f'reco_taun_pizero1_px']
            taup_py = df[f'reco_taup_charged_py'] + df[f'reco_taup_pizero1_py']
            taun_py = df[f'reco_taun_charged_py'] + df[f'reco_taun_pizero1_py']
            taup_pt = np.sqrt(taup_px**2 + taup_py**2)
            taun_pt = np.sqrt(taun_px**2 + taun_py**2)
            return np.sqrt((taup_px + taun_px)**2 + (taup_py + taun_py)**2)
        even_df['vis_pt'] = compute_vis_pt(even_df, prefix)
        odd_df['vis_pt'] = compute_vis_pt(odd_df, prefix)
        even_df = even_df[even_df['vis_pt'] > extra_pt_cut]
        odd_df = odd_df[odd_df['vis_pt'] > extra_pt_cut]
        if mix_df is not None:
            mix_df['vis_pt'] = compute_vis_pt(mix_df, prefix)
            mix_df = mix_df[mix_df['vis_pt'] > extra_pt_cut]
    return even_df, odd_df, mix_df
    #return read(cfg[f'{prefix}even']), read(cfg[f'{prefix}odd']), mix_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--option', choices=['gen', 'gen_ts', 'recoRun3', 'recoNu', 'recoNu_ts', 'recoNu_hybrid'],
                        default='gen', help="Reconstruction method to use.")
    parser.add_argument('--output-dir', default='.', help="Directory for output PDFs.")
    parser.add_argument('--useMLP', action='store_true')
    parser.add_argument('--GENfilter', action='store_true',
                        help="Use true_ prefix for DM/prong masks instead of reco_.")
    parser.add_argument('--hide-errors', action='store_true',
                        help="Hide Poisson error bands on the bins (shown by default).")
    parser.add_argument('--leptonic_mode', default=0, type=int, choices=[0,1,2],
                        help="If 0 use hadronic decay modes, for 1 use semileptonic, for 2 use fully leptonic (not currently supported).")

    args = parser.parse_args()

    if args.leptonic_mode == 2:
        raise NotImplementedError("Fully leptonic mode not currently supported.")

    do_DM10= False

    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/logs", exist_ok=True)
    os.makedirs(f"{args.output_dir}/{args.option}", exist_ok=True)

    even_df, odd_df, mix_df = load_data(prefix='sl_' if args.leptonic_mode == 1 else '')


    use_map = not args.useMLP
    dm_pfx = 'true' if args.GENfilter else 'reco'
    even_df = add_DM(even_df, dm_prefix=dm_pfx)
    odd_df = add_DM(odd_df, dm_prefix=dm_pfx)
    even_df = compute_phicp_all(even_df, args.option, use_map=use_map, output_dir=args.output_dir)
    odd_df  = compute_phicp_all(odd_df,  args.option, use_map=use_map, output_dir=args.output_dir)
    if mix_df is not None:
        mix_df = add_DM(mix_df, dm_prefix=dm_pfx)
        mix_df = compute_phicp_all(mix_df, args.option, use_map=use_map, output_dir=args.output_dir)

    if args.leptonic_mode == 1:
        dm_combs = [[100, 0], [100,1], [100,2], [100,10]]
        if args.option in ('gen_ts', 'recoNu_ts', 'recoNu_hybrid'):
            dm_combs += [[100, 11]]
    else: 
        dm_combs = [[0, 0], [0,1], [1,1], [2,2], [1,2], [0,2], [10,10], [0,10], [1,10], [2,10]]
        if args.option in ('gen_ts', 'recoNu_ts', 'recoNu_hybrid'):
            dm_combs += [[0, 11], [1,11], [2,11], [10,11], [11,11]]

    for dm_taup, dm_taun in dm_combs:

        dm_mask = lambda df, p=dm_taup, n=dm_taun: ((df['taup_DM'] == p) & (df['taun_DM'] == n)) | ((df['taun_DM'] == n) & (df['taup_DM'] == p))
        even = even_df[dm_mask(even_df)]
        odd  = odd_df[dm_mask(odd_df)]

        print(f"DM{dm_taup}-DM{dm_taun}: {len(even)} CP even, {len(odd)} CP odd events")

        fig, ax = plt.subplots(figsize=(8, 6))
        bin_edges = np.linspace(0, 2 * np.pi, 21)
        hide = args.hide_errors
        even_counts = plot_phicp_histogram(ax, even, bin_edges, 'phiCP', 'CP-even', 'red',   hide)
        odd_counts  = plot_phicp_histogram(ax, odd,  bin_edges, 'phiCP', 'CP-odd',  'blue',  hide)
        if mix_df is not None:
            mix = mix_df[dm_mask(mix_df)]
            mix = mix[single_prong_mask(mix)]
            plot_phicp_histogram(ax, mix, bin_edges, 'phiCP', 'CP-mix', 'green', hide)
        avg = 0.5 * (even_counts + odd_counts)
        asymmetry = np.mean(np.abs(even_counts - odd_counts) / avg)

        significance = 0 
        for i in range(len(even_counts)):
            b_est = (odd_counts[i] + even_counts[i])*0.5*4
            #temp = odd_counts[i] - even_counts[i] + (even_counts[i]+b_est)*np.log((even_counts[i]+b_est)/(odd_counts[i]+b_est)) if even_counts[i] > 0 and odd_counts[i] > 0 else 0
            temp = (odd_counts[i] - even_counts[i])**2
            significance += temp
        #significance = np.sqrt(2 * significance)
        significance = np.sqrt(significance)

        

        ax.set_xlabel(r'$\phi_{CP}$')
        ax.set_title(f'DM{dm_taup} - DM{dm_taun} - {options[args.option]["label"]}')
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 0.28)
        ax.legend()
        ax.text(0.05, 0.95, f'Asymmetry: {asymmetry:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        ax.text(0.05, 0.85, f'Asymmetry (quadrature): {significance:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontweight='bold')
        out = f"{args.output_dir}/{args.option}/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")

        # save numpy arrays to remake plots in future
        np.savez(
            f"{args.output_dir}/logs/DM{dm_taup}DM{dm_taun}_{options[args.option]['tag']}.npz",
            even_counts=even_counts,
            odd_counts=odd_counts,
            bin_edges=bin_edges,
        )

if __name__ == '__main__':
    main()
