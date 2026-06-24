import argparse
import logging
import math

import numba as nb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import vector
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MET covariance is assumed constant across events (no per-event estimate available).
met_cov_xx = 184.0
met_cov_xy = 0.0
met_cov_yy = 184.0

# Higgs-window constraint used for the constrained FastMTT fit.
constraint_setting = "Window"
constraint_window = np.array([123.0, 127.0])

# mass likelihood shaping parameters, see compute_fastmtt.
delta = 1 / 1.15
reg_order = 6

# events per chunk: bounds memory use and sets progress bar granularity.
batch_size = 200_000

# core algorithm copied from HiggsDNA's higgs_dna/tools/ditau/fastmtt.py

@nb.jit(nopython=True, parallel=False)
def compute_fastmtt(
    N,
    pt_1,
    eta_1,
    phi_1,
    mass1,
    pt_2,
    eta_2,
    phi_2,
    mass2,
    met_x,
    met_y,
    metcov_xx,
    metcov_xy,
    metcov_yx,
    metcov_yy,
    decay_type_1,
    decay_type_2,
    m_ele,
    m_muon,
    m_tau,
    m_pion,
    delta,
    reg_order,
    constraint,
    constraint_setting,
    constraint_window
):
    fastmttMass_values = np.zeros(N, dtype=np.float32)
    fastmttPt_values = np.zeros(N, dtype=np.float32)
    fastmttPt1_values = np.zeros(N, dtype=np.float32)
    fastmttPt2_values = np.zeros(N, dtype=np.float32)

    mass_dict = {0: m_ele, 1: m_muon, 2: m_tau}

    for i in range(N):

        # grab the correct masses based on tau decay type
        # tau decay_type: 0 ==> leptonic to electron,
        #                 1 ==> leptonic to muon,
        #                 2 ==> leptonic to hadronic
        if (decay_type_1[i] != 2):
            m1 = mass_dict[decay_type_1[i]]
        else:
            m1 = mass1[i]
        if (decay_type_2[i] != 2):
            m2 = mass_dict[decay_type_2[i]]
        else:
            m2 = mass2[i]

        # store visible masses
        m_vis_1 = m1
        m_vis_2 = m2

        # determine minimum and maximum possible masses
        m_vis_min_1, m_vis_max_1 = 0, 0
        m_vis_min_2, m_vis_max_2 = 0, 0
        if (decay_type_1[i] == 0):
            m_vis_min_1, m_vis_max_1 = m_ele, m_ele
        if (decay_type_1[i] == 1):
            m_vis_min_1, m_vis_max_1 = m_muon, m_muon
        if (decay_type_1[i] == 2):
            m_vis_min_1, m_vis_max_1 = m_pion, 1.5
        if (decay_type_2[i] == 0):
            m_vis_min_2, m_vis_max_2 = m_ele, m_ele
        if (decay_type_2[i] == 1):
            m_vis_min_2, m_vis_max_2 = m_muon, m_muon
        if (decay_type_2[i] == 2):
            m_vis_min_2, m_vis_max_2 = m_pion, 1.5
        if (m_vis_1 < m_vis_min_1):
            m_vis_1 = m_vis_min_1
        if (m_vis_1 > m_vis_max_1):
            m_vis_1 = m_vis_max_1
        if (m_vis_2 < m_vis_min_2):
            m_vis_2 = m_vis_min_2
        if (m_vis_2 > m_vis_max_2):
            m_vis_2 = m_vis_max_2

        # store both tau candidate four vectors
        leg1 = vector.obj(pt=pt_1[i], eta=eta_1[i], phi=phi_1[i], mass=m_vis_1)
        leg2 = vector.obj(pt=pt_2[i], eta=eta_2[i], phi=phi_2[i], mass=m_vis_2)

        # store visible mass of ditau pair
        m_vis = math.sqrt(2 * leg1.pt * leg2.pt * (math.cosh(leg1.eta - leg2.eta) - math.cos(leg1.phi - leg2.phi)))

        # correct initial visible masses
        if (decay_type_1[i] == 2 and m_vis_1 > 1.5):
            m_vis_1 = 0.3
        if (decay_type_2[i] == 2 and m_vis_2 > 1.5):
            m_vis_2 = 0.3

        # invert met covariance matrix, calculate determinant
        metcovinv_xx, metcovinv_yy = metcov_yy[i], metcov_xx[i]
        metcovinv_xy, metcovinv_yx = - metcov_xy[i], - metcov_yx[i]
        metcovinv_det = (metcovinv_xx * metcovinv_yy - metcovinv_yx * metcovinv_xy)

        if (metcovinv_det < 1e-10):
            continue

        # perform likelihood scan
        # see http://cms.cern.ch/iCMS/jsp/openfile.jsp?tp=draft&files=AN2019_032_v3.pdf
        met_const = 1 / (2 * math.pi * math.sqrt(metcovinv_det))
        min_likelihood, x1_opt, x2_opt = 999, 0.01, 0.01
        mass_likelihood, met_transfer = 0, 0

        initialise = True

        # scan over weights for each ditau four-vector
        for x1 in np.arange(0.01, 1, 0.01):
            for x2 in np.arange(0.01, 1, 0.01):
                x1_min = min(1, math.pow((m_vis_1 / m_tau), 2))
                x2_min = min(1, math.pow((m_vis_2 / m_tau), 2))
                if ((x1 < x1_min) or (x2 < x2_min)):
                    continue

                # test weighted four-vectors
                leg1_x1, leg2_x2 = leg1 * (1 / x1), leg2 * (1 / x2)
                ditau_test = vector.obj(px=leg1_x1.px + leg2_x2.px,
                                        py=leg1_x1.py + leg2_x2.py,
                                        pz=leg1_x1.pz + leg2_x2.pz,
                                        E=leg1_x1.E + leg2_x2.E)
                nu_test = vector.obj(px=ditau_test.px - leg1.px - leg2.px,
                                     py=ditau_test.py - leg1.py - leg2.py,
                                     pz=ditau_test.pz - leg1.pz - leg2.pz,
                                     E=ditau_test.E - leg1.E - leg2.E)
                test_mass = ditau_test.mass

                if constraint_setting == "Window":
                    if (((test_mass < constraint_window[0]) or (test_mass > constraint_window[1])) and constraint):
                        continue

                # calculate mass likelihood integral
                m_shift = test_mass * delta
                if (m_shift < m_vis):
                    continue
                x1_min = min(1.0, math.pow((m_vis_1 / m_tau), 2))
                x2_min = max(math.pow((m_vis_2 / m_tau), 2),
                             math.pow((m_vis / m_shift), 2))
                x2_max = min(1.0, math.pow((m_vis / m_shift), 2) / x1_min)
                if (x2_max < x2_min):
                    continue
                J = 2 * math.pow(m_vis, 2) * math.pow(m_shift, -reg_order)
                I_x2 = math.log(x2_max) - math.log(x2_min)
                I_tot = I_x2
                if (decay_type_1[i] != 2):
                    I_m_nunu_1 = math.pow((m_vis / m_shift), 2) * (math.pow(x2_max, -1) - math.pow(x2_min, -1))
                    I_tot += I_m_nunu_1
                if (decay_type_2[i] != 2):
                    I_m_nunu_2 = math.pow((m_vis / m_shift), 2) * I_x2 - (x2_max - x2_min)
                    I_tot += I_m_nunu_2
                mass_likelihood = 1e9 * J * I_tot

                # calculate MET transfer function
                residual_x = met_x[i] - nu_test.x
                residual_y = met_y[i] - nu_test.y
                pull2 = (residual_x * (metcovinv_xx * residual_x + metcovinv_xy * residual_y) + residual_y * (metcovinv_yx * residual_x + metcovinv_yy * residual_y))
                pull2 /= metcovinv_det
                met_transfer = met_const * math.exp(-0.5 * pull2)

                # calculate final likelihood, store if minimum
                likelihood = -met_transfer * mass_likelihood

                if constraint and constraint_setting == "BreitWigner":
                    mH = 125.0
                    GammaH = 0.004
                    deltaM = test_mass * test_mass - mH * mH
                    mG = test_mass * GammaH
                    BreitWigner_likelihood = 1 / (deltaM * deltaM + mG * mG)
                    likelihood = likelihood * BreitWigner_likelihood

                if initialise:
                    min_likelihood = likelihood
                    x1_opt, x2_opt = x1, x2
                    initialise = False
                else:
                    if (likelihood < min_likelihood):
                        min_likelihood = likelihood
                        x1_opt, x2_opt = x1, x2

        leg1_x1, leg2_x2 = leg1 * (1 / x1_opt), leg2 * (1 / x2_opt)
        p4_ditau_opt = vector.obj(px=leg1_x1.px + leg2_x2.px,
                                  py=leg1_x1.py + leg2_x2.py,
                                  pz=leg1_x1.pz + leg2_x2.pz,
                                  E=leg1_x1.E + leg2_x2.E)

        mass_opt = p4_ditau_opt.mass
        pt_opt = p4_ditau_opt.pt
        pt1_opt = pt_1[i] / x1_opt
        pt2_opt = pt_2[i] / x2_opt

        fastmttMass_values[i] = mass_opt
        fastmttPt_values[i] = pt_opt
        fastmttPt1_values[i] = pt1_opt
        fastmttPt2_values[i] = pt2_opt

    return fastmttMass_values, fastmttPt_values, fastmttPt1_values, fastmttPt2_values


def get_args():
    parser = argparse.ArgumentParser(description="Add FastMTT mass/pt predictions to a ditau dataframe (parquet)")
    parser.add_argument('--input_file', type=str, required=True, help="Input parquet file.")
    parser.add_argument('--output_file', type=str, required=True, help="Output parquet file (must differ from input_file).")
    return parser.parse_args()


def build_visible_leg(df, leg):
    px = (df[f'reco_{leg}_charged_px'] + df[f'reco_{leg}_pizero1_px']).to_numpy()
    py = (df[f'reco_{leg}_charged_py'] + df[f'reco_{leg}_pizero1_py']).to_numpy()
    pz = (df[f'reco_{leg}_charged_pz'] + df[f'reco_{leg}_pizero1_pz']).to_numpy()
    e = (df[f'reco_{leg}_charged_e'] + df[f'reco_{leg}_pizero1_e']).to_numpy()
    vis = vector.array({'px': px, 'py': py, 'pz': pz, 'E': e})
    return vis.pt, vis.eta, vis.phi, vis.mass


def add_fastmtt(df):
    # tt channel only for now: both legs are hadronic (decay_type 2).
    pt_1, eta_1, phi_1, mass1 = build_visible_leg(df, 'taup')
    pt_2, eta_2, phi_2, mass2 = build_visible_leg(df, 'taun')

    met_x = df['reco_met_px'].to_numpy()
    met_y = df['reco_met_py'].to_numpy()

    nevents = len(df)
    decay_type_1 = np.full(nevents, 2, dtype=np.uint8)
    decay_type_2 = np.full(nevents, 2, dtype=np.uint8)

    m_ele, m_muon, m_tau, m_pion = 0.51100e-3, 0.10566, 1.77685, 0.13957

    metcov_xx_arr = np.full(nevents, met_cov_xx)
    metcov_xy_arr = np.full(nevents, met_cov_xy)
    metcov_yy_arr = np.full(nevents, met_cov_yy)

    logger.info("Running FastMTT (constraint=True)")
    mass_vals, pt_vals, pt1_vals, pt2_vals = compute_fastmtt(
        nevents,
        pt_1, eta_1, phi_1, mass1,
        pt_2, eta_2, phi_2, mass2,
        met_x, met_y,
        metcov_xx_arr, metcov_xy_arr, metcov_xy_arr, metcov_yy_arr,
        decay_type_1, decay_type_2,
        m_ele, m_muon, m_tau, m_pion,
        delta, reg_order, True, constraint_setting, constraint_window,
    )
    df['FastMTT_mass'] = mass_vals
    df['FastMTT_pt'] = pt_vals
    df['FastMTT_pt_1'] = pt1_vals
    df['FastMTT_pt_2'] = pt2_vals

    return df


def main():
    args = get_args()

    parquet_file = pq.ParquetFile(args.input_file)
    n_batches = math.ceil(parquet_file.metadata.num_rows / batch_size)

    writer = None
    for record_batch in tqdm(parquet_file.iter_batches(batch_size=batch_size), total=n_batches, desc="FastMTT"):
        df = record_batch.to_pandas()
        df = add_fastmtt(df)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(args.output_file, table.schema)
        writer.write_table(table)
    writer.close()


if __name__ == "__main__":
    main()
