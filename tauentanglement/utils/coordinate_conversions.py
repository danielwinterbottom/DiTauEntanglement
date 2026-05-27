import numpy as np
import pandas as pd


def ConvertToPolar(df,prefix):
    # convert the px, py, pz columns with given prefix to polar coordinates (pt, eta, phi)
    px = df[f'{prefix}x']
    py = df[f'{prefix}y']
    pz = df[f'{prefix}z']

    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    # to avoid division by zero
    theta = np.arccos(np.clip(pz / p, -1.0, 1.0))
    eta = -np.log(np.tan(theta / 2))
    phi = np.arctan2(py, px)

    df[f'{prefix}_pt'] = pt
    df[f'{prefix}_eta'] = eta
    df[f'{prefix}_phi'] = phi

    # drop the original x, y, z columns
    df = df.drop(columns=[f'{prefix}x', f'{prefix}y', f'{prefix}z'])

    return df

def ConvertToCartesian(df,prefix):
    # convert the pt, eta, phi columns with given prefix to cartesian coordinates (px, py, pz)
    pt = df[f'{prefix}_pt']
    eta = df[f'{prefix}_eta']
    phi = df[f'{prefix}_phi']

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    df[f'{prefix}x'] = px
    df[f'{prefix}y'] = py
    df[f'{prefix}z'] = pz

    # drop the original pt, eta, phi columns
    df = df.drop(columns=[f'{prefix}_pt', f'{prefix}_eta', f'{prefix}_phi'])

    return df

# make a similar function for converting back the predictions which aren't labelled with a prefix
def ConvertPredictionsToCartesian(predictions, output_features):
    # predictions is a numpy array of shape [N_events, N_features]
    predictions_df = pd.DataFrame(predictions, columns=output_features)

    # find all prefixes by removing the _pt, _eta, _phi suffixes
    prefixes = set()
    for col in predictions_df.columns:
        if col.endswith('_pt'):
            prefixes.add(col[:-3])
        elif col.endswith('_eta'):
            prefixes.add(col[:-4])
        elif col.endswith('_phi'):
            prefixes.add(col[:-4])

    for prefix in prefixes:
        predictions_df = ConvertToCartesian(predictions_df, prefix)

    # return as numpy array
    return predictions_df.values

def _get_vec(df, pref: str) -> np.ndarray:
    """Return (N,3) array from columns {pref}px, {pref}py, {pref}pz."""
    return np.stack(
        [df[f"{pref}px"].to_numpy(),
         df[f"{pref}py"].to_numpy(),
         df[f"{pref}pz"].to_numpy()],
        axis=1
    ).astype(float)

def _build_nrk_basis_from_visible_tau(
    df,
    charged_prefix: str,
    pi0_prefix: str,
    eps: float = 1e-12,
):
    """
    Build event-by-event orthonormal basis (n_hat, r_hat, k_hat) using:
      p_hat = (0,0,-1)
      k_hat = unit(pi + pi0)
      n_hat = unit(p_hat x k_hat)
      r_hat = unit(p_hat - k_hat (p_hat·k_hat))

    Returns:
      n, r, k as (N,3) arrays
    """

    v_vis = _get_vec(df, charged_prefix) + _get_vec(df, pi0_prefix)

    # --- k_hat ---
    k_norm = np.linalg.norm(v_vis, axis=1, keepdims=True)

    k = np.zeros_like(v_vis)
    np.divide(v_vis, k_norm, out=k, where=(k_norm > eps))

    # beam axis
    N = len(df)
    p = np.zeros((N, 3), dtype=float)
    p[:, 2] = -1.0

    n = np.cross(p, k)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)

    n_unit = np.zeros_like(n)
    np.divide(n, n_norm, out=n_unit, where=(n_norm > eps))
    n = n_unit

    cosTheta = np.sum(p * k, axis=1, keepdims=True)

    r = p - k * cosTheta
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)

    r_unit = np.zeros_like(r)
    np.divide(r, r_norm, out=r_unit, where=(r_norm > eps))
    r = r_unit

    # fallback basis if everything vanished
    zero_mask = ((n_norm <= eps) & (k_norm <= eps)).ravel()

    n[zero_mask] = [1.0, 0.0, 0.0]
    r[zero_mask] = [0.0, 1.0, 0.0]
    k[zero_mask] = [0.0, 0.0, 1.0]

    return n, r, k

def ConvertToOrthonormalNRK(
    df,
    prefix_to_convert: str,
    charged_prefix: str,
    pi0_prefix: str,
    out_prefix = None,
    drop_xyz: bool = True,
    eps: float = 1e-12,
    keep_basis: bool = False,
):
    if out_prefix is None:
        out_prefix = prefix_to_convert

    n, r, k = _build_nrk_basis_from_visible_tau(df, charged_prefix, pi0_prefix, eps=eps)
    v = _get_vec(df, prefix_to_convert)

    df[f"{out_prefix}n"] = np.sum(v * n, axis=1)
    df[f"{out_prefix}r"] = np.sum(v * r, axis=1)
    df[f"{out_prefix}k"] = np.sum(v * k, axis=1)

    if keep_basis:
        for i, comp in enumerate(["x", "y", "z"]):
            df[f"{out_prefix}n{comp}"] = n[:, i]
            df[f"{out_prefix}r{comp}"] = r[:, i]
            df[f"{out_prefix}k{comp}"] = k[:, i]

    if drop_xyz:
        df = df.drop(columns=[f"{prefix_to_convert}px", f"{prefix_to_convert}py", f"{prefix_to_convert}pz"])

    return df


def ConvertFromOrthonormalNRK(
    df,
    prefix_to_convert: str,  # expects {prefix}n,{prefix}r,{prefix}k
    charged_prefix: str,
    pi0_prefix: str,
    out_prefix = None,  # writes {out}px,{out}py,{out}pz
    drop_nrk: bool = False,
    eps: float = 1e-12,
):
    if out_prefix is None:
        out_prefix = prefix_to_convert

    n, r, k = _build_nrk_basis_from_visible_tau(df, charged_prefix, pi0_prefix, eps=eps)

    vn = df[f"{prefix_to_convert}n"].to_numpy(dtype=float)[:, None]
    vr = df[f"{prefix_to_convert}r"].to_numpy(dtype=float)[:, None]
    vk = df[f"{prefix_to_convert}k"].to_numpy(dtype=float)[:, None]

    v = vn * n + vr * r + vk * k

    df[f"{out_prefix}px"] = v[:, 0]
    df[f"{out_prefix}py"] = v[:, 1]
    df[f"{out_prefix}pz"] = v[:, 2]

    if drop_nrk:
        df = df.drop(columns=[f"{prefix_to_convert}n", f"{prefix_to_convert}r", f"{prefix_to_convert}k"])

    return df


def convert_coordinates_pred(arr, coordinates, output_features, taup_charged, taup_pizero, taun_charged, taun_pizero):
    if coordinates == 'polar':
        return ConvertPredictionsToCartesian(arr, output_features)
    elif coordinates == 'onorm':
        return ConvertFromOrthonormalNRK_Predictions(arr, taup_charged=taup_charged, taup_pi0=taup_pizero,
                                                     taun_charged=taun_charged, taun_pi0=taun_pizero)
    return arr


def ConvertFromOrthonormalNRK_Predictions(
    predictions,
    taup_charged,
    taup_pi0,
    taun_charged,
    taun_pi0,
    eps: float = 1e-12,
):

    charged_name = "charged"

    # create a dataframe with predictions and visible tau decay products
    column_names = ['taup_nu_n', 'taup_nu_r', 'taup_nu_k','taun_nu_n', 'taun_nu_r', 'taun_nu_k']
    taup_charged_columns = [f'reco_taup_{charged_name}_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    taup_pi0_columns = [f'reco_taup_pizero1_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    taun_charged_columns = [f'reco_taun_{charged_name}_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    taun_pi0_columns = [f'reco_taun_pizero1_{comp}' for comp in ['E', 'px', 'py', 'pz']]
    visible_data = np.concatenate([taup_charged, taup_pi0, taun_charged, taun_pi0], axis=1)
    visible_column_names = taup_charged_columns + taup_pi0_columns + taun_charged_columns + taun_pi0_columns
    all_data = np.concatenate([predictions, visible_data], axis=1)
    column_names += visible_column_names
    df = pd.DataFrame(all_data, columns=column_names)


    # now call ConvertFromOrthonormalNRK on the dataframe, first for nubar tau+ neutrino
    charged_prefix = f'reco_taup_{charged_name}_'
    pi0_prefix = 'reco_taup_pizero1_'
    df = ConvertFromOrthonormalNRK(
        df,
        prefix_to_convert='taup_nu_',
        charged_prefix=charged_prefix,
        pi0_prefix=pi0_prefix,
        drop_nrk=True,
        eps=eps,
    )
    # then for nubar tau- neutrino
    charged_prefix = f'reco_taun_{charged_name}_'
    pi0_prefix = 'reco_taun_pizero1_'
    df = ConvertFromOrthonormalNRK(
        df,
        prefix_to_convert='taun_nu_',
        charged_prefix=charged_prefix,
        pi0_prefix=pi0_prefix,
        drop_nrk=True,
        eps=eps,
    )

    # ensure it is ordered like taup_nu_px, taup_nu_py, taup_nu_pz, taun_nu_px, taun_nu_py, taun_nu_pz
    ordered_columns = [
        'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
        'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
    ]
    df = df[ordered_columns]
    # return as numpy array
    return df.values
