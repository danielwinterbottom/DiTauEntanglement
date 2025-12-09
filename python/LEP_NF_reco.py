import argparse

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--stages', '-s', help='a list of stages to run', type=int, nargs="+")
    argparser.add_argument('--model_name', '-m', help='the name of the model output name', type=str, default='dummy_ditau_nu_regression_model_residualNN')
    argparser.add_argument('--n_epochs', '-n', help='number of training epochs', type=int, default=10)
    argparser.add_argument('--n_trials', '-t', help='number of hyperparameter optimization trials', type=int, default=100)
    argparser.add_argument('--inc_reco_taus', help='whether to include the taus reconstructed by the analytical model as inputs', action='store_true')
    args = argparser.parse_args()

    input_variables = ['dmin_x', 'dmin_y', 'dmin_z',
                       'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz', 'reco_taup_pi1_e',
                       'reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz',
                       'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz', 'reco_taup_pizero1_e',
                       'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz', 'reco_taun_pi1_e',
                       'reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz',
                       'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz', 'reco_taun_pizero1_e',
                       'BS_x', 'BS_y', 'BS_z',
                       'taup_haspizero', 'taun_haspizero']

    output_variables = [
        'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
        'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
    ]

    # stage one prepares the dataframe
    if 1 in args.stages:
        import uproot3
        input_file_name = '/vols/cms/dw515/HH_reweighting/DiTauEntanglement/batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events_extravars.root'
        tree = uproot3.open(input_file_name)['new_tree']

        variables = [
            'taup_npi', 'taup_npizero',
            'reco_taup_pi1_px', 'reco_taup_pi1_py', 'reco_taup_pi1_pz', 'reco_taup_pi1_e',
            'reco_taup_pi1_ipx', 'reco_taup_pi1_ipy', 'reco_taup_pi1_ipz',
            'reco_taup_pizero1_px', 'reco_taup_pizero1_py', 'reco_taup_pizero1_pz', 'reco_taup_pizero1_e',
            'taun_npi', 'taun_npizero',
            'reco_taun_pi1_px', 'reco_taun_pi1_py', 'reco_taun_pi1_pz', 'reco_taun_pi1_e',
            'reco_taun_pi1_ipx', 'reco_taun_pi1_ipy', 'reco_taun_pi1_ipz',
            'reco_taun_pizero1_px', 'reco_taun_pizero1_py', 'reco_taun_pizero1_pz', 'reco_taun_pizero1_e',
            'reco_mass',
            'BS_x', 'BS_y', 'BS_z',
            'taup_nu_px', 'taup_nu_py', 'taup_nu_pz',
            'taun_nu_px', 'taun_nu_py', 'taun_nu_pz'
        ]

        df = tree.pandas.df(variables)
        # filter the non pi and pipizero decay modes
        df = df[(df['taup_npi'] == 1) & (df['taup_npizero'] < 2)]
        df = df[(df['taun_npi'] == 1) & (df['taun_npizero'] < 2)]

        # now we add a bool depending on whether tau is pi or pipizero, then delete the npi and npizero columns
        df['taup_haspizero'] = df['taup_npizero'] > 0
        df['taun_haspizero'] = df['taun_npizero'] > 0
        df = df.drop(columns=['taup_npi', 'taup_npizero', 'taun_npi', 'taun_npizero'])

        # also apply a reco_mass cut to select events close to the Z pole with little boost
        df = df[(df['reco_mass'] > 91)]
        # now remove the reco_mass column
        df = df.drop(columns=['reco_mass'])

        # compute the d_min vector by subrtacting the 2 impact parameters
        df['dmin_x'] = df['reco_taup_pi1_ipx'] - df['reco_taun_pi1_ipx']
        df['dmin_y'] = df['reco_taup_pi1_ipy'] - df['reco_taun_pi1_ipy']
        df['dmin_z'] = df['reco_taup_pi1_ipz'] - df['reco_taun_pi1_ipz']       

        df.to_pickle('ditau_nu_regression_ee_to_tauhtauh_dataframe.pkl')

    else: # load the dataframe
        import pandas as pd
        df = pd.read_pickle('ditau_nu_regression_ee_to_tauhtauh_dataframe.pkl')

    #print the names of all the columns and information on the number of events in the dataframe
    print('Columns in dataframe:', df.columns.tolist())
    print('Number of events in dataframe:', len(df))
    # print number of input variables
    print('Number of input variables:', len(input_variables))