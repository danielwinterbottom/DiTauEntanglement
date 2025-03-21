import pandas as pd
import uproot
import os

def read_root_in_chunks(filename, treename, output_name='df.pkl', nchunks=10, variables=None, verbosity=0):

    """
    Read a ROOT file into dataframe in chunks and save each chunk as a pickle file.
    Parameters:
    - filename (str): The name of the ROOT file to read.
    - treename (str): The name of the TTree to read.
    - output_name (str): The name of the output pickle file.
    - nchunks (int): The number of chunks to divide the data into.
    - verbosity (int): The level of verbosity for printing information.
    """

    # Check if the file exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    if filename.endswith('.pkl'):
        raise ValueError('Filename must end with .pkl')

    if verbosity > 0: print('Preparing dataframes for sample: %s using %i chunks' %(f,nchunks))
    # Open the root file
    tree = uproot.open(filename)[treename]

    # Get the total number of entries
    num_entries = tree.num_entries

    chunksize = int(num_entries/nchunks)

    if verbosity > 0: print('Total events in sample = %(num_entries)i, number of events per chunk = %(chunksize)g' % vars())

    # Iterate over the chunks
    for i in range(nchunks):
        # Calculate start and stop indices for the current chunk
        start = i * chunksize
        stop = min((i + 1) * chunksize, num_entries)
        if verbosity > 1: print ('Processing chunk %(i)i, start and stop indices = %(start)s and %(stop)s' % vars())

        # Read the current chunk into a dataframe
        if variables is None: variables = tree.keys()
        # filter any variables that are not in the tree
        variables = [x for x in variables if x in tree.keys()]
        if verbosity > 1: print('Reading variables: %s' % variables)
        df = tree.arrays(variables, library="pd", entry_start=start, entry_stop=stop) 

        if verbosity > 1: 
            print('Number of events in chunk = %i' % len(df))
            print("First Entry in chunk:")
            print(df.head(1))
            print("Last Entry in chunk:")
            print(df.tail(1))

        # save dataframe then delete

        output = output_name.replace('.pkl', '_chunk_%i.pkl' % i)

        if verbosity > 1: print('Writing %(output)s\n' % vars())
        
        df.to_pickle(output)
        del df

def PrepareDataframes(filename1, filename2, treename1, treename2, nchunks=10, verbosity=0):

    """
    Prepare dataframes from ROOT files containing different friend trees.
    The Two trees contain different variables for the same events.
    """

    variables = [
        'taup_px','taup_py','taup_pz','taup_e',
        'taun_px','taun_py','taun_pz','taun_e',
        'taup_nu_px','taup_nu_py','taup_nu_pz','taup_nu_e',
        'taun_nu_px','taun_nu_py','taun_nu_pz','taun_nu_e',
        'taup_npi','taup_npizero',
        'taun_npi','taun_npizero',
        'reco_taup_pi1_px','reco_taup_pi1_py','reco_taup_pi1_pz','reco_taup_pi1_e',
        'reco_taup_pi2_px','reco_taup_pi2_py','reco_taup_pi2_pz','reco_taup_pi2_e',
        'reco_taup_pi3_px','reco_taup_pi3_py','reco_taup_pi3_pz','reco_taup_pi3_e',
        'reco_taup_pizero1_px','reco_taup_pizero1_py','reco_taup_pizero1_pz','reco_taup_pizero1_e',
        'reco_taup_pizero2_px','reco_taup_pizero2_py','reco_taup_pizero2_pz','reco_taup_pizero2_e',
        'reco_taun_pi1_px','reco_taun_pi1_py','reco_taun_pi1_pz','reco_taun_pi1_e',
        'reco_taun_pi2_px','reco_taun_pi2_py','reco_taun_pi2_pz','reco_taun_pi2_e',
        'reco_taun_pi3_px','reco_taun_pi3_py','reco_taun_pi3_pz','reco_taun_pi3_e',
        'reco_taun_pizero1_px','reco_taun_pizero1_py','reco_taun_pizero1_pz','reco_taun_pizero1_e',
        'reco_taun_pizero2_px','reco_taun_pizero2_py','reco_taun_pizero2_pz','reco_taun_pizero2_e',
        'reco_taup_vx','reco_taup_vy','reco_taup_vz',
        'reco_taun_vx','reco_taun_vy','reco_taun_vz',
        'reco_taup_pi1_ipx','reco_taup_pi1_ipy','reco_taup_pi1_ipz',
        'reco_taun_pi1_ipx','reco_taun_pi1_ipy','reco_taun_pi1_ipz',
        'reco_Z_px','reco_Z_py','reco_Z_pz','reco_Z_e',
        'taup_pi1_vz','taup_vz',
    ]

    read_root_in_chunks(filename1, treename1, output_name='output/df.pkl', nchunks=nchunks, variables=variables, verbosity=verbosity)
    read_root_in_chunks(filename2, treename2, output_name='output/df_friend.pkl', nchunks=nchunks, variables=variables, verbosity=verbosity)


    # for each chunk concatenate along columns since they have the same event order and save the concatenated dataframe
    for i in range(nchunks):
        # load the two dataframes
        df = pd.read_pickle('output/df_chunk_%i.pkl' % i)
        df_friend = pd.read_pickle('output/df_friend_chunk_%i.pkl' % i)

        # concatenate along columns
        df = pd.concat([df, df_friend], axis=1)

        # the variable in the first dataframe called "taup_pi1_vz" should match the variables from the second dataframe called "taup_vz"
        # drop any events where this is not the case, printing how many events are dropped and then drop these collumns from the dataframe
        if verbosity>0: print(f"Number of events before dropping: {len(df)}")
        df = df[df["taup_pi1_vz"] == df["taup_vz"]]
        if verbosity>0: print(f"Number of events after dropping unmatched events from the two trees: {len(df)}")
        # drops nans
        df = df.dropna()
        if verbosity>0: print(f"Number of events after dropping nans: {len(df)}")
        df = df.drop(columns=["taup_pi1_vz", "taup_vz"])

        # save the concatenated dataframe
        if verbosity>0: print('Writing output/df_chunk_%i.pkl' % i)
        df.to_pickle('output/df_chunk_%i.pkl' % i)
        # delete the two dataframes
        del df
        del df_friend
        # delete the friends pkl file
        os.remove('output/df_friend_chunk_%i.pkl' % i)


if __name__ == '__main__':
    filename1 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events.root"
    filename2 = "batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/pythia_events_extravars.root"
    treename1 = "tree"
    treename2 = "new_tree"

    PrepareDataframes(filename1, filename2, treename1, treename2)
