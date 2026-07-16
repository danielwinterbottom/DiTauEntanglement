# TauPolaris

TauPolaris is a package for reconstructing and studying tau polarimetric vectors. It provides:

- Generation of ditau event samples (e+e- and LHC processes, via Pythia + Delphes), including a from-scratch Python reimplementation of TauSpinner's spin-correlation reweighting, so events can be produced/reweighted under different CP hypotheses without depending on the external TauSpinner library.
- Training of machine-learning models -- primarily conditional normalizing flows with transformer-based conditioning -- to reconstruct tau polarimetric vectors and full tau kinematics from reconstructed decay products.
- Exact computation of polarimetric vectors when the true tau decay products are known, independent of any ML model -- useful as ground truth for training/evaluation, or as a standalone calculation.


# Setup environment

        conda env create -f env.yml
        conda activate taupolarisenv


# New Setup gude:

need to run this to get the package
```
pip install -e .
```



# Quick(-ish) example

This example uses the reduced datasets for that it can be run reasonably quickly.

Prepare dataframes:

```
python taupolaris/scripts/prepare_inputs.py -c taupolaris/config/LHC.yaml
```


Run Nflows training:


```
python taupolaris/scripts/train.py -c taupolaris/config/LHC.yaml
```


Train a "normal" neural network with MSE loss to compare to:

```
python taupolaris/scripts/train.py -c taupolaris/config/LHC.yaml --useMLP
```

Can use the taupolaris/utils/batch_submission.py script to run any python script as a batch job e.g using one of the GPU nodes.
For running the training on the batch use:

	python taupolaris/utils/batch_submission.py -c "python taupolaris/scripts/train.py -c taupolaris/config/LHC.yaml" --gpu --runtime 86400 --job_name model_NFlows_LHC_onnorm_reco_May08


Test Nflows model:

```
python taupolaris/scripts/evaluate.py -c taupolaris/config/LHC.yaml
```

Test MLP model (haven't tested this explicitly yet)
```
python taupolaris/scripts/evaluate.py -c taupolaris/config/LHC.yaml --useMLP
```

## Instructions for pp->H training

# Produce MC samples

Note for now no smearing is applied to the gen quantities

	python scripts/submit_pythia_jobs.py --no_lhe --cmnd_file scripts/pythia_cmnd_dm0and1  -j ppToHToTauTau_DM0and1_CPEven -n 1000 --extra="--phi=1.5708 --extra_vars --pythia_process=ppToHToTauTau"

which will produce 10000 per event (10M total)

To produce CP-odd Higgs bosons use --phi=0, or for max-mix use --phi=0.7854, or max-mix with the opposite interference use -0.7854. Any other intermediate values can be used as well of course.


Can then hadd these files together. For convinience we seperate 100k events to be used purely for validation:

	for x in ppToHToTauTau_DM0and1_CPEven; do
		hadd -f ${x}/pythia_events_validation.root ${x}/job_output_*/pythia_events_[0-9].root
                hadd -f ${x}/pythia_events_training.root ${x}/job_output_*/pythia_events_[0-9][0-9]*.root;
done

# Produce dataframes:

        python python/LEP_NF_reco.py -s 1 --ppHTraining

# Train the NFlows model:

	python python/LEP_NF_reco.py -s 3 --ppHTraining --dataframe_name ditau_nu_regression_ppToHToTauTau_DM0and1_CPEven_dataframe.pkl -m model_ppToH_NFlows_v0 -n 3

# Test the NFlows model:

	python python/LEP_NF_reco.py -s 4 --ppHTraining --dataframe_name ditau_nu_regression_ppToHToTauTau_DM0and1_CPEven_validation_dataframe.pkl -m model_ppToH_NFlow_v0



# Event Generation

## Setup Madgraph

Download and untar madgraph:

	wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.5.7.tar.gz -O MG5_aMC_v3.5.7.tar.gz
	tar -xvf MG5_aMC_v3.5.7.tar.gz

Note you may need to update the version

## Install HEPMC3

        git clone https://gitlab.cern.ch/hepmc/HepMC3.git
        cd HepMC3
	cmake -DCMAKE_INSTALL_PREFIX=../hepmc3-install ./
	make -j8 
        make -j8 install
        cd -

## Install Pythia8

Install pythia8:

        wget https://pythia.org/download/pythia83/pythia8310.tar
	cd pythia8310
	./configure --with-hepmc3=$CONDA_PREFIX --prefix=$CONDA_PREFIX --arch=LINUX --with-python-include=$CONDA_PREFIX/include/python3.9/ --with-python-bin=$CONDA_PREFIX/bin/
        make -j8
	make -j8 install
        ln -s $CONDA_PREFIX/lib/pythia8.so $CONDA_PREFIX/lib/python3.9/site-packages/pythia8.so
	cd -


## Generate events with Madgraph

To generate events with entanglement included e.g for tau->pinu decays:

	import model sm-lepton_masses
	add model taudecay_UFO

	generate e+ e- > ta+ ta- / h, ta+ > pi+ vt~, ta- > pi- vt	

To remove entanglment use:

	generate e+ e- > ta+{R} ta-{L} / h, ta+ > pi+ vt~, ta- > pi- vt	
	add process e+ e- > ta+{L} ta-{R} / h, ta+ > pi+ vt~, ta- > pi- vt	
	add process e+ e- > ta+{L} ta-{L} / h, ta+ > pi+ vt~, ta- > pi- vt	
	add process e+ e- > ta+{R} ta-{R} / h, ta+ > pi+ vt~, ta- > pi- vt	

This thread has some more info on generating taus with different helicities: https://answers.launchpad.net/mg5amcnlo/+question/694581


To produce gridpacks for both entanglement and no entanglement scenarios

	./MG5_aMC_v3_5_7/bin/mg5_aMC -f scripts/mg_lep_command 

This will produce the gridpacks:

	ee_to_tauhtauh_inc_entanglement/run_01_gridpack.tar.gz 
	ee_to_tauhtauh_no_entanglement/run_01_gridpack.tar.gz 

To run some events locally do:

	cd test
	tar -xvf ../ee_to_tauhtauh_inc_entanglement/run_01_gridpack.tar.gz 
	./run.sh 1000 1 

For larger submission the lhe files can be produced as batch jobs, e.g to produce 10M events foot both scenarios:

	python scripts/submit_gridpack_batch_jobs.py --events_per_job 10000 --total_events 10000000 -i ee_to_tauhtauh_inc_entanglement/run_01_gridpack.tar.gz -o ee_to_tauhtauh_inc_entanglement
	python scripts/submit_gridpack_batch_jobs.py --events_per_job 10000 --total_events 10000000 -i ee_to_tauhtauh_no_entanglement/run_01_gridpack.tar.gz -o ee_to_tauhtauh_no_entanglement

To run the parton shower (pythia) locally use:

	python scripts/shower_events.py -c scripts/pythia_cmnd  -i batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_2000_Njob_1000/job_output_0/events_0.lhe -n -1

Then to smear the gen-level quantities and compute the spin observables used to test entanglement you can run

	python scripts/compute_spin_vars.py -i pythia_output.root -o pythia_events_extravars.root

You can submit both of the previous commands as as batch jobs using:

	python scripts/submit_pythia_jobs.py --cmnd_file scripts/pythia_cmnd  -i batch_job_outputs/ee_to_tauhtauh_inc_entanglement_Ntot_10000000_Njob_10000/ -j pythia_inc_entanglement -n 1000
	python scripts/submit_pythia_jobs.py --cmnd_file scripts/pythia_cmnd  -i batch_job_outputs/ee_to_tauhtauh_no_entanglement_Ntot_10000000_Njob_10000/ -j pythia_no_entanglement -n 1000

## List of Produced samples:

Many samples have been produced already, a non-exhastive list is below:

~30M ee->Z->tautau LEP events, where taus are forced to decay into either DM=0 or DM=1: 

	/vols/cms/dw515/HH_reweighting/DiTauEntanglement/batch_job_outputs/ee_to_tauhtauh_dm0and1only_inc_entanglement_Ntot_30000000_Njob_10000/pythia_events_extravars.root

~30M ee->Z->tautau LEP events with no entanglement, where taus are forced to decay into either DM=0 or DM=1: 

	/vols/cms/dw515/HH_reweighting/DiTauEntanglement/batch_job_outputs/ee_to_tauhtauh_dm0and1only_no_entanglement_Ntot_30000000_Njob_10000/pythia_events_extravars.root

Smaller datasets for quicker training:

ee->Z->tautau LEP events, where taus are forced to decay into either DM=0 or DM=1: 
	/vols/cms/dw515/HH_reweighting/DiTauEntanglement/batch_job_outputs/ee_to_tauhtauh_dm0and1only_inc_entanglement_Ntot_30000000_Njob_10000/pythia_events_extravars_reduced.root

ee->Z->tautau LEP events with no entanglement, where taus are forced to decay into either DM=0 or DM=1: 
	/vols/cms/dw515/HH_reweighting/DiTauEntanglement/batch_job_outputs/ee_to_tauhtauh_dm0and1only_no_entanglement_Ntot_30000000_Njob_10000/pythia_events_extravars_reduced.root

### Running Delphes

Install with condor:
	conda install --channel conda-forge delphes

Run delphes starting from a .hepmc file:
	DelphesHepMC3 $CONDA_PREFIX/cards/delphes_card_CMS.tcl delphes_output_pp.root pythia_events_test.hepmc

### generating LHC events

This is run in 3 stages. First pythia is run to produce .hepmc files, then delphes is run to do the detector smearing, and then the outputs of delphes are processed to reconstruct the taus and their decay products using HPS-like algorithms


To run all 3 stages together as batch jobs use:

	python taupolaris/generation/submit_pythia_jobs_LHC.py -c taupolaris/generation/configs/pythia_cmnd_dm0and1 -j ppToHToTauTau_DM0and1_CPOdd_May07 --extra="--phi=0" -n 3000

This runs 10000 events per job. The phi angle changes the CP nature of the higgs. phi=0 = CP-odd, 1.5708 = CP-even, +/-0.7854 = max-mix (+/- determines sign of inteference term)
