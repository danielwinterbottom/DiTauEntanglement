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


# Quick example 

Prepare dataframes:

```
python taupolaris/scripts/prepare_inputs.py -c taupolaris/config/LHC_Transformer_HadronicOnly.yaml
```


Run Nflows training:

```
python taupolaris/scripts/train.py -c taupolaris/config/LHC_Transformer_HadronicOnly.yaml
```


Train a "normal" neural network with MSE loss to compare to:

```
python taupolaris/scripts/train.py -c taupolaris/config/LHC.yaml --useMLP
```

Can use the taupolaris/utils/batch_submission.py script to run any python script as a batch job e.g using one of the GPU nodes.
For running the training on the batch use:

	python taupolaris/utils/batch_submission.py -c "python taupolaris/scripts/train.py -c taupolaris/config/LHC_Transformer_HadronicOnly.yaml" --gpu --runtime 86400 --job_name training_job


Test Nflows model:

```
python taupolaris/scripts/evaluate.py -c taupolaris/config/LHC_Transformer_HadronicOnly.yaml
```

Test MLP model (haven't tested this explicitly yet)
```
python taupolaris/scripts/evaluate.py -c taupolaris/config/LHC.yaml --useMLP
```

# Event Generation

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

### Installing Delphes

Install with condor:
	conda install --channel conda-forge delphes

### Generating LHC events

This is run in 3 stages. First pythia is run to produce .hepmc files, then delphes is run to do the detector smearing, and then the outputs of delphes are processed to reconstruct the taus and their decay products using HPS-like algorithms


To run all 3 stages together as batch jobs use:

	python taupolaris/generation/submit_pythia_jobs_LHC.py -c taupolaris/generation/configs/pythia_cmnd_maindm -j ppToHToTauTau_CPOdd --extra="--phi=0" -n 3000

This runs 10000 events per job. The phi angle changes the CP nature of the higgs. phi=0 = CP-odd, 1.5708 = CP-even, +/-0.7854 = max-mix (+/- determines sign of inteference term)

To run 3 stages seperatly:

	python taupolaris/generation/shower_events.py -c taupolaris/generation/configs/pythia_cmnd_maindm -n 10000 -o ppToHToTauTau_CPOdd/job_output/pythia_events.hepmc --seed 1 --phi=0

	DelphesHepMC3 taupolaris/generation/configs/delphes_card_CMS.tcl ppToHToTauTau_CPOdd/job_output/delphes_output.root ppToHToTauTau_CPOdd/job_output/pythia_events.hepmc

	python taupolaris/generation/run_delphes.py -i ppToHToTauTau_CPOdd/job_output/delphes_output.root -o ppToHToTauTau_CPOdd/job_output/reco_events.root
