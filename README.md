# DiTauEntanglement


# Setup environment

        conda env create -f env.yml
        conda activate DiTauEntanglement

# Setup Madgraph

Download and untar madgraph:

	wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.5.7.tar.gz -O MG5_aMC_v3.5.7.tar.gz
	tar -xvf MG5_aMC_v3.5.7.tar.gz

Note you may need to update the version

# Install HEPMC3

        git clone https://gitlab.cern.ch/hepmc/HepMC3.git
        cd HepMC3
	cmake -DCMAKE_INSTALL_PREFIX=../hepmc3-install ./
	make -j8 
        make -j8 install
        cd -

# Install Pythia8

Install pythia8:

        wget https://pythia.org/download/pythia83/pythia8310.tar
	cd pythia8310
	./configure --with-hepmc3=$CONDA_PREFIX --prefix=$CONDA_PREFIX --arch=LINUX --with-python-include=$CONDA_PREFIX/include/python3.9/ --with-python-bin=$CONDA_PREFIX/bin/
        make -j8
	make -j8 install
        ln -s $CONDA_PREFIX/lib/pythia8.so $CONDA_PREFIX/lib/python3.9/site-packages/pythia8.so
	cd -


# Generate events with Madgraph

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


# Install BaysFlow

Need to pyse python version 3.10 or 3.11.
Setup a condor environment with this version:

	conda env create -f env_py_3p10.yml

Then you can install BayesFlow using

	pip install git+https://github.com/bayesflow-org/bayesflow.git
