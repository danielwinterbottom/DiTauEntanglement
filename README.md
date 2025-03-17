# DiTauEntanglement

# Setup Madgraph

Download and untar madgraph:

	wget https://launchpad.net/mg5amcnlo/3.0/3.6.x/+download/MG5_aMC_v3.5.7.tar.gz -O MG5_aMC_v3.5.7.tar.gz
	tar -xvf MG5_aMC_v3.5.7.tar.gz

Note you may need to update the version

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
