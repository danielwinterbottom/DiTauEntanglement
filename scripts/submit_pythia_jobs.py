import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'Name of input directory which should contain subdirectories named job_output_0, job_output_1, etc. with the LHE files in them that are named events_0.lhe, events_1.lhe, etc.')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--N_jobs', '-n', help= 'Number of job to submit', default=-1, type=int)
parser.add_argument('--job_name', '-j', help= 'Name of job to submit', default='pythia')
parser.add_argument('--no_lhe', action='store_true', help= 'If set, do not use LHE files as input to Pythia8 i.e generate ee->tautau events directly in Pythia8')
parser.add_argument('--extra', default='', help= 'Extra options for the shower_events.py script')

args = parser.parse_args()

current_dir=os.getcwd()
input = args.input

def Submit(job, jobname, N):
   print(jobname)

   job_file_name = 'jobs/job_%(jobname)s.job' % vars()
   job_file_out_name = 'jobs/job_%(jobname)s_$(CLUSTER)_$(PROCESS).job' % vars()

   out_str =  'executable = %s\n' % job
   out_str += 'getenv = True\n'
   out_str += 'arguments = \"$(CLUSTER) $(PROCESS)\"\n' % vars()
   out_str += 'output = %(job_file_out_name)s.out\n' % vars()
   out_str += 'error = %(job_file_out_name)s.err\n' % vars()
   out_str += 'log = %(job_file_out_name)s.log\n' % vars()
   out_str += '+MaxRuntime = 10800\n'
   out_str += 'queue %i' % N
   job_file = open(job_file_name,'w')
   job_file.write(out_str)
   job_file.close()

   os.system('condor_submit %(job_file_name)s' % vars())

#python scripts/compute_spin_vars_pp.py -i {args.input}/job_output_$2/pythia_events_$2.root -o {args.input}/job_output_$2/pp_pythia_events_extravars_$2.root\n\ - equivalent for pp collisions
if not args.no_lhe:
    out_string = f'#!/bin/sh\n\
    echo \"Cluster = $1 Process = $2\"\n\
    cd {current_dir}/\n\
    gunzip {args.input}/job_output_$2/events_$2.lhe.gz\n\
    python scripts/shower_events.py -c {args.cmnd_file}  -i {args.input}/job_output_$2/events_$2.lhe -o {args.input}/job_output_$2/pythia_events_$2.hepmc --seed $2 -n -1 {args.extra}\n\
    #python scripts/compute_spin_vars.py -i {args.input}/job_output_$2/pythia_events_$2.root -o {args.input}/job_output_$2/pythia_events_extravars_$2.root\n\
    #python scripts/compute_spin_vars.py -i {args.input}/job_output_$2/pythia_events_$2.root -o {args.input}/job_output_$2/pythia_events_extravars_$2_LHE.root --useLHE\n\
    python scripts/compute_spin_vars.py -i {args.input}/job_output_$2/pythia_events_$2.root -o {args.input}/job_output_$2/pythia_events_extravars_$2.root\n\
    ' % vars()
else:
    out_string = f'#!/bin/sh\n\
    echo \"Cluster = $1 Process = $2\"\n\
    cd {current_dir}/\n\
    mkdir -p {args.input}/job_output_$2/\n\
    python scripts/shower_events.py -c {args.cmnd_file} -n 10000 -o {args.input}/job_output_$2/pythia_events_$2.hepmc --seed $2 {args.extra}\n\
    python scripts/compute_spin_vars.py -i {args.input}/job_output_$2/pythia_events_$2.root -o {args.input}/job_output_$2/pythia_events_extravars_$2.root\n\
    ' % vars()

os.system('mkdir -p jobs')

name = args.job_name

print('Writing job file: jobs/parajob_%(name)s.sh' % vars())
with open("jobs/parajob_%(name)s.sh" % vars(), "w") as output_file:
    output_file.write(out_string)
os.system('chmod +x jobs/parajob_%(name)s.sh' % vars())

print('Submitting jobs')
Submit('jobs/parajob_%(name)s.sh' % vars(), name, args.N_jobs)
