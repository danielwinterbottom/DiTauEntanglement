import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'Name of input directory which should contain subdirectories named job_output_0, job_output_1, etc. with the LHE files in them that are named events_0.lhe, events_1.lhe, etc.')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--N_jobs', '-n', help= 'Number of job to submit', default=-1, type=int)
parser.add_argument('--job_name', '-j', help= 'Name of job to submit', default='pythia')
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

out_string = f'#!/bin/sh\n\
echo \"Cluster = $1 Process = $2\"\n\
cd {current_dir}/\n\
gunzip {args.input}/job_output_$2/events_$2.lhe.gz\n\
python scripts/shower_events.py -c {args.cmnd_file}  -i {args.input}/job_output_$2/events_$2.lhe -o {args.input}/job_output_$2/pythia_events_$2.hepmc --seed $2 -n -1\n\
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
