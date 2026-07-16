import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help= 'Name of input directory which should contain subdirectories named job_output_0, job_output_1, etc. with the LHE files in them that are named events_0.lhe, events_1.lhe, etc.')
parser.add_argument('--cmnd_file', '-c', help= 'Pythia8 command file')
parser.add_argument('--N_jobs', '-n', help= 'Number of job to submit', default=-1, type=int)
parser.add_argument('--job_name', '-j', help= 'Name of job to submit', default='pythia')
parser.add_argument('--no_lhe', default=1, help= 'If set, do not use LHE files as input to Pythia8 i.e generate ee->tautau events directly in Pythia8', type=int)
parser.add_argument('--extra', default='', help= 'Extra options for the shower_events.py script')
parser.add_argument('--start_seed', default=0, help= 'First seed number to use for the jobs, seeds will be assigned sequentially starting from this number', type=int)

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

start_seed = args.start_seed

out_string = f'#!/bin/sh\n\
echo \"Cluster = $1 Process = $2\"\n\
SEED=$(($2 + {start_seed}))\n\
echo "Seed = $SEED"\n\
cd {current_dir}/\n' % vars()

if args.no_lhe == 0:
    out_string += f'rm {args.job_name}/job_output_$SEED/*.{{root,hepmc}}\n\
gunzip {args.input}/job_output_$SEED/events_$SEED.lhe.gz\n\
python taupolaris/generation/shower_events.py -c {args.cmnd_file}  -i {args.input}/job_output_$SEED/events_$SEED.lhe -o {args.input}/job_output_$SEED/pythia_events_$SEED.hepmc --seed $SEED -n -1 {args.extra}\n\
' % vars()
else:
    nperjob = 10000
    out_string += f'mkdir -p {args.job_name}/job_output_$SEED/\n\
rm {args.job_name}/job_output_$SEED/*.{{root,hepmc}}\n\
python taupolaris/generation/shower_events.py -c {args.cmnd_file} -n {nperjob} -o {args.job_name}/job_output_$SEED/pythia_events_$SEED.hepmc --seed $SEED {args.extra}\n'

out_string += f'DelphesHepMC3 taupolaris/generation/configs/delphes_card_CMS.tcl {args.job_name}/job_output_$SEED/delphes_output_$SEED.root {args.job_name}/job_output_$SEED/pythia_events_$SEED.hepmc\n\
python taupolaris/generation/run_delphes.py -i {args.job_name}/job_output_$SEED/delphes_output_$SEED.root -o {args.job_name}/job_output_$SEED/reco_events_$SEED.root\n\
rm {args.job_name}/job_output_$SEED/pythia_events_$SEED.hepmc\n\
#rm {args.job_name}/job_output_$SEED/pythia_events_$SEED.root\n\
rm {args.job_name}/job_output_$SEED/delphes_output_$SEED.root'

os.system('mkdir -p jobs')

name = args.job_name

print('Writing job file: jobs/parajob_%(name)s.sh' % vars())
with open("jobs/parajob_%(name)s.sh" % vars(), "w") as output_file:
    output_file.write(out_string)
os.system('chmod +x jobs/parajob_%(name)s.sh' % vars())

print('Submitting jobs')
Submit('jobs/parajob_%(name)s.sh' % vars(), name, args.N_jobs)
