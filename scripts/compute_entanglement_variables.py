import argparse
import ROOT
import numpy as np
import uproot
import pandas as pd
from entanglement_funcs import EntanglementVariables
from array import array


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, help="Input ROOT file.")
parser.add_argument('--n_events', '-n', help='Maximum number of events to process', default=-1, type=int)
parser.add_argument('--n_skip', '-s', help='skip n_events*n_skip', default=0, type=int)
parser.add_argument("-p", "--prefix", help="Input ROOT file.", default="true", type=str) #options="true","ana_pred","pred","alt_pred"

args = parser.parse_args()

with uproot.open(args.input) as file:
    tree = file["tree"]

    total_entries = tree.num_entries
    entry_start = args.n_skip * args.n_events if args.n_events != -1 else 0
    entry_stop = total_entries if args.n_events == -1 else min(entry_start + args.n_events, total_entries)

    df = tree.arrays(entry_start=entry_start, entry_stop=entry_stop, library="pd")

#example below applys a cut on cosTheta
#df = df[abs(df['cosTheta']) < 0.2334]
#df = df[df['mass'] > 1000]
#df = df[abs(df['cosTheta']) < 0.5]
#df = df[(df['taup_npi'] == 1) & (df['taup_npizero'] == 0) & (df['taun_npi'] == 1) & (df['taun_npizero'] == 0)]
#print number of rows in the dataframe
print('Events processed:', len(df))


prefix = args.prefix

# rename prefix variables to remove prefix
df["cosn_plus"] = df[prefix+"_cosn_plus"]
df["cosn_minus"] = df[prefix+"_cosn_minus"]
df["cosr_plus"] = df[prefix+"_cosr_plus"]
df["cosr_minus"] = df[prefix+"_cosr_minus"]
df["cosk_plus"] = df[prefix+"_cosk_plus"]
df["cosk_minus"] = df[prefix+"_cosk_minus"]

df["cosncosn"] = df["cosn_plus"]*df["cosn_minus"]
df["cosrcosr"] = df["cosr_plus"]*df["cosr_minus"]
df["coskcosk"] = df["cosk_plus"]*df["cosk_minus"]
df["cosncosr"] = df["cosn_plus"]*df["cosr_minus"]
df["cosncosk"] = df["cosn_plus"]*df["cosk_minus"]
df["cosrcosk"] = df["cosr_plus"]*df["cosk_minus"]
df["cosrcosn"] = df["cosr_plus"]*df["cosn_minus"]
df["coskcosn"] = df["cosk_plus"]*df["cosn_minus"]
df["coskcosr"] = df["cosk_plus"]*df["cosr_minus"]



def ComputeEntanglementVariables(df, verbose=False):

    # note currently not sure where the minus signs come from below but they are needed to get the correct matrix, although it doesn't change the entanglement variables at all anyway...
    C11 = -df["cosncosn"].mean()*9
    C22 = -df["cosrcosr"].mean()*9
    C33 = -df["coskcosk"].mean()*9
    C12 = -df["cosncosr"].mean()*9
    C13 = -df["cosncosk"].mean()*9
    C23 = -df["cosrcosk"].mean()*9
    C21 = -df["cosrcosn"].mean()*9
    C31 = -df["coskcosn"].mean()*9
    C32 = -df["coskcosr"].mean()*9

    C11_err = df["cosncosn"].std()*9/np.sqrt(len(df))
    C22_err = df["cosrcosr"].std()*9/np.sqrt(len(df))
    C33_err = df["coskcosk"].std()*9/np.sqrt(len(df))
    C12_err = df["cosncosr"].std()*9/np.sqrt(len(df))
    C13_err = df["cosncosk"].std()*9/np.sqrt(len(df))
    C23_err = df["cosrcosk"].std()*9/np.sqrt(len(df))
    C21_err = df["cosrcosn"].std()*9/np.sqrt(len(df))
    C31_err = df["coskcosn"].std()*9/np.sqrt(len(df))
    C32_err = df["coskcosr"].std()*9/np.sqrt(len(df))

    Bplus1 = -df["cosn_plus"].mean() * 3
    Bplus2 = -df["cosr_plus"].mean() * 3
    Bplus3 = -df["cosk_plus"].mean() * 3
    Bminus1 = df["cosn_minus"].mean() * 3
    Bminus2 = df["cosr_minus"].mean() * 3
    Bminus3 = df["cosk_minus"].mean() * 3
    Bplus1_err = df["cosn_plus"].std() * 3 / np.sqrt(len(df))
    Bplus2_err = df["cosr_plus"].std() * 3 / np.sqrt(len(df))
    Bplus3_err = df["cosk_plus"].std() * 3 / np.sqrt(len(df))
    Bminus1_err = df["cosn_minus"].std() * 3 / np.sqrt(len(df))
    Bminus2_err = df["cosr_minus"].std() * 3 / np.sqrt(len(df))
    Bminus3_err = df["cosk_minus"].std() * 3 / np.sqrt(len(df))

    Bplus = np.array([Bplus1, Bplus2, Bplus3])
    Bminus = np.array([Bminus1, Bminus2, Bminus3])
    Bplus_err = np.array([Bplus1_err, Bplus2_err, Bplus3_err])
    Bminus_err = np.array([Bminus1_err, Bminus2_err, Bminus3_err])

    C = np.array([[C11, C12, C13],
                  [C21, C22, C23],
                  [C31, C32, C33]])

    C_err = np.array([[C11_err, C12_err, C13_err],
                  [C21_err, C22_err, C23_err],
                  [C31_err, C32_err, C33_err]])

    #compute c_cp which is defined as sum(|Cij-Cji|) for i < j
    c_cp = abs(C[0, 1] - C[1, 0]) + abs(C[0, 2] - C[2, 0]) + abs(C[1, 2] - C[2, 1])

    print(f"c_cp = {c_cp}")
    

    con, m12 = EntanglementVariables(C)
   
    latex = True
    if verbose:
        if latex:
            # Print in LaTeX format
            print('\\begin{split}')
            print('C = \\begin{bmatrix}')
            print(f'{C11:.2f} \\pm {C11_err:.2f}~& {C12:.2f} \\pm {C12_err:.2f}~& {C13:.2f} \\pm {C13_err:.3f} \\\\')
            print(f'{C21:.2f} \\pm {C21_err:.2f}~& {C22:.2f} \\pm {C22_err:.2f}~& {C23:.2f} \\pm {C23_err:.3f} \\\\')
            print(f'{C31:.2f} \\pm {C31_err:.2f}~& {C32:.2f} \\pm {C32_err:.2f}~& {C33:.2f} \\pm {C33_err:.3f}')
            print('\\end{bmatrix},~')

            print('B^{+} = \\begin{bmatrix}')
            print(f'{Bplus1:.2f} \\pm {Bplus1_err:.2f} \\\\')
            print(f'{Bplus2:.2f} \\pm {Bplus2_err:.2f} \\\\')
            print(f'{Bplus3:.2f} \\pm {Bplus3_err:.2f}')
            print('\\end{bmatrix},~')

            print('B^{-} = \\begin{bmatrix}')
            print(f'{Bminus1:.2f} \\pm {Bminus1_err:.2f} \\\\')
            print(f'{Bminus2:.2f} \\pm {Bminus2_err:.2f} \\\\')
            print(f'{Bminus3:.2f} \\pm {Bminus3_err:.2f}')
            print('\\end{bmatrix},~\\\\')

            print(f'\\mathcal{{C}}[\\rho] = {con:.3f},~')
            print(f'\\text{{m}}_{{12}} = {m12:.3f}')
            print('\\end{split}')
        else:
            
            np.set_printoptions(precision=3,suppress=True)
            # print C along with its errors as +/- after each element
            print('C with errors = ')
            print(np.array([[f'{C11:.3f} +/- {C11_err:.3f}', f'{C12:.3f} +/- {C12_err:.3f}', f'{C13:.3f} +/- {C13_err:.3f}'],
                             [f'{C21:.3f} +/- {C21_err:.3f}', f'{C22:.3f} +/- {C22_err:.3f}', f'{C23:.3f} +/- {C23_err:.3f}'],
                             [f'{C31:.3f} +/- {C31_err:.3f}', f'{C32:.3f} +/- {C32_err:.3f}', f'{C33:.3f} +/- {C33_err:.3f}']]))

            print ('Bplus with errors = ')
            print(np.array([[f'{Bplus1:.3f} +/- {Bplus1_err:.3f}'], 
                             [f'{Bplus2:.3f} +/- {Bplus2_err:.3f}'], 
                             [f'{Bplus3:.3f} +/- {Bplus3_err:.3f}']]))
            print ('Bminus with errors = ')
            print(np.array([[f'{Bminus1:.3f} +/- {Bminus1_err:.3f}'], 
                             [f'{Bminus2:.3f} +/- {Bminus2_err:.3f}'], 
                             [f'{Bminus3:.3f} +/- {Bminus3_err:.3f}']]))

            print('C = ')
            print(C)
            print('concurrence = %.4f' % con)
            print('m12 = %.3f' % m12)
    return(con, m12)

con, m12 = ComputeEntanglementVariables(df, True)
exit()
N = 100  # Number of bootstrap samples to generate
bootstrap_samples = []

bs_con_vals = array('d')
bs_m12_vals = array('d')

# Generate N bootstrap samples
for i in range(N):
    sample = df.sample(n=len(df), replace=True)
    bootstrap_samples.append(sample)
    sample_con, sample_m12 = ComputeEntanglementVariables(sample)

    bs_con_vals.append(sample_con)
    bs_m12_vals.append(sample_m12)

#bs_con_vals = np.random.normal(loc=0, scale=1, size=100)

print('\nconcurrence = %.4f +/- %.4f' %(con,np.std(bs_con_vals)))
print('m12 = %.4f +/- %.4f' %(m12,np.std(bs_m12_vals)))

bs_con_vals = np.array(bs_con_vals)
bs_m12_vals = np.array(bs_m12_vals)

# get assymetric errors
mean_con = np.mean(bs_con_vals)
con_hi = np.sqrt(np.mean((bs_con_vals[bs_con_vals >= mean_con] - mean_con)**2))
con_lo = np.sqrt(np.mean((mean_con - bs_con_vals[bs_con_vals < mean_con])**2))

mean_m12 = np.mean(bs_m12_vals)
m12_hi = np.sqrt(np.mean((bs_m12_vals[bs_m12_vals >= mean_m12] - mean_m12)**2))
m12_lo = np.sqrt(np.mean((mean_m12 - bs_m12_vals[bs_m12_vals < mean_m12])**2))

print('\nconcurrence = %.4f +/- +%.4f/%.4f' %(con,con_hi,con_lo))
print('m12 = %.4f +/- +%.4f/%.4f' %(m12,m12_hi,m12_lo))

# use percentiles instead to get error
con_perc_lo = np.percentile(bs_con_vals, 16)
con_perc_hi = np.percentile(bs_con_vals, 84)
m12_perc_lo = np.percentile(bs_m12_vals, 16)
m12_perc_hi = np.percentile(bs_m12_vals, 84)
print('\nconcurrence = %.4f +/- +%.4f/%.4f' %(con,con_perc_hi-con,con_perc_lo-con))
print('m12 = %.4f +/- +%.4f/%.4f' %(m12,m12_perc_hi-m12,m12_perc_lo-m12))
