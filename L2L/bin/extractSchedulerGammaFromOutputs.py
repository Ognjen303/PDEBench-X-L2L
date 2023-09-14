import re
import os
from pathlib import Path


# THIS SCRIPT IS USED TO PROCESS THE DATA FROM THE LEARNING2LEARN + PDEBENCH
# NEURAL NETWORK TRAINING OUTPUT FILES.

# THIS SCRIPT WAS USED MORE FOR DOING A 'SANITY-CHECK' OF MY RESULT
# JUST TO SEE IF I GET THE EXPECTED SCHEDULER GAMMA VALUES.

# AFTER THE TRAINING OF NEURAL NETWORK HAS FINISHED
# Change the relevant path, the path should look like e.g.

# folderPath = '/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/L2L/results/12206922_v1-L2L-PDEBench-ES_lr_1e-04_psize_7_gens_10_batchsize_100/simulation/work' 

folderPath = '/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/L2L/results/12257630-L2LPDEBench-ES_ScGamma_1.00e-03_lr_5e-03_psize_7_gens_17_batchsize_1000/simulation/work/'

schedulerGammas = []

schedulerGammaRegex = re.compile('The traj scheduler_gamma is: ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?')


for folderName, subfolders, filenames in os.walk(folderPath):
    
    if 'stdout' in filenames:
        
        stdoutFile = open(Path(folderName) / 'stdout', 'r')
        content = stdoutFile.read()
        
        mo_gamma = schedulerGammaRegex.search(content)
        
        gamma_string = mo_gamma.group()
        scheduler_gamma = float(gamma_string.split(' ')[-1])
        print(scheduler_gamma)
        schedulerGammas.append(scheduler_gamma)