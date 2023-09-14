import re
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# THIS SCRIPT IS USED TO PROCESS THE DATA FROM THE LEARNING2LEARN + PDEBENCH
# NEURAL NETWORK TRAINING OUTPUT FILES.

# AFTER THE TRAINING OF NEURAL NETWORK HAS FINISHED
# Change the relevant path, the path should look like e.g.

# folderPath = '/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/L2L/results/12206922_v1-L2L-PDEBench-ES_lr_1e-04_psize_7_gens_10_batchsize_100/simulation/work'

folderPath = '/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/L2L/results/12262173-100GPU_Testrun_L2LPDEBench-ES_lr_5e-03_psize_31_gens_17_batchsize_1000/simulation/work'


baselineModelRmse = (1e-3, 1.68e-2)
lrs = []

valRmses = []
val_nRmses = []
val_cRmses = []
val_bRmses = []
testRmses = []
fitnesses = []
generation_index = []

# If the output file structure of a individual gets changed, then you must also change these Regexes
# Note: The easiest way to generate a suitble Regex is via: https://regex-generator.olafneumann.org

# This Regex matches e.g. 'The learning rate was: 0.005' in the output file
learningRateRegex = re.compile(r'The learning rate was: ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?')

# This Regex matches e.g. 'Test RME: 0.001234' in the output file
testRmseRegex = re.compile(r'Test RMSE: ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?')

# This Regex matches e.g. 'Validation RMSE: 0.001234' in the output file
valRmseRegex = re.compile(r'Validation RMSE: ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?')

# This Regex matches e.g. 'The fitness is: [0.01595898 0.02780193 0.00615715 0.03753364]' in the output file
fitnessRegex = re.compile(r'The fitness is: \[[^\]]*\]')

# This Regex matches a floating point number (with optional exponent) e.g. '0.0123'
floatingPointNumberRegex = re.compile('([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?')


for folderName, subfolders, filenames in os.walk(folderPath):
    # print('The current folder is ' + folderName)

    if 'stdout' in filenames:      
        
        generation_index.append(folderName.split('_')[-2])
        
        stdoutFile = open(Path(folderName) / 'stdout', 'r')
        content = stdoutFile.read()
        # print(f'The length of content is {len(content)}')
        
        # mo stands for 'Match' object, since search() returns this type
        mo_lr = learningRateRegex.search(content)
        mo_val_rmse = valRmseRegex.search(content)
        mo_test_rmse = testRmseRegex.search(content)
        mo_fitness_vector = fitnessRegex.search(content)
        
        # .group() returns a string of the actual matched text
        # lr_string should be equal to e.g.: 'The learning rate was: 0.005'
        lr_string = mo_lr.group()
        lr = float(lr_string.split(' ')[-1])
        lrs.append(lr)
        
        # test_rmse_string should be equal to e.g.: 'Test RME: 0.001234'
        test_rmse_string = mo_test_rmse.group()
        test_rmse = float(test_rmse_string.split(' ')[-1])
        testRmses.append(test_rmse)
        
        # fitness_vector_string should be equal to e.g.: 'The fitness is: [0.01595898 0.02780193 0.00615715 0.03753364]'
        fitness_vector_string = mo_fitness_vector.group()
        
        # .findall() returns a list, in this case fitness_entries is [('0.02021489', ''), ('0.04168874', ''), ('0.01485859', ''), ('0.04310259', '')]
        fitness_entries = floatingPointNumberRegex.findall(fitness_vector_string)
        
        fitness_list = [float(entry[0]) for entry in fitness_entries]
        
        # fitness_list contains in order: RMSE, normalized RMSE, conserved RMSE, boundary RMSE
        valRmses.append(fitness_list[0])
        val_nRmses.append(fitness_list[1])
        val_cRmses.append(fitness_list[2])
        val_bRmses.append(fitness_list[3])
        
        fitness = np.average(fitness_list)
        fitnesses.append(fitness)


assert len(lrs) == len(valRmses)
assert len(lrs) == len(val_nRmses)
assert len(lrs) == len(val_cRmses)
assert len(lrs) == len(val_bRmses)
assert len(lrs) == len(testRmses)
assert len(lrs) == len(fitnesses)

lrs = np.array(lrs)
valRmses = np.array(valRmses)
val_nRmses = np.array(val_nRmses)
val_cRmses = np.array(val_cRmses)
val_bRmses = np.array(val_bRmses)
testRmses = np.array(testRmses)
fitnesses = np.array(fitnesses)



# generation_index is a list which tells you which individual belongs to which generation
generation_index = np.array(generation_index, dtype=int)

# number_of_gens is the total number of generations
number_of_gens = len(np.unique(generation_index))


#print(f'generation_index list: {generation_index}')
print('Start of for loop.')


# We want to find the average fitness of each generation and
# we want to find the individual with the best (lowest) fitness in each generation
# so we initialize these emty arrays
avg_fitness_of_each_gen = np.empty(number_of_gens)
idxOfBestIndivOfEachGen = np.empty(number_of_gens, dtype=int)


for generation in range(number_of_gens):
    
    #print(f'generation: {generation}')
    
    indices = [i for i, x in enumerate(generation_index) if x == generation]
    # print(f'indices_type: {type(indices)}')
    
    # fitnesses[indices] returns a list of fitnesses which are from the same generation
    # e.g. all fitnesses from the 3rd generation
    
    #print(f'fitnesses[indices]: {np.array(fitnesses)[indices]}')
    average_fitness = np.average(fitnesses[indices])
    avg_fitness_of_each_gen[generation] = average_fitness
    
    # Now we want to find the individual with the best (lowest) fitness in each generation
    
    # idx_fit_dict is a dictionary with a key as index and value as fitness
    idx_fit_dict = {key: value for key, value in zip(indices, fitnesses[indices])}
    
    # Find the lowest fitness in the dictionary and return the corresponding index. Store this index
    min_idx = min(idx_fit_dict, key=idx_fit_dict.get)    
    min_fitness = idx_fit_dict[min_idx]
    
    idxOfBestIndivOfEachGen[generation] = min_idx
    
    
    # ----------------SANITY CHECK------------ 
    # The below is gives the same best fitnesses as the L2L framework automatically
    # prints this out in the error file inside of ./errors/
    
    print(f'-- End of generation {generation} --')
    print(f'Evaluated {len(indices)} individuals.')
    print(f'Best Fitness: {min_fitness:.4f}')
    print(f'Average Fitness: {average_fitness:.4f}')

    
    

# Find the values with the best fitness and lowest RMSE
# According to the validation dataset
min_fitness_index = np.argmin(fitnesses)
# min_val_rmse_index = np.argmin(valRmses)

print(f'The lowest and best fitness is: {fitnesses[min_fitness_index]:.2e} and the corresponding validation and test RMSE are {valRmses[min_fitness_index]:.2e} and {testRmses[min_fitness_index]:2e} respectively. The learning rate is: {lrs[min_fitness_index]:.2e}, and the corresponding generation is: {generation_index[min_fitness_index]}.\n')

# print(f'The lowest validation RMSE is: {valRmses[min_val_rmse_index]:.2e}. It was achieved in generation {generation_index[min_val_rmse_index]} with the learning rate: {lrs[min_val_rmse_index]:.2e}. The corresponding test RMSE is: {testRmses[min_val_rmse_index]:.2e}')



# ----------------- PLOTTING ----------------



# Define a custom colormap transitioning from light yellow to dark blue-green
light_yellow = mcolors.hex2color('#FFFFCC')
dark_blue_green = mcolors.hex2color('#006622')
num_colors = number_of_gens

custom_colors = []
for i in range(num_colors):
    r = light_yellow[0] + (dark_blue_green[0] - light_yellow[0]) * i / (num_colors - 1)
    g = light_yellow[1] + (dark_blue_green[1] - light_yellow[1]) * i / (num_colors - 1)
    b = light_yellow[2] + (dark_blue_green[2] - light_yellow[2]) * i / (num_colors - 1)
    custom_colors.append((r, g, b))

# Create a custom colormap using the defined colors
cmap = mcolors.ListedColormap(custom_colors)


did_you_change_the_name_of_the_plots = False

# YOU NEED TO CHANGE THE NAME OF THE PLOTS WHENEVER
# plt.savefig() IS CALLED IN THE CODE BELOW TO AVOID OVERWRITING
# THE ALREADY EXISTING PLOTS
assert did_you_change_the_name_of_the_plots == True







# IF YOU WANT TO PLOT ALL THE INDIVIDUALS, THEN OMIT THE FOLLOWING LINES
# IF YOU WANT TO PLOT ONLY THE "BEST" INDIVIDUALS IN EACH GENERATION
# (i.e. THE ONES WITH THE LOWEST FITNESS) THEN INCLUDE THE FOLLOWING LINES



print(f'idxOfBestIndivOfEachGen: {idxOfBestIndivOfEachGen}')
print(f'type of idxOfBestIndivOfEachGen: {type(idxOfBestIndivOfEachGen)}')


lrs = lrs[idxOfBestIndivOfEachGen]
valRmses = valRmses[idxOfBestIndivOfEachGen]
val_nRmses = val_nRmses[idxOfBestIndivOfEachGen]
val_cRmses = val_cRmses[idxOfBestIndivOfEachGen]
val_bRmses = val_bRmses[idxOfBestIndivOfEachGen]
testRmses = testRmses[idxOfBestIndivOfEachGen]
fitnesses = fitnesses[idxOfBestIndivOfEachGen]
generation_index = generation_index[idxOfBestIndivOfEachGen]







# --------- PLOT OF VALIDATION RMSE VS LEARNING RATE ---------


scatter = plt.scatter(lrs, valRmses, c = generation_index, cmap=cmap)

# Create a discrete colorbar using the 'tab10' colormap
colorbar = plt.colorbar(scatter, ticks=np.arange(30))
colorbar.set_label('Generation')

# Plot a single red 'X' at the baseline model score coordinates
# the * operator unpacks the tuple into separate arguments
baselineModelRmse = (1e-3, 1.68e-2)
plt.scatter(*baselineModelRmse, color='red', marker='x', s=100, label='PDEBench paper score')

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Optimizee learning rate')
plt.ylabel('Val RMSE', fontsize=8)
plt.title('Val RMSE')
plt.legend()
plt.savefig('plots/BestFitnessPerGen_32GPUs_valRMSE.png')
plt.close()




# --------- PLOT OF VALIDATION normalized RMSE VS LEARNING RATE ---------

scatter = plt.scatter(lrs, val_nRmses, c = generation_index, cmap=cmap)

# Create a discrete colorbar using the 'tab10' colormap
colorbar = plt.colorbar(scatter, ticks=np.arange(30))
colorbar.set_label('Generation')

# Plot a single red 'X' at the baseline model score coordinates
# the * operator unpacks the tuple into separate arguments
baselineModel_nRmse = (1e-3, 2.95e-2)
plt.scatter(*baselineModel_nRmse, color='red', marker='x', s=100, label='PDEBench paper score')

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Optimizee learning rate')
plt.ylabel('Val nRMSE', fontsize=8)
plt.title('Val nRMSE')
plt.legend()
plt.savefig('plots/BestFitnessPerGen_32GPUs_nvalRMS.png')
plt.close()




# --------- PLOT OF VALIDATION conserved RMSE VS LEARNING RATE ---------

scatter = plt.scatter(lrs, val_cRmses, c = generation_index, cmap=cmap)

# Create a discrete colorbar using the 'tab10' colormap
colorbar = plt.colorbar(scatter, ticks=np.arange(30))
colorbar.set_label('Generation')

# Plot a single red 'X' at the baseline model score coordinates
# the * operator unpacks the tuple into separate arguments
baselineModel_cRmse = (1e-3, 1.06e-2)
plt.scatter(*baselineModel_cRmse, color='red', marker='x', s=100, label='PDEBench paper score')

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Optimizee learning rate')
plt.ylabel('Val cRMSE', fontsize=8)
plt.title('Val cRMSE')
plt.legend()
plt.savefig('plots/BestFitnessPerGen_32GPUs_val_cRMSE.png')
plt.close()



# --------- PLOT OF VALIDATION boundary RMSE VS LEARNING RATE ---------

scatter = plt.scatter(lrs, val_bRmses, c = generation_index, cmap=cmap)

# Create a discrete colorbar using the 'tab10' colormap
colorbar = plt.colorbar(scatter, ticks=np.arange(30))
colorbar.set_label('Generation')

# Plot a single red 'X' at the baseline model score coordinates
# the * operator unpacks the tuple into separate arguments
baselineModel_bRmse = (1e-3, 3.04e-2)
plt.scatter(*baselineModel_bRmse, color='red', marker='x', s=100, label='PDEBench paper score')

plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Optimizee learning rate')
plt.ylabel('Val bRMSE', fontsize=8)
plt.title('Val bRMSE')
plt.legend()
plt.savefig('plots/BestFitnessPerGen_32GPUs_val_bRMSE.png')
plt.close()



# --------- PLOT OF FITNESS VS LEARNING RATE ---------

scatter = plt.scatter(lrs, fitnesses, c = generation_index, cmap=cmap)

# Create a discrete colorbar using the 'tab10' colormap
colorbar = plt.colorbar(scatter, ticks=np.arange(30))
colorbar.set_label('Generation')


plt.xscale('linear')
plt.yscale('log')
plt.xlabel('Optimizee learning rate')
plt.ylabel('Best Fitness of each generation')
plt.title('Fitness is average of RMSE, nRMSE, cRMSE and bRMSE with 500Epochs', fontsize=10)

plt.tight_layout()
plt.savefig('plots/BestFitnessPerGen_32GPUs_Fitness_is_avg_of_RMSE_nRMSE_cRMSE_bRMSE.png')
plt.close()



# --------- PLOT OF AVG FITNESS FOR EACH GENERATION VS GENERATION ---------

scatter = plt.scatter(np.arange(number_of_gens), avg_fitness_of_each_gen)

plt.xlabel('Generation')
plt.ylabel('Avg fitness')
plt.title('Fitness is average of RMSE, nRMSE, cRMSE and bRMSE with 500Epochs', fontsize=10)

plt.tight_layout()
plt.savefig('plots/18h_run_32GPUs_AvgFitnessForEachGeneration.png')
plt.show()
