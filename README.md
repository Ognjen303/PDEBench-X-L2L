
Please look at ```Project report.pdf``` for details about the project. 
The repo consists of two parts. PDEBench (~1st month work) and PDEBench + L2L ( ~2nd month work).
# PDEBench
To setup the environment run
```
bash create_env.sh
```
## pdebench/models
Below is a description of each folder added by me inside of ```pdebench/models```. You can find [here](https://github.com/pdebench/PDEBench#directory-tour) the original present directories.

### run_training

Here you can find the .sh files which you can run using the sbatch command in commandline It contains all the scripts necessary to run different experiments. TODO: Figure out how to clean up this folder, there is a lot of redundant code.

### errors

Contains all the error files. An error file should be empty if the code ran correctly. You get an error file when you run a .sh file inside run_training.

### outputs

Contains all the outputs files. The output files are the results of the experiments. You get an output file when you run a .sh file inside run_training.

### pretrained_models

Contains the pretrained models which can be found [here](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987). Please look at the README inside of ```./pretrained_models/``` for more instructions on how to run them.

### newly_trained_models

Here I manually (by copy-pasting) placed all newly trained NNet models. Once training is finished you can find the newly trained model inside of e.g. ```../../data/1D/Advection/Train/```. The NN model should have a .pt and .pickle file.



# PDEBench + Learning 2 Learn
  
## Installation
Clone the repository. Then in command line run

```
cd pdebench-daad
bash create_couple_env.sh
```
Next we download a dataset. For example, lets download the Advection dataset. For more datasets please follow the instructions given [here](https://github.com/pdebench/PDEBench#data-download)
```
# Activate the environment
. ../couple_env/bin/activate
# Download the Advection dataset
cd pdebench/data_download/
python download_direct.py --root_folder $proj_home/data --pde_name advection
```
Once the download finished, please check if the downloaded dataset(e.g. a file called 1D_Advection_Sols_beta4.0.hdf5) is inside a path like:

```/pdebench-daad/pdebench/data/1D/Advection/Train/```

If not, then please put 1D_Advection_Sols_beta4.0.hdf5 in such a path. This is also the path where later the trained model weights will be saved to.

To run training go to ```pdebench-daad/L2L/bin```  and run
 ```
sbatch launch_l2l_pdebench_Unet_es.sh
```
The errors are saved in ```pdebench-daad/L2L/bin/errors``` and outputs in ```pdebench-daad/L2L/bin/outputs```

To plot the results with only one hyperparameter training go into ```data_processing.py```. Here we need to change the ```folderPath```  valiable. To do this, open the generated outputs file in ```pdebench-daad/L2L/bin/outputs``` and somewhere from line 40 onward you should copy a path that ends with ```/simulation/work``` and set that as folderPath.


# Authors and acknowledgment

Many thanks to my mentor, Marcel Aach, without whom this project wouldn't have happened.

# Project status

In development...
