##########################################

ADDED FILES AND DIRECTORIES BY OGI

##########################################



->run_training
  
  In this folder you can find the .sh files which you can run using the sbatch command in commandline
  It contains all the scripts necessary to run different experiments.
  
  TODO: Figure out how to clean up this folder, there is a lot of redundant code.


->errors
  
  Contains all the error files. An error file should be empty if the code ran correctly.
  
  You get an error file when you run a .sh file inside run_training.  
  
  
->outputs

  Contains all the outputs files. The output files are the results of the experiments.
  
  You get an output file when you run a .sh file inside run_training.


->pretrained_models
  |_Advection
  |_Burgers
  etc...
  |_README.txt
  
  
  pretrained_models contains the pretrained models which can be found at 
  
  https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2987
  
  Look in the README.txt inside pretrained_models for more instuctions.
  

->newly_trained_models
  
  Here I manually (by copy-pasting) placed all newly trained models. Once training is finished you can find the
  newly trained model inside e.g. ../../data/1D/Advection/Train/
  
  The model should have a .pt and .pickle file





You can find on the PDEBench github site the original directory structure in 'Directory Tour' at the bottom of the repo webpage.