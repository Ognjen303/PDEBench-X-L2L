from l2l.utils.experiment import Experiment
from l2l.optimizees.cifar.optimizee import CIFAROptimizeeParameters, CIFAROptimizee
from l2l.optimizers.evolution import GeneticAlgorithmOptimizer, GeneticAlgorithmParameters
from l2l.optimizers.crossentropy.distribution import NoisyGaussian

import numpy as np


def run_experiment():
    name = 'L2L-CIFAR-GA'
    experiment = Experiment("../results/")
    jube_params = {"exec": "srun -n 1 -c 8 --gpus=1 --exact python"}
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          log_stdout=True,
                                                          jube_parameter=jube_params)
    optimizee_seed = 200
    
    # MARCEL SOLUTION: Here you pass the parameters that you wish to optimize
    optimizee_parameters = CIFAROptimizeeParameters(lr=1e-5, momentum=0.3, dampening=0.1, weight_decay=1e-3)
    ## Innerloop simulator
    optimizee = CIFAROptimizee(traj, optimizee_parameters)
    
    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = GeneticAlgorithmParameters(
        seed=optimizer_seed, pop_size=5, cx_prob=0.5,
        mut_prob=0.3, n_iteration=3,
        ind_prob=0.02,
        tourn_size=15, mate_par=0.5,
        mut_par=1)
    
    optimizer = GeneticAlgorithmOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                          optimizee_fitness_weights=(1.,),
                                          parameters=optimizer_parameters,
                                          optimizee_bounding_func=optimizee.bounding_func)
    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizee_parameters=optimizee_parameters,
                              optimizer_parameters=optimizer_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    run_experiment()
    

if __name__ == '__main__':
    main()
    