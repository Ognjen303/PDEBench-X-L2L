from l2l.utils.experiment import Experiment
from l2l.optimizees.cifar.optimizee import CIFAROptimizeeParameters, CIFAROptimizee
from l2l.optimizers.crossentropy import CrossEntropyOptimizer, CrossEntropyParameters
from l2l.optimizers.crossentropy.distribution import NoisyGaussian

import numpy as np


def run_experiment():
    name = 'L2L-CIFAR-CE'
    experiment = Experiment(root_dir_path='../results')
    
    jube_params = {"exec": "srun -n 1 -c 8 --gpus=1 --exact python"}
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          log_stdout=True,
                                                          jube_parameter=jube_params)
    optimizee_seed = 200
    
    # Here you pass the parameters that you wish to optimize
    optimizee_parameters = CIFAROptimizeeParameters(lr=1e-5, momentum=0.3, dampening=0.1, weight_decay=1e-3)
    ## Innerloop simulator
    optimizee = CIFAROptimizee(traj, optimizee_parameters)
    
    ## Outerloop optimizer initialization
    optimizer_parameters = CrossEntropyParameters(pop_size=5, rho=0.9, 
                                        smoothing=0.0, temp_decay=0, 
                                        n_iteration=2,
                                        distribution=NoisyGaussian(noise_magnitude=1., noise_decay=0.99),
                                        stop_criterion=np.inf, seed=102)
    
    optimizer = CrossEntropyOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
        optimizee_fitness_weights=(1.,), # 1. means you are doing maximization and -1. for minimizaiton 
        parameters=optimizer_parameters,
        optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    run_experiment()


if __name__ == '__main__':
    main()
