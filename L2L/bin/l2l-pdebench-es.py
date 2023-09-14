from l2l.utils.experiment import Experiment
from l2l.optimizees.pdebench.optimizee import PDEBENCHOptimizeeParameters, PDEBENCHOptimizee
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from l2l.optimizers.crossentropy.distribution import NoisyGaussian

import os
import numpy as np
import time

# The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv
import argparse


def parsIni():  
    
    parser = argparse.ArgumentParser(description='L2L PDEBench ES Params')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='Learning rate')
    # parser.add_argument('--scheduler-step', type=int, metavar='SS', help='Period of learning rate decay.')
    # parser.add_argument('--scheduler-gamma', type=float, metavar='SG', help='Multiplicative factor of learning rate decay. Default: 0.1.')
    parser.add_argument('--config', type=str, metavar='CFG', help='Config file of a NNet in PDEBench.')
    parser.add_argument('--pop-size', type=int, default=15, metavar='PSIZE', help='Population size is the number of spawned individuals inside the inner loop optimizee.')
    parser.add_argument('--n-iteration', type=int, default=10, metavar='NITER', help='Number of generations that the outer loop runs for.')
    parser.add_argument('--epochs', type=int, default=500, metavar='E', help='Number of epoch in the NN training of the inner loop.')
    parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size of the NN in the inner loop.')
    parser.add_argument('--noise-std', type=float, default=0.01, metavar='NOISE', help='The standard deviation of the Gaussian sampling of next hyperparameter in optimizer.')
    parser.add_argument('--optimizer-lr', type=float, default=0.1, metavar='OPLR', help='The learning rate of the optimizer.')
    return parser


def run_experiment(args):
    
    print(f'args.lr: {args.lr}')
    # print(f'args.scheduler_gamma: {args.scheduler_gamma}')
    print(f'args.config: {args.config}')
    print(f'args.pop_size: {args.pop_size}')
    print(f'args.n_iteration: {args.n_iteration}')
    print(f'args.epochs: {args.epochs}')
    print(f'args.batch_size: {args.batch_size}')
    print(f'args.noise_std: {args.noise_std}')
    print(f'args.optimizer_lr: {args.optimizer_lr}')
    
    name = f"{os.environ['SLURM_JOB_ID']}-100GPU_Testrun_L2LPDEBench-ES_lr_{args.lr:.0e}_psize_{args.pop_size}_gens_{args.n_iteration}_batchsize_{args.batch_size}"
    experiment = Experiment("../results/" )
    
    
    # -n is the 'task count'; your tasks will be laid out on the nodes in the order of the file. 
    # -c is the number of CPUs per task
    # one srun command is ran for one individual
    # CUDA_VISIBLE_DEVICES=$(($index%4))
    
    # THE BELOW NEXT LINE IS THE ONE THAT RUNS THE BEST UP UNTILL NOW!!!
    jube_params = {"exec": "srun -n 1 -c 10 --mem-per-cpu 8000 --gpu-bind=per_task:1 --exact python"}
    
    # Command from 26. Juli
    # jube_params = {"exec": "CUDA_VISIBLE_DEVICES=0 srun -c 1 --gpus-per-task 1 --exclusive --exact python"} # we would like to run this command 4 times on 1 node, but we probably get 4 nodes instead
    
    # Number of CUDA system devices
    print(f'number of system CUDA devices(before jube cmd) is: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          log_stdout=True,
                                                          jube_parameter=jube_params)
    # optimizee_seed = 200
    
    # MARCEL SOLUTION: Here you pass the parameters that you wish to optimize
    optimizee_parameters = PDEBENCHOptimizeeParameters(lr=args.lr, config=args.config, epochs=args.epochs, batch_size=args.batch_size)
    
    # Innerloop simulator
    optimizee = PDEBENCHOptimizee(traj, optimizee_parameters)

    # Outerloop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = EvolutionStrategiesParameters(
        learning_rate=args.optimizer_lr,
        noise_std=args.noise_std,
        mirrored_sampling_enabled=False,
        fitness_shaping_enabled=True,
        pop_size=args.pop_size,
        n_iteration=args.n_iteration,
        stop_criterion=np.Inf,
        seed=optimizer_seed)

    optimizer = EvolutionStrategiesOptimizer(
        traj,
        optimizee_create_individual=optimizee.create_individual,
        optimizee_fitness_weights=(-0.25, -0.25, -0.25, -0.25), # 1. means you are doing maximization and -1. for minimizaiton 
        parameters=optimizer_parameters,
        optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)
    # End experiment
    experiment.end_experiment(optimizer)


def main():
    
    parser = parsIni()
    
    # Convert argument strings to objects and assign them as attributes of the namespace.
    args = parser.parse_args()
    run_experiment(args)
    
    # Pick the best individual with the lowest Validation RMSE(i.e. best fitness)
    

if __name__ == '__main__':
    main()