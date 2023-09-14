from l2l.utils.experiment import Experiment
from l2l.optimizees.cifar.optimizee import CIFAROptimizeeParameters, CIFAROptimizee
from l2l.optimizers.evolutionstrategies import EvolutionStrategiesParameters, EvolutionStrategiesOptimizer
from l2l.optimizers.crossentropy.distribution import NoisyGaussian

import os
import numpy as np
import time
import argparse


def parsIni():
    parser = argparse.ArgumentParser(description='L2L Cifar ES Params')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.999, metavar='M',
                    help='momentum in SGD')
    parser.add_argument('--dampening', type=float, default=0.02, metavar='D',
                    help='dampening in SGD')
    parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='W',
                    help='weight decay in SGD')
    parser.add_argument('--pop-size', type=int, default=15, metavar='P',
                    help='population size in EvolutionStrategiesParameters')
    parser.add_argument('--denominator', type=int, default=4, metavar='De',
                    help='denominator of the modulo operator in srun')
    return parser


def run_experiment(args):
    
    print(f'args.lr: {args.lr}')
    print(f'args.momentum: {args.momentum}')
    print(f'args.dampening: {args.dampening}')
    print(f'args.weight_decay: {args.weight_decay}')
    print(f'args.pop_size: {args.pop_size}')
    print(f'args.denominator: {args.denominator}')
    
    name = f'L2L-CIFAR-ES_denom_{args.denominator}_lr_{args.lr:.0e}_mom_{args.momentum:.0e}_damp_{args.dampening:.0e}_wd_{args.weight_decay:.0e}_pop_{args.pop_size}'
    experiment = Experiment("../results/")
    
    
    # -n is the 'task count'; your tasks will be laid out on the nodes in the order of the file. 
    # -c is the number of CPUs per task
    # one srun command is ran for one individual
    # CUDA_VISIBLE_DEVICES=$(($index%4))
    
    jube_params = {"exec": "srun -n 1 -c 32 --gpu-bind=per_task:1 --exact python"}
    
    # Last command from 26. Juli
    # jube_params = {"exec": "CUDA_VISIBLE_DEVICES=0 srun -c 1 --gpus-per-task 1 --exclusive --exact python"} # we would like to run this command 4 times on 1 node, but we probably get 4 nodes instead
    
    # MEETING WITH DR. DIAZ 2nd august 2023
    # You HAVE to specify the no. of processes -n (else the default is 1, we think..., depends on slurm version)
    # She usually fixes also the no. of threads per cpu. The new version of slurm from 2nd half of 2022 has the tendency to overload cpus. We don't want that. 
    
    # Q: Can I have multiple sruns per node? A: Yes.
    # --exact is very important because it tells the srun to allocate the exact amount of reources requested.
    
    
    
    # Number of CUDA system devices
    print(f'number of system CUDA devices(before jube cmd) is: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    
    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          log_stdout=True,
                                                          jube_parameter=jube_params)
    optimizee_seed = 200
    
    # MARCEL SOLUTION: Here you pass the parameters that you wish to optimize
    optimizee_parameters = CIFAROptimizeeParameters(lr=args.lr, momentum=args.momentum, dampening=args.dampening, weight_decay=args.weight_decay)
    ## Innerloop simulator
    optimizee = CIFAROptimizee(traj, optimizee_parameters)

    ## Outerloop optimizer initialization
    optimizer_seed = 1234
    optimizer_parameters = EvolutionStrategiesParameters(
        learning_rate=0.1,
        noise_std=0.1,
        mirrored_sampling_enabled=False,
        fitness_shaping_enabled=True,
        pop_size=args.pop_size,
        n_iteration=2,
        stop_criterion=np.Inf,
        seed=optimizer_seed)

    optimizer = EvolutionStrategiesOptimizer(
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
    
    parser = parsIni()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('--- %s seconds ---' % (time.time() - start_time))
