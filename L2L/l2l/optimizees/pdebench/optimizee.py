from collections import namedtuple
import numpy as np
from l2l.optimizees.optimizee import Optimizee


# modules needed from train_models_forward.py
import sys
import os
import yaml
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pdebench', 'models', 'unet')))


# PDEBENCHOptimizeeParameters = namedtuple('PDEBENCHOptimizeeParameters', ['lr', 'scheduler_gamma', 'config', 'epochs', 'batch_size'])
PDEBENCHOptimizeeParameters = namedtuple('PDEBENCHOptimizeeParameters', ['lr', 'config', 'epochs', 'batch_size'])


class PDEBENCHOptimizee(Optimizee):
    """
    This is the base class for the Optimizees, i.e. the inner loop algorithms. Often, these are the implementations that
    interact with the environment. Given a set of parameters, it runs the simulation and returns the fitness achieved
    with those parameters.
    """

    def __init__(self, traj, parameters):
        """
        This is the base class init function. Any implementation must in this class add a parameter add its parameters
        to this trajectory under the parameter group 'individual' which is created here in the base class. It is
        especially necessary to add all explored parameters (i.e. parameters that are returned via create_individual) to
        the trajectory.
        """
        super().__init__(traj)
        
        self.lr = parameters.lr
        #self.scheduler_gamma = parameters.scheduler_gamma
        self.config = parameters.config
        self.epochs = parameters.epochs
        self.batch_size = parameters.batch_size
        print(f'In __init__ of optimizee.py self.lr is {self.lr}')
        #print(f'scheduler_gamma is {self.scheduler_gamma}.')
        print(f'The config file is {self.config}.')
        print(f'Number of epochs for NN training is {self.epochs} and the batch size is {self.batch_size}.')
        
        
        # create_individual can be called because __init__ is complete except for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        traj.individual.f_add_parameter('lr', self.lr)
        #traj.individual.f_add_parameter('scheduler_gamma', self.scheduler_gamma)
    

    def create_individual(self):
        """
        Create one individual i.e. one instance of parameters. This instance must be a dictionary with dot-separated
        parameter names as keys and parameter values as values. This is used by the optimizers via the
        function create_individual() to initialize the individual/parameters. After that, the change in parameters is
        model specific e.g. In simulated annealing, it is perturbed on specific criteria

        :return dict: A dictionary containing the names of the parameters and their values
        """
        
        # Note that lr doesn't need to be initialy 'randomized' as the nn weights 
        # in the mnist example, because over there they do not give an initial value
        # to those nn weights, while we do give an initial value for lr in l2l-pdebench-ga.py
        
        print(f'In create_individual() of optimizee.py the learning rate is: {self.lr}')
        #print(f'In create_individual() of optimizee.py the scheduler gamma is: {self.scheduler_gamma}')
        
        # return dict(lr=self.lr, scheduler_gamma=self.scheduler_gamma)
        return dict(lr=self.lr)
    
    
    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        
        # return individual
        # return {'lr': np.clip(individual['lr'], a_min=1e-3, a_max=9e-3),
                # 'scheduler_gamma': np.clip(individual['scheduler_gamma'], a_min=0.1, a_max=0.9)}
        
        return {'lr': np.clip(individual['lr'], a_min=1e-3, a_max=1e-2)}
    

    def simulate(self, traj):
        """
        This is the primary function that does the simulation for the given parameter given (within :obj:`traj`)

        :param  ~l2l.utils.trajectory.Trajectory traj: The trajectory that contains the parameters and the
            individual that we want to simulate. The individual is accessible using `traj.individual` and parameter e.g.
            param1 is accessible using `traj.param1`

        :return: a :class:`tuple` containing the fitness values of the current run. The :class:`tuple` allows a
            multi-dimensional fitness function.

        """
        
        print("Start simulation inside of simulate() in optimizee.py...")
        
        self.lr = traj.individual.lr
        print(f"The traj lr is: {traj.individual.lr}")
        
        # self.scheduler_gamma = traj.individual.scheduler_gamma
        # print(f'The traj scheduler_gamma is: {traj.individual.scheduler_gamma}')
        
        
        # CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_Adv.yaml ++args.model_name='Unet' ++args.filename='../../data/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.ar_mode=True ++args.pushforward=True ++args.unroll_step=20 ++args.if_training=True ++args.epochs=2
        
        print("I'm inside of simulate() in of optimizee.py.")
        
        
        # TODO: Perhaps figure out a better way than to just 'hardcode' the config_yaml_path
        config_yaml_path = '/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/pdebench/models/config/args/' + self.config
        
        with open(config_yaml_path, 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
        
        print(f'Now opening the config file for the NN:\n{config_yaml_path}')
            
        # This should print 1
        print('Actual number CUDA Devices: ', torch.cuda.device_count())
        
        # Number of CUDA system devices
        print(f'os CUDA visible system device index: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        print('Now training a NN from PDEBench...')
        
        fitness = None
        
        if yaml_data['model_name'] == 'FNO':
            from pdebench.models.fno.train import run_training as run_training_FNO
            print('FNO')
            
            # TODO: run_training_FNO() at the moment doens't return any rmse or fitness. Fix that.
            fitness = run_training_FNO(
                if_training=yaml_data['if_training'],
                continue_training=yaml_data['continue_training'],
                num_workers=yaml_data['num_workers'],
                initial_step=yaml_data['initial_step'],
                t_train=yaml_data['t_train'],
                in_channels=yaml_data['in_channels'],
                out_channels=yaml_data['out_channels'],
                epochs=self.epochs,
                learning_rate=self.lr,
                batch_size=self.batch_size, # yaml_data['batch_size']
                unroll_step=yaml_data['unroll_step'],
                ar_mode=yaml_data['ar_mode'],
                pushforward=yaml_data['pushforward'],
                scheduler_step=yaml_data['scheduler_step'],
                scheduler_gamma=yaml_data['scheduler_gamma'], # self.scheduler_gamma * 100
                model_update=yaml_data['model_update'],
                flnm='../../../../../../pdebench/data/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5',
                single_file=yaml_data['single_file'],
                base_path = None, # TODO: Figure out how to implement this....
                # The issue is that in train_models_forward.py in line 188 there is
                # base_path=cfg.args.data_path, and you would expect that in e.g. args/config_Adv.yaml or config.yaml there is a data_path variable, but there isn't...hence I'm confused 
                reduced_resolution=yaml_data['reduced_resolution'],
                reduced_resolution_t=yaml_data['reduced_resolution_t'],
                reduced_batch=yaml_data['reduced_batch'],
                plot=False,
                channel_plot=0,
                x_min=-1,
                x_max=1,
                y_min=-1,
                y_max=1,
                t_min=0,
                t_max=5,
                model_to_load=yaml_data['model_to_load']
            )
            
            
        elif yaml_data['model_name'] == 'Unet':
            from train import run_training as run_training_Unet
            print('Unet')
            print(f"num_workers in config file is: {yaml_data['num_workers']}")
            
            # Unet returns validation rmse as fitness, which is of type tuple
            # The test rmse is here called rmse
            fitness, rmse = run_training_Unet(
                if_training=yaml_data['if_training'],
                continue_training=yaml_data['continue_training'],
                num_workers=yaml_data['num_workers'],
                initial_step=yaml_data['initial_step'],
                t_train=yaml_data['t_train'],
                in_channels=yaml_data['in_channels'],
                out_channels=yaml_data['out_channels'],
                epochs=self.epochs,
                learning_rate=self.lr,
                batch_size=self.batch_size, # yaml_data['batch_size']
                unroll_step=yaml_data['unroll_step'],
                ar_mode=yaml_data['ar_mode'],
                pushforward=yaml_data['pushforward'],
                scheduler_step=yaml_data['scheduler_step'],
                scheduler_gamma=yaml_data['scheduler_gamma'], # self.scheduler_gamma * 100,
                model_update=yaml_data['model_update'],
                flnm='../../../../../../pdebench/data/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5',
                single_file=yaml_data['single_file'],
                reduced_resolution=yaml_data['reduced_resolution'],
                reduced_resolution_t=yaml_data['reduced_resolution_t'],
                reduced_batch=yaml_data['reduced_batch'],
                
                # TODO: The following variables should be imported from
                # /p/scratch/raise-ctp2/stefanovic1/pdebench-daad/pdebench/models/config/config.yaml
                # figure out how to do that
                plot=False,
                channel_plot=0,
                x_min=-1,
                x_max=1,
                y_min=-1,
                y_max=1,
                t_min=0,
                t_max=5,
                model_to_load=yaml_data['model_to_load']
              )
            
        elif yaml_data['model_name'] == 'PINN':
            from pdebench.models.pinn.train import run_training as run_training_PINN
            
            print("PINN")
            
            # Run training PINN returns current_rmse as fitness
            fitness = run_training_PINN(
                scenario=yaml_data['scenario'],
                epochs=self.epochs, # epochs = yaml_data['epochs']
                learning_rate=self.lr,
                model_update=yaml_data['model_update'],
                flnm='1D_Advection_Sols_beta4.0.hdf5',
                seed=yaml_data['seed'], # TODO: I don't think seed=yaml_data['seed'] 
                input_ch=yaml_data['input_ch'],
                output_ch=yaml_data['output_ch'],
                root_path='/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/pdebench/data/1D/Advection/Train/', # TODO Figure out how to set this via e.g. the startscript
                val_num=yaml_data['val_num'],
                if_periodic_bc=yaml_data['if_periodic_bc'],
                aux_params=yaml_data['aux_params'],
                save_path='/p/scratch/raise-ctp2/stefanovic1/pdebench-daad/pdebench/models/newly_trained_models/Advection/PINN' # TODO Figure out how to set this via e.g. the startscript
            )
        
        assert fitness is not None
        
        print(f'The fitness is: {fitness}')
        print(f'The test rmse is: {rmse}')
        
        # replacing NaN values with 10000
        x = np.isnan(fitness)
        fitness[x] = np.array(10000, dtype=np.float32)
        rmse[x] = np.array(10000, dtype=np.float32)
        
        print('End of NN training.')
        
        # if isnan(fitness):
        #     fitness = 10000
        #     rmse = 10000
        # elif fitness is None:
        #     raise ValueError('fitness cannot be None.')
        
        # command = "CUDA_VISIBLE_DEVICES='0' python3 -u ../train_models_forward.py +args=config_Adv.yaml ++args.model_name='Unet' ++args.filename='../../data/1D/Advection/Train/1D_Advection_Sols_beta4.0.hdf5' ++args.ar_mode=True ++args.pushforward=True ++args.unroll_step=20 ++args.if_training=True ++args.epochs=2 ++args.learning_rate=self.lr"
               
        
        
        print(f'The fitness type is: {type(fitness)}. The fitness has the following elements: ')
        print(f'RMSE: {fitness[0]:.2e}, nRMSE: {fitness[1]:.2e}, cRMSE: {fitness[2]:.2e}, bRMSE: {fitness[3]:.2e}')
        print(f'The learning rate was: {self.lr}')
        print(f'The test RMSE is: {rmse}')
        
        # The returned values are, in order:
        # RMSE, normalized RMSE, conserved RMSE, boundary RMSE
        return [fitness[0], fitness[1], fitness[2], fitness[3]]
