from collections import namedtuple

import numpy as np

from l2l.optimizees.optimizee import Optimizee

CIFAROptimizeeParameters = namedtuple('CIFAROptimizeeParameters', ['lr', 'momentum', 'dampening', 'weight_decay'])

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from .resnet import ResNet18


class CIFAROptimizee(Optimizee):
    """
    Implements a simple function optimizee. Functions are generated using the FunctionGenerator.
    NOTE: Make sure the optimizee_fitness_weights is set to (-1,) to minimize the value of the function. If it is set to (1,) then maximization is performed.

    :param traj:
        The trajectory used to conduct the optimization.

    :param parameters:
        Instance of :func:`~collections.namedtuple` :class:`.MNISTOptimizeeParameters`

    """

    def __init__(self, traj, parameters):
        super().__init__(traj)

        self.lr = parameters.lr
        self.momentum = parameters.momentum
        self.dampening = parameters.dampening
        self.weight_decay = parameters.weight_decay
        
        # create_individual can be called because __init__ is complete except for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        traj.individual.f_add_parameter('lr', self.lr)
        traj.individual.f_add_parameter('momentum', self.momentum)
        traj.individual.f_add_parameter('dampening', self.dampening)
        traj.individual.f_add_parameter('weight_decay', self.weight_decay)

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.trainset = torchvision.datasets.CIFAR10(
                    root="/p/scratch/raise-ctp2/cifar10/data/", train=True, download=False, transform=transform_train)
        
        self.testset = torchvision.datasets.CIFAR10(
                    root="/p/scratch/raise-ctp2/cifar10/data/", train=False, download=False, transform=transform_test)

#         #self.net = models.resnet18()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.net = ResNet18()
        
#         print("Current device: ", self.device)

#         self.net.to(self.device)
        
        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=256,
            shuffle=True,
            num_workers=10) # num_workers is the number of CPUs that perform data loading
        
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=256,
            shuffle=False,
            num_workers=10)
        
        print("ResNet init done with lr: ", self.lr)
        print("ResNet init done with momentum: ", self.momentum)
        print("ResNet init done with dampening: ", self.dampening)
        print("ResNet init done with weight_decay: ", self.weight_decay)

        return dict(lr=self.lr, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay)

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        
        # return individual
        return {'lr': np.clip(individual['lr'], a_min=1e-8, a_max=1),
                'momentum': np.clip(individual['momentum'], a_min=1e-4, a_max=0.9999999),
                'dampening':np.clip(individual['dampening'], a_min=0, a_max=0.999),
                'weight_decay':np.clip(individual['weight_decay'], a_min=1e-7, a_max=0.01)}
    
    
    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~l2l.utils.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the chosen function
        """

        print("Start simulation...")
        self.lr = traj.individual.lr
        self.momentum = traj.individual.momentum
        self.dampening = traj.individual.dampening
        self.weight_decay = traj.individual.weight_decay
        
        print(f"Current lr: {self.lr:.4f}, momentum: {self.momentum:.4f}")
        print(f"Current dampening: {self.dampening:.4f}, weight_decay: {self.weight_decay:.4f}")
        
        # Number of CUDA system devices
        print(f'os CUDA visible system device index: {os.environ["CUDA_VISIBLE_DEVICES"]}')
        
        # Number of CUDA devices available
        # This should return 1
        print('Actual number CUDA Devices: ', torch.cuda.device_count())
        
        
        # We force the optimizee to run on one GPU
        # self.device = torch.device(f"cuda:{int(os.environ['CUDA_VISIBLE_DEVICES'])}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device('cuda')
        # self.device = torch.device('cpu')
        
        # Define the Neural Network
        self.net = ResNet18()
        
        print("Current device: ", self.device)

        self.net.to(self.device)
        
        
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, 
                                   dampening=self.dampening, weight_decay=self.weight_decay)
        
        running_train_correct = 0
        
        # Training loop
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            running_trian_correct = 0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                pred = outputs.argmax(dim=1, keepdim=True)
                loss.backward()
                self.optimizer.step()

                running_train_correct += pred.eq(labels.view_as(pred)).sum().item()

        current_train_acc = running_train_correct / len(self.trainset)
        print("Current Train Accuracy: ", current_train_acc)
        
        
        total_test_loss = 0
        current_test_correct = 0
        running_test_correct = 0
        # prepare net for testing and loop over test dataset
        self.net.eval()
        
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                pred = outputs.argmax(dim=1, keepdim=True)
                
                total_test_loss = total_test_loss + loss
                running_test_correct += pred.eq(labels.view_as(pred)).sum().item()
        
            current_test_acc = running_test_correct / len(self.testset)
        
        print(f'Current test accuracy: {current_test_acc}')
        
        # We can pass any fitness function we want, but we choose accuracy
        return [current_test_acc]
