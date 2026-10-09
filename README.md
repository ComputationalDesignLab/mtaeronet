# Transonic wing aerodynamic analysis and uncertainty quantification using multi-task deep learning

This repository provides an open-source implementation of the multi-task surrogate models and hyperparameter optimization methods introduced in the work titled "**Transonic wing aerodynamic analysis and uncertainty quantification using multi-task deep learning**". This can be used to reproduce the results presented in the paper and use the methods for further research and development.

# Abstract

## Multi-task Aerodynamics Net (MTAeroNet)

This work introduces the Multi-task Aerodynamics Net (MTAeroNet) surrogate modeling architecture, which has a main network to predict integrated aerodynamic quantities and multiple sub-networks to predict various surface quantities for a three-dimensional body in transonic flow. A graphical overview of the modeling architecture is shown in the diagram below. In this implementation and work, each sub-network and the main network of the modeling architecture are MLP networks. This is done to promote simplicity within the modeling architecture while still maintaining prediction accuracy. 


## Multiobjective Hyperparameter Optimization

This work also introduces a multiobjective optimization problem formulation for selecting the optimal hyperparameters for multi-task aerodynamic surrogate models. There are two objectives in the problem formulation. One objective is the mean of the normalized root mean square errors (NRMSE) in the integrated coefficients, and the other objective is the mean NRMSE in the surface quantities across the surface of the three-dimensional body in transonic flow. An open-source implementation of this formulation and its solution procedure is provided, which uses Optuna and Bayesian optimization with Tree-structured Parzen estimators to solve the optimization problem. Additionally, single objective optimization problem formulations and an implementation of their solution through Optuna is also provided for models that are not multi-task in nature but are also investigated in the work.  

# Installation and setup

The best way to run the implementation is through a conda environment. Miniconda is the recommended way to create a new conda environment with a Python installation that has a version of at least 3.12. Follow the steps below to create the new conda environment and use this implementation. 

1. Create a new conda environment.

2. Install the requirements for the environment.
   
3. Download the latest release of the code as a zip file and include it in your working directory.

The relevant code and implementations for the methods are located in the src folder in the repo.  

# Common Research Model wing dataset

# Running the code

# Citation 

