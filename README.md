# Image in DCM
This code is a research framework for integrating image and tabular data in DCM models. While it is configured to work with the MIT moral machine dataset, it is sufficiently modular to change datasets with little modifications. It was developed to gather experiment results found in the following paper: [Images in Discrete Choice Modeling: Addressing Data Isomorphism in Multi-Modality Inputs]() [1]

## Setup
The code is tested on python 3.10.12 . 
You may create an environment and install requirements with:

```pip install -r requirements.txt```

### Data
The experiments are setup to function with the [MIT moral machine datasets](https://osf.io/3hvt2/) [2]. You will require the full file SharedResponses.csv as well as the survey responses SharedResponsesSurvey.csv. 

To prepare the data, you must only change the raw_path directory in ```conf/data/default_process.yaml```. Then call:

```cd data && python preprocess_data.py ```

### Run code
The code uses hydra to run, and visualization of results solely rely on cloud loggers such as ClearML or Weight and Biases (wandb). Setting up your environment would require having an account with either cloud logger. Then you may also change your own hardware and file path configurations in ```conf/platform```. 

While most experiments were run with overrides using hydra convention, the core combination of trainers, dataloaders and settings are found in the ```conf/exps``` folder. As there are no default exps activated in the main conf.yaml file, you must call it as a new config in the command line.
A typical line to run the code from the main directory would resemble like this: 

```python run.py platform=laptop +exps=debug_mode trainer.epochs=5```

or

```python run.py platform=scitas +exps=imCorr_composedMasking data.controlled_mask_scale=0.8,0.9,1```

for a multirun ( in this example, the platform is using hydra's submitit launcher for SLURM based clusters. ).

NOTE that image based models will not run without access to the tiles used to create the images. Ours is set in ```default_process.yaml``` in the path variable ```data.tiles_folder```. Further details given in 'missing required data' section below.  


## Adding Models

Once you define a new model class - add how to call it in ```model_utils.py```. This is directed by a configuration in trainer yaml files, notably you would define a new yaml with a new value for ```trainer.name```.

If you define a new trainer, make sure it is invoked correctly in ```trainer_utils.py``` using the parameter trainer.type.

If you define a new loss, define it in ```trainer_utils.py```

If you use a new dataset, it may require lots of changes in the ```data/``` folder, but should work with little changes if kept as a pytorch dataloader. 

## Missing Required Data
A crucial part of the data, notably the tiles, are not available here as long as we currently do not have the rights to share them publicly. This may change in the future. 

Please contact authors to discuss best possible arrangements in the meantime if necessary.  

## References

[1] Sifringer B, Alahi A. (2023). *Images in Discrete Choice Modeling: Addressing Data Isomorphism in Multi-Modality Inputs*. arXiv preprint

[2] Awad, E., Dsouza, S., Kim, R., Schulz, J., Henrich, J., Shariff, A., ... & Rahwan, I. (2018). *The moral machine experiment*. Nature, 563(7729), 59-64.