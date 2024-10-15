# Opganoid Morphology analyser 

the appconsits of two parts: 
1) A Jupyter Notebook to select your Image folder and Gerenate the statistics 
2) A Dash application for the interactive statistical evaluation 

Boths are work in Progress

Download the repository locally and install the requirements files 

Requirements are python3 to be installed

We suggest using a local Visual Studio Code instance with a dedicateded conda environment 

[how to install VSC](https://code.visualstudio.com/download)  
[how to install conda on windows ](https://docs.anaconda.com/anaconda/install/windows/)


in VSC open a Termain alnd move to the folder you would like to intall this setup get the repository 

```{shell}
git pull https://github.com/KJPMolgenLab/Stat_dash.git

# move to the folder 
cd Stat_dash

# create a new conda environemnt with alle the prerequisites installes 

conda env create -f environment.yml

# activate the new environment 

conda activate organoid_clean

```

Now your kernel should be ready for the Jupyter notebook 

within VSC you can now open the [organoid_analysis.ipynb notbook](organoid_analysis.ipynb) and follow the instructions 


