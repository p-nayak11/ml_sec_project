# Environment Setup

1. Download the most recent version have Anaconda or Miniconda.
2. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Use the command: `conda env create -f env_setup.yml`
    - This should create an environment named `data_viz_env`. 
3. Activate the conda environment:
    - Windows command: `activate data_viz_env` 
    - MacOS / Linux command: `conda activate data_viz_env`

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)