Installation Guide
- Python version 3.10.10 is used for the experiments
- Install the required libraries from the requirements.txt
    - use: pip install -r path/to/requirements.txt


NOTE: For clarity, notebooks are inside folders for each experiments. When running experiments notebooks, one might need to move the notebook to the main directory. This is due to dependencies on the modules. 


Experiments:
1. Main experiments
    - there are 5 datasets (synthetic, google, Euro Stoxx, power, temperature) 
    - there are 3 seeds (seeds 100, 300, 500)
    - for every dataset for every seed a notebook is created which outputs results
    - the results are stored in results folders within the main_results folder and can be inspected using results.ipynb

2. Ablation experiments
    - ACI grid search
        - The file ACI_grid_search.ipynb is the file in which grid search is performed
        - Using aci_optimized_comparison.ipynb, the parameters found in the previous grid search can be evaluated on the test set (random seed 300)
        - These results are exported and can be inspected using results_aci_grid_search.ipynb
    - Quantile tunning
        - Experiments on synthetic data can be found in quantile_tuning-SYNTHETIC.ipynb
        - Experiments on google data can be found in quantile_tuning-SYNTHETIC.ipynb
    - Multi-step forecasting
        - Experiments and results can be found in multi-step-forecasting.ipynb

3. Data 
    - Real-life data can be found within the data folder
    - Synthetic data is found in the synthetic_data folder

