# machine_learning

This file outlines the order of the processes to be performed to train a NN.


**1. Data collection from the database.**      
    *file: collect_data.py    
    *input: flown trajectories and flights' id information    
    *output: pickled DataFrame containing selected trajectories + id info + entry conditions
    
**2. Data processing and additional information.**      
    *file: preprocess_data.py     
    *input: pickled DataFrame from the previous step       
    *output: pickled DataFrame containing features and labels for a NN   

**3. Data visualization for outliers inspection.**    
    *file: visualize_data.py          
    *input: pickled DataFrame from the previous step          
    *output: plot illustrating the AOR boundaries, first observations and entry points; in case of outliers, update of step 2

**4. Data selection and scaling.**       
    *file: prep_data_for_nn.py        
    *input: pickled DataFrame from step 3 (including outliers removal based on step 4)       
    *output: pickled DataFrames containing selected and scaled features and labels; split on train, validation and test subsets.

**5. Neural Network training.**      
    *file: machine_learning.py     
    *input: pickled DataFrames from the previous step.    

*rand_forest.py* performs prediction using Random Forest and provides features importance. 
