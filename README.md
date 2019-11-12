# Predict Zillow Logerror

***Purpose:*** 

Find features that drive logerror (Zestimate) of properties to minimize prediction errors.

What features do you think will influcence Zestimate?
Hypothesis: Factors such as location (latitude, and longititude), value, squarefootage will influence how a property gets it's Zestimate.

***Process:***
  - Planning : Predict drivers of logerror, what type of variables to research, which models will make the best predictions?
  - Acquire Zillow dataset through sql query
  - Clean dataset: Qualify variables, fill missing values
  - Explore: Create visuals of variables with clustering and distributions, and run statistical analysis to find significance in variable subgroups.
  - Model: Create models to predict logerror. Evaluate the models with root mean square error to minimize the prediction errors.
  
***Files to use:***
  - env.py - To connect to sql database you will need a username, password, and host.
  - zillow.csv - If you don't have a connection to sql, we have provided the csv file that contains the queried result from sql.
  - acquire.py - functions to read sql and csv
  - prepare.py - functions to prep the dataframe (cleaning, fill nulls, etc.)
  - model.py - function to support exploring (cluster_exam), and run dataframes through models.
  

