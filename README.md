# Telco costumer churn - MLOps pipeline

This project demonstrates the complete lifecycle of a machine learning model for predicting
customer churn. The project includes data versioning with DVC (Data Version Control)
to ensure that datasets and their different versions are tracked. Training of 
different machine learning models for the estimation of the customers churn 
probabilities. The machine learning application will be containerized using Docker,
A CI/CD pipeline based on GitHub Actions will be implemented to automatically run tests.
Finally, the trained churn prediction model will be exposed through a REST API, allowing external applications or services to send customer data and receive churn predictions programmatically.

## Dataset
For the analysis performed in this project the Telco Customer Churn dataset (Kaggle, by BlastChar)
was used. The dataset contains more than 7000 customers and the target is a binary variable (Yes/No).

## Programming language and libraries
All the analysis have been performed using Python and jupyter notebooks. All the libraries needed
to run the python files and notebooks are stored in the 'requirements.txt' file.
They can be installed in a virtual environment with "pip install -r requirements.txt"

## Workflow
At first the 'telco_churn.csv' file was downloaded and stored in \data\raw\telco_churn.csv.
At this point an Exploratory Data Analysis (EDA) was performed with \notebooks\EDA.ipynb to 
look at our data and decide which procedures must be used to clean and prepare the data.
After that with src\data_prep.py we perform the data cleaning, by running this file
the train and test processed datasets will be stored in the \data\processed folder
and the preprocessor trained on the training dataset will be stored in the \models folder.
The train and test datasets will be used in the \src\train.py to train the machine learning
models. Each model will be stored in the \models folder and the metrics and features
coefficient/importance will be stored using the json format in the metrics.json file.
