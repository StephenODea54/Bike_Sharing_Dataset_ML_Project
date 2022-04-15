# Bike_Sharing_Dataset_ML_Project
## Overview
This repository holds all of the files used for Eastern University's DTSC 691 - Data Science Capstone: Applied Data Science. The goal of this project is to build a supervised regression model to predict the number of individuals that will rent a bike at any given time, on any given day. This dataset was made publicly available by the company Capital-Bikeshare in Washington D.C. Since making the data public, Fanaee-T, Hadi, and Joao Gama have added weather and seasonal information to the dataset.

The main focus of model construction was not soley based on predictive performance. Rather, the goal was to build an ML model that was practical and robust. If a less complex modeled performed similarly, then that model would be chosen since less flexible models exhibit higher levels of interpretibility. Ultimately, a Random Forest model was chosen due to its performance on unseen data and its reliability in terms of stable predictions. A Gradient Boosted Tree actually outperformed the Random Forest in terms of predictiveness but there were concerns of overfitting.

An ML user interface was built in Streamlit but due to the large size of the model's .pkl file I am unable to upload it onto GitHub. The original plan was to host the application on Heroku. Future work includes reducing the size of the .pkl file.

## Summary
The structure of this repo is as follows:
  - Data: Contains the 'Hour.csv' file taken directly from the UCI Machine Learning Repository at [Bike-Sharing Dataset](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
  - Figures: Visualziations created during the EDA portion of the model construction. Includes the distribution of all variables as well as different bivariate graphs such as scatterplots and boxplots. Also includes model diagnostics such as Q-Q Plots and Feature Importance plots.
  - Sandbox.ipynb: Contains the process-flow of the project. The notebook also contains notes as to help explain the rationale and methodology of all model justifications.
  - UI: Contains the code used to build the Streamlit app. No .pkl is available due to file size but running model_building.py will create the file. App is fully functional otherwise.

## Model Comparisons
Below is a comparison of how each model performed on the training and testing set. Note that a Randomized Search was used in an effort to keep computation times down. Also, the interactions you see specified for the regression models are between the variables "hr + workingday" and "hum + weathersit". For more information, consult Sandbox.ipynb or check out "Hr_Holiday_Interaction.jpg" and "Humidity_Weathersit_Interaction.jpg" in the Figures folder.

It is important to note that the training RMSEs for all non-parametric models were constructed using 5-Fold CV due to overfitting concerns whereas the training RMSEs for the parametric models were found using the entire training set.

| Model | Train RMSE | Test RMSE | Percentage Increase |
| --- | :---: | :---: | :---: |
| Linear Regression w/ Log-Transformation & Interactions | 88.5667 | 91.5580 | 3.38% |
| Poisson Regression w/ Log-Link & Interactions | 81.6309 | 82.8643 | 1.51% |
| Decision Tree w/ Default Hyperparameters | 95.5451 | 92.6805 | -3.00 % |
| Random Forest w/ Randomized Hyperparameter Tuning | 69.2991 | 64.7621 | -6.55% |
| Gradient Boosted Tree w/ Randomized Hyperparameter Tuning | 39.2181 | 60.8703 | 55.21% |
