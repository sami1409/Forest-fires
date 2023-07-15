Description
The code performs linear regression on a dataset containing information about forest fires. It imports necessary libraries and modules, reads the dataset, performs data preprocessing and scaling, trains a linear regression model, and makes predictions.

Dependencies
The following libraries are required to run the code:

numpy
pandas
matplotlib
seaborn
scikit-learn
Make sure these libraries are installed in your Python environment before running the code.

Usage
Ensure that the 'forestfires.csv' dataset file is located in the same directory as the code file.
Run the code in a Python environment.
Code Explanation
The code starts by importing the necessary libraries: numpy, pandas, matplotlib, seaborn, and scikit-learn.
The 'forestfires.csv' dataset is read using pandas and stored in the variable 'df'.
Exploratory data analysis (EDA) is performed on the dataset to gain insights and understand the data distribution. Several visualizations using seaborn are created to display the histograms of numerical columns.
Data preprocessing is done by removing rows where the 'FFMC' value is less than 80 or the 'ISI' value is greater than 25. The filtered dataset is stored back in the 'df' variable.
Further visualizations are created after the data preprocessing step to observe the changes in the data distribution.
Categorical columns 'month' and 'day' are mapped to numerical values for compatibility with the linear regression model.
The dataset is split into training and testing sets using a test size of 0.2 and a random seed of 101.
Standard scaling is applied to the numerical features of the training and testing sets using the StandardScaler from scikit-learn.
A linear regression model is created and trained on the scaled training data.
Predictions are made on the training set, and the root mean squared error (RMSE) is calculated and printed.
The actual and predicted values for the first 10 instances from the training set are printed.
Predictions are made on the scaled testing set, and the actual and predicted values for the first 10 instances are printed.
