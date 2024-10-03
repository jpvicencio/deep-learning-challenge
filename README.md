# deep-learning-challenge
Purpose
The goal of this project is to create a robust predictive model that can assist Alphabet Soup in selecting applicants for funding based on historical data. Using machine learning and neural networks, various features can be analyzed to to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. Understanding which organizations are most likely to succeed will enable maximizing the impact of the foundation's funding initiatives.

Objectives

Data Preprocessing: Clean and prepare the dataset for analysis by removing irrelevant columns, encoding categorical variables, and scaling numerical features.

Model Development: Create a deep learning model to predict the success of funding applications using TensorFlow and Keras.

Model Optimization: Enhance the model's performance through iterative adjustments and fine-tuning, aiming for an accuracy greater than 75%.

Process

Data Processing - ETL
1. Load and preprocess the charity_data.csv dataset.
2. Identify target and feature variables.
3. Handle categorical variables and scale numerical features.

Machine Learning Models
1. Initial Model: Develop a baseline neural network to classify application success.
2. Optimization: Experiment with model architecture, including the number of layers, activation functions, and training epochs to improve accuracy.

Evaluation 
1. Assess model performance using accuracy and loss metrics.
2. Save the final model weights for future use.

Results: 
Data Preprocessing

What variable(s) are the target(s) for your model? The variable which served as target for the model is 'IS_SUCCESSFUL' since we're looking for applications with likelihood of success if funded by Alphabet Soup.

What variable(s) are the features for your model? The variables for the model are APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT   

What variable(s) should be removed from the input data because they are neither targets nor features? EIN and NAME are the variables dropped from input data because they are identifiers which do not provide predictive information for the model.

Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why? Model 3 has the following details:
Total Number of Layers: 3 (2 hidden layers + 1 output layer)
Total Number of Neurons: 
        First Hidden Layer: 128 neurons, 
        Second Hidden Layer: 64 neurons
        Output Layer: 1 neuron
Total: 128 + 64 + 1 = 193 neurons
Activation Functions:
        ReLU: Used in both hidden layers (2 times).
        Sigmoid: Used in the output layer (1 time).

Were you able to achieve the target model performance? No. Model #3 has an accuracy: 0.7278 and loss: 0.5500.

What steps did you take in your attempts to increase model performance?
To increase model performance, I increased neurons, add dropout for regularization, increased epochs to the training regimen and added complexity, early dropping and early stopping.

Summary

The deep learning model developed for predicting the success of funding applications for Alphabet Soup has produced a performance accuracy of approximately 72.78% with a loss of 0.5500. The target variable for this classification task was 'IS_SUCCESSFUL', while the features included relevant attributes such as APPLICATION_TYPE, AFFILIATION, and INCOME_AMT. Variables like EIN and NAME were removed as they did not contribute predictive value.
The model architecture comprised three layers: two hidden layers with 128 and 64 neurons, respectively, and an output layer with a single neuron utilizing the sigmoid activation function for binary classification. Despite efforts to optimize the model through increasing the number of neurons, applying dropout for regularization, and adjusting training epochs, we did not reach the target accuracy of 75%.

Recommendation for a Different Model

To potentially improve performance on this classification problem, features can be further reassessed and reduced.

Feature Reduction Strategies:
    Feature Importance Analysis: Utilize feature importance scores from models to identify and retain the most impactful features.
    Correlation Analysis: Examine correlation matrices to remove highly correlated features that do not contribute additional predictive power.
    Dimensionality Reduction Techniques: Apply methods like PCA (Principal Component Analysis) to condense the feature set while retaining variance.
