# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)

## Table of Content
- [Project Overview](#overview)
- [Project Components](#components)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [ Executing Program](#Executing_Program)
  - [Data Cleaning](#cleaning)
  - [Training Classifier](#training)
  - [Starting the Web App](#starting)
- [Conclusion](#conclusion)
- [Files](#files)
- [Software Requirements](#sw)
- [Credits and Acknowledgements](#credits)

<a id='overview'></a>
## Project Overview

This Project is part of Udacity Data Science Nanodegree Program in collaboration with <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a>. In this project I will apply data engineering to analize disaster data from Figure Eight to build a model for an API to classify disaster messages.. 

The data contains real messages that were sent during disaster events. The objective if to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Once the model is ready, we will be displaying the result in a Flask Web app, such that anyone should be able to input a new message and get classification results in several categories. The web app will also display visualizations of the data.


<a id='components'></a>
## Project Components

1. ETL Pipeline

File ..data/process_data.py, contains the data cleaning pipeline. 

  - Loads the 'messages' and 'categories' datasets.
  - Merges the two datasets.
  - Cleans the data.
  - Stores it in a SQLite database.


2. ML Pipeline

File ..models/train_classifier.py, contains the ML pipeline.

  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file
  
3. Flask Web App

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you

![image](https://user-images.githubusercontent.com/35266145/130718444-dec881e6-d8fc-4a22-9b42-f8e0f8bf20fa.png)


<a id='Executing_Program'></a>
##  Executing Program

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


