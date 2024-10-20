# Disaster Response Pipeline Project

## Summary:
This project is to build a machine learning pipeline that classifies disaster response messages. The dataset contains real messages that were sent during disaster events. These messages are categorized into multiple categories. 
The project processes the data, trains a machine learning model. It also provides a web interface for classifying new messages and visualizing the data.

The project consists of:
- **ETL pipeline:** Extracts, transforms, and loads data into a SQLite database.
- **ML pipeline:** Trains a multi-output classifier to categorize the messages.
- **Web application:** Visualizes the data and allows users to input new messages for classification.

## How to Run the Python Scripts and Web App
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Explanation of Files:

- **app/**: 
    - `run.py`: Flask web application to display and classify messages.
    - **templates/**: HTML templates for rendering the web pages.
  
- **data/**:
    - `disaster_messages.csv`: Dataset of disaster-related messages.
    - `disaster_categories.csv`: Dataset of categories for the messages.
    - `process_data.py`: Script for cleaning and processing the data and saving it into a SQLite database.
    - `DisasterResponse.db`: The SQLite database created after running `process_data.py`.

- **models/**:
    - `train_classifier.py`: Script for training the machine learning model and saving it as a `.pkl` file.
    - `classifier.pkl`: Saved model after training.

## Github
- Link: https://github.com/shinshinIruka/udacity_datascientist_project2.git


## Acknowledgment: 
- The disaster data is from Appen (formerly Figure 8) 
