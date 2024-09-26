import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml

# First, let's load the config file to get all our settings.
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def load_and_prepare_data():
    # Load the dataset. We'll assume it's in CSV format.
    data = pd.read_csv(config['data_path'])
    
    # Convert any categorical data into numbers. 
    # For example, 'previous_dropout_reports' and 'behavioral_issues' could be yes/no or similar.
    data['previous_dropout_reports'] = LabelEncoder().fit_transform(data['previous_dropout_reports'])
    data['behavioral_issues'] = LabelEncoder().fit_transform(data['behavioral_issues'])

    # Now we select the features we care about (as defined in config.yaml) and the target column.
    x = data[config['features']]
    y = data[config['target_column']]

    # We'll scale the numerical columns so they're all on the same scale. 
    # This is generally a good idea when working with different ranges of data.
    scaler = StandardScaler()
    x[['average_grade', 'attendance_rate', 'number_of_absences', 'homework_submission_rate',
       'participation_score', 'parental_involvement', 'extracurricular_activities']] = scaler.fit_transform(x[['average_grade', 'attendance_rate', 'number_of_absences', 
                                                                                                               'homework_submission_rate', 'participation_score', 
                                                                                                               'parental_involvement', 'extracurricular_activities']])

    # We'll split the data into training and test sets so we can evaluate how well the model performs on unseen data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = config['train_test_split']['test_size'], random_state = config['train_test_split']['random_state'])
    return x_train, x_test, y_train, y_test
