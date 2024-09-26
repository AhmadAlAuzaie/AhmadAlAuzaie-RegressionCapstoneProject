from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import yaml
from data_preparation import load_and_prepare_data

# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def training_model():
    x_train, x_test, y_train, y_test = load_and_prepare_data()

    # Initialize model
    model = RandomForestClassifier(n_estimators = config['model']['hyperparameters']['n_estimators'],
                                   max_depth = config['model']['hyperparameters']['max_depth'],
                                   random_state = config['model']['hyperparameters']['random_state'])
    
    # Train the model
    model.fit(x_train, y_train)

    # Predict on test data
    y_pred = model.predict(x_test)

    # Evaluate the model
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, 'student_dropout_model.pkl')

if __name__ == '__main__':
    training_model()
