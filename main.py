""" Script to train a Random Forest Classifier model and predict the category of a CSV file """

import os
import chardet
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from collections import Counter
import pickle

#  Import datasets
walkdataset = "data/walk.csv"
cardataset = "data/car.csv"
rundataset = "data/run.csv"
walkdata = pd.read_csv(walkdataset, sep="\t", encoding='utf-16')
cardata = pd.read_csv(cardataset, sep="\t", encoding='utf-16')
rundata = pd.read_csv(rundataset, sep="\t", encoding='utf-16')

#  prepare data
walkdata = walkdata.drop(walkdata.index[0])  # remove row 2 and 3 as they were wrong
cardata = cardata.drop(cardata.index[1])
cardata = cardata.drop(cardata.index[1])  # Remove row 2 and 3 as they were wrong
rundata = rundata.drop(rundata.index[1])

#  Add labels to understand which data is which
walkdata["category"] = "Walking"
cardata["category"] = "Driving"
rundata["category"] = "Running"

#  Merge datasets into one to make it easier for training
data = pd.concat([walkdata, cardata, rundata])
data = data.dropna()  # Remove rows with missing values

#  Feature selection. We chose speed as it seems the most relevant
features = ["Speed (km/h)", "Heading", "Altitude (m)", "Total Distance (km)"]
X = data[features]
y = data["category"]

#  Split data into training and testing. 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SimpleImputer to fill missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

trainornot = input("Do you want to train the model? (yes/no): ")
if trainornot.lower() == "yes":
    # Create a Random Forest Classifier model with 100 trees and a random state of 42
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)  # Train the model using the training sets

    # Save model
    with open('model/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    # Generate a classification report
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Generate accuracy score
    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred))

if trainornot.lower() == "no":
    # Load model
    with open('model/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Get data folder directory
    current_directory = os.getcwd() + "/data"

    # List all CSV files in the current directory
    csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]

    # Print the list of CSV files
    print("\nCSV files in the current directory:")

    filecount = 0
    for csv_file in csv_files:
        filecount = filecount + 1
        print(filecount, "-", csv_file)

    count = 0
    predictions_list = []  # Lists to store results
    accuracy_list = []

    while True:
        var = count + 1
        # Prompt user to choose a CSV file to predict
        file = input("\nChoose a file to predict or 'done' to finish: ")

        # Exit the program if the user types 'done'
        if file.lower() == "done":
            exit()

        if file > str(len(csv_files)):
            print("Invalid file number")
            continue

        if file == '0':
            print("Invalid file number")
            continue

        file = csv_files[int(file) - 1]

        print("\nYou chose:", file)
        print("File size: ", os.path.getsize(current_directory + "/" + file) / 1000, "KB")

        # Detect the encoding of the CSV file
        with open(current_directory + "/" + file, 'rb') as rawdata:
            result = chardet.detect(rawdata.read(10000))  # You can adjust the buffer size as needed

        # Read the CSV file using the detected encoding
        data = pd.read_csv(current_directory + "/" + file, encoding=result['encoding'], sep="\t")

        # Preprocess the data
        data = data.iloc[2:]
        X = data[features]
        X = imputer.transform(X)

        # Run the AI model on the entire file and print the most common prediction
        predictions = model.predict(X)
        prediction_counts = Counter(predictions)
        most_common_prediction = prediction_counts.most_common(1)[0][0]
        print("Prediction:", "\033[92m", most_common_prediction, "\033[0m")
        endprediction = prediction_counts[most_common_prediction] / len(predictions)  # Calculate accuracy
        accuracy = endprediction * 100
        print("Accuracy: ", accuracy, "%")
