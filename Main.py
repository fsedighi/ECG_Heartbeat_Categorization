# Importing requred libraries and modules
import numpy as np
from Utils.DataUtils.DataImputation import impute_missing_values
from Utils.DataUtils.DataLoader import ECGDataLoader
import Utils.DataUtils.DataProcessing as data_processing
from Utils.TrainingUtils import train_model
from sklearn.model_selection import train_test_split

"""""
Loading data
"""""

# Create an instance of ECGDataLoader
train_data_loader = ECGDataLoader("Dataset\mitbih_train.csv")
test_data_loader = ECGDataLoader("Dataset\mitbih_test.csv")

# Load the ECG dataset from the CSV file
train_data_loader.load_ecg_dataset_csv()
test_data_loader.load_ecg_dataset_csv()

# Get the dataset
train_data = train_data_loader.get_dataset()
test_data = test_data_loader.get_dataset()

"""""
Data preprocessing
"""""

# Apply imputation to address missing data
train_data = impute_missing_values(train_data, method='forward')
test_data = impute_missing_values(test_data, method='forward')

# # Split the data into train and holdout data
train_data, holdout_data = train_test_split(train_data, test_size=0.1, random_state=42, stratify=train_data.iloc[:, -1])

train_data = data_processing.augment_dataset(train_data, augmentation_factor= 5)
holdout_data = data_processing.add_noise_dataset(holdout_data, level = 0.02)

x_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
x_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
x_holdout, y_holdout = holdout_data.iloc[:, :-1].values, holdout_data.iloc[:, -1].values

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
x_holdout = x_holdout.reshape((x_holdout.shape[0], x_holdout.shape[1], 1))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]


"""""
Model training
"""""

model, history = train_model(x_train, y_train, epochs=200)


"""""
Model evaluation
"""""
evaluation = model.evaluate(x_test, y_test, verbose=1)