Project Structure

This project is organized as follows:

Dataset: Contains the dataset files
DeepLearning: Includes the code for the deep learning model used in the project.
Utils: Includes modules written for each task.
tests: Includes one test file.
**Notebooks: contains the main responses:
Three Jupyter notebooks for result analysis, each accompanied by documented answers and references alongside the code.
ArrhythmiaClassificationDocumentation.ipynb: This notebook was run on Google Colab to achieve faster training. The modules were loaded from Google Drive. If you want to run it locally, replace the importing path with the following code:

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from Utils.DataUtils.DataImputation import impute_missing_values
from Utils.DataUtils.DataLoader import ECGDataLoader
import Utils.DataUtils.DataProcessing as data_processing
import Utils.DataUtils.DataAnalysing as data_analysing
import DeepLearning.TransformerBasedModel as transformer_based_model
from Utils.TrainingUtils import train_model, train_model_with_hyperparameter_tuning
from Utils.EvaluationUtils import show_learning_curves, create_confusion_matrix, metrics_calculation
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Result:

After applying different preprocessing steps, and methods to train efficiently, the model achieved 97% accuracy on the testing set.