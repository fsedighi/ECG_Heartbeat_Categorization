import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from sklearn.metrics import precision_score, recall_score, f1_score


def show_learning_curves(history):
    # Set the seaborn theme
    sns.set_theme(style="whitegrid")

    # Plot the training curve
    plt.figure(figsize=(12, 6))  # Adjust the figure size as per your preference

    # Plotting the accuracy curve
    plt.subplot(1, 2, 1)  # Create a subplot for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting the loss curve
    plt.subplot(1, 2, 2)  # Create a subplot for loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()
    
def create_confusion_matrix(y_test, predictions):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Define class labels (if available)
    class_labels = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

    # Create a table for the confusion matrix
    table = tabulate(cm, headers=class_labels, showindex=class_labels, tablefmt="fancy_grid")

    # Print the table
    print("Confusion Matrix:")
    print(table)
    return cm

def metrics_calculation(y_test, predictions):
    
    # Calculate precision, recall, and F1 score for each class
    class_metrics = {
        'precision': precision_score(y_test, predictions, average=None),
        'recall': recall_score(y_test, predictions, average=None),
        'f1': f1_score(y_test, predictions, average=None)
    }

    # Calculate overall precision, recall, and F1 score
    overall_metrics = {
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted'),
        'f1': f1_score(y_test, predictions, average='weighted')
    }

    return overall_metrics, class_metrics
    

    
    


    
