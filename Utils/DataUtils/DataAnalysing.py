import matplotlib.pyplot as plt
import seaborn as sns

def check_null_values(dataframe):
    # Check for null values in the DataFrame
    if dataframe.isnull().values.any():
        print("Null values found in the CSV file.")
    else:
        print("No null values found in the CSV file.")

def visualize_feature_statistics(features):
    # we compute the statistical properties using the describe() method, which provides the mean, standard deviation, quartiles, and other relevant statistics for each feature.
    # Compute the statistical properties of the features
    stats = features.describe()
    # Transpose the statistics DataFrame for better visualization
    stats = stats.transpose()    
    return stats

def plot_class_distribution(label):   
    label_counts = label.value_counts()   
    # Set Seaborn style
    sns.set(style='whitegrid')
    # Plotting the distribution of classes using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Distribution of Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_features_correlation(features):
    # Calculate the correlation matrix
    #A high correlation between two features means that there is a strong linear relationship between them.
    corr_matrix = features.corr()
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(12, 10))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Correlation Matrix')
    plt.xticks(range(0, len(features.columns), 5), range(0, len(features.columns), 5), rotation=90)
    plt.yticks(range(0, len(features.columns), 5), range(0, len(features.columns), 5))
    plt.show()
    
def plot_density(dataframe, start_index=None, end_index=None, feature_indices=None):
    if feature_indices is not None:
        selected_features = dataframe.iloc[:, feature_indices]
        title = 'Density Plots: Feature Comparison'
    elif start_index is not None and end_index is not None:
        selected_features = dataframe.iloc[:, start_index:end_index]
        title = f'Density Plots: Features {start_index} to {end_index}'
    else:
        raise ValueError("Either 'feature_indices' or 'start_index' and 'end_index' must be provided.")
    
    # Plot density plots for the selected features
    selected_features.plot.density(figsize=(10, 6))
    plt.xlabel('Feature Values')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
def plot_signals(signals):
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Iterate over each signal and plot it
    for i, signal in enumerate(signals):
        ax.plot(signal, label=f"Heartbeat Signal {i+1}")

    ax.set_title('Heartbeat Signals')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()

    # Show the plot
    plt.show()