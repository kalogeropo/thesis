import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_experiment(df, avgs, prune_pers, current_best, title=''):
    
    # get rid of similarity label 
    temp_columns = df.columns.str.replace(r'-\d\.\d+', '', regex=True)
    
    df = pd.DataFrame({"model": temp_columns, "avg_pre": avgs, "%prune": prune_pers})
    df = df.sort_values(by="avg_pre", ascending=False)
    
    ax = df.plot.bar(x="model", y="avg_pre", rot=60, figsize=(8, 5), title=title)
    ax.set_ylabel('precision')

    ax.plot([current_best for _ in range(df.shape[0])], color='red', linestyle='-', label="Old Best Model")

    ax2 = ax.twinx()
    # plot prune percentage line
    ax2.plot(df["%prune"].values, color='green', linestyle='--', marker='o')
    ax2.set_ylabel('% Prune')

    # Remove the legend
    ax.legend().remove()

    # Display the chart
    plt.show()


def compare_models(df):
    """
        Compare the performance of each query between each pair of models
        in order to find the model that has the biggest diff between all others
    """
    models = df.columns

    # Create a new DataFrame to store the positive counts
    positives = pd.DataFrame(0, index=models, columns=models)

    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                pairwise_diff = df[models[i]] - df[models[j]]  # Subtract pairwise columns
                positive_count = (pairwise_diff > 0).sum()  # Count positive differences
                positives.loc[models[i], models[j]] = positive_count

    return positives


def plot_heatmap(cm):
    # Create a heatmap plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')

    # Set plot title and labels
    plt.title('Number of Queries Where One Model Performs Better')

    # Display the plot
    plt.show()



def evaluate_models(positives, avgs, prunes, *coeffs):
    
    # unpack percentages for the linear combination
    a, b, c, = coeffs

    # When we sum the values by rows, we are essentially aggregating the number of positive differences for each model.
    # The row sum represents the total count of queries where a given model outperforms the other models. 
    mpd = positives.sum(axis=1) / ((positives.shape[1]-1) * 100)

    model_values = (a * mpd + b * avgs + c * prunes/100).round(4)
    
    return model_values.to_dict()