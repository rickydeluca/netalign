import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_algorithm_results(files):
    # Read data from each file and combine them into a single DataFrame
    dfs = [pd.read_csv(file) for file in files]
    data = pd.concat(dfs, ignore_index=True)

    # Combine p_add and p_rm to get the total noise level
    data['noise_level'] = data['p_add'] + data['p_rm']

    # Group by noise_level and calculate the average accuracy for each algorithm
    grouped_data = data.groupby(['model', 'noise_level'])['avg_acc'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    for algorithm in data['model'].unique():
        algorithm_data = grouped_data[grouped_data['model'] == algorithm]
        plt.plot(algorithm_data['noise_level'], algorithm_data['avg_acc'],
                 marker='o', label=f'{algorithm}', linestyle='-', linewidth=2)

    plt.xlabel('Noise Level')
    plt.ylabel('Average Accuracy')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot algorithm results.')
    parser.add_argument('files', nargs='+', help='List of CSV files containing algorithm results')
    args = parser.parse_args()

    plot_algorithm_results(args.files)