import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_algorithm_results(files):
    # Read data from each file and combine them into a single DataFrame
    dfs = [pd.read_csv(file) for file in files]
    data = pd.concat(dfs, ignore_index=True)

    # Set p_rm to 0, using only p_add for the noise level
    data_p_add = data[data['p_add'] == 0].copy()  # Use .copy() to avoid SettingWithCopyWarning
    data_p_add['noise_level'] = data_p_add['p_rm']

    # Group by noise_level and calculate the average best epoch and standard deviation for each algorithm
    grouped_data = data_p_add.groupby(['model', 'noise_level']).agg({'avg_best_epoch': 'mean', 'std_dev': 'mean'}).reset_index()

    # Plotting with Matplotlib
    plt.figure(figsize=(10, 6))
    for algorithm in data_p_add['model'].unique():
        algorithm_data = grouped_data[grouped_data['model'] == algorithm]

        # Plot the best epoch line
        plt.plot(algorithm_data['noise_level'], algorithm_data['avg_best_epoch'], marker='o', label=f'{algorithm}', linestyle='-', linewidth=2)

        # Plot the shaded area for standard deviation
        plt.fill_between(algorithm_data['noise_level'], algorithm_data['avg_best_epoch'] - algorithm_data['std_dev'],
                         algorithm_data['avg_best_epoch'] + algorithm_data['std_dev'], alpha=0.2)

    plt.xlabel('Noise (Remove)')
    plt.ylabel('Average Best Epoch')
    plt.title('Best Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot algorithm results.')
    parser.add_argument('files', nargs='+', help='List of CSV files containing algorithm results')
    args = parser.parse_args()

    plot_algorithm_results(args.files)
