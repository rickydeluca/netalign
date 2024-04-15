import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_algorithm_results(files):
    # Read data from each file and combine them into a single DataFrame
    dfs = [pd.read_csv(file) for file in files]
    data = pd.concat(dfs, ignore_index=True)

    # Set p_add to 0, using only p_rm for the noise level
    data_p_rm = data[data['p_add'] == 0].copy()  # Use .copy() to avoid SettingWithCopyWarning
    data_p_rm['noise_level'] = data_p_rm['p_rm']

    # Group by noise_level and calculate the average accuracy and standard deviation for each algorithm
    grouped_data = data_p_rm.groupby(['model', 'noise_level']).agg({'avg_acc': 'mean', 'std_dev': 'mean'}).reset_index()

    # Plotting with Matplotlib
    plt.figure(figsize=(10, 6))
    for algorithm in data_p_rm['model'].unique():
        algorithm_data = grouped_data[grouped_data['model'] == algorithm]

        # Plot the accuracy line
        plt.plot(algorithm_data['noise_level'], algorithm_data['avg_acc'], marker='o', label=f'{algorithm}', linestyle='-', linewidth=2)

        # Plot the shaded area for standard deviation
        plt.fill_between(algorithm_data['noise_level'], algorithm_data['avg_acc'] - algorithm_data['std_dev'],
                         algorithm_data['avg_acc'] + algorithm_data['std_dev'], alpha=0.2)

    plt.xlabel('p_rm')
    plt.ylabel('Average Accuracy')
    plt.title('Algorithm Comparison with Standard Deviation (p_add=0)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot algorithm results.')
    parser.add_argument('files', nargs='+', help='List of CSV files containing algorithm results')
    args = parser.parse_args()

    plot_algorithm_results(args.files)
