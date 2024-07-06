import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_algorithm_results(files):
    # Combine files in a DataFrame
    dfs = [pd.read_csv(file) for file in files]
    data = pd.concat(dfs, ignore_index=True)

    # Choose between 'addition' or 'removal' noise
    filtered_data = data[data['p_add'] == 0].copy()
    filtered_data['noise_level'] = filtered_data['p_rm']

    # Group by noise_level and calculate the average accuracy and standard deviation for each algorithm
    grouped_data = filtered_data.groupby(['model', 'noise_level']).agg({'avg_time': 'mean'}).reset_index()

    # Plotting with Matplotlib
    plt.figure(figsize=(10, 6))
    for algorithm in filtered_data['model'].unique():
        algorithm_data = grouped_data[grouped_data['model'] == algorithm]

        # Plot the accuracy line
        plt.plot(algorithm_data['noise_level'], algorithm_data['avg_time'], marker='o', label=f'{algorithm}', linestyle='-', linewidth=2)

    xlabel_name = "Noise"
    plt.xlabel(xlabel_name)
    plt.ylabel('Average Time')
    plt.title('Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot algorithm results.')
    parser.add_argument('--files', nargs='+', help='List of CSV files containing algorithm results')
    args = parser.parse_args()

    plot_algorithm_results(args.files)