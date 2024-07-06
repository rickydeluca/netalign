import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_algorithm_results(files, noise):
    # Combine files in a DataFrame
    dfs = [pd.read_csv(file) for file in files]
    data = pd.concat(dfs, ignore_index=True)

    # Choose between 'addition' or 'removal' noise
    if noise == 'add':
        filtered_data = data[data['p_rm'] == 0].copy()  
        filtered_data['noise_level'] = filtered_data['p_add']
    else:
        filtered_data = data[data['p_add'] == 0].copy()
        filtered_data['noise_level'] = filtered_data['p_rm']

    # Group by noise_level and calculate the average accuracy and standard deviation for each algorithm
    grouped_data = filtered_data.groupby(['model', 'noise_level']).agg({'avg_acc': 'mean', 'std_acc': 'mean'}).reset_index()

    # Plotting with Matplotlib
    plt.figure(figsize=(10, 6))
    for algorithm in filtered_data['model'].unique():
        algorithm_data = grouped_data[grouped_data['model'] == algorithm]

        # Plot the accuracy line
        plt.plot(algorithm_data['noise_level'], algorithm_data['avg_acc'], marker='o', label=f'{algorithm}', linestyle='-', linewidth=2)

        # Plot the shaded area for standard deviation
        plt.fill_between(algorithm_data['noise_level'], algorithm_data['avg_acc'] - algorithm_data['std_acc'],
                         algorithm_data['avg_acc'] + algorithm_data['std_acc'], alpha=0.2)

    xlabel_name = "Noise (add)" if noise == 'add' else "Noise (rm)"
    plt.xlabel(xlabel_name)
    plt.ylabel('Average Accuracy')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()


import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_accuracy_vs_noise(folder_path):
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize figures for the two plots
    fig_add, ax_add = plt.subplots(figsize=(12, 8))
    fig_rm, ax_rm = plt.subplots(figsize=(12, 8))
    
    for file in csv_files:
        # Read CSV file
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # Filter for addition noise (p_add > 0 and p_rm == 0)
        df_add = df[(df['p_add'] > 0) & (df['p_rm'] == 0)]
        if not df_add.empty:
            noise_levels_add = df_add['p_add']
            avg_acc_add = df_add['avg_acc']
            std_acc_add = df_add['std_acc']
            
            # Plotting for addition noise
            ax_add.plot(noise_levels_add, avg_acc_add, marker='o', linestyle='dotted', label=file)
            ax_add.fill_between(noise_levels_add, avg_acc_add - std_acc_add, avg_acc_add + std_acc_add, alpha=0.2)
        
        # Filter for removal noise (p_rm > 0 and p_add == 0)
        df_rm = df[(df['p_rm'] > 0) & (df['p_add'] == 0)]
        if not df_rm.empty:
            noise_levels_rm = df_rm['p_rm']
            avg_acc_rm = df_rm['avg_acc']
            std_acc_rm = df_rm['std_acc']
            
            # Plotting for removal noise
            ax_rm.plot(noise_levels_rm, avg_acc_rm, marker='o', linestyle='dotted', label=file)
            ax_rm.fill_between(noise_levels_rm, avg_acc_rm - std_acc_rm, avg_acc_rm + std_acc_rm, alpha=0.2)
    
    # Adding plot details for addition noise plot
    ax_add.set_title('Model Accuracy vs. Noise Level (Noise in Addition)')
    ax_add.set_xlabel('Noise Level (p_add)')
    ax_add.set_ylabel('Accuracy')
    ax_add.legend()
    ax_add.grid(True)
    
    # Adding plot details for removal noise plot
    ax_rm.set_title('Model Accuracy vs. Noise Level (Noise in Removal)')
    ax_rm.set_xlabel('Noise Level (p_rm)')
    ax_rm.set_ylabel('Accuracy')
    ax_rm.legend()
    ax_rm.grid(True)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Plot algorithm results.')
    # parser.add_argument('--files', nargs='+', help='List of CSV files containing algorithm results')
    # parser.add_argument('--noise', type=str, default='add', help='Choose between addition or removal noise. Default: add')
    parser = argparse.ArgumentParser(description='Plot algorithm results.')
    parser.add_argument('--dir', type=str, help='Path to directory with accuracy vs noise results')
    args = parser.parse_args()

    # plot_algorithm_results(args.files, args.noise)
    plot_accuracy_vs_noise(args.dir)