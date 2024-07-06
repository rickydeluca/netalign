import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_with_std(folder_path):
    # Get list of CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize figures for the two plots
    plt.figure(figsize=(12, 8))
    fig_add, ax_add = plt.subplots(figsize=(12, 8))
    fig_rm, ax_rm = plt.subplots(figsize=(12, 8))
    
    for file in csv_files:
        # Read CSV file
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # Filter for addition noise (p_add > 0 and p_rm = 0)
        df_add = df[(df['p_add'] > 0) & (df['p_rm'] == 0)]
        if not df_add.empty:
            train_ratios_add = df_add['train_ratio']
            avg_acc_add = df_add['avg_acc']
            std_acc_add = df_add['std_acc']
            
            # Plotting for addition noise
            ax_add.plot(train_ratios_add, avg_acc_add, marker='o', linestyle='dotted', label=file)
            ax_add.fill_between(train_ratios_add, avg_acc_add - std_acc_add, avg_acc_add + std_acc_add, alpha=0.2)
        
        # Filter for removal noise (p_rm > 0 and p_add = 0)
        df_rm = df[(df['p_rm'] > 0) & (df['p_add'] == 0)]
        if not df_rm.empty:
            train_ratios_rm = df_rm['train_ratio']
            avg_acc_rm = df_rm['avg_acc']
            std_acc_rm = df_rm['std_acc']
            
            # Plotting for removal noise
            ax_rm.plot(train_ratios_rm, avg_acc_rm, marker='o', linestyle='dotted', label=file)
            ax_rm.fill_between(train_ratios_rm, avg_acc_rm - std_acc_rm, avg_acc_rm + std_acc_rm, alpha=0.2)
    
    # Adding plot details for addition noise plot
    ax_add.set_title('Model Accuracy vs. Train Ratio (Noise in Addition)')
    ax_add.set_xlabel('Train Ratio')
    ax_add.set_ylabel('Accuracy')
    ax_add.legend()
    ax_add.grid(True)
    
    # Adding plot details for removal noise plot
    ax_rm.set_title('Model Accuracy vs. Train Ratio (Noise in Removal)')
    ax_rm.set_xlabel('Train Ratio')
    ax_rm.set_ylabel('Accuracy')
    ax_rm.legend()
    ax_rm.grid(True)
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot algorithm results.')
    parser.add_argument('--dir', type=str, help='Path to directory with train ratio results')
    args = parser.parse_args()
    plot_accuracy_with_std(args.dir)