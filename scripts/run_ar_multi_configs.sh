#!/bin/bash

# Configuration files folder
config_folder="configs"

# Prefix of the configuration file
file_prefix="ar_gine"

# Loop through all configuration files with the specified prefix
for config_file in "$config_folder/$file_prefix"*.yaml; do
    if [ -e "$config_file" ]; then
        python train_eval.py -c "$config_file"
    else
        echo "No matching configuration files found."
    fi
done
