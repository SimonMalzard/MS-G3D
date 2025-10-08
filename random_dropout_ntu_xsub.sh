#!/bin/bash

# Define the list of values for miss_amount
drop_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Absolute path to the YAML config file
config_file="/datadrive2/MS-G3D/config/nturgbd120-cross-subject/test_joint.yaml"
work_dir="/datadrive2/MS-G3D/pretrain_eval/ntu120/xset/joint"

# Print the current working directory and check if the config file exists
echo "Current working directory: $(pwd)"
if [ ! -f "$config_file" ]; then
    echo "Error: Config file does not exist at $config_file"
    exit 1
else
    echo "Config file exists at $config_file"
fi

# Loop over each value
for drop_value in "${drop_values[@]}"
do
    echo "Running script with miss_amount: $drop_value"
    
    # Update the YAML config file with the new miss_amount value using sed
    #sed -i "s/miss_amount: [0-9]\+/miss_amount: $miss_value/" "$config_file"
    sed -i "s/drop_rate: [0-9]*\.[0-9]\+/drop_rate: $drop_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'drop_rate' "$config_file" | awk '{print $2}')
    if [ "$updated_value" != "$drop_value" ]; then
        echo "Error: Failed to update drop_rate to $drop_value in $config_file"
        exit 1
    else
        echo "Successfully updated drop_rate to $drop_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'drop_amount' "$config_file"

    # Run the Python script with the specified command
    python3 main.py --config ./config/nturgbd120-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu120/xsub/joint --weights pretrained-models/ntu120-xsub-joint.pt

    echo "Finished running script with miss_amount: $miss_value"
    echo "---------------------------------------------"
done