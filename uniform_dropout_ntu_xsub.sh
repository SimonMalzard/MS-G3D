#!/bin/bash

# Define the list of values for miss_amount
FPS_values=(3 5 10 15 30)

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
for fps_value in "${FPS_values[@]}"
do
    echo "Running script with miss_amount: $fps_value"

    # Update the YAML config file with the new miss_amount value using sed
    #sed -i "s/miss_amount: [0-9]\+/miss_amount: $miss_value/" "$config_file"
    sed -i "s/FPS: [0-9]*\.[0-9]\+/FPS: $fps_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'FPS' "$config_file" | awk '{print $2}')
    if [ "$updated_value" != "$fps_value" ]; then
        echo "Error: Failed to update FPS to $fps_value in $config_file"
        exit 1
    else
        echo "Successfully updated FPS to $fps_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'FPS' "$config_file"

    # Run the Python script with the specified command
    python3 main.py --config ./config/nturgbd120-cross-subject/test_joint.yaml --work-dir pretrain_eval/ntu120/xsub/joint --weights pretrained-models/ntu120-xsub-joint.pt

    echo "Finished running script with miss_amount: $miss_value"
    echo "---------------------------------------------"
done