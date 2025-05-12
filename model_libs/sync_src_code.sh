#!/bin/bash

# Check if src directory exists
if [ ! -d "src_code" ]; then
    echo "Error: 'src_code' directory not found"
    exit 1
fi

# Find all directories
run_dirs=$(find configs -type d -name "*-*")

# Check if any run directories were found
if [ -z "$run_dirs" ]; then
    echo "Error: No directories starting with 'run-' found"
    exit 1
fi

# Loop through each Python file in the src directory
for src_file in src_code/*.py; do
    # Check if file exists (in case there are no .py files)
    [ -e "$src_file" ] || continue
    
    # Get the base name of the file
    filename=$(basename "$src_file")
    
    # Loop through each run directory
    for run_dir in $run_dirs; do
        # Check if the file already exists in the run directory
        if [ -e "$run_dir/$filename" ]; then
            # If it exists, remove it
            rm "$run_dir/$filename"
            echo "Removed existing file: $run_dir/$filename"
        fi

        # Create a hard link in the run directory
        ln "$src_file" "$run_dir/$filename"
        echo "Linked $src_file to $run_dir/$filename"
    done
done

echo "All Python scripts have been linked to the run directories"