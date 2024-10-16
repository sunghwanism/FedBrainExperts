#!/bin/bash

MRI_DIR="/NFS/Users/moonsh/data/FLData/Image"

declare -A dataset_dict=(
    [0]="HCP"
    [1]="SLIM"
    [2]="COBRE"
    [3]="UCLA-CNP"
    [4]="OAS3"
    [5]="ADNI"
    [6]="CamCAN"
    [7]="DecNef"
    [8]="DLBS"
    [9]="IXI"
    [10]='SALD'
)

for idx in "${!dataset_dict[@]}"; do
    dataset_name="${dataset_dict[$idx]}"
    dataset_dir="${MRI_DIR}/${dataset_name}"
    

    if [ ! -d "$dataset_dir" ]; then
        echo "Directory $dataset_dir does not exist, skipping..."
        continue
    fi

    BAD_QC_FILE="bad_qc_files_${dataset_name}.txt"

    if [ -f "$BAD_QC_FILE" ]; then
        rm "$BAD_QC_FILE"
    fi

    for mri in "$dataset_dir"/*; do
        if [ -f "$mri" ]; then
            
            output_file="${mri%.*}_brain_extracted"

            bet "$mri" "$output_file" -R -f 0.5 -g 0

            if [ ! -f "${output_file}.nii.gz" ]; then
                echo "$mri has QC issues (brain extraction failed)"
                echo "$mri" >> "$BAD_QC_FILE"
                continue
            fi

            brain_vol=$(fslstats "${output_file}.nii.gz" -V | awk '{print $1}')
            echo "Processing $mri ... "
            echo $brain_vol

            if [ $brain_vol -lt 570000 ]; then
                echo "$mri has QC issues (small brain volume)"
                echo "$mri" >> "$BAD_QC_FILE"
                rm -f "${mri}"
            fi

            rm -f "${output_file}.nii.gz"
        fi
    done

    echo "QC process completed for dataset: $dataset_name"
done
