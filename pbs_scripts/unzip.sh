#!/bin/bash

# Launch a single job for each unzip, so it goes faster.
# cd /scratch/xe2/cb8590/lidar  # This means the .o and .e files save here

for f in *.zip; do
    name="${f%.zip}"

    qsub -N "unzip_${name}" <<EOF
#!/bin/bash
#PBS -l mem=4GB
#PBS -l ncpus=1
#PBS -l jobfs=1GB
#PBS -P xe2
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/xe2
#PBS -q normal

cd /scratch/xe2/cb8590/lidar

outdir="${name}/laz_files"
mkdir -p "\$outdir"

echo "Unzipping $f -> \$outdir"
unzip -q "$f" -d "\$outdir"
mv "$f" "$name/"
EOF
    # break   # stop after the first zip

done

# Can use this if I want to place the out and err files in a specific spot
#PBS -o /scratch/xe2/cb8590/lidar/${name}/unzip.out
#PBS -e /scratch/xe2/cb8590/lidar/${name}/unzip.err
