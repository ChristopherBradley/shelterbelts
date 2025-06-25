#!/bin/bash

# specify working directory and storage directory:
wd=/home/147/cb8590/Projects/shelterbelts 
dir=/g/data/xe2/cb8590/shelterbelts/tasmania_testdata
tmpdir=/scratch/xe2/cb8590/shelterbelts  

# params
buffer=2.5  #km 
start='2019-01-01'
end_='2019-04-01'

# Automatically submit a whole bunch of jobs for lots of locations
coordinates_file="tasmania_tiles_testing.csv"

# Loop through each line in the file
while IFS=, read -r lon lat; do
    # Trim any leading or trailing whitespace
    lat=$(echo $lat | xargs)
    lon=$(echo $lon | xargs)

    # Generate the stub by formatting lat, lon, removing minus signs, and replacing decimal points with underscores
    stub="$(printf "%.2f_%.2f" $lat $lon | sed 's/-//' | tr '.' '_')"

    ## Run first job
    job_id1=$(qsub -v wd=$wd,stub=$stub,dir=$dir,lat=$lat,lon=$lon,buffer=$buffer,start_time=$start,end_time=$end_ run_sentinel.sh)
    echo "First job submitted for stub $stub with ID $job_id1"

    # echo "Latitude: $lat, Longitude $lon"

done < "$coordinates_file"
done < "$coordinates_file"