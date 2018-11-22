#!/bin/bash
DATE='Nov15'
DATASET=$1
ALGORITHM=$2
ADDITIONAL_SUFFIX='full2'
IPYNB_FILE="${DATASET}_train.ipynb"

JOB_NAME="$DATE-$DATASET-$ALGORITHM"
OUTPUT_FOLDER="$DATE-$DATASET-$ALGORITHM"
LOGFILE="$DATE-$DATASET-$ALGORITHM"
SCRIPT_FILE="${IPYNB_FILE%.*}.py"
APPEND_STR=0
while [ -f "$SCRIPT_FILE" ];do
    let APPEND_STR++
    SCRIPT_FILE="${IPYNB_FILE%.*}_$APPEND_STR.py"
done


if [ "$ALGORITHM" = "full_data" ];then
    [ "$#" -eq 3 ] || { echo "insufficient arguments" 2>&1 ; exit 1; }
    CLASS_BATCH_SIZE=$3
    JOB_NAME="$JOB_NAME-${CLASS_BATCH_SIZE}classes"
    OUTPUT_FOLDER="$OUTPUT_FOLDER-${CLASS_BATCH_SIZE}classes"
    LOGFILE="$LOGFILE-${CLASS_BATCH_SIZE}classes"
elif [ "$ALGORITHM" = "SupportNet_exemplars_set_size" ];then
    [ "$#" -eq 3 ] || { echo "insufficient arguments" 2>&1 ; exit 1; }
    EXEMPLARS_SET_SIZE=$3
    JOB_NAME="$JOB_NAME-${EXEMPLARS_SET_SIZE}exemplars"
    OUTPUT_FOLDER="$OUTPUT_FOLDER-${EXEMPLARS_SET_SIZE}exemplars"
    LOGFILE="$LOGFILE-${EXEMPLARS_SET_SIZE}exemplars"    
fi

if [ -n "$ADDITIONAL_SUFFIX" ];then
    JOB_NAME="$JOB_NAME-$ADDITIONAL_SUFFIX"
    OUTPUT_FOLDER="$OUTPUT_FOLDER-$ADDITIONAL_SUFFIX"
    LOGFILE="$LOGFILE-$ADDITIONAL_SUFFIX"
fi
LOGFILE="${LOGFILE}.log"

printf "IPYNB_FILE=${IPYNB_FILE}\nJOB_NAME=${JOB_NAME}\nSCRIPT_FILE=${SCRIPT_FILE}\nConfirm? [y]"
read choice
if ! [ "$choice" = "y" ]; then echo "Cancelled" 1>&2;exit 1; fi

if [ -d "$OUTPUT_FOLDER" ] || [ -f "$LOGFILE" ];then
    echo "ERROR: history folder or log file already exists" 1>&2
    exit 1
fi

if [ -f "$IPYNB_FILE" ];then
    jupyter nbconvert --to python $IPYNB_FILE --stdout > $SCRIPT_FILE|| {
    echo "ERROR: failed to convert $IPYNB_FILE"
    exit 1
}
else
    echo "ERROR: cannot find ipynb file $IPYNB_FILE" 1>&2
    exit 1
fi

sbatch << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=1-00:00:00 # DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --output $LOGFILE
module purge
module load slurm
module load cuda/9.0.176
module load cudnn/7.0.3-cuda9.0.176
module load anaconda
source activate tf
if [ "$ALGORITHM" = "full_data" ];then
    python -u "$SCRIPT_FILE" "$ALGORITHM" "$OUTPUT_FOLDER" "$CLASS_BATCH_SIZE"
elif [ "$ALGORITHM" = "SupportNet_exemplars_set_size" ]; then
    python -u "$SCRIPT_FILE" "$ALGORITHM" "$OUTPUT_FOLDER" "$EXEMPLARS_SET_SIZE"
else
    python -u "$SCRIPT_FILE" "$ALGORITHM" "$OUTPUT_FOLDER"
fi
source deactivate
EOF
