
PYTHON=python3
CONFIG='demo'
LAYER=2
MODE=0

while getopts d:m:l: option
do
    case "$option" in
    d)
         CONFIG=$OPTARG
         ;;
    m)
         MODE=$OPTARG
         ;;
    l)
         LAYER=$OPTARG
         ;;
    esac
done

SCRIPT=experiment_PCANet${LAYER}_main.py
echo "Run configuration $CONFIG, layer $LAYER, mode $MODE"
$PYTHON $SCRIPT --config_files PCANet${LAYER}_configs/2dfourier2_$file.json --run_type $MODE





