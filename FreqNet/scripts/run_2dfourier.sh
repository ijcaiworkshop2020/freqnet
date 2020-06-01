MODE=0

files=("bg_img_rot" "basic" "bg_rand" "rot" 'bg_img')
run_modes=("d" "t" "f")

echo "now running ${run_modes[$MODE]}"
for file in "${files[@]}"
do
  python3 experiment_PCANet2_main.py --config_files PCANet2_configs/2dfourier2_$file.json --run_type $run_modes
done

