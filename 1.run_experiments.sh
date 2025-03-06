# CPU experiments, saved in cpu.txt
mkdir -p ./cpu_scripts/output
mkdir -p ./gpu_scripts/output
cd ./cpu_scripts
./run_experiments.sh
cd ..
# GPU experiments, saved in gpu.txt
cd ./gpu_scripts
./run_experiments.sh
# Now open and run 2.validation.ipynb
