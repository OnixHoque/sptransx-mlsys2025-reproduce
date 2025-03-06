source activate ../env_sptransx
python transe-fastkg.py fb15k
python transe-torchkge.py fb15k
python transe-pyg.py fb15k
python transr-fastkg.py fb15k
python transr-torchkge.py fb15k
python transh-fastkg.py fb15k
python transh-torchkge.py fb15k
python toruse-fastkg.py fb15k
python toruse-torchkge.py fb15k
conda deactivate
source activate ../env_sptransx_dglke
python transe-dglke.py fb15k
python transr-dglke.py fb15k
conda deactivate
cat ./output/* > ../cpu.txt