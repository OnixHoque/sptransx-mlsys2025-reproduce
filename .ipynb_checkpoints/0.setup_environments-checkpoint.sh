# module load gcc conda
conda create --prefix=env_sptransx python=3.9 -y
source activate ./env_sptransx
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
pip install tqdm
pip install torch_geometric
pip install torchkge
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
git clone https://github.com/HipGraph/iSpLib.git
cd iSpLib
./configure
make
cd ..
conda deactivate
conda create --prefix=env_sptransx_dglke python=3.8 -y
source activate ./env_sptransx_dglke
conda install cudatoolkit=11.0 -c pytorch -y
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu110==0.5.3
cd ./dglke-installation/dgl-ke/python/
pip install .
cd ../../../
pip install ogb tqdm psutil
conda deactivate