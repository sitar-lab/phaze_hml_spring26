# conda env create -f environment.yml -y
# conda activate phaze_env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
pip3 install transformers==4.45.2 --no-cache-dir
pip3 install graphviz pygraphviz --no-cache-dir
# pip3 install transformer-engine[pytorch]
pip3 install accelerate --no-cache-dir

# Megatron specific installations
######################################################
#pip3 install megatron-lm
pip3 install git+https://github.com/NVIDIA/Megatron-LM.git --no-cache-dir
pip3 install six --no-cache-dir
pip3 install pybind11 --no-cache-dir
pip3 install ninja --no-cache-dir

pip3 install scipy --no-cache-dir
pip3 install networkx --no-cache-dir
pip3 install gurobipy --no-cache-dir
pip3 install wandb --no-cache-dir

######################################################

git clone https://github.com/Accelergy-Project/accelergy.git
cd accelergy
git checkout 0278a565187dc019ca40043ed486bf94b645327e
pip3 install .
accelergy
cd ..

git clone https://github.com/Accelergy-Project/accelergy-cacti-plug-in.git
cd accelergy-cacti-plug-in
git checkout ba5468303c27b4a1a317742a4eaf147065b907e5
pip3 install .

git clone https://github.com/HewlettPackard/cacti.git 
cd cacti
make
export PATH=$(pwd):${PATH}
cd ../..

git clone https://github.com/Accelergy-Project/accelergy-table-based-plug-ins.git
cd accelergy-table-based-plug-ins/
git checkout 223039ffbf0e034f3b09c2b80074ad398fbaf03e 
pip3 install . --no-build-isolation
cd ..

git clone https://github.com/Accelergy-Project/accelergy-library-plug-in.git
cd accelergy-library-plug-in
git checkout 0cab62c3631dbbe9a7925ff795285619a1bd6538
pip3 install .
cd ..

cp Estimator/arch_configs/area_files/*.csv $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-table-based-plug-ins/set_of_table_templates/data/.
cp -r Estimator/arch_configs/area_files/tablePluginData $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-library-plugin/library
cp -r accelergy-cacti-plug-in/cacti $CONDA_PREFIX/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in

cd Solver/device_placement/
g++ device_placement.cpp -o device_placement
cd ../..

###### apex installation
git clone https://github.com/NVIDIA/apex
cd apex
# pip3 install -v --disable-pip-version-check --no-cache-dir  --no-build-isolation  --global-option="--cpp_ext" --global-option="--cuda_ext" ./
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..
cp -r third_party_for_phaze/phaze-megatron-lm/megatron/*  $CONDA_PREFIX/lib/python3.10/site-packages/megatron/
