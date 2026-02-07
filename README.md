# Phaze (CS/ECE 8803 HML Lab 2 version)

Phaze is a framework to perform the co-optimization between accelerator architecture search and model partitioning for distributed training. For more details, please refer to our ICML 2024 paper, [Integrated Hardware Architecture and Device Placement Search](https://openreview.net/pdf?id=ucl3B05EsX).

## Installation

To install the dependencies for Phaze, run:

```bash
conda env create -f environment.yml
export CXX=$(which g++)
export CC=$(which gcc)
./setup.sh
```
**Note on build Time:** Creating the conda environment and installing the packages may take some time ~10  minutes (apex will take an additional ~15 mintues)

Once installation is all done, add the following path variables in `~/.bashrc`:
```bash
export THIRD_PARTY_PATH=$(pwd)/phaze_hml_spring26/third_party_for_phaze
export WHAM_PATH=$THIRD_PARTY_PATH/wham/
export SUNSTONE_PATH=$THIRD_PARTY_PATH/sunstone/
export PYTHONPATH=$THIRD_PARTY_PATH:$WHAM_PATH:$SUNSTONE_PATH:$PYTHONPATH
export PYTHONPATH=$(pwd)/.conda/envs/phaze_env/lib/python3.10/site-packages/megatron/fused_kernels/build:$PYTHONPATH

export CXX=$(which g++)
export CC=$(which gcc)
```

Refresh you shell at your home directory by running: 
```bash
cd ~
source ~/.bashrc
```
**Important**: Check that the `PYTHONPATH` is configured correctly, and all the paths listed are valid, with: 
```bash
echo $PYTHONPATH
```

### Obtain a Gurobi License

Phaze uses Gurobi 10.0.1 to solve the ILP formulations. To run the ILP solver, obtain a Gurobi license from the [The Gurobi Website](https://www.gurobi.com/).

- Create an Gurobi WLS license and place the `gurobi.lic` file in you home directory. 

## Debugging for setup

### **1. Troubleshooting Apex Installation**

If you encounter a CUDA version mismatch error during the `apex` installation process:

**Error:** "Cuda extensions are being compiled with a version of Cuda that does not match the version used to compile Pytorch binaries..."

You can resolve this using one of the two methods below:

1. **Method 1: Bypass the Version Check**

    For minor version mismatches, it is generally safe to skip the strict version check.

    1. Open `apex/setup.py`.
    2. Locate and comment out lines 84–92 (the `if bare_metal_version != torch_binary_version:` block).
    3. Save the file and restart the installation. (Rerun the last part of  `setup.sh` (starting from  `cd apex`))

2. **Method 2: Align PyTorch with System CUDA**

    Check your system's CUDA version using `nvcc --version`, then install the PyTorch build that matches that version. For example, if your system is running CUDA 12.4, install PyTorch 2.5.1 as follows:
    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    ```

Rerun the last part of  `setup.sh` (starting from  `cd apex`)

**Note:** Once the build starts, you should see logs indicating extensions are being compiled:
* `building 'apex_C' extension`
* `building 'amp_C' extension`

**Note on Compilation Time:** Compiling these extensions from source typically takes approximately 15 to 20 minutes depending on your system resources.

### **2. Troubleshooting building with C++**
If you every see errors such as `x86_64-conda-linux-gnu-cc: fatal error: cannot execute 'cc1plus':` when building device_placement or when running task0. Try running the command below and run again: 

```bash
export CXX=$(which g++)
export CC=$(which gcc)
```


## Quick start (HML students): 
Follow your lab outline to run the scripts for each task in `/hml_scripts`



## Quick Start (Original Phaze instructions, Ignore for HML students)

We provide scripts to run the experiments described in the paper.

The following example command searches for the optimal architecture configuration and device placement strategy for the specified `model` and list of microbatch sizes. It stores the throughput estimations for the explored architectures in `/Solver/output`:

```bash
cd scripts
./<model.sh> "<microbatch_sizes>"
```

## Phaze Execution and Code Structure

Phaze can be executed with the following command:
```bash
python3 phaze.py --phaze_model <model_name> --phaze_exec_type <execution_mode> 
 --phaze_micro_batch_size <microbatch_sizes> --phaze_max_tmp_width <tmp> \
--phaze_sequence_length <seq_len>  --phaze_hbm_size <hbm>
```

### Inputs
- `model_name` = Bert, GPT, OPT, llama2 variants
- `execution_mode` = ["run_solver", "prepopulate_estimates", "extract_graph"]
- `seq_len`= Sequence length of the model
- `micro_batch_size` = List of microbatch sizes to explore
- `max_tmp_width` = Maximum Tensor Model Parallel width for megatron models

### Execution Modes

Phaze has 3 execution modes: 

- `extract_graph`
  - Extracts the graph from the training script (`GraphExtractor/graph_extract.py`)
  - Stores torch.fx graphmodule in `GraphExtractor/out/<model>` folder
- `prepopulate_estimates`
  - Runs `extract_graph` or load from file
  - Generates valid architecture configurations if `Estimator/arch_configs/cores.json` does not exist, otherwise loads from file.
  - Generates estimates for all the operators in the graph and stores the output in `Estimator/estimates/<model>`
    - Estimator is executed per node and per architectural configuration using Sunstone
- `run_solver`
    - Runs `extract_graph` and `prepopulate_estimates` or load from file
    - Runs the ILP solver to get per-layer latency estimates
        - All model latency and memory estimates, per layer are stored in `Solver/output/` folder
    - Solver runs dynamic program for each model and `hbm` size 

### Code Structure
```bash
/                           : PHAZE_ROOT
|-- GraphExtractor          : Extract model operator graphs
|-- Estimator               : Generate architectures and estimate latencies
|-- Solver                  : ILP and DP solver
|-- third_party_for_phaze
|   |-- Wham                : For operator mapping and estimating area
|   |-- Sunstone            : For estimating operator latency
|   |-- Megatron            : For Megatron Models
|-- phaze.py                : Python source for Phaze
```

## Citation
If you use Phaze in your research, please cite our paper:

```
@inproceedings{phaze,
    author={Wang, Irene and Tarnawski, Jakub and Phanishayee, Amar and Mahajan, Divya},
    title={Integrated Hardware Architecture and Device Placement Search}, 
    booktitle={International Conference on Machine Learning},
    year={2024}
}
```
