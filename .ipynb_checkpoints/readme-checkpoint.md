# Artifact Evaluation Reproduction for "SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations", MLSys 2025

This repository contains artifacts and workflows for reproducing experiments from Md Saidul Hoque Anik and Ariful Azad's MLSys 2025 paper. The scripts will produce the table depicted in Figure 7 of the paper for the _FB15K_ dataset.

| **SparseTransX: Efficient Training of Translation-Based Knowledge Graph Embeddings Using Sparse Matrix Operations**

# Hardware Prerequisite
The CPU and GPU experiments were run on dedicated CPU/GPU (single) nodes of NERSC Perlmutter. Their configurations are given below. The parameters are set to maximize CPU/GPU utilization. Similar configurations are recommended to reproduce the results.

## For CPU Experiments
`AMD EPYC 7763 (Milan) CPU with 64 cores and 512GB DDR4 memory`

## For GPU Experiments
`1 x NVIDIA A100-SXM4 GPU with 40 GB VRAM`

# Software Prerequisite
The experiments were tested on the following configuration.
- GCC 12.2
- Conda 24.9.1
- Python 3.9 (3.8 for DGLKE)
- PyTorch 2.3.1 (1.7.1 for DGLKE)
- CudaToolKit 12.1 (11.0 for DGLKE)

# Environment Installation
To set up the environments, run the following command. It will create two venvs. One specific to DGLKE, another for the rest.

    ./0.setup_environments.sh

# Run Workflow
To run the experiments, execute the following command. It will generate the training time of a single minibatch training for various models and frameworks of *FB15K* dataset. The outputs will be saved in `cpu.txt` and `gpu.txt`.
    
    ./1.run_experiments.sh

# Validation of Results
To generate the table of Figure 7 in the paper, execute the Jupyter notebook `2.validation.ipynb`. It will parse the generated text files and produce the tables for CPU and GPU for *FB15k* dataset.