# Description

Spatom is a state-of-the-art software to predict protein-protein interaction sites(PPIS). This Repository provides intermediate data of Spatom and Python scripts which are for users to reproduce Spatom.
                                                                                
This software is free to use, modify, redistribute without any restrictions, except including the license provided with the distribution. 

# Requirements

Python 3.8.5

torch 1.10

torch-geometric 2.0.2

numpy 1.21.3

DSSP (user should install [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/), the following command can be used to install DSSP)

```
$ conda install -c salilab dssp # for Linux
$ conda install -c speleo3 dssp # for Windows
```

freesasa 2.1.0

Biopython 1.7.9

sklean 0.23.2

matplotlib 3.4.0

If runing on GPU, Spatom needs

cuda 10.2

# Usage

We provide training and testing protein PDB files (stored in path_to_Spatom-Reproducing/Data/DBD/data/pre_pdb) , PSSM files (stored in path_to_Spatom-Reproducing/Data/DBD/data/feature/pssm), DSSP files (stored in path_to_Spatom-Reproducing/Data/DBD/data/feature/dssp), labels (stored in path_to_Spatom-Reproducing/Data/DBD/data/label_542.txt) of each protein and pretrained model (/home/yuting/haonan/Spatom-Reproduce/result/model) for users to reproducing the results in this paper.

Users can run feature_data.py to extract raw features (stored in path_to_Spatom-Reproducing/Data/DBD/data/feature_data), and run feature.py to generate processed data (stored in path_to_Spatom-Reproducing/Data/DBD/data/feature_extract), and run train.py or test.py to train or test the model.

# Dataset and feature data

Spatom uses datasets from [Protein-Protein Docking Benchmark 5.5](https://zlab.umassmed.edu/benchmark/), and all features were created by ourselves. 

# Contact

Any questions, problems, bugs are welcome and should be dumped to Haonan Wu <hnw.bio@outlook.com>.

Created on Feb. 10, 2023.

