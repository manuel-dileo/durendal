# Deep learning framework for temporal heterogeneous networks
A discrete-time GNN-based deep learning framework for temporal heterogeneous networks forecasting.

## Table of Contents

* [General Information](#general-information)
* [Repository organization](#repository-organization)
* [Dataset](#dataset)
* [Experiments](#experiments)
* [Additional information](#additional-information)
* [Contact](#contact)

## General Information

This repository contains additional material related to the work titled "A discrete-time deep learning framework for temporal heterogeneous networks forecasting".

## Repository organization
This repository is organized as follows:
- The folder `src` contains the Python source file that defines the components of our framework. You can import this Python file into your projects to create your own model for your prediction task.
- The folders `gdelt18`, `icews18`, `taobaoth`, and `steemitth` contain the code to reproduce the monorelational experiments and perform the data preprocessing steps for each dataset. The experiments are described in jupyter notebooks. The py files in these folders contain the code to obtain the datasets and the definition of the custom DURENDAL and baseline models to perform the specific future link prediction task.
- The folder `multirelational` contains the code to reproduce the experiments on multirelational future link predictions and the effectiveness of model design. Experiments can be executed using the terminal. Python files contain the ready-to-use multirelational architectures, the training and evaluation code, and the repurposed heterogeneous GNNs to a temporal setting.

## Dataset
We conducted the experimental evaluation of the framework over four temporal heterogeneous network datasets on future link prediction tasks. 

We also extend the set of benchmarks for TNHs by introducing two novel high-resolution temporal heterogeneous graph datasets derived from an emerging Web3 platform and a well-established e-commerce website. 

For `GDELT18`, `ICEWS18`, and `TaobaoTH`, we download the source data from the [PyG library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). We release the code to compute data preprocessing and obtain the graph snapshots representation to train and test DURENDAL. 

For `SteemitTH`, we collect the data using the [Steemit API](https://developers.steem.io/). Due to privacy reasons on personal data like usernames and textual content, we can't release the dataset. To patch this problem, we provide an anonymized version of our data. This version represents the final mathematical objects that are used to feed the models. To be compliant with IRB, we publicly release the heterogenous network of Steemit without text-based features but features will be available upon request. Note that performing future link prediction on SteemitTH without node features may lead to different results compared to the one described in the paper. Further details about data gathering and preprocessing for SteemitTH can be found in the data-related concerns section of the paper and in the supplementary information.

## Experiments
To reproduce the experiments described in the paper for the multirelational link prediction task you can run the following command:
```
cd multirelational/
python run.py --seed <seed_value> --dataset <dataset_name> --model <model_name>
```
To reproduce the experiments described in the paper for the effectiveness of model design you can run the following command:
```
cd multirelational/
python repurpose.py --model <model_name> --update <update_function>
```

For the monorelational link prediction task, you can run the notebooks contained in the folder dedicated to each dataset.

Note that the code to run the experiments is the same for all the datasets; there are small changes just related to relation and model names. To inspect an annotated version of the work, you can refer to code and notebooks in the `steemitth` folder. 

Note also that results related to the `SteemitTH` dataset may be different from the one presented in the paper since we do not include textual features, which we can not release to be compliant with IRB. Textual features are available upon request.

## Additional information
For information concerning training details, model architecture, hardware resources, and computational time for individual experiments, please read the paper and its appendix.

## Contact
For any clarification or further information please do not hesitate to contact me.
