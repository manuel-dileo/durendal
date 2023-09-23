# Durendal
Graph deep learning framework for temporal heterogeneous networks. Submitted @ ICLR2024

## Table of Contents

* [General Information](#general-information)
* [Repository organization](#repository-organization)
* [Dataset](#dataset)
* [Experiments](#experiments)
* [Additional information](#additional-information)
* [Contact](#contact)

## General Information

This repository contains additional material related to the work titled "DURENDAL: Graph deep learning framework for temporal heterogeneous network" submitted at ICLR2024. 

We summarize our main contributions as follows: 1) we propose a novel graph deep learning framework that allows an easy repurposing of any heterogenous GNNs to a dynamic setting; 2) we introduce two different update schemes for obtaining temporal heterogeneous node embeddings, highlighting their strengths and weaknesses and their practical use scenarios; and 3) we define some minimal requirements datasets must satisfy to be useful testing grounds for temporal heterogeneous graph learning models, extending the set of benchmarks for THNs by introducing two novel high-resolution THNs datasets.

## Repository organization
This repository is organized as follows:
- The folder `src` contains the Python source file that defines the components of the DURENDAL architecture. You can import this Python file into your projects to create your own DURENDAL model for your prediction task.
- The folders `gdelt18`, `icews18`, `taobaoth`, and `steemitth` contain the code to reproduce the experiments and perform the data preprocessing steps for each dataset. The experiments are described in jupyter notebooks. The py files in these folders contain the code to obtain the datasets and the definition of the custom DURENDAL and baseline models to perform the specific future link prediction task.
- The folder `repurposing` contains the code to reproduce the experiments on the effectiveness of the DURENDAL model design. Experiments are described in the jupyter notebook while the Python file contains the ready-to-use DURENDAL repurposed architectures.
- The folder `multirelational` contains the code to reproduce the experiments on multirelational future link predictions. Experiments are described in the jupyter notebook while the Python file contains the ready-to-use multirelational architectures.
## Dataset
We conducted the experimental evaluation of the framework over four temporal heterogeneous network datasets on future link prediction tasks. 

We also extend the set of benchmarks for TNHs by introducing two novel high-resolution temporal heterogeneous graph datasets derived from an emerging Web3 platform and a well-established e-commerce website. 

For `GDELT18`, `ICEWS18`, and `TaobaoTH`, we download the source data from the [PyG library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). We release the code to compute data preprocessing and obtain the graph snapshots representation to train and test DURENDAL. 

For `SteemitTH`, we collect the data using the [Steemit API](https://developers.steem.io/). Due to privacy reasons on personal data like usernames and textual content, we can't release the dataset. To patch this problem, we provide an anonymized version of our data. This version represents the final mathematical objects that are used to feed the models. To be compliant with IRB, we publicly release the heterogenous network of Steemit without text-based features but features will be available upon request. Note that performing future link prediction on SteemitTH without node features may lead to different results compared to the one described in the paper. Further details about data gathering and preprocessing for SteemitTH can be found in the data-related concerns section of the paper and in the supplementary information.

## Experiments
To reproduce the experiments described in the paper you can run the notebooks contained in the folder dedicated to each dataset. The experiments related to the effectiveness of model design are reported in the `repurposing` folder. The experiments related to the multirelation future link prediction task are reported in the `multirelaitonal` folder. 

Note that the code to run the experiments is the same for all the datasets; there are small changes just related to relation and model names. To inspect an annotated version of the work, you can refer to code and notebooks in the `steemitth` folder. 

Note also that results related to the `SteemitTH` dataset may be different from the one presented in the paper since we do not include textual features, which we can not release to be compliant with IRB. Textual features are available upon request.

## Additional information
For information concerning training details, model architecture, hardware resources, and computational time for individual experiments, please read the paper and its appendix.

## Contact
For any clarification or further information please do not hesitate to contact name dot surname at the institution.
