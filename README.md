# Durendal
Roland-based graph deep learning framework for temporal heterogeneous networks. Submitted @ NeurIPS23

## Dataset
We conducted the experimental evaluation of the framework over four temporal heterogeneous network datasets on future link prediction tasks. 

We also extend the set of benchmarks for TNHs by introducing two novel high-resolution temporal heterogeneous graph datasets derived from an emerging Web3 platform and a well-established e-commerce website. 

For GDELT18, ICEWS18 and TAOBAOTH, we download the source data from the [PyG library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html). We release the code to compute data preprocessing and obtain the graph snapshots representation to train and test DURENDAL. 

For SteemitTH, we collect the data using the [Steemit API](https://developers.steem.io/). Due to privacy reasons on personal data like username and textual content, we can't release the dataset. To patch this problem, we provide an anonymized version of our data. This version represents the final mathematical objects that are use to feed the models. To be compliant with IRB, we publicly release the heterogenous network of Steemit without text-based features but features will be available upon request. Note that performing future link prediction on SteemitTH without node features may lead to different results compared to the one described in the paper. 

## Additional information
For information concerning training details, model architecture, hw resources and computational time for individual experiments, please see the appendix.pdf file.


