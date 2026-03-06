# Spider

Spider is a flexible and unified framework for simulating ST data by leveraging spatial neighborhood patterns among cells through a conditional optimization algorithm. By inputting spatial patterns extracted from real data or user-specified transition matrix, Spider assigns cell type labels through the BSA algorithm that improves the computational efficiency of conventional simulated annealing algorithm. Gene expression profiles for individual cell types can be generated either using real or simulated data by Splatter or scDesign2. Finally, spot-level ST data can be simulated by aggregating cells inside generated spots using 10X spatial encoding scheme or user-specified generation rules. ![](./figures/Figure1.png) 
## Manuscript 
Please see our manuscript [Yang, Wei et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.05.21.541605v1) in BioRxiv to learn more. 
## The Key features of Spider
* Characterize spatial patterns by cell type composition and their transition matrix. 
* Flexible to implement most existing simulation methods as special cases.
* Design the batched simulated annealing algorithm to generate ST data for 1 million cells in < 5 minutes.
* Generate various scenarios of the tumor immune micro-environment by capturing the dynamic changes in diverse immune cell components and transition matrices.
* Provide customized data generation APIs for special application scenarios, such as the tissue layer structure implemented by Napari interface and some regular structures for reference.

# Software dependencies
anndata 
matplotlib 
numba 
numpy 
pandas
PyQt5
scanpy 
scikit_learn
scipy
seaborn
squidpy
# installation
Install Spider via PyPI by using:

```         
pip install st-spider
```

