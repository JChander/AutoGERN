# AutoGERN
We present AutoGERN, a GNN framework tailored for GRN inference from scRNA-seq data.

## Dependencies
cudatoolkit>=11.9
cudnn>=8.9  
networkx>=3.1
numpy>=1.24
pandas>=2.0
python>=3.8
pytorch>=2.4
pyg>=2.6
scikit-learn>=1.3

## Usage
We provided an example dataset under data/mDC/ for running gene regulatory inference using AutoGERN. It takes gene expression (**N✖M**) and an adjacent matrix (**M✖M**) representing the prior regulatory graph as input.
* For **normal data separation**, run `python main_LP.py --dataset data/mDC/TF+500/`</br>
+ For **strict data separation**, run `python main_LP_hardsplit.py --dataset data/mDC_HS/TF+500/`</br>
