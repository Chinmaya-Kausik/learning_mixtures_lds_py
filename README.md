# Learning Mixtures of Linear Dynamical Systems (Python)

* This repository implements Chen and Poor's algorithm from [this ICML paper](https://arxiv.org/abs/2201.11211) in Python. 
* All relevant code is in the "code" folder. 
* The main 4 subroutines are implemented in subspace_est.py, clustering.py, classification.py and model_estimation_vectorized.py respectively. There are unvectorized versions of the model estimation subroutine too, which take less memory but more time.

This was a part of a reproducibility report for the paper. All experiments are performed in files whose names start with "exp_". We noticed that the authors did not split the datasets into parts to be used in the clustering and subspace estimation subroutines, which is needed for theoretical guarantees. Splitting improved the performance in certain aspects, and experiments that include this splitting have the suffix "_split." More details can be found in the reproducibility report, to be uploaded soon.
