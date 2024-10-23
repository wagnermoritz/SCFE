# S-CFE: Simple Counterfactual Explanations

This repository contains the code for the paper "S-CFE: Simple Counterfactual Explanations". To run the code, execute the `main.py` file with the desired arguments, e.g.

```
python main.py --dataset Wine --alg SCFE_KDE --stepsize 0.01 --sparsity 2 --plausibility 5 --n_neighbors 3 --prox zero --model DNN --resdir ./Results/
```

where `sparsity` is the parameter beta in the paper and `plausibility` is the parameter theta in the paper.

## Requirements
PyTorch, TorchVision, scikit-learn, Pandas, Numpy