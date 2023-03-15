# Regularization in Extracting WFA from RNNs
Welcome to our repository of *Regularization in Extracting Weighted Finite Automata from Recurrent Neural Networks*!

## Main Contributions
Our main contributions are:
+ An empirical rule to deal with the missing data;
+ A regularization approach to repair the overfitting problem in extracting;
+  A Dropout tactic to supply data and regularize in extracting.


## Quick Start

Run demo_extended.ipynb to show datails for implementation.
To use, change preprocess.py , line 125 to
    return transition_count, kmeans, state_weightes, all_prediction_container
change main.py , line 75 to
    transition_count, kmeans, state_weightes, all_prediction_container = get_transitions(model, train_dataset, CLUSTER)
