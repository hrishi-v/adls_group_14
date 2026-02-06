# Lab 2

## Task 1


### Sampling Method Perf. Evaluation

In order to evaluate each sampling method, we first run each sampler on the entire search space provided in the tutorials. 

Optuna's `RandomSampler()` will randomly select each hyperparameter value from the search space, and we found that this produced a best test accuracy of 0.8321 after 30 trials. However, as shown in the graph, the sampler struggles to find hyperparameter combinations that improves the best accuracy as the number of trials increases. As the number of trials increases, the trial accuracies also do not become more consistent. For example, in trial 13, the accuracy of the model constructed was only 0.5, while the best accuracy up to this point was over 0.8, meaning that a large number of trials may be required to obtain a model architecture with a near-optimal test accuracy.

Optuna's `TPESampler()` will use Guassian Mixure Models (GMMs), where one GMM `l(x)` is trained using the hyperparameters which has given test accuracies within the top 25% of all models evaluated, and another GMM `g(x)` is trained based on the hyperparameters which has given test accuracies within the bottom 75% of all models evaluated. This split is controlled by a parameter to TPESampler known as gamma, and helps balance exploration of new hyperparameters with exploitation of the existing best hyperparameters found. The TPESampler will pick the set of hyperparameters which will maximise `l(x)/g(x)`. While this method gradually converges to trialling better hyperparameters, there are instances where it may trial the same set of hyperparameters multiple times, since it may not explore the search space as aggressively as the number of trials increases, which may mean that the TPESampler becomes "stuck" at a local optimal set of hyperparameters in the search space. This may be mitigated through increasing the value of gamma.

Optuna's `GridSampler()` will perform a grid search over the entire search space, so it may require a large amount of trials before finding hyperparameters that give a good test accuracy. In the search space, there are 153600 possible hyperparameter combinations, so for a limited number of trials, Optuna's `GridSampler()` is unlikely to be effective. In order to limit the search space, the number of linear layers in the Bert encoder is set to be the same as the number of linear layers in the best model architecture found by the `TPESampler()`, and the `num_layers`, `num_heads` and `hidden_size` is also limited to only include values which formed part of a hyperparameter combination which achieved an accuracy of above 0.8 with the `TPESampler()`. This means that the search space was limited to 60 hyperparameter combinations.

In the graph below, the best accuracy achieved against the number of trials is plotted for Optuna's `RandomSampler()`, `TPESampler()`, the `GridSampler()` with the entire search space, and the `GridSampler()` with a limited search space determined by the `TPESampler()`. 


![Best Accuracy vs Number of Trials for different Optuna Samplers](best_accuracy.png)



In the graph below, the trial accuracy achieved against the number of trials is plotted for Optuna's `RandomSampler()`, `TPESampler()`, the `GridSampler()` with the entire search space, and the `GridSampler()` with a limited search space determined by the `TPESampler()`. 


![Trial Accuracy vs Number of Trials for different Optuna Samplers](trial_accuracy.png)


## Task 2


In this task, the test accuracy will be computed on the quantised and pruned model, so the objective of the study is to maximise the test accuracy of the compressed model, making the search compression-aware. Optuna `TPESampler()` yielded the best results in Task 1, so it will be used to run the compression-aware search. For the first experiment, the model will be trained for 1 epoch then compressed, so the objective is to maximise the test accuracy of the model found using Post-Training Quantisation and Pruning. For the second experiment, the model will be trained for 1 epoch, compressed, then trained for 1 more epoch to investigate whether fine-tuning the compressed model can recover the test accuracy to be similar to the best model found in Task 1 without compression.


![](best_accuracy_cas.png)


