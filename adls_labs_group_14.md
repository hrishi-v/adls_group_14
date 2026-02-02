# Lab 0

## Tutorial 1

### Key Definitions and Summary

The main flow of this tutorial is from pretrained model (Bert specifically) to FX Graph (in FX IR). This is then raised to the Mase IR, upon which we can run passes.

FX Graph - A compute graph which offers a high-level representation of the computation. It is also PyTorch native, meaning that each operator in the graph correlates to a Python object or callable. We can transform and optimize the graph, and regenerate the Python to run it!

Mase IR - The key benefit of Mase IR is that it offers a common abstraction layer for both hardware and software workloads. It also incorporates information regarding the workload that is to be run by the graph.

### What we do

What we essentially do is to take a model, generate a MaseGraph of it, such that we can execute passes. These passes are able to analyse or transform nodes in the graph.

Writing an analysis pass can be done using the `get_logger` API from Machop; in this tutorial we specifically use to count the number of dropout layer (6 of them). This is important as they only have meaning in the training stage, and can be removed for inference.

Once those have been picked out, the graph is again exported to have further optimisations done on it.

## Tutorial 2

### Key Definitions and Summary

This tutorial focuses on how we finetune pretrained models from HuggingFace. In this example we do so with two options - first, SFT (Supervised Fine Tuning), second, LoRA ([Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)). The model is for sentiment analysis of the IMDb dataset.

### What we do

We take the tokenized IMDb dataset, and the model is loaded as a MaseGraph. We can then apply both methods of finetuning with the Machop `trainer` method.

With tools available to us in Mase, we can inject the LoRA adapter and perform a pass of the graph. We generate a custom MaseGraph using arguments.


**Task:** Remove the `attention_mask` and `labels` arguments from the `hf_input_names` list and re-run the following cell. Use `mg.draw()` to visualize the grah in each case. Can you see any changes in the graph topology? Can you explain why this happens?

When the graph is drawn without the attention mask and label arguments, those values aren't passed into the model, and don't appear on the graph topology.

LoRA allows us to achieve high accuracy whilst limiting memory usage, allowing control of parameters whilst fine-tuning memory hungry models like LLMs.

# Lab 1

## Tutorial 3

### Key Definitions and Summary

We take a finetuned model and run the PTO (Post-Training Quantization) pass from Mase on it. This will reduce accuracy, whilst also reducing the precision of weights and biases, such that the model size reduces.

**Finetuned Model Accuracy:** 83.44%
**Post-PTQ Accuracy:** 81.58%
**Post-QAT Accuracy:** 84.04%

We then try to restore the accuracy from before quantization using QAT (Quantization-Aware Training). This includes the model back into the training loop after the quantization pass, such that the model can optimize the new, lower-resolution weights for the dataset. As we can see from the results, this step results in a better accuracy than even before PTQ, with a lower memory requirement.

From the weights/biases present in the finetuned model, we observed that the range of values primarily lied between -1 and 1, allowing us to utilise very few bits allocated for the integer. This limited our search space to between 1 and 10 integer bits, where we found using more than 5 integer bits didn't provide a greater post-PTQ accuracy.

![Graph to show how the number of integer bits affect the accuracy.](labs_media/tut3_integer_bits.png)

Graph to show how the number of integer bits affect the accuracy.

Having determined the minimum number of bits required to preserve integer precision, we could allocate up to all 27 remaining bits for fractional precision. When search that space, we found our first peak of post-QAT accuracy at 5 fractional bits, with little improvement from beyond even 3 fractional bits of precision - meaning we could essentially use a Q5.3 if really pressed by memory constraints.

![Graph to show how the number of integer bits affect the accuracy.](labs_media/tut3_fractional_bits.png)

Graph to show how the number of fractional bits affect the accuracy.

Overall though, from the graphs, we can see that QAT really improves our accuracy across the board, especially so with lower number of integer bits, so much so that it can really seem agnostic to the number of integer bits - from 2 up to 7 bits we see less than a percentile improvement.

## Tutorial 4

### Key Definitions and Summary

Pruning is used to reduce the size and complexity of the neural networks we build, by removing parameters or structural components. The goal is to produce a more efficient model that maintains a lot of the original model's performance.

- Structured Pruning: removes whole structures from the network.
- Unstructed Pruning: removes individual weights or connections from the network.

![Graph to show how Random and L1-Norm pruning affect accuracy.](labs_media/tutorial_4_output.png)

Graph to show how Random and L1-Norm pruning affect accuracy.

As sparsity increases, accuracy drops for both methods, reflecting the loss of model capacity as a larger fraction of parameters are removed. We can see that L1-Norm pruning is significantly more robust, with higher accuracy than random pruning for all values of sparsity.

Random pruning leads to a rapid drop in performance beyond sparsity 0.5, with accuracy collapsing to near random choice at sparsities of 0.7 and above. This indicates that random removal of weights quickly disrupts critical model structure.

L1-norm pruning has a higher accuracy for all sparsity levels. Performance drops more gradually, remaining somewhat stable up to sparsity 0.7 before a sharp decline at 0.9. This suggests that magnitude based pruning is more effective in preserving informative parameters by preferentially removing weights with lower contribution.

Finetuning shows a massive recovery for random pruning at lower sparsity levels (0.1-0.5) but L1-norm requires less finetuning help until very high levels of sparsity levels, at which point it presents moderate recovery.