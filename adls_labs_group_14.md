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

We then try to restore the accuracy from before quantization using QAT (Quantization-Aware Training). This is 

![Graph to show the change of accuracy against number of fractional bits](labs_media/tutorial3_output.png)


## Tutorial 4

![Alt text](labs_media/tutorial4_output.png)

