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

LoRA allows us to achieve high accuracy whilst limiting memory usage, allowing control of parameters whilst fine-tuning memory hungry models like LLMs. Note that LoRA does not change inference time, as the low-rank matrices are merged back in, leaving the model with the same dimensions as before LoRA is applied.

$$
y = X (W + AB) + b
$$

All we do is change the lower rank matrices A and B, freezing the larger matrix W.

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

Graph to show how the number of integer bits affect the accuracy. All these tests for integer precision were conducted with the fractional width left at 4. 

Having determined the minimum number of bits required to preserve integer precision (5), we could allocate up to all 27 remaining bits for fractional precision. When search that space, we found our first peak of post-QAT accuracy at 5 fractional bits, with little improvement from beyond even 3 fractional bits of precision.

![Graph to show how the number of fractional bits affect the accuracy.](labs_media/tut3_fractional_bits.png)

Graph to show how the number of fractional bits affect the accuracy.

Overall though, from the graphs, we can see that QAT really improves our accuracy across the board, especially so with lower number of integer bits, so much so that it can really seem agnostic to the number of integer bits - from 2 up to 7 bits we see less than a percentile improvement.

We can conclude that due to the significant accuracy improvements post-QAT, even 1 integer bit only loses us two percent accuracy, so using an 8-bit wide, Q3.5 format is probably a good mix of precision and speed/storage savings.

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


# Lab 3

## Tutorial 6

### Task 1: Per-Layer Bit Width Search

In the original template code, all `LinearInteger` layers use the same hardcoded configuration (width=8, frac_width=4). We modified the search to let Optuna choose width [8, 16, 32] and fractional width [2, 4, 8] independently for each layer. This configuration is as requested by the question. A graph of the improving model accuracy is shown below. On each Optuna trial, if the accuracy of the configuration is higher, the best configuratin of the best model so far is updated. At the end only the best model is retained.

![Graph to show improving model accuracies as Optuna tries more models.](labs_media/tut6_task1_search_progress.png)

### Task 2: Multi-Precision Type Search

As requested in the question, the cnofiguration search is then extended to include multiple precision types: `Integer`, `MinifloatDenorm`, `MinifloatIEEE`, and `Log` etc. The below graph shows the performance of these different precision types.

![Graph to show improving model accuracies as Optuna tries more models.](labs_media/tut6_task2_search_progress.png)

`MinifloatDenorm` achieves the best accuracy (86.02%), with `MinifloatIEEE` close behind (85.99%). To consider the case where a model could use a mix of all of these strategies, Optuna was configured to search a mix of these precision types. To give this run a higher chance of reaching an optimal model, 50 trials were used for this larger search space. Shown below is the graph of the improving accuracies against the trial index. 

![Graph to show improving model accuracies as Optuna tries more models.](labs_media/tut6_task2_mixed_search_progress.png)

Lets visualise the distribution of these different layer types in the final chosen best model:

![Graph to show improving model accuracies as Optuna tries more models.](labs_media/tut6_task2_mixed_layer_dist.png)

# Lab 4

## `torch.compile`

This makes PyTorch models run faster by optimising the model and input data. It is a JiT compiler that optimises for specific hardware. It will essentially compile from Python to machine code at runtime, analysing slow parts of the code that it can recompile as needed.

The three main components of this compiler are:

- TorchDynamo: Captures the PyTorch graph by hooking into CPython.
- AOT Autograd: Captures the backward pass (gradients) ahead of time.
- TorchInductor: The backend that generates the optimized code (often using OpenAI Triton for GPUs).


## Model Runtime

The runtime of the optimised model is higher, when run for 5 iterations. However, when the iteration count is increased to 50, the optimised model is indeed faster. 

This demonstrates that the compilation does indeed make the model run faster on average, but at 5 iterations the compilation overhead isn't outweighed by the model speedup.

| Hardware | Iterations | Baseline Runtime (s) | Optimised Runtime (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** | 5 | 1.8944 | 11.4648 | 0.17x |
| **CPU** | 50 | 1.5581 | 1.1766 | 1.32x |
| **CPU** | 200 | 1.6352 | 1.3449 | 1.22x |
| **GPU (RTX 4080)**| 5 | 0.0261 | 0.0205 | 1.27x |
| **GPU (RTX 4080)**| 50 | 0.0251 | 0.0205 | 1.22x |
| **GPU (RTX 4080)**| 200 | 0.0250 | 0.0206 | 1.21x |

On a GPU, the runtime does improve, but by a smaller amount, and the compilation benefits aren't scaled on increasing numbers of runs. This implies either that the baseline GPU performance was already highly efficient, leaving less room for JIT improvements.

## Kernel Fusion

This is advantageous as it reduces the number of memory accesses as well as the number of kernel launches, making numerous small operations cheaper by fusing them together. 

For example, a linear layer followed immediately by a ReLU operation on the data, need not be executed by two separate kernels (with a corresponding read and write back to global memory for each), and instead have it's intermediate values in registers whilst the ReLU is operated, saving bandwidth to the global memory on GPU.

PyTorch does provide these fused kernels for common operations, an example being SDPA (Scaled Dot-Product Attention).

| Hardware   |   Iterations |   Original Runtime (s) |   Fused Runtime(s) |   Speedup |
|:---------|-------------:|---------------:|------------:|----------:|
| CPU      |            5 |       0.005323 |    0.003605 |  1.476353 |
| CPU      |           50 |       0.005025 |    0.003576 |  1.405349 |
| CPU      |          200 |       0.005110 |    0.003851 |  1.326901 |
| GPU (RTX 4080)     |            5 |       0.000136 |    0.000071 |  1.920646 |
| GPU (RTX 4080)     |           50 |       0.000068 |    0.000019 |  3.607179 |
| GPU (RTX 4080)     |          200 |       0.000070 |    0.000019 |  3.611273 |

With kernel fusion on the GPU, we can clearly see the benefits in terms of speedup. The fused kernel reaches speedup of 3.6x as opposed to 1.3x on the CPU. This is because GPUs are memory-bound, with their operation speed limited by memory latency when reading/writing values from main memory.

On the GPU, amortization effects are stronger at higher iterations. At a low number of iterations, fixed overheads like CUDA kernel launch latency and Python dispatch make up a larger percentage of the total runtime.

The CPU does benefit from better cache locality due to the fused kernel, but the massive memory latency paid by GPUs isn't present, meaning the performance boost is a little more modest.

## Custom Kernels

These are the most powerful way to optimise code performance, but require a bit more legwork. We should write these kernels manually, when we want to implement a custom operation, as opposed to ones provided by PyTorch, Intel or NVIDIA as some examples.

Our custom kernel is for dequantization, making the quantized values back into floating-point values. 

### MXINT8 Quantization*


The CPU algorithm to convert from MXINT8 to BFloat16 is explained below.

```cpp
    for (int i = 0; i < M; ++i) {
        auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8; // The signed bit is the MSB of the 16-bit BFloat16 representation
        auto exp = static_cast<uint16_t>(hScales[i / group_size]) << 7; // Take the shared exponent, shift by 7 since you want to align 1 bit past sign bit
        auto mantissa_abs = abs(hX[i]); // Taking an unsigned 8-bit representation
        auto frac = static_cast<uint16_t>((mantissa_abs & 0x3F) << 1); // Since BFloat16 has an implicit leading bit, we take 6 bits (S | i | 6-bit fraction) (not the signed bit or the MS positive bit), then shift left by 1, since MXINT8 has a 6-bit fraction while BFloat16 has a 7-bit fraction
        auto out = cutlass::bfloat16_t::bitcast(sign | exp | frac); // Or all the bits
        auto dont_need_abs = bool(mantissa_abs & 0x40); // Check the value of the bit that is implcitely 1 in BFloat16
        auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0)); 
        y[i] = dont_need_abs ? out : out - bias; // Subtract the bias if the bit that is implicit is zero, since it will be 1 in BFloat16 (bias represents ±1.0 × 2^exp)
    }
```

If both the weights and activation are quantized to the format, the GPU is able to leave them in that format and compute the answer, as opposed to having to dequantize either one or the other, requiring more computation. With both weight and activation in integer form, the hardware is able to use a mathematical trick, performing the full operation in integer math, only using expensive floating point scales once at the end of the block.

With quantized values, the physical hardware (integer multipliers) take up less space on the silicon chip, saving area and allowing designers to fit more multipliers on the chip.

### dont_need_abs and bias

These are computed from the input value. They are used to determine if the MXINT8 value has a real leading one in its mantissa - that's what (mantissa_abs && 0x40) is doing (is our value 1.0.. or 0.0...). If so, we know we can allow the hardware to insert it's leading 1 as it would do for typical floating point values. If not, we make sure to subtract the bias (correctly signed, shifted, 1.00000) at the end, representing values smaller than 1 correctly.

### cta_tiler and local_tile

The `cta_tiler` is used to split up the representations of the full matrices into tiles, such that the work can be divided across CTAs (Compute Thread Arrays). We give a desired CTA tile size (shape), such that the `cta_tiler` can split up the tensors across the CTAs. 

This separates the global tensor into a grid of tiles. The `local_tile` function then selects a specific tile corresponding to the CTA's coordinates (like `blockIdx.x` and `blockIdx.y`). Finally the CTA can now use the view of that single tile to perform computation on.

This doesn't actually "copy" the data in a literal sense from global memory, instead using a pointer into global memory generated by the combination of `cta_tiler` and `local_tile`.

### layout_sX and local_partition

Having generated the tiles of global memory in the previous step, we want an efficient way to copy one tile of global memory to one tile of our shared memory. To speed this up, we can allow the threads in the CTA to copy their own subtensor of data.

We define the thread layout so we can partition the global memory tensors data and shared memory tensors. In this operation `local_partition` is a lot like `local_tile` except each thread gets one element of data assigned to it per thread tile.

This means that each thread can participate in the copy operation as they all own a different subtensor of the tile that is to be copied.

### Why is the saved GPU memory not what we expect?
Only the torch.nn.Linear is quantized
```python
for layer_name, layer in model.named_modules():
        if not isinstance(layer, torch.nn.Linear):
            continue
        if "classifier" in layer_name:
            continue
```
Specifically, the following were not quantized:
1. Encoder layers
2. Output layers (poolers, classifier, dropout)

Therefore a lot of the model is not quantized so it would not achieve the theoretical memory savings.
