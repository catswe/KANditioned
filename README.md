# KANditioned: Fast, Conditioned Training of Kolmogorov-Arnold Networks (KANs) via Lookup Interpolation ~~and Discrete Cosine Transform~~

Training is accelerated by orders of magnitude through exploiting the structure of the linear (C⁰) B-spline (see Fig. 1) with uniformly spaced control points. Because the intervals are uniform, evaluating spline(x) reduces to a constant-time index calculation, followed by looking up the two relevant control points and linearly interpolating between them. This contrasts with the typical summation over basis functions typically seen in splines, reducing the amount of computation required and enabling effective sublinear scaling across the control points dimension.

## Install

```
pip install kanditioned
```

## Usage
> [!IMPORTANT]  
> It is highly recommended to use this layer with torch.compile, which will provide very significant speedups (Triton kernel is coming sometimes later, but I found torch.compile to provide very satisfactory performance), in addition to a normalization layer before each KANLayer.

```python
from kanditioned.kan_layer import KANLayer

layer = KANLayer(in_features=3, out_features=3, init="random_normal", num_control_points=8)
layer.visualize_all_mappings(save_path="kan_mappings.png")
```
### Arguments

- **in_features** (int)  
  Size of each input sample.
- **out_features** (int)  
  Size of each output sample.
- **init** (str) – initialization method:  
  - `"random_normal"`: Slopes drawn from a normal distribution, then normalized so each “neuron” has unit weight norm.  
  - `"identity"`: Identity mapping (requires `in_features == out_features`). Output initially equals input.  
  - `"zero"`: All splines initialized to zero.
- **num_control_points** (int, default = `32`)  
  Number of uniformly spaced control points per input feature.
- **spline_width** (float, default = `4.0`)  
  Domain of the spline: `[-spline_width / 2, spline_width / 2]`. Beyond that, the spline will linearly extrapolate.

#### Methods:

    visualize_all_mappings(save_path=path[optional]) - this will plot out the shape of each spline and its corresponding input and output feature

## Figure

![Linear B-spline example](https://raw.githubusercontent.com/cats-marin/KANditioned/main/image-1.png)

**Figure 1.** Linear B-spline example (each triangle-like shape is a basis):

## Roadmap
- ~~Use F.embedding_bag~~
- Update package with cleaned up, efficient Discrete Cosine Transform (with rank-2 correction) and parallel scan (prefix sum) parameterizations.
    - Both provide isotropic O(1) condition scaling for the discrete second difference penalty, as opposed to O(N^4) conditioning for the naive B-spline parameterization. This only matters if you care about regularization.
    - May add linearDCT variant first. Although it's O(N^2), it's more parallelized and optimized on GPU for small N since it's essentially a matmul with weight being a DCT matrix
- Proper baselines against MLP and various other KAN implementations on backward and forward passes
    <!-- - https://github.com/ZiyaoLi/fast-kan -->
    <!-- - https://github.com/Blealtan/efficient-kan -->
    <!-- - https://github.com/1ssb/torchkan -->
    <!-- https://github.com/quiqi/relu_kan -->
    <!-- https://github.com/Jerry-Master/KAN-benchmarking -->
    <!-- https://github.com/KindXiaoming/pykan -->
    <!-- https://github.com/mintisan/awesome-kan -->
- Add sorting on indices as an option (potentially radix sort, which is common optimization on embedding) to improve speed through coalesced access
- Add in feature-major input variant
- Run benchmarks and further optimize memory locality
    - Feature-major input variant versus batch-major input variant
    - Interleaved indices [l1, u1, l2, u2, ...] versus stacked indices [l1, l2, ..., u1, u2, ...]
- Add optimized Triton kernel
- Update visualize_all_mappings method to something like .plot with option for plotting everything
- Research adding Legendre polynomials parameterization
    - Preliminary: does not seem to offer much benefits or have isotropic penalty conditioning
- Polish writing

## Open To Collaborators. Contributions Are Welcomed!

## LICENSE
This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
