
# TensorFlow Implementation of Kolmogorov-Arnold Network (KAN)

*Update,* see [KANX Repo](https://github.com/Mattral/KANX/) for the maintained library implementation.

---

#### Introduction
The provided implementation includes a customizable neural network architecture based on Kolmogorov-Arnold Networks (KANs), 
utilizing TensorFlow's API. KANs aim to efficiently approximate multivariate functions by employing nonlinear transformations 
with fewer parameters compared to traditional deep neural networks.

#### Modules and Dependencies
- **TensorFlow:** Main library providing tools for machine learning and neural network construction.

#### Classes and Functions

##### `KANLinear` Layer
- **Description:** Custom TensorFlow layer implementing a linear transformation followed by a B-spline transformation as part of a KAN.
- **Parameters:**
  - `in_features`: Integer, number of input features.
  - `out_features`: Integer, number of output features.
  - `grid_size`: Integer, number of grid points for B-spline basis.
  - `spline_order`: Integer, order of the spline (degree is `spline_order - 1`).
  - `activation`: String, activation function to use after summing base and spline outputs.
  - `regularization_factor`: Float, factor for L2 regularization.
  - `grid_range`: Tuple, range of the grid used in B-spline transformation.
- **Methods:**
  - `build_grid`: Initializes the grid used for B-spline transformations.
  - `call`: Computes the output of the layer using both linear and spline transformations.
  - `compute_spline_output`: Calculates the output from the spline transformation.

##### `B_batch_tf` Function
- **Description:** Computes B-spline basis values for input values using a specified grid and order.
- **Parameters:**
  - `x`: TensorFlow Tensor, input values.
  - `grid`: TensorFlow Tensor, grid points for the splines.
  - `k`: Integer, order of B-spline.
  - `extend`: Boolean, whether to extend the grid to handle boundaries.
- **Returns:** TensorFlow Tensor of B-spline basis values.

##### `extend_grid_tf` Function
- **Description:** Extends a given grid by a specified number of points on both ends.
- **Parameters:**
  - `grid`: TensorFlow Tensor, original grid points.
  - `k_extend`: Integer, number of points to extend on each side.
- **Returns:** Extended grid.

##### `KAN` Class
- **Description:** Sequential model that aggregates multiple `KANLinear` layers to form a complete KAN.
- **Parameters:**
  - `layers_configurations`: List of dictionaries, configurations for each `KANLinear` layer.

##### `get_activations` Function
- **Description:** Utility function to fetch activations from a specified layer in the model.
- **Parameters:**
  - `model`: TensorFlow model from which to fetch activations.
  - `model_inputs`: Input data to the model.
  - `layer_name`: Optional name of the layer to specifically fetch activations.
- **Returns:** Activations from the specified layer or all layers if none specified.

#### Notes and Improvements
1. **Error Handling:** Consider adding error handling for potential issues with input types and values.
2. **Efficiency:** Analyze and optimize the computation of B-spline basis, which can be critical for performance.
3. **Documentation:** Ensure each method and function is accompanied by comprehensive docstrings in the code.

### Conclusion
This documentation provides an overview and detailed explanation of each component in the TensorFlow implementation of KAN. 
For practical use, ensure proper testing and validation of the functions, especially around the numerical stability of the B-spline calculations.

# Kolmogorov-Arnold Networks (KANs) Overview

## Introduction
- **Kolmogorov-Arnold Networks (KANs)** represent a novel neural network architecture inspired by the Kolmogorov-Arnold representation theorem.
- They differ from traditional Multi-Layer Perceptrons (MLPs) by featuring learnable activation functions on edges instead of fixed activation functions on nodes.

## How KANs Work
- **Node Functionality:** Nodes in KANs sum incoming signals without applying non-linearities.
- **Edge Functionality:** Edges contain spline-based learnable activation functions, allowing for precise local adjustments and optimization of univariate functions.

## Advantages of KANs
- **Accuracy and Interpretability:** KANs can optimize both compositional structures and univariate functions, leading to improved accuracy and interpretability.
- **Flexibility with Functions:** They are particularly adept at modeling complex, low-dimensional functions accurately.

## Challenges
- **Training Speed:** KANs currently train significantly slower than MLPs, identified as an engineering challenge that may be optimized in future developments.

## Implications and Potential Applications
- **Efficiency:** KANs could potentially create more compact and efficient models, reducing the computational expense.
- **Interpretability:** The learnable activation functions enhance the interpretability of the models, crucial for applications requiring transparency, like healthcare.
- **Few-shot Learning:** KANs might outperform existing architectures in learning from fewer examples.
- **Knowledge Representation and Reasoning:** They could potentially enhance the ability of models to represent and manipulate complex, structured knowledge.
- **Multimodal Learning:** KANs could lead to more effective and efficient multimodal models by leveraging their ability to learn and optimize compositional structures.

## Conclusion
- **Significance:** Kolmogorov-Arnold Networks mark a significant step forward in neural network design, promising to advance the capabilities and applications of machine learning models.
- **Future Research:** Ongoing research will likely focus on overcoming the current limitations and expanding the practical applications of KANs.


## Contributions and Feedback

Bug reports, clarifications, and experimental extensions are welcome.
If you are exploring KAN interpretability or visualization, contributions in that direction are especially encouraged.

