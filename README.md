![Neurons](https://github.com/user-attachments/assets/57dfeed2-29db-4b3e-b6b8-879f42c209b4)

Sourced Neuron is a comprehensive Python library designed to advance computational neuroscience through a structured, multi-scale modeling framework inspired by Sourceduty’s science principles. The library integrates biophysical realism with adaptive systems modeling, allowing researchers to simulate neuronal dynamics ranging from single ion-channel kinetics to emergent network-level criticality. By combining deterministic conductance-based equations with stochastic input modeling and adaptive parameter tuning, Sourced Neuron provides a unified platform for constructing predictive neural simulations. The architecture is modular, enabling users to transition seamlessly between simplified integrate-and-fire abstractions and detailed multi-compartment Hodgkin–Huxley–style formulations. Each function is engineered to promote reproducibility, parameter transparency, and experimental alignment, making the library suitable for theoretical exploration, data-driven fitting, and hypothesis testing.

![Neuron Models](https://github.com/user-attachments/assets/e25a537d-030e-4c2e-a0f2-626142c2052a)

Beyond individual neuron simulation, Sourced Neuron emphasizes network-level intelligence modeling, incorporating plasticity mechanisms such as Hebbian learning and spike-timing-dependent plasticity, dynamic energy consumption estimation, oscillatory analysis, and attractor state detection. Advanced analytical tools allow researchers to measure synchronization, entropy, and system criticality, bridging cellular biophysics with large-scale neural computation theory. The inclusion of parameter sweeps, sensitivity analysis, adaptive time-stepping, and optimization routines ensures that models remain both computationally efficient and biologically grounded. Designed for extensibility and research-grade simulation, Sourced Neuron aims to support neuroscientists, AI researchers, and computational modelers in building scalable, interpretable, and experimentally relevant models of neural systems.

Functions:
--------------

#### CORE LIFECYCLE & INITIALIZATION
1.  `InitNeu(params)`: This function initializes neuron state variables and core parameters.
2.  `SetWeight(w)`: Configures the initial synaptic weight values for incoming signals.
3.  `SetBias(b)`: Establishes the threshold offset for the neuron's firing activation.
4.  `GetState()`: Retrieves the current electrochemical or mathematical state of the unit.
5.  `ResetNeu()`: Clears all temporal data and returns the neuron to a resting state.
6.  `InitXavier()`: Implements Xavier Initialization to keep signal variance consistent.
7.  `InitHe()`: Specifically scales weights for neurons using ReLU-based activations.
8.  `CopyNeu()`: Creates a deep copy of the neuron for genetic algorithms.
9.  `SetID(name)`: Assigns a unique identifier to the neuron for tracking.
10. `CheckHealth()`: Validates that all internal pointers and arrays are intact.
11. `CloneState()`: Transfers the current potential and state to a new instance.
12. `Version()`: Returns the current build version of the Sourced_Neuron lib.
13. `Shutdown()`: Safely closes all data streams and saves the final state.
14. `MainLoop()`: The master function that orchestrates all sub-processes.

#### SIGNAL PROCESSING & MATHEMATICS
15. `ProcInput(data)`: The primary function for receiving and scaling raw input vectors.
16. `SumSignals()`: Performs the weighted summation of all current dendritic inputs.
17. `FireImpulse()`: Executes the output transmission if the internal threshold is met.
18. `Integrate(dt)`: Performs temporal integration for continuous-time simulations.
19. `Leakage(rate)`: Simulates the natural decay of potential in a resting neuron.
20. `SynapticDelay()`: Introduces a time lag between input reception and summation.
21. `ScaleInput(s)`: Multiplies all incoming signals by a constant scaling factor.
22. `BiasShift(s)`: Shifts the activation function along the x-axis.
23. `PulseSync()`: Aligns the firing cycle with a global system clock.
24. `ParallelSum()`: Uses multi-threading to speed up input summation.

#### ACTIVATION FUNCTIONS (NON-LINEARITY)
25. `ActReLU(x)`: Applies the Rectified Linear Unit activation to introduce non-linearity.
26. `ActSigmoid(x)`: Normalizes the output signal between a range of 0 and 1.
27. `ActTanh(x)`: Maps inputs to a range between -1 and 1 for zero-centered data.
28. `ActSoftmax(v)`: Converts a vector of values into a probability distribution.
29. `ActLeaky(x, a)`: A modified ReLU that prevents "dying neurons" by allowing a small gradient.
30. `ActELU(x, a)`: Exponential Linear Unit for faster learning and better noise handling.
31. `ActSwish(x)`: A self-gated activation function optimized for deep architectures.
32. `PulseWidth(t)`: Modulates the duration of an impulse in spiking neuron models.
33. `StepFunc()`: A binary activation function for simple logic-gate simulations.
34. `LinearFunc()`: Passes the input directly to output without transformation.
35. `SoftPlus(x)`: A smooth approximation of the ReLU function.
36. `MishAct(x)`: A self-regularized non-monotonic activation function.

#### LEARNING & OPTIMIZATION
37. `CalcError(tar)`: Measures the difference between the current output and target value.
38. `BackProp(err)`: Calculates the gradient of the loss function regarding weights.
39. `UpdWeight(lr)`: Adjusts internal weights based on the calculated gradient and learning rate.
40. `UpdBias(lr)`: Updates the bias parameter to refine the firing threshold over time.
41. `SetLR(rate)`: Dynamically adjusts the learning rate parameter for the optimizer.
42. `DecayLR(step)`: Reduces the learning rate over time to converge on a global minimum.
43. `ClipGrad(val)`: Prevents "exploding gradients" by capping the maximum gradient value.
44. `L1Reg(lambda)`: Adds Lasso regularization to encourage sparse weight matrices.
45. `L2Reg(lambda)`: Adds Ridge regularization to prevent excessively large weights.
46. `MomUpdate(v)`: Applies momentum to weight updates to bypass local minima.
47. `AdamOpt()`: Implements the Adam Optimization logic.
48. `RMSProp()`: Normalizes the gradient using a moving average of squared gradients.
49. `AdaGrad()`: Adjusts learning rates based on the frequency of parameter updates.
50. `GetGrads()`: Exports the current gradient values for external analysis.
51. `ResetGrads()`: Zeroes out gradients before a new training iteration.
52. `SetTarget(t)`: Defines the ground-truth value for supervised learning.
53. `BatchSize(n)`: Sets how many inputs are processed before a weight update.

#### TOPOLOGY & CONNECTIVITY
54. `SynapsePrune()`: Removes low-weight connections to optimize network efficiency.
55. `Inhibit(n2)`: Logic for one neuron to actively suppress the firing of another.
56. `Excite(n2)`: Logic for increasing the membrane potential of a target neuron.
57. `GetLayer()`: Returns the index of the layer this neuron belongs to.
58. `BindTo(peer)`: Establishes a permanent synaptic link to a specific peer neuron.
59. `Unbind(peer)`: Severs the link between the neuron and a peer unit.
60. `GetSynCount()`: Returns the total number of active synaptic connections.
61. `MapTopology()`: Returns the spatial coordinates of the neuron in a 3D grid.
62. `DistanceTo(n)`: Calculates the Euclidean distance to another neuron.
63. `PruneOrphan()`: Automatically deletes the neuron if it has no connections.

#### VALIDATION & METRICS
64. `CheckNaN()`: Scans for "Not a Number" errors in the weight or output matrices.
65. `CheckLimit()`: Ensures weights do not exceed pre-defined safety boundaries.
66. `PeakVoltage()`: Records the highest potential reached during a cycle.
67. `FreqCheck()`: Calculates the firing frequency over a set time window.
68. `LogLoss()`: Calculates the logarithmic loss for classification tasks.
69. `MSELoss()`: Computes the Mean Squared Error for regression tasks.
70. `HuberLoss()`: Applies a loss function that is less sensitive to outliers.
71. `SetMaxWeight()`: Clips weights to a specific maximum absolute value.
72. `SetMinWeight()`: Sets a floor for weights to prevent dead connections.
73. `IsSaturated()`: Detects if the neuron is stuck at the limits of its activation.
74. `VerifyArch()`: Cross-references current params against the master architecture.
75. `CheckDep()`: Verifies that required libraries like NumPy are installed.

#### DATA PERSISTENCE & I/O
76. `SaveModel(path)`: Serializes the current neuron parameters to a file.
77. `LoadModel(path)`: Restores weights and biases from a saved configuration file.
78. `LogActivity()`: Records firing history and internal states for debugging.
79. `StreamIn()`: Opens a data buffer for real-time sensor or stream input.
80. `FlushQueue()`: Clears the input buffer to prevent data overflow.
81. `WatchWeight()`: Attaches a listener to detect sudden changes in weight values.
82. `LatentState()`: Accesses hidden variables not exposed to the output layer.
83. `ExportJSON()`: Formats neuron data into a JSON string for web usage.
84. `ImportCSV()`: Loads a weight matrix from a standard CSV file.
85. `TracePath()`: Follows the signal flow through the neuron for auditing.

#### PERFORMANCE & HARDWARE
86. `DropOut(p)`: Randomly deactivates the neuron during training to prevent overfitting.
87. `NormBatch()`: Standardizes the inputs across a batch to accelerate training.
88. `PlotNeuron()`: Generates a visual representation of the neuron's current behavior.
89. `Refractory(p)`: Implements a "cool-down" period where the neuron cannot fire.
90. `SetPrecision()`: Toggles between float32 and float16 for memory optimization.
91. `MutateWeight()`: Introduces random noise into weights for evolutionary testing.
92. `Crossover(n2)`: Blends parameters with another neuron to create a "child" unit.
93. `IsFiring()`: Boolean check to see if the neuron is currently in an active state.
94. `HeatMap()`: Outputs a matrix of synaptic strengths for visualization.
95. `NoiseGen(std)`: Injects Gaussian noise to improve model robustness.
96. `AvgEnergy()`: Estimates the computational cost or power usage of the neuron.
97. `EnableGPU()`: Offloads mathematical operations to NVIDIA CUDA kernels.
98. `SharedMem()`: Configures neurons to share a memory space for weights.
99. `UnitTests()`: Runs a suite of internal tests to ensure functional integrity.
100. `Benchmark()`: Measures the time taken to process 1,000,000 signals.

Prototype Lib
--------------

Sourced Neuron is a prototype PyPI library designed to provide a modular, research-oriented framework for computational neuron modeling grounded in Sourceduty’s science principles. As an experimental package, it integrates biophysical realism with scalable abstraction layers, allowing users to move seamlessly from single-compartment membrane models to multi-compartment morphologies and emergent network dynamics within a unified API. The library is structured around concise, high-efficiency functions that encapsulate membrane potential updates, ion channel kinetics, synaptic plasticity mechanisms (including Hebbian and spike-timing-dependent rules), stochastic input generation, and attractor-state analysis, while also supporting parameter sweeps, sensitivity testing, entropy estimation, and criticality detection. Designed for compatibility with standard scientific Python ecosystems (NumPy, SciPy, Matplotlib), Sourced Neuron emphasizes reproducibility, transparency of parameters, and computational efficiency, making it suitable for rapid prototyping, hypothesis testing, and educational exploration. As a prototype PyPI release, it focuses on clean architecture, extensibility, and clear documentation standards, encouraging community-driven expansion toward GPU acceleration, dataset integration, and interoperability with established neuroscience simulators while maintaining a lightweight core optimized for conceptual clarity and experimental modeling workflows.

--------------
https://chatgpt.com/g/g-675f752981348191a84d20f6f15cfb2b-neuron-modelling
<br>
https://sourceduty.com/
