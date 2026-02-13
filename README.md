![Neurons](https://github.com/user-attachments/assets/57dfeed2-29db-4b3e-b6b8-879f42c209b4)

Sourced Neuron is a comprehensive Python library designed to advance computational neuroscience through a structured, multi-scale modeling framework inspired by Sourceduty’s science principles. The library integrates biophysical realism with adaptive systems modeling, allowing researchers to simulate neuronal dynamics ranging from single ion-channel kinetics to emergent network-level criticality. By combining deterministic conductance-based equations with stochastic input modeling and adaptive parameter tuning, Sourced Neuron provides a unified platform for constructing predictive neural simulations. The architecture is modular, enabling users to transition seamlessly between simplified integrate-and-fire abstractions and detailed multi-compartment Hodgkin–Huxley–style formulations. Each function is engineered to promote reproducibility, parameter transparency, and experimental alignment, making the library suitable for theoretical exploration, data-driven fitting, and hypothesis testing.

![Neuron Models](https://github.com/user-attachments/assets/e25a537d-030e-4c2e-a0f2-626142c2052a)

Beyond individual neuron simulation, Sourced Neuron emphasizes network-level intelligence modeling, incorporating plasticity mechanisms such as Hebbian learning and spike-timing-dependent plasticity, dynamic energy consumption estimation, oscillatory analysis, and attractor state detection. Advanced analytical tools allow researchers to measure synchronization, entropy, and system criticality, bridging cellular biophysics with large-scale neural computation theory. The inclusion of parameter sweeps, sensitivity analysis, adaptive time-stepping, and optimization routines ensures that models remain both computationally efficient and biologically grounded. Designed for extensibility and research-grade simulation, Sourced Neuron aims to support neuroscientists, AI researchers, and computational modelers in building scalable, interpretable, and experimentally relevant models of neural systems.

Functions:
--------------

1.  `InitNeu(params)`: This function initializes neuron state variables and core parameters.
2.  `UpdVolt(state)`: This function updates membrane potential dynamics.
3.  `IonDyn(state)`: This function simulates ion channel conductance behavior.
4.  `UpdWgt(weights)`: This function modifies synaptic weights.
5.  `DetSpike(state)`: This function detects spike threshold crossings.
6.  `RstSpike(state)`: This function resets membrane potential after a spike.
7.  `LeakCur(voltage)`: This function computes passive leak current.
8.  `NaCur(voltage)`: This function calculates sodium current flow.
9.  `KCur(voltage)`: This function calculates potassium current flow.
10. `CaCur(voltage)`: This function calculates calcium current flow.
11. `UpdGate(vars)`: This function integrates gating variable kinetics.
12. `StepSim(state)`: This function advances the simulation by one timestep.
13. `PoisIn(rate)`: This function generates Poisson-distributed spike input.
14. `SynDel(spikes)`: This function applies synaptic transmission delays.
15. `SynCur(weights)`: This function computes synaptic current.
16. `HebbUpd(activity)`: This function applies Hebbian plasticity updates.
17. `StdpUpd(timing)`: This function performs spike-timing-dependent plasticity.
18. `HomeoScale(weights)`: This function applies homeostatic synaptic scaling.
19. `FireRate(spikes)`: This function calculates firing rate over time.
20. `BurstDet(spikes)`: This function detects bursting patterns.
21. `ISIComp(spikes)`: This function computes interspike intervals.
22. `RefracDyn(state)`: This function enforces refractory period dynamics.
23. `InjCur(state)`: This function injects external current stimulus.
24. `BuildNet(topology)`: This function constructs a neural network structure.
25. `NetStep(network)`: This function simulates one network activity step.
26. `ConnMat(nodes)`: This function generates a connectivity matrix.
27. `NoiseAdd(state)`: This function adds stochastic membrane noise.
28. `LfpEst(network)`: This function estimates local field potential signals.
29. `PhaseSync(spikes)`: This function measures phase synchronization.
30. `OscDet(signal)`: This function detects oscillatory activity bands.
31. `EnergyUse(state)`: This function estimates metabolic energy consumption.
32. `FitModel(data)`: This function optimizes model parameters to fit data.
33. `ParamSweep(ranges)`: This function performs parameter space exploration.
34. `SensAnal(model)`: This function evaluates parameter sensitivity.
35. `AdaptDT(state)`: This function adjusts timestep dynamically.
36. `SaveSim(data)`: This function stores simulation results.
37. `LoadSim(file)`: This function loads stored simulation data.
38. `PlotVolt(trace)`: This function visualizes membrane voltage traces.
39. `PlotRaster(spikes)`: This function generates raster plots.
40. `TransFunc(signal)`: This function computes neuron transfer function.
41. `Entropy(spikes)`: This function estimates spike train entropy.
42. `CritDetect(network)`: This function identifies critical network dynamics.
43. `AttrState(network)`: This function analyzes attractor states.
44. `MultiComp(geometry)`: This function simulates multi-compartment neurons.
45. `DendInteg(inputs)`: This function models dendritic integration.
46. `AxonProp(spike)`: This function simulates axonal spike propagation.
47. `GliaLink(state)`: This function models neuron–glia coupling.
48. `NtRelease(spike)`: This function simulates neurotransmitter release.
49. `PlastStable(weights)`: This function assesses plasticity stability.
50. `ExportJSON(model)`: This function exports model configuration to JSON.

Prototype Lib
--------------

Sourced Neuron is a prototype PyPI library designed to provide a modular, research-oriented framework for computational neuron modeling grounded in Sourceduty’s science principles. As an experimental package, it integrates biophysical realism with scalable abstraction layers, allowing users to move seamlessly from single-compartment membrane models to multi-compartment morphologies and emergent network dynamics within a unified API. The library is structured around concise, high-efficiency functions that encapsulate membrane potential updates, ion channel kinetics, synaptic plasticity mechanisms (including Hebbian and spike-timing-dependent rules), stochastic input generation, and attractor-state analysis, while also supporting parameter sweeps, sensitivity testing, entropy estimation, and criticality detection. Designed for compatibility with standard scientific Python ecosystems (NumPy, SciPy, Matplotlib), Sourced Neuron emphasizes reproducibility, transparency of parameters, and computational efficiency, making it suitable for rapid prototyping, hypothesis testing, and educational exploration. As a prototype PyPI release, it focuses on clean architecture, extensibility, and clear documentation standards, encouraging community-driven expansion toward GPU acceleration, dataset integration, and interoperability with established neuroscience simulators while maintaining a lightweight core optimized for conceptual clarity and experimental modeling workflows.

--------------
https://chatgpt.com/g/g-675f752981348191a84d20f6f15cfb2b-neuron-modelling
<br>
https://sourceduty.com/
