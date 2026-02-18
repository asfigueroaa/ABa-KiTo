# ABa-KiTo: Agent-Based Kinetics via Topology

This repository contains Python and Julia code, and Jupyter Notebooks for reproducing the results presented in the manuscript: *Green Growth Meets Koopman: A Data-Driven Understanding of Economic Green Transitions in an Agent-Based Model* ( https://doi.org/10.5281/zenodo.18671445 )

# ABa-KiTo

The computational framework ABa-KiTo extends the MokiTo framework (Molecular Kinetics via Topology, for more details see https://arxiv.org/abs/2412.20580) to ABM data. 

The framework involves three stages:

1. Exploration of system state space.
   
2. Construction of the χ-function with the ISOKANN algorithm. We use the Julia package ISOKANN.jl (https://axsk.github.io/ISOKANN.jl/dev/)

3. Clustering of simulation data filtered by the associated χ-function to obtain clusters of states that are dynamically close, i.e., not only in terms of χ-values but also spatially close. A second
part of this step is an edge assignment to construct a graph representation which captures the system’s topological structure. The χ-function serves as ordering parameter that highlights the
dominant kinetic pathways between macrostates in this graph.
