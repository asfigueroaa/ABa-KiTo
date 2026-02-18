# ABa-KiTo

Agent-Based Kinetics via Topology (ABa-KiTo) is a computational framework that extracts and analyses topological insights to better understand emergent phenomena of agent-based model simulations of complex socio-economic systems. This computational framework extends the MoKiTo framework (Molecular Kinetics via Topology).

The framewrork involves three stages:

1. Exploration of system state space.
   
2. Construction of the χ-function with the ISOKANN algorithm. We use the Julia package ISOKANN.jl .

3. Clustering of simulation data filtered by the associated χ-function to obtain clusters of states that are dynamically close, i.e., not only in terms of χ-values but also spatially close. A second part of this step is an edge assignment to construct a graph representation which captures the system’s topological structure. The χ-function serves as ordering parameter that highlights the dominant kinetic pathways between macrostates in this graph.



In particular, this package contains the workflow for the analysis f the ABM analysed in the article: Green Growth Meets Koopman: A Data-Driven Understanding of Economic Green Transitions in an Agent-Based Model (https://doi.org/10.5281/zenodo.18671445)

For full dataset visit: 
