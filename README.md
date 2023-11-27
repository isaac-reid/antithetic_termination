# antithetic_termination

Repo to accompany the paper 'Quasi-Monte Carlo Graph Random Features', to appear at NeurIPS 2023 as a spotlight paper (https://arxiv.org/abs/2305.12470). We present a novel quasi-Monte Carlo (QMC) mechanism to improve the convergence of the recently-introduced class of graph random features (https://arxiv.org/pdf/2305.00156.pdf), which induces negative correlations between the lengths of random walks on a graph. We call it *antithetic termination*. To our knowledge, this is the first rigorously-studied QMC scheme for kernels defined on combinatorial objects.

`<q-GRFs_implementation.ipynb>` reproduces Fig. 2, the central result of the paper which shows that antithetic termination indeed improves the quality of Monte Carlo estimates of the graph kernel compared to i.i.d. sampling.  

