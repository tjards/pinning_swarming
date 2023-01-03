# Adaptive Pinning Control using Network Controlability Gramian This project implements adaptive pinning control on a network of dynamic agents. Energy efficient pins are automatically selected using the Network Controlability Gramian.We consider the network as a Graph, $G=${ $V$, $E$ } such that:- Vertices ( $V$ ) are the agents (nodes)- Edges ( $E$ ) are a set of links between the nodes in the form of ordered pairs from the Cartesian Product $V$ $\times$ $V$, or *E=* {*(a,b)* such that *a* is in *V* and *b* is in *V*}- *G* is simple, or *(a,a)* not in *E* for all *a* in *V*  - *G* is undirected, or *(a,b)* in *E* <=> *(b,a)* in *E*- Nodes *i* and *j* are neighbours if they share an edge- Adjacency matrix (*A={a_ij}*) with dimensions *NxN* describes the connections between nodes where *a_ij* is 1 if *i* and *j* are neighbours or 0 otherwise# PlotsBelow is are plots with adaptive pins, selected to conserve energy using the Network Controlability Gramian:<p float="center">    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_10.gif" width="45%" />    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/convergence_pin_10.png" width="45%" />    </p><p float="center">    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_9.gif" width="45%" />    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/convergence_pin_9.png" width="45%" />    </p># References 1. Kléber M. Cabral, Sidney N. Givigi, and Peter T. Jardine, [Autonomous assembly of structures using pinning control and formation algorithms](https://ieeexplore-ieee-org.proxy.queensu.ca/document/9275901) in 2020 IEEE International Systems Conference (SysCon), 07 Dec 20202. Erfan Nozari, Fabio Pasqualetti, and Jorge Cortes,[Heterogeneity of Central Nodes Explains the Benefits of Time-Varying Control in Complex Dynamical Networks](https://arxiv.org/abs/1611.06485) in arXiv, 26 Dec 20183. Fabio Pasqualetti, Sandro Zampieri, and Francesco Bullo, [Controllability Metrics, Limitations and Algorithms for Complex Networks](https://ieeexplore-ieee-org.proxy.queensu.ca/stamp/stamp.jsp?tp=&arnumber=6762966) in IEEE Transactions on Control of Network Systems, 11 Mar 20144. The code is opensource but, if you reference this work in your own reserach, please cite me. I have provided an example bibtex citation below:`@techreport{Jardine-2022,  title={Adaptive Pinning Control using Network Controlability Gramian},  author={Jardine, P.T.},  year={2022},  institution={Royal Military College of Canada, Kingston, Ontario},  type={GitHub Code Repository},}`5. Alternatively, you can cite any of my related papers, which are listed in [Google Scholar](https://scholar.google.com/citations?hl=en&user=RGlv4ZUAAAAJ&view_op=list_works&sortby=pubdate). 