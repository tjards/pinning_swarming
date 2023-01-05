# Adaptive Pinning Control using Network Controlability Gramian Note: this implementation has been incorporated into a larger project [here](https://github.com/tjards/swarming_sim).This project implements adaptive pinning control on a network of dynamic agents. Energy efficient pins are automatically selected using the Network Controlability Gramian.We consider the network as a **Graph**, $G=${ $V$, $E$ } such that:- **Vertices** ( $V$ ) are the agents (nodes)- **Edges** ( $E$ ) are a set of links in the form of ordered pairs - Edges are defined by the Cartesian Product $V \times V$, $E=$ {( $a$ , $b$ ) such that $a \in V$ and $b \in V$}- $G$ is simple, or ( $a$ , $a$) $\notin E~\forall~a \in V$  - $G$ is undirected, or ( $a$ , $b$ ) $\in E <=>$ ( $b$ , $a$) $\in E$- Nodes $i$ and $j$ are **neighbours** if they share an edge- **Adjacency** matrix $A=$ { $a_{ij}$ } with dimensions $N \times N$ describes the connections between nodes where $a_{ij}$ is $1$ if $i$ and $j$ are neighbours or $0$ otherwise- **Component** set of the graph is a set with no neighbours outside itselfWe define a **pin** for each component such that it consumes the minimal amount of energy to control its respective component. Here, we achieve this by computing the Network Controlability Gramian:$W_j=$ $\sum\limits_{h=0}^{H-1}$ $A_d^hb_jb_j^T\left(A_d^T\right)^h$where:- $A_d = AD^{-1}$- $D$ is the diagonal **augmented in-degree matrix** of $A$, composed by the sum of the columns of $A$ on its diagonal (or a $1$ when this sum is $0$)- $b_j$ is a column vector with all terms equal to $0$ except at the pin location $j$- $H$ is the control horizonThe **trace** of $W_j$ is inversely proportional to the average **energy** required to control the graph component when $j$ is selected as the pin. Therefore, we compute the trace of $W_j$ for each candidate pin (within a given component set) and select the one with the lowest value in order to minimize the control energy.The graph is built using **Olfati-Saber flocking** (Reference 4, below) such that the agents are considered to be *connected* when they are within radius $r$ of eachother. These connections are what define the component sets of the graph. The pins share a common target, which draws the separate components together. As the components meet, they combine to form a larger component. New pins are computed in **real-time**, until all agents form part of a single component representing the entire graph. This convergence is guaranteed, so long as the components are observable, controlable, stable, and share the same target.# PlotsBelow are plots with adaptive pins, selected to conserve energy using the Network Controlability Gramian:<p float="center">    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_big2.gif" width="70%" />    </p><p float="center">    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_big2_e.png" width="40%" />    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_big2_s.png" width="70%" />    </p><p float="center">    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_10.gif" width="45%" />    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/convergence_pin_10.png" width="45%" />    </p><p float="center">    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/animation_pin_9.gif" width="45%" />    <img src="https://github.com/tjards/pinning_swarming/blob/master/Figs/convergence_pin_9.png" width="45%" />    </p># References 1. Kléber M. Cabral, Sidney N. Givigi, and Peter T. Jardine, [Autonomous assembly of structures using pinning control and formation algorithms](https://ieeexplore-ieee-org.proxy.queensu.ca/document/9275901) in 2020 IEEE International Systems Conference (SysCon), 07 Dec 20202. Erfan Nozari, Fabio Pasqualetti, and Jorge Cortes,[Heterogeneity of Central Nodes Explains the Benefits of Time-Varying Control in Complex Dynamical Networks](https://arxiv.org/abs/1611.06485) in arXiv, 26 Dec 20183. Fabio Pasqualetti, Sandro Zampieri, and Francesco Bullo, [Controllability Metrics, Limitations and Algorithms for Complex Networks](https://ieeexplore-ieee-org.proxy.queensu.ca/stamp/stamp.jsp?tp=&arnumber=6762966) in IEEE Transactions on Control of Network Systems, 11 Mar 20144. Reza Olfati-Saber, ["Flocking for Multi-Agent Dynamic Systems: Algorithms and Theory"](https://ieeexplore.ieee.org/document/1605401) in IEEE Transactions on Automatic Control, Vol. 51 (3), 3 Mar 2006.5. The code is opensource but, if you reference this work in your own reserach, please cite me. I have provided an example bibtex citation below:`@techreport{Jardine-2022,  title={Adaptive Pinning Control using Network Controlability Gramian},  author={Jardine, P.T.},  year={2022},  institution={Royal Military College of Canada, Kingston, Ontario},  type={GitHub Code Repository},}`6. Alternatively, you can cite any of my related papers, which are listed in [Google Scholar](https://scholar.google.com/citations?hl=en&user=RGlv4ZUAAAAJ&view_op=list_works&sortby=pubdate). 