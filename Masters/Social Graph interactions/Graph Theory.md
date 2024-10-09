# Graph Theory
An undirected and unweighted graph can be written down in two ways 
- Edgelist: a list of edges
- ![](Pasted%20image%2020240911092403.png)
-  adjacency matrix 
- ![](Pasted%20image%2020240911092509.png) 
Moreover, 
The book refers to nodes as N links (edges) as L, the average degree for each network \<k\> = \<$k_{in}$\>=\<$k_{out}$\> 
### Degree
let $k_i$ be the degree of the $i^{th}$ node. which is the number of links that node has (for undirected graph). The total number of links that a node has $L=\frac {1}{2} \sum_{i=1}^Nk_i$. 
![](Pasted%20image%2020240911095821.png)
#### Average Degree 
Average degree for an undirected network is $<k>=\frac {1}{N} \sum_{i=1}^Nk_i=\frac{2L}{N}$.  In directed networks we distinguish between incoming degree, $k_i^{in}$, representing the number of links that point to node $i$, and outgoing degree, $k_i^{out}$, representing the number of links that point from node $i$ to other nodes. Finally, a node’s total degree, $ki$, is given by $k_i=k_i^{in}+ k_i^{out}$ and the total number of links is given by$L=\sum_{i=1}^Nk_i^{out}=\sum_{i=1}^Nk_i^{in}$. the average incoming degree and out going degrees for a directed graph is 
$k<in>=\frac {1}{N} \sum_{i=1}^Nk_i^{in}=k<out>=\frac {1}{N} \sum_{i=1}^Nk_i^{out}$ 

#### Degree Distribution 
The _degree distribution_, $p_k$, provides the probability that a randomly selected node in the network has degree _k_. Since _pk_ is a probability, it must be normalized $\sum_{k=1}^\infty p_k=1$. For a network with N nodes the degree distribution is normalized histogram given by $p_k=\frac {N_k}{N}$. where $N_k$ is the number of degree-k nodes. Hence the number of degree-k nodes can be obtained from the degree distribution as $N_k=p_k N$. 
The degree distribution allows for scale-free network because the calculation of most network properties requires us to know _pk_. For example, the average degree of a network can be written as $<k>=\sum_{k=0}^\infty kp_k$ The other reason is that the precise functional form of _pk_ determines many network phenomena, from network robustness to the spread of viruses.
 **Example**
 ![](Pasted%20image%2020240911101603.png)
 Degree Distribution  
The degree distribution of a network is provided by the ratio (2.7).
1. For the network in (a) with _N_ = 4 the degree distribution is shown in (b).
2. We have _p1_ = 1/4 (one of the four nodes has degree _k1_1 = 1), _p2_ = 1/2 (two nodes have _k3 = k4_ = 2), and _p3_ = 1/4 (as _k2_ = 3). As we lack nodes with degree _k_ › 3, _pk_ = 0 for any _k_ › 3.
3. A one dimensional lattice for which each node has the same degree _k_ = 2.
4. The degree distribution of (c) is a Kronecker’s delta function, _pk_ = _δ_(_k_ - 2)

![](Pasted%20image%2020240911101810.png)Degree Distribution of a Real Network  
In real networks the node degrees can vary widely.

1. A layout of the protein interaction network of yeast. Each node corresponds to a yeast protein and links correspond to experimentally detected binding interactions. Note that the proteins shown on the bottom have self-loops, hence for them _k_=2.
2. The degree distribution of the protein interaction network shown in (a). The observed degrees vary between _k_=0 (isolated nodes) and _k_=92, which is the degree of the most connected node, called a _hub_. There are also wide differences in the number of nodes with different degrees: Almost half of the nodes have degree one (i.e. _p1_=0.48), while we have only one copy of the biggest node (i.e. _p92_ = 1/_N_=0.0005).
3. The degree distribution is often shown on a log-log plot, in which we either plot log _pk_ in function of ln _k_, or, as we do in (c), or we use logarithmic axes. The advantages of this representation are discussed in Chapter 4.

### Adjacency Matrix
The adjacency matrix of a directed network of N nodes has N rows and N columns, its elements being

$A_{ij}= 1$ if there is a link pointing from node _j_ to node _i_

$A_{ij} = 0$ if nodes _i_ and _j_ are not connected to each other
The adjacency matrix of an undirected network has two entries for each link, e.g. link (1, 2) is represented as $A_{12} = 0$ and $A_{21} = 0$. Hence, the adjacency matrix of an undirected network is symmetric, $A_{ij} = A_{ji}$. The degree of node can be directly obtained from the elements of the adjacency matrix. For undirected networks a node’s degree is a sum over either the rows or the columns of the matrix. For directed networks the sums over the adjacency matrix’ rows and columns provide the incoming and outgoing degrees, respectively. ![](Pasted%20image%2020240911102649.png)
### Complete Graph
is a graph that every node is connected to another $L_{max}=\frac{N(N-1)}{2}$ , The average degree of a complete graph is $‹k› = N - 1$.  real networks _L_ is much smaller than _Lmax_, reflecting the fact that most real networks are sparse. We call a network _sparse_ if $L<< L_{max}$. 
### Weighted Networks
For _weighted networks_ the elements of the adjacency matrix carry the weight of the link as $A_{ij}=w_{ij}$ 
![](Pasted%20image%2020240911120530.jpg)
### Bipartite Networks
A _bipartite graph_ (or _bigraph_) is a network whose nodes can be divided into two disjoint sets U and V such that each link connects a U-node to a V-node. In other words, if we color the U-nodes green and the V-nodes purple, then each link must connect nodes of different colors. We can generate two projections for each bipartite network. The first projection connects two U-nodes by a link if they are linked to the same V-node in the bipartite representation. The second projection connects the V-nodes by a link if they connect to the same U-node![](Pasted%20image%2020240911105539.jpg)
A bipartite network has two sets of nodes, U and V. Nodes in the U-set connect directly only to nodes in the V-set. Hence there are no direct U-U or V-V links. Projection U is obtained by connecting two U-nodes to each other if they link to the same V-node in the bipartite representation. A wellknown example is the Hollywood actor network, in which one set of nodes corresponds to movies (U), and the other to actors (V). 

**Tripartite Network**  

1. The construction of the tripartite recipe-ingredient- compound network, in which one set of nodes are recipes, like Chicken Marsala; the second set corresponds to the ingredients each recipe has (like flour, sage, chicken, wine, and butter for Chicken Marsala); the third set captures the flavor compounds, or chemicals that contribute to the taste of each ingredient.
2. The _ingredient_ or the _flavor network_ represents a projection of the tripartite network. Each node denotes an ingredient; the node color indicating the food category and node size indicates the ingredient’s prevalence in recipes. Two ingredients are connected if they share a significant number of flavor compounds. Link thickness represents the number of shared compounds
![](Pasted%20image%2020240911105839.jpg)
### Paths and Distances
![Paths.](https://networksciencebook.com/images/ch-02/figure-2-12.jpg)

Paths  

1. A path between nodes _i<sub>0</sub>_ and _i<sub>n</sub>_ is an ordered list of _n_ links P = {(_i<sub>0</sub>_, _i<sub>1</sub>_), (_i<sub>1</sub>_, _i<sub>2</sub>_), (_i<sub>2</sub>_, _i<sub>3</sub>_), ... ,(_i<sub>n-1</sub>_, _i<sub>n</sub>_)}. The length of this path is n. The path shown in orange in (a) follows the route 1→2→5→7→4→6, hence its length is _n_ = 5.
2. The shortest paths between nodes 1 and 7, or the distance _d<sub>17</sub>_, correspond to the path with the fewest number of links that connect nodes 1 to 7. There can be multiple paths of the same length, as illustrated by the two paths shown in orange and grey. The network diameter is the largest distance in the network, being _d<sub>max</sub>_ = 3 here.
#### Shortest Path
The shortest path between nodes _i_ and _j_ is the path with the fewest number of links and  is often called the distance between nodes _i_ and _j_, and is denoted by _d<sub>ij</sub>_, or simply _d_.  In an undirected network _d<sub>ij</sub>_ = _d<sub>ji</sub>_, i.e. the distance between node i and j is the same as the distance between node _i_ and _j_. In a directed network often _d<sub>ij</sub>_ ≠ _d<sub>ji</sub>_. Furthermore, in a directed network the existence of a path from node _i_ to node _j_ does not guarantee the existence of a path from _j_ to _i_.

**Pathology**  
![](Pasted%20image%2020240911112134.jpg)
1. Path  
    A sequence of nodes such that each node is connected to the next node along the path by a link. Each path consists of _n_+1 nodes and _n_ links. The length of a path is the number of its links, counting multiple links multiple times. For example, the orange line 1 → 2 → 5 → 4 → 3 covers a path of length four.
2. Shortest Path (Geodesic Path, _d_)  
    The path with the shortest distance _d_ between two nodes. We also call _d_ the distance between two nodes. Note that the shortest path does not need to be unique: between nodes 1 and 4 we have two shortest paths, 1→ 2→ 3→ 4 (blue) and 1→ 2→ 5→ 4 (orange), having the same length _d<sub>1,4</sub>_ =3.
3. Diameter (_d<sub>max</sub>_)  
    The longest shortest path in a graph, or the distance between the two furthest nodes. In the graph shown here the diameter is between nodes 1 and 4, hence _d<sub>max</sub>_=3.
4. Average Path Length (_〈d〉_)  
    The average of the shortest paths between all pairs of nodes. For the graph shown on the left we have _〈d〉_=1.6, whose calculation is shown next to the figure.
5. Cycle  
    A path with the same start and end node. In the graph shown on the left we have only one cycle, as shown by the orange line.
6. Eulerian Path  
    A path that traverses each link exactly once. The image shows two such Eulerian paths, one in orange and the other in blue.
7. Hamiltonian Path  
    A path that visits each node exactly once. We show two Hamiltonian paths in orange and in blue.
![](Pasted%20image%2020240911112411.png)
#### Network Diameter
The _diameter_ of a network, denoted by _d<sub>max</sub>_, is the maximum shortest path in the network. In other words, it is the largest distance recorded between _any_ pair of nodes. One can verify that the diameter of the network shown above is _d<sub>max</sub>_ = 3. For larger networks the diameter can be determined using the BFS algorithm.
#### Average Path Length

The _average path length_, denoted by _〈d〉_, is the average distance between all pairs of nodes in the network. For a directed network of _N_ nodes, _〈d〉_ is $d= \frac{1}{N(N-1)} \sum_{i,j=1,N;i\neq j} d_{i,j}$   is measured only for node pairs that are in the same component. We can use the BFS algorithm to determine the average path length for a large network. determine the distances between the first node and all other nodes in the network then determine the distances between the second node and all other nodes but the first one and so on. 

#### BFS
1. Start at node _i_, that we label with “0”.
2. Find the nodes directly linked to _i_. Label them distance “1” and put them in a queue.
3. Take the first node, labeled _n_, out of the queue (_n_ = 1 in the first step). Find the unlabeled nodes adjacent to it in the graph. Label them with _n_ + 1 and put them in the queue.
4. Repeat step 3 until you find the target node _j_ or there are no more nodes in the queue.
5. The distance between _i_ and _j_ is the label of _j_. If _j_ does not have a label, then _d<sub>ij</sub>_ = ∞.
The computational complexity of the BFS algorithm, representing the approximate number of steps the computer needs to find _dij_ on a network of _N_ nodes and _L_ links, is _O(N + L)_.![](Pasted%20image%2020240911113312.jpg)
### Connectedness

In an undirected network nodes _i_ and _j_ are _connected_ if there is a path between them. They are _disconnected_ if such a path does not exist, in which case we have _d<sub>ij</sub>_ = ∞. This is illustrated in ![](Pasted%20image%2020240911113645.jpg)which shows a network consisting of two disconnected clusters. While there are paths between any two nodes on the same cluster (for example nodes 4 and 6), there are no paths between nodes that belong to different clusters (nodes 1 and 6).

A _network is connected_ if all pairs of nodes in the network are connected. A _network is disconnected_ if there is at least one pair with _d<sub>ij</sub>_ = ∞. we call its two subnetworks _components_ or _clusters_. A _component_ is a subset of nodes in a network, so that there is a path between any two nodes that belong to the component, but one cannot add any more nodes to it that would have the same property.

If a network consists of two components, a properly placed single link can connect them, making the network connected. Such a link is called a _bridge_. In general a bridge is any link that, if cut, disconnects the network.


For a disconnected network the adjacency matrix can be rearranged into a block diagonal form, such that all nonzero elements in the matrix are contained in square blocks along the matrix’ diagonal and all other elements are zero. Each square block corresponds to a component. We can use the tools of linear algebra to decide if the adjacency matrix is block diagonal, helping us to identify the connected components. or just use BFS

Connected and Disconnected Networks  

1. A small network consisting of two disconnected components. Indeed, there is a path between any pair of nodes in the (1,2,3) component, as well in the (4,5,6,7) component. However, there are no paths between nodes that belong to the different components.  
    The right panel shows the adjacently matrix of the network. If the network has disconnected components, the adjacency matrix can be rearranged into a block diagonal form, such that all nonzero elements of the matrix are contained in square blocks along the diagonal of the matrix and all other elements are zero.
2. The addition of a single link, called a _bridge_, shown in grey, turns a disconnected network into a single connected component. Now there is a path between every pair of nodes in the network. Consequently the adjacency matrix cannot be written in a block diagonal form.

**Finding the Connected Components of a Network**
1. Start from a randomly chosen node _i_ and perform a BFS (BOX 2.5). Label all nodes reached this way with _n_ = 1.
2. If the total number of labeled nodes equals _N_, then the network is connected. If the number of labeled nodes is smaller than _N_, the network consists of several components. To identify them, proceed to step 3.
3. Increase the label _n_ → _n_ + 1. Choose an unmarked node _j_, label it with _n_. Use BFS to find all nodes reachable from _j_, label them all with _n_. Return to step 2.
### Clustering Coefficient
The clustering coefficient captures the degree to which the neighbours of a given node link to each other. For a node _i_ with degree _ki_ the _local clustering coefficient_ is defined as $C_i=\frac{2L_i}{K_i(K_i-1)}$ where _L<sub>i</sub>_ represents the number of links between the _k<sub>i</sub>_ neighbours of node _i_. Note that _C<sub>i</sub>_ is between 0 and 1. 

- _C<sub>i</sub>_ = 0 if none of the neighbours of node _i_ link to each other.
- _Ci_ = 1 if the neighbours of node _i_ form a complete graph, i.e. they all link to each other.
- _Ci_ is the probability that two neighbors of a node link to each other. Consequently _C_ = 0.5 implies that there is a 50% chance that two neighbours of a node are linked.
In summary _C<sub>i</sub>_ measures the network’s local link density: The more densely interconnected the neighborhood of node _i_, the higher is its local clustering coefficient.

![Clustering Coefficient.](https://networksciencebook.com/images/ch-02/figure-2-16.jpg)
Clustering Coefficient  

1. The local clustering coefficient, _C<sub>i</sub>_ , of the central node with degree _k<sub>i</sub>_ = 4 for three different configurations of its neighbourhood. The local clustering coefficient measures the local density of links in a node’s vicinity.
2. A small network, with the local clustering coefficient of each nodes shown next to it. We also list the network’s average clustering coefficient _〈C〉_, according to the equation below, and its global clustering coefficient$C_Δ = \frac {3\times NumberOfTriangles}{NumberOfConnectedTriples}$  Note that for nodes with degrees _k<sub>i</sub>_ = 0,1, the clustering coefficient is zero.
The degree of clustering of a whole network is captured by the _average clustering coefficient_, _〈C〉_, representing the average of _C<sub>i</sub>_ over all nodes _i_ = 1, ..., _N_ $〈C〉=\frac{1}{N}\sum_{i=1}^NC_i$  In line with the probabilistic interpretation _〈C〉_ is the probability that two neighbors of a randomly selected node link to each other.

# Random Networks
There are two definitions of a random network:
- _G_(_N_, _L_) Model: N labeled nodes are connected with _L_ randomly placed links. Erdős and Rényi used this definition in their string of papers on random networks
- _G_(_N_, _p_) Model: Each pair of N labeled nodes is connected with probability _p_, a model introduced by Gilbert.
### Number of Links
The probability that a random network has exactly _L_ links is the product of three terms:
- The probability that L of the attempts to connect the _N_(_N_-1)/2 pairs of nodes have resulted in a link, which is _p<sup>L</sup>_
- The probability that the remaining _N_(_N_-1)/2 - _L_ attempts have not resulted in a link, which is $(1-p)^{\frac {N(N-1)}{2}-L}$ 
- A combinational factor, $\binom{\frac {N(N-1)}{2}}{L}$ counting the number of different ways we can place _L_ links among _N_(_N_-1)/2 node pairs.
Hence the probability that a particular realization of a random network has exactly _L_ links as
$$
PL=\binom{\frac {N(N-1)}{2}}{L}(1-p)^{\frac {N(N-1)}{2}-L}p^L
$$
Which is a binomial distribution the expected number of links in a random graph is
$$
<L>=\sum_{L=0}^{\frac{N(N-1)}{2}}L_{PL}=p\frac{N(N-1)}{2}
$$
Hence ‹_L_› is the product of the probability _p_ that two nodes are connected and the number of pairs we attempt to connect, which is $L_{max}=\frac{N(N-1)}{2}$

Using the above we can obtain the average degree of a random network
$$
<K>=\frac{2<L>}{N}=p(N-1)
$$
Hence ‹_k_› is the product of the probability _p_ that two nodes are connected and (_N_-1), which is the maximum number of links a node can have in a network of size _N_.

In summary the number of links in a random network varies between realizations. Its expected value is determined by _N_ and _p_. If we increase _p_ a random network becomes denser: The average number of links increase linearly from ‹_L_› = 0 to _L<sub>max_</sub> and the average degree of a node increases from ‹_k_› = 0 to ‹_k_› = _N_-1

### Binomial Distribution: Mean and Variance
![](Pasted%20image%2020240912162934.png)
### Degree Distribution
![](Pasted%20image%2020240912163255.jpg)
The exact form of the degree distribution of a random network is the binomial distribution(left half). For _N_ ›› ‹_k_› the binomial is well approximated by a Poisson distribution (right half). As both formulas describe the same distribution,they have the identical properties, but they are expressed in terms of different parameters: The binomial distribution depends on _p_ and _N_, while the Poisson distribution has only one parameter, ‹_k_›. It is this simplicity that makes the Poisson form preferred in calculations
#### Binomial Distribution
In a random network the probability that node _i_ has exactly _k_ links is the product of three terms
- The probability that _k_ of its links are present, or $p^k$
- The probability that the remaining (_N_-1-_k_) links are missing, or $(1-p)^{N-1-k}$
- The number of ways we can select _k_ links from _N_- 1 potential links a node can have, or $\binom{N-1}{k}$ 
Consequently the degree distribution of a random network follows the binomial distribution which is the product of all three components. = p<sub>k</sub>
#### Poisson Distribution
Most real networks are sparse, meaning that for them ‹_k_› ‹‹ N. In this limit the degree distribution (3.7) is well approximated by the Poisson distribution
$$
p_k=e^{-<k>}\frac {<k>^k}{k!}
$$

- The exact result for the degree distribution is the binomial form, thus the equation above represents only an approximation to the binomial form valid in the ‹_k_› ‹‹ _N_ limit. As most networks of practical importance are sparse, this condition is typically satisfied.
- The advantage of the Poisson form is that key network characteristics, like <‹_k_›, ‹_k<sup>2</sup>_› and _σ<sub>k</sub>_ , have a much simpler form The mean of the distribution (first moment)  depending on a single parameter, ‹_k_›.
- The Poisson distribution in above does not explicitly depend on the number of nodes _N_. Therefore, it predicts that the degree distribution of networks of different sizes but the same average degree ‹_k_› are indistinguishable from each other

- Both distributions have a peak around ‹_k_›. If we increase _p_ the network becomes denser, increasing ‹_k_› and moving the peak to the right.
- The width of the distribution (dispersion) is also controlled by _p_ or ‹_k_›. The denser the network, the wider is the distribution, hence the larger are the differences in the degrees.
### The Evolution of a Random Network
let N<sub>G</sub> be the largest connecting network
- For _p_ = 0 we have ‹_k_› = 0, hence all nodes are isolated. Therefore the largest component has size _N<sub>G</sub>_ = 1 and _N<sub>G</sub>_/_N_→0 for large _N_.
- For _p_ = 1 we have ‹_k_›= _N_-1, hence the network is a complete graph and all nodes belong to a single component. Therefore _N<sub>G</sub>_ = _N_ and _N<sub>G</sub>_/_N_ = 1.

Once ‹_k_› exceeds a critical value, _NG_/_N_ increases, signaling the rapid emergence of a large cluster that we call the _giant component_. The condition for the emergence of the giant component is ⟨k⟩=1 or p<sub>c</sub> =1/(N-1). we have a giant component if and only if each node has on average more than one link.

The emergence of the giant component is only one of the transitions characterizing a random network as we change ‹_k_›. We can distinguish four topologically distinct regimes![](Pasted%20image%2020240912171936.jpg)

1. The relative size of the giant component in function of the average degree ‹_k_› in the Erdős-Rényi model. The figure illustrates the phase tranisition at ‹_k_› = 1, responsible for the emergence of a giant component with nonzero _NG_
2. A sample network and its properties in the four regimes that characterize a random network.
##### Subcritical Regime
0 ‹ ‹_k_› ‹ 1 and  (_p_ ‹ 1/_N_ )

For ‹_k_› = 0 the network consists of _N_ isolated nodes. Increasing ‹_k_› means that we are adding $N‹k› = pN(N-1)/2$  links to the network. Yet, given that ‹_k_› ‹ 1, we have only a small number of links in this regime, hence we mainly observe tiny clusters.

We can designate at any moment the largest cluster to be the giant component. Yet in this regime the relative size of the largest cluster, _NG_/_N_, remains zero. The reason is that for ‹_k_› ‹ 1 the largest cluster is a tree with size _NG_ ~ _lnN_, hence its size increases much slower than the size of the network. Therefore _NG_/_N_ ≃ ln_N_/_N_→0 in the _N_→∞ limit

 The subcritical regime the network consists of numerous tiny components, whose size follows the exponential distribution. Hence these components have comparable sizes, lacking a clear winner that we could designate as a giant component.

##### Critical Point
‹_k_› = 1
The critical point separates the regime where there is not yet a giant component (‹_k_› ‹ 1) from the regime where there is one (‹_k_› › 1). At this point the relative size of the largest component is still zero. the size of the largest component is $N_G \approx N^{2/3}$. Consequently $N_G$ grows much slower than the network’s size, so its relative size decreases as $N_G/N \approx N^{-1/3}$ in the _N_→∞ limit.

 At the critical point most nodes are located in numerous small components, whose size distribution follows. The power law form indicates that components of rather different sizes coexist. These numerous small components are mainly trees, while the giant component may contain loops. Note that many properties of the network at the critical point resemble the properties of a physical system undergoing a phase transition.

##### Supercritical Regime
 ‹_k_› › 1 _p_ › 1/_N_
This regime has the most relevance to real systems, as for the first time we have a giant component that looks like a network. In the vicinity of the critical point the size of the giant component varies as
$\frac {N_G}{N}\approx ‹k› - 1$ or $N_G \approx (p-p_c)N$ 
where _pc_ is given by p<sub>c</sub> =1/(N-1). In other words, the giant component contains a finite fraction of the nodes. The further we move from the critical point, a larger fraction of nodes will belong to it. Note that $\frac {N_G}{N}\approx ‹k› - 1$ is valid only in the vicinity of ‹_k_› = 1. For large ‹_k_› the dependence between _NG_ and ‹_k_› is nonlinear.

the supercritical regime numerous isolated components coexist with the giant component, their size distribution. These small components are trees, while the giant component contains loops and cycles. The supercritical regime lasts until all nodes are absorbed by the giant component.

##### Connected Regime
‹_k_› › ln_N_ _p_ › ln_N_/_N_
For sufficiently large _p_ the giant component absorbs all nodes and components, hence $N_G≃ N$. In the absence of isolated nodes the network becomes connected. The average degree at which this happens depends on _N_ as $‹k› = \ln k$. Note that when we enter the connected regime the network is still relatively sparse, as $\ln N / N → 0$ for large _N_. The network turns into a complete graph only at ‹_k_› = _N_ - 1.
the random network model predicts that the emergence of a network is not a smooth, gradual process: The isolated nodes and tiny components observed for small ‹_k_› collapse into a giant component through a phase transition. As we vary ‹_k_› we encounter four topologically distinct regimes(see image above). 

The discussion offered above follows an empirical perspective, fruitful if we wish to compare a random network to real systems. A different perspective, with its own rich behavior, is offered by the mathematical literature.
![](Pasted%20image%2020240922150221.png)
### Real Networks are Supercritical


Two predictions of random network theory are of direct importance for real networks:

1. Once the average degree exceeds ‹_k_› = 1, a giant component should emerge that contains a finite fraction of all nodes. Hence only for ‹_k_› › 1 the nodes organize themselves into a recognizable network.
2. For ‹_k_› › lnN all components are absorbed by the giant component, resulting in a single connected network.
![]({800D3126-2C53-4816-8206-1E1D37116512}.png)
# The Scale-Free Property
_A scale-free network is a network whose degree distribution follows a power law_.
## Power Laws and Scale-Free network
Power distribution law $p_k=k^{-\gamma}$ the exponent is the degree exponent 
The degree distribution  
## Discrete Formalism
As node degrees are positive integers, _k_ = 0, 1, 2, ..., the discrete formalism provides the probability _p<sub>k</sub>_ that a node has exactly _k_ links
$p_k=Ck^{-\gamma}$ where he constant _C_ is determined by the normalization condition. 
$$
\sum_{k=1}^\infty k^{-\gamma} =1
$$
hence
$$
C=\frac{1}{\sum_{k=1}^\infty k^{-\gamma}}=\frac{1}{\zeta(\gamma)}
$$
where _ζ (γ)_ is the Riemann-zeta function. Thus for _k_ > 0 the discrete power-law distribution has the form $p_k=\frac{k^{-\gamma]}}{\zeta(\gamma)}$ which diverges to 0 
## Continuum Formalism
In analytical calculations it is often convenient to assume that the degrees can have any positive real value. In this case we write the power-law degree distribution as $p(k)=Ck^{-\gamma}$  using the normalization condition 
$$
\int_{k_{min}}^\infty p(k)dk=1
$$
we obtain 
$$
C=\frac{1}{\int_{k_{min}}^\infty p(k)dk}=(\gamma-1)k_{min}^{\gamma -1}
$$
Here _kmin_ is the smallest degree for which the power law  $p_k=\frac{k^{-\gamma]}}{\zeta(\gamma)}$  holds
Note that _pk_ encountered in the discrete formalism has a precise meaning: it is the probability that a randomly selected node has degree _k_. In contrast, only the integral of _p(k)_ encountered in the continuum formalism has a physical interpretation is the probability that a randomly chosen node has degree between _k1_ and _k2_.

## Hubs
The main difference between a random and a scale-free network comes in the _tail_ of the degree distribution, representing the high-_k_ region of _pk_![](Pasted%20image%2020240918141440.jpg)
- For small _k_ the power law is above the Poisson function, indicating that a scale-free network has a large number of small degree nodes, most of which are absent in a random network.
- For _k_ in the vicinity of _〈k〉_ the Poisson distribution is above the power law, indicating that in a random network there is an excess of nodes with degree _k_ ≈_〈k〉_ .
- For large k the power law is again above the Poisson curve. The difference is particularly visible if we show _pk_ on a log-log plot ([Image 4.4b](https://networksciencebook.com/#figure-4-4)), indicating that the probability of observing a high-degree node, or _hub_, is several orders of magnitude higher in a scale-free than in a random network.
**Poisson vs. Power-law Distributions**
1. Comparing a Poisson function with a power-law function (_γ_= 2.1) on a linear plot. Both distributions have _〈k〉_= 11.
2. The same curves as in (a), but shown on a log-log plot, allowing us to inspect the difference between the two functions in the high-_k_ regime.
3. A random network with _〈k〉_= 3 and _N_ = 50, illustrating that most nodes have comparable degree _k_≈_〈k〉_
4. A scale-free network with _γ_=2.1 and _〈k〉_=3, illustrating that numerous small-degree nodes coexist with a few highly connected hubs. The size of each node is proportional to its degree.
### The Largest Hub 
How does the network size affect the size of its hubs? To answer this we calculate the maximum degree, _kmax_, called the _natural cutoff_ of the degree distribution _pk_. It represents the expected size of the largest hub in a network.

![](Pasted%20image%2020240918143754.png)
As ln_N_ is a slow function of the system size, (4.17) tells us that the maximum degree will not be significantly different from _kmin_. For a Poisson degree distribution the calculation is a bit more involved, but the obtained dependence of _kmax_ on _N_ is even slower than the logarithmic dependence predicted by (4.17)
For a scale-free network, according to (4.12) and (4.16), the natural cutoff follows

$$
K_{max}=k_{min}N^{\frac{1}{\gamma -1}}
$$
Hence the larger a network, the larger is the degree of its biggest hub

#### Hubs are Large in Scale-free Networks
The key difference between a random and a scale-free network is rooted in the different shape of the Poisson and of the power-law function: In a random network most nodes have comparable degrees and hence hubs are forbidden. Hubs are not only tolerated, but are expected in scale-free networks.
the more nodes a scalefree network has, the larger are its hubs. Indeed, the size of the hubs grows polynomially with network size, hence they can grow quite large in scalefree networks.![](Pasted%20image%2020240918144526.jpg)
1. The degrees of a random network follow a Poisson distribution, rather similar to a bell curve. Therefore most nodes have comparable degrees and nodes with a large number of links are absent.
2. A random network looks a bit like the national highway network in which nodes are cities and links are the major highways. There are no cities with hundreds of highways and no city is disconnected from the highway system.
3. In a network with a power-law degree distribution most nodes have only a few links. These numerous small nodes are held together by a few highly connected hubs.
4. A scale-free network looks like the air-traffic network, whose nodes are airports and links are the direct flights between them. Most airports are tiny, with only a few flights. Yet, we have a few very large airports, like Chicago or Los Angeles, that act as major hubs, connecting many smaller airports.
## The Meaning of Scale-Free
To best understand the meaning of the scale-free term, we need to familiarize ourselves with the moments of the degree distribution.
![](Pasted%20image%2020240918144807.png)
![](Pasted%20image%2020240918145617.png)
![](Pasted%20image%2020240918150036.png)
For any exponentially bounded distribution, like a Poisson or a Gaussian, the degree of a randomly chosen node is in the vicinity of _〈k〉_. Hence _〈k〉_ serves as the network’s _scale_. For a power law distribution the second moment can diverge, and the degree of a randomly chosen node can be significantly different from _〈k〉_. Hence _〈k〉_ does not serve as an intrinsic scale. As a network with a power law degree distribution lacks an intrinsic scale. The scale-free name captures the lack of an internal scale, a consequence of the fact that nodes with widely different degrees coexist in the same network. This feature distinguishes scale-free networks from lattices, in which all nodes have exactly the same degree (_σ_ = 0), or from random networks, whose degrees vary in a narrow range (_σ_ = _〈k〉_1/2)_. As we will see in the coming chapters, this divergence is the origin of some of the most intriguing properties of scale-free networks, from their robustness to random failures to the anomalous spread of viruses.
## Universality

### Plotting the Degree Distribution
The degree distributions shown in this chapter are plotted on a double logarithmic scale, often called a log-log plot. The main reason is that when we have nodes with widely different degrees, a linear plot is unable to display them all. To obtain the clean-looking degree distributions shown throughout this book we use logarithmic binning, ensuring that each datapoint has sufficient number of observations behind it. 
### Measuring the Degree Exponent

A quick estimate of the degree exponent can be obtained by fitting a straight line to _pk_ on a log-log plot.Yet, this approach can be affected by systematic biases, resulting in an incorrect _γ_
## Ultra-Small Property
 Do hubs affect the small world property?
 The calculations support this expectation, finding that _distances in a scale-free network are smaller than the distances observed in an equivalent random network_
 ![](Pasted%20image%2020240918153152.png)
###  Anomalous Regime (γ = 2)
  for _γ_ = 2 the degree of the biggest hub grows linearly with the system size, i.e. _kmax_ ~ _N_. This forces the network into a _hub and spoke_ configuration in which all nodes are close to each other because they all connect to the same central hub. In this regime the average path length does not depend on _N_.
### Ultra-Small World (2 ‹ γ ‹ 3)
  Equation (4.22) predicts that in this regime the average distance increases as lnln_N_, a significantly slower growth than the ln_N_ derived for random networks. We call networks in this regime _ultra-small_, as the hubs radically reduce the path length. ![](Pasted%20image%2020240918153341.jpg)
  Distances in Scale-free Networks  

1. The scaling of the average path length in the four scaling regimes characterizing a cale-free network: constant (_γ_ = 2), lnln_N_ (2 ‹ _γ_ ‹ 3), ln_N_/lnln_N_ (_γ_ = 3), ln_N_ (_γ_ › 3 and random networks). The dotted lines mark the approximate size of several real networks. Given their modest size, in biological networks, like the human protein-protein interaction network (PPI), the differences in the node-to-node distances are relatively small in the four regimes. The differences in _〈d〉_ is quite significant for networks of the size of the social network or the WWW. For these the small-world formula significantly underestimates the real _〈d〉_.
2. Distance distribution for networks of size _N_ = 102, 104, 106, illustrating that while for small networks (_N_ = 102) the distance distributions are not too sensitive to _γ_, for large networks (_N_ = 106) _pd_ and _〈d〉_ change visibly with _γ_.
### Critical Point (γ = 3)

This value is of particular theoretical interest, as the second moment of the degree distribution does not diverge any longer. We therefore call _γ_ = 3 the _critical point_. At this critical point the ln_N_ dependence encountered for random networks returns. Yet, the calculations indicate the presence of a double logarithmic correction _lnln_N_ , which shrinks the distances compared to a random network of similar size.

### Small World (γ > 3)
While hubs continue to be present, for _γ_ > 3 they are not sufficiently large and numerous to have a significant impact on the distance between the nodes.
The more effectively they shrink the distances between nodes.

conclusion is supported by [Image 4.12a](https://networksciencebook.com/#figure-4-12), which shows the scaling of the average path length for scale-free networks with different _γ_. The figure indicates that while for small _N_ the distances in the four regimes are comparable, for large _N_ we observe remarkable differences.

In summary the scale-free property has several effects on network distances:

- Shrinks the average path lengths. Therefore most scale-free networks of practical interest are not only “small”, but are “ultra-small”. This is a consequence of the hubs, that act as bridges between many small degree nodes.
- Changes the dependence of _〈d〉_ on the system size, as predicted by (4.22). The smaller is _γ_, the shorter are the distances between the nodes.
- Only for _γ_ › 3 we recover the ln_N_ dependence, the signature of the small-world property characterizing random networks ([Image 4.12](https://networksciencebook.com/#figure-4-12)).
## The Role of the Degree Exponent
- _γ_ varies from system to system, prompting us to explore how the properties of a network change with _γ_.
- For most real systems the degree exponent is above 2, making us wonder: Why don’t we see networks with _γ_ ‹ 2?
![](Pasted%20image%2020240918154901.jpg)
### Anomalous Regime (γ≤ 2)

For _ γ _ ‹ 2 the exponent 1/( _γ _− 1) in (4.18) is larger than one, hence the number of links connected to the largest hub grows faster than the size of the network. This means that for sufficiently large _ N_ the degree of the largest hub must exceed the total number of nodes in the network, hence it will run out of nodes to connect to. Similarly, for _γ_ ‹ 2 the average degree _〈k〉_ diverges in the _N_ → ∞ limit. These odd predictions are only two of the many anomalous features of scale-free networks in this regime. They are signatures of a deeper problem: Large scale-free network with _γ_ ‹ 2, that lack multi-links, cannot exist (BOX 4.6).

### Scale-Free Regime (2 ‹ γ ‹ 3)

In this regime the first moment of the degree distribution is finite but the second and higher moments diverge as _N_ → ∞. Consequently scalefree networks in this regime are ultra-small. Equation (4.18) predicts that _kmax_ grows with the size of the network with exponent 1/(_γ_ - 1), which is smaller than one. Hence the market share of the largest hub, _kmax_ /_N_, representing the fraction of nodes that connect to it, decreases as _kmax_/_N_ ~ _N_-(γ-2)/(γ-1).

As we will see in the coming chapters, many interesting features of scale-free networks, from their robustness to anomalous spreading phenomena, are linked to this regime.

### Random Network Regime (γ › 3)

According to (4.20) for _γ_ > 3 both the first and the second moments are finite. For all practical purposes the properties of a scale-free network in this regime are difficult to distinguish from the properties a random network of similar size. For example (4.22) indicates that the average distance between the nodes converges to the small-world formula derived for random networks. The reason is that for large _γ_ the degree distribution _pk_ decays sufficiently fast to make the hubs small and less numerous.

Note that scale-free networks with large _γ_ are hard to distinguish from a random network. Indeed, to document the presence of a power-law degree distribution we ideally need 2-3 orders of magnitude of scaling, which means that _kmax_ should be at least 10<sup>2</sup> - 10<sup>3</sup> times larger than _kmin_. By inverting (4.18) we can estimate the network size necessary to observe the desired scaling regime, finding
![](Pasted%20image%2020240918155653.png)
In summary, we find that the behavior of scale-free networks is sensitive to the value of the degree exponent _γ_. Theoretically the most interesting regime is 2 ‹ _γ_ ‹ 3, where _〈k2〉_ diverges, making scale-free networks ultra-small. Interestingly, many networks of practical interest, from the WWW to protein interaction networks, are in this regime.
![](Pasted%20image%2020240918155804.png)
![](Pasted%20image%2020240918155818.png)

# The Barabási-Albert Model
## Growth and Preferential Attachment
Why are hubs and power laws absent in random networks? The answer emerged in 1999, highlighting two hidden assumptions of the Erdős-Rényi model, that are violated in real networks. Next we discuss these assumptions separately
### Networks Expand Through the Addition of New Nodes
The random network model assumes that we have a _fixed_ number of nodes, _N_. Yet, _in real networks the number of nodes continually grows thanks to the addition of new nodes_.
### Nodes Prefer to Link to the More Connected Nodes

The random network model assumes that we randomly choose the interaction partners of a node. Yet, _most real networks new nodes prefer to link to the more connected nodes_, a process called _preferential attachment_

In summary, the random network model differs from real networks in two important characteristics:

1. **Growth**  
    Real networks are the result of a growth process that continuously increases _N_. In contrast the random network model assumes that the number of nodes, _N_, is fixed.
2. **Preferential Attachment**  
    In real networks new nodes tend to link to the more connected nodes. In contrast nodes in random networks randomly choose their interaction partners.

## The Barabási-Albert Model    

The recognition that growth and preferential attachment coexist in real networks has inspired a minimal model called the _Barabási-Albert_ model, which can generate scale-free networks. Also known as the BA model or the _scale-free model_, it is defined as follows:

We start with _m0_ nodes, the links between which are chosen arbitrarily, as long as each node has at least one link. The network develops following two steps

**Growth**
At each timestep we add a new node with _m_ ($≤m_0$) links that connect the new node to _m_ nodes already in the network.
**Preferential attachment**  
The probability _Π(k)_ that a link of the new node connects to node _i_ depends on the degree _ki_ as $$\Pi(k_i)=\frac{k_i}{\sum_j k_j} \tag{5.1}$$ 
Preferential attachment is a probabilistic mechanism: A new node is free to connect to _any_ node in the network, whether it is a hub or has a single link. Equation (5.1) implies, however, that if a new node has a choice between a degree-two and a degree-four node, it is twice as likely that it connects to the degree-four node.

In summary, the Barabási-Albert model indicates that two simple mechanisms, _growth_ and _preferential attachment_, are responsible for the emergence of scale-free networks. The origin of the power law and the associated hubs is a _rich-gets-richer phenomenon_ induced by the coexistence of these two ingredients. To understand the model’s behavior and to quantify the emergence of the scale-free property.
![](Pasted%20image%2020240922153847.png)
![](Pasted%20image%2020240922153919.png)
## Degree Dynamics
In the model an existing node can increase its degree each time a _new_ node enters the network. This new node will link to _m_ of the _N(t)_ nodes already present in the system. The probability that one of these links connects to node _i_ is given by (5.1)

Let us approximate the degree _ki_ with a continuous real variable, representing its expectation value over many realizations of the growth process. The rate at which an existing node _i_ acquires links as a result of new nodes connecting to it is
![](Pasted%20image%2020240922154743.png)
![](Pasted%20image%2020240922154947.png)
$$K_i(t)=m(\frac{t}{t_i})^\beta \tag{5.7}$$
We call _β_ the _dynamical exponent_ and has the value $\beta = \frac{1}{2}$ 
![](Pasted%20image%2020240922155316.jpg)
- The degree of each node increases following a power-law with the same dynamical exponent _β_ =1/2 (a) Hence all nodes follow the same dynamical law.
- The growth in the degrees is sublinear (i.e. _β_ < 1). This is a consequence of the growing nature of the Barabási-Albert model: Each new node has more nodes to link to than the previous node. Hence, with time the existing nodes compete for links with an increasing pool of other nodes.
- The earlier node _i_ was added, the higher is its degree _ki(t)_. Hence, hubs are large because they arrived earlier, a phenomenon called _first-mover advantage_ in marketing and business.
- The rate at which the node _i_ acquires new links is given by the derivative of (5.7) $$\frac{dk_i(t)}{dt}=\frac {m}{2}\frac{1}{\sqrt{t_it}} \tag{5.8}$$ 
 indicating that in each time step older nodes acquire more links (as they have smaller _ti_). Furthermore the rate at which a node acquires links decreases with time as $t^{−1/2}$. Hence, fewer and fewer links go to a node.
 Degree Dynamics

1. The growth of the degrees of nodes added at time _t_ =1, 10, 10<sup>2</sup>, 10<sup>3</sup>, 10<sup>4</sup>, 10<sup>5</sup> (continuous lines from left to right) in the Barabási-Albert model. Each node increases its degree following (5.7). Consequently at any moment the older nodes have higher degrees. The dotted line corresponds to the analytical prediction (5.7) with _β_ = 1/2.
2. Degree distribution of the network after adding _N_ = 102, 104, and 106 nodes, i.e. at time _t_ = 102, 104, and 106 (illustrated by arrows in (a)). The larger the network, the more obvious is the power-law nature of the degree distribution. Note that we used linear binning for _pk_ to better observe the gradual emergence of the scale-free state.
![](Pasted%20image%2020240922160343.png)
## Degree Distribution
A number of analytical tools are available to calculate the degree distribution of the Barabási-Albert network. The simplest is the _continuum theory_ It predicts the degree distribution.
$$
p(k)\approx 2m^{1/\beta}k^{-\gamma} \tag{5.9}
$$
$$
\gamma = (1/\beta)+1=3 \tag{5.10}
$$
Therefore the degree distribution follows a power law with degree exponent _γ_=3, in agreement with the numerical results. Moreover (5.10) links the degree exponent, _γ_, a quantity characterizing the network topology, to the dynamical exponent _β_ that characterizes a node’s temporal evolution, revealing a deep relationship between the network's topology and dynamics.

While the continuum theory predicts the correct degree exponent, it fails to accurately predict the pre-factors of (5.9). The correct pre-factors can be obtained using a master or rate equation approach or calculated exactly using the LCD model (BOX 5.3). Consequently the _exact degree distribution_ of the Barabási-Albert model is 
$$
p_k=\frac{2m(m+1)}{k(k+1)(k+2)} \tag{5.11}
$$
![](Pasted%20image%2020240922160813.jpg)
1. We generated networks with _N_=100,000 and _m0_=_m_=1 (blue), 3 (green), 5 (grey), and 7 (orange). The fact that the curves are parallel to each other indicates that _γ_ is independent of _m_ and _m0_. The slope of the purple line is -3, corresponding to the predicted degree exponent _γ_=3. Inset: (5.11) predicts $p_k\approx 2m^2$, hence $p_k/ 2m^2$ should be independent of _m_. Indeed, by plotting $p_k/ 2m^2$, the data points shown in the main plot collapse into a single curve.
2. The Barabási-Albert model predicts that _pk_ is independent of _N_. To test this we plot _pk_ for _N_ = 50,000 (blue), 100,000 (green), and 200,000 (grey), with _m0_=_m_=3. The obtained _pk_ are practically indistinguishable, indicating that the degree distribution is _stationary_, i.e. independent of time and system size.

![](Pasted%20image%2020240922160801.png)
In summary, the analytical calculations predict that the Barabási-Albert model generates a scale-free network with degree exponent _γ_=3. The degree exponent is independent of the _m_ and _m0_ parameters. Furthermore, the degree distribution is stationary (i.e. time invariant), explaining why networks with different history, size and age develop a similar degree distribution.