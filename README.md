<a name="br1"></a> 

Application of Machine Learning Techniques for

Performance Enhancement in Information

Retrieval Models Using Graphs

George Kontogiannis

Department of Computer Engineering & Informatics

University of Patras

September 2, 2023



<a name="br2"></a> 

Contents

[Introduction](#br3)[ ](#br3)[to](#br3)[ ](#br3)[Laplacian](#br3)[ ](#br3)[Graphs](#br3)

[Spectral](#br4)[ ](#br4)[Embedding](#br4)

[Spectral](#br10)[ ](#br10)[Clustering](#br10)

[Old](#br14)[ ](#br14)[Models:](#br14)[ ](#br14)[GSB,](#br14)[ ](#br14)[GSBW](#br14)

[Proposed](#br21)[ ](#br21)[Models](#br21)

[Experimental](#br27)[ ](#br27)[Procedure](#br27)[ ](#br27)[and](#br27)[ ](#br27)[Analysis](#br27)

[Evaluation](#br28)[ ](#br28)[of](#br28)[ ](#br28)[Models](#br28)

[Model](#br38)[ ](#br38)[Selection](#br38)[ ](#br38)[and](#br38)[ ](#br38)[Analysis](#br38)

[Bibliography/References](#br43)



<a name="br3"></a> 

Graph Laplacians: Overview

▶

▶

▶

(Weighted) Adjacency matrix: A

P

n

j=1

Degree of vertex v ∈ V : d =

a<sub>ij</sub>

i

i

Degree matrix D: diag(d , . . . , d )

1

n

▶

Laplacians:

Unnormalized: L = D − A

(1)

(2)

−1/2

LD<sup>−1/2</sup>

Normalized: L = D



<a name="br4"></a> 

The Spectral Embedding Problem

Our goal:

▶

Start with a graph G characterized by its Laplacian matrix L.

▶

Seek a low-dimensional representation X (e.g., a matrix) of

the graph, keeping connected nodes close.

Figure: Graph

Figure: Embeddings in R<sup>3</sup>

The challenge: How to deﬁne and achieve this representation?



<a name="br5"></a> 

Objective Function: Intuition

We wish to create a representation where:

▶

Nodes that are close in the graph remain close in the

low-dimensional space.

▶

▶

The sum of squared diﬀerences between connected nodes is

minimized.

Diﬀerences are weighted by the strength of the connection:

A .

ij



<a name="br6"></a> 

Objective Function: Formulation

▶

Optimization Problem:

X

2

min

X:X<sup>T</sup> 1=0

A ∥X − X ∥

ij

i

j

i,j∈V

▶

▶

Centering Constraint: X<sup>T</sup> 1 = 0

Triviality Warning: Without additional constraints, the

solution X = 0 is trivial.



<a name="br7"></a> 

Optimization Problem

We seek the optimal representation by minimizing:

min

tr(X<sup>T</sup> LX)

X:X<sup>T</sup> 1=0,X<sup>T</sup> X=I<sub>K</sub>

▶

▶

T

X X = I : This forces the embedding to fully occupy the

K

vector space by ensuring that the coordinates are uncorrelated

and have equal (positive) variance, eﬀectively making X<sup>T</sup> X a

normalized covariance matrix.

Lemma 1:

X<sup>n</sup>

1

2

tr(X<sup>T</sup> LX) =

A ∥X − X ∥

2

ij

i

j

i,j=1



<a name="br8"></a> 

Solution to the Problem

▶

Theorem 1:

<sup>K</sup>X+1

k=2

min

tr(X<sup>T</sup> LX) =

λ<sub>k</sub>

X:X<sup>T</sup> 1=0,X<sup>T</sup> X=I<sub>K</sub>

▶

Proof: First consider the problem without the centering and

orthogonality constraints,

min

X:diag(X<sup>T</sup> X)=I<sub>K</sub>

tr(X<sup>T</sup> LX)

The Lagrangian for this optimization problem is

T

T

L = tr(X LX) − Λ(diag(X X) − I )

K

where Λ is the diagonal matrix of Lagrange multipliers.

Taking the gradient gives

∇<sub>X</sub> L = 2LX − 2ΛX = 0

which implies that X is a matrix of eigenvectors of L, and the

theorem follows.



<a name="br9"></a> 

Eigenvalues and Embedding

▶

We skip the ﬁrst eigenvalue λ<sub>1</sub> = 0 due to centering constraint

▶

The eigenvectors X corresponding to the K smallest

eigenvalues of L, λ ≤ λ ≤ . . . ≤ λ<sub>K+1</sub>, provide the optimal

2

3

solution.

▶

Taking the orthogonality constraint into account, we have:

tr(X<sup>T</sup> LX) = tr(Λ), Λ = diag(λ<sub>2</sub>, . . . , λ<sub>K+1</sub>

)



<a name="br10"></a> 

Spectral Clustering: Introduction

▶

Spectral clustering uses graph theory and linear algebra to

cluster data.

▶

▶

Does not make assumptions about the shape or size of

clusters.

Particularly eﬀective when clusters are non-convex.

Figure: Karate Club Graph Analysis. Left: Original structure. Right:

Result of Spectral Clustering highlighting the two main factions within

the club.



<a name="br11"></a> 

Spectral Clustering: Basic Idea

▶

Build a similarity graph: nodes represent data points and edge

weights represent similarity.

▶

▶

Utilize the graph’s Laplacian to reveal cluster structure.

Eigenvalues and eigenvectors of the Laplacian capture this

structure.

▶

Cluster the eigenvector representation using k-means or

another clustering method.



<a name="br12"></a> 

Spectral Clustering: Algorithm Steps

1\. Construct a similarity matrix and form a weighted adjacency

matrix W .

2\. Compute the Unnormalized Laplacian L.

2\.1 Method 1 (Standard):

▶

Calculate ﬁrst k eigenvectors u<sub>1</sub>, . . . , u<sub>k</sub> of L.

2\.2 Method 2 (Shi and Malik, 2000):

▶

Solve the generalized eigenproblem Lu = λDu to get

u , . . . , u .

1

k

2\.3 Method 3 (Normalized Laplacian by Ng, Jordan, and

Weiss, 2002):

▶

Compute L.

Calculate ﬁrst k eigenvectors u<sub>1</sub>, . . . , u<sub>k</sub> of L.

▶

▶

Construct matrix T ∈ R

n×k

from U by normalizing its rows to

u

have a norm of 2. Speciﬁcally, set t =

ij

.

ij

P

(

2

)<sup>1/2</sup>

ik

u

k

3\. Let U ∈ R

n×k

contain u , . . . , u as columns.

1

k

4\. For each i, let y ∈ R correspond to the i-th row of U (or T

k

i

for Method 3).

5\. Cluster the new data representations in R<sup>k</sup> using k-means into

clusters C , . . . , C .

1

k



<a name="br13"></a> 

Eigengap Heuristic

Deﬁnition

The eigengap is the diﬀerence between consecutive eigenvalues,

λ<sub>i</sub> and λ<sub>i+1</sub>

.

▶

▶

▶

Utilized for dimension selection in spectral embedding.

Large eigenvalues capture high variance or spread.

The eigengap represents a substantial diﬀerence in data

spread signifying more distinct data partitions.

▶

Mathematically:

k = arg max(λ − λ<sub>i+1</sub>).

i

i



<a name="br14"></a> 

GSB Model Overview

▶

Extension of Set-Based Model.

▶

▶

▶

▶

Each text represented as a weighted undirected graph.

Text indexing using graphs.

Weight computation using graph analysis.

Node signiﬁcance through K-core Decomposition.



<a name="br15"></a> 

Weight Computation Theorems

Theorem

For a node N in text D with term frequency tf , the internal edge

i

j

i

weight (N , N ) is calculated as:

i

i

tf × (tf + 1)

i

i

W<sub>in</sub> =

2

Theorem

For nodes N and N in text D with term frequencies tf and tf ,

i

j

j

i

j

the external edge weight (N , N ) is calculated as:

i

j

W<sub>out</sub> = tf<sub>i</sub> × tf<sub>j</sub> when N<sub>i</sub≯= N<sub>j</sub>



<a name="br16"></a> 

GSBW Model Overview

▶

Addresses GSB limitations.

▶

▶

▶

Divides text into windows (sentences/paragraphs).

Generates discrete graphs with content-based edges.

Enhances content relationship representation.



<a name="br17"></a> 

Fixed Window Approach

▶

Divide text into ﬁxed-size windows.

▶

▶

Calculate weights based on words in each window.

Let document D = {t , t , t , . . . , t }

1

2

3

10

Example (Constant Window):

W<sub>size</sub> = 3









 {t , t , t }, 

1

2

3





{t , t , t },

4

5

6

Windows =

 {t , t , t }, 

7

8

9

{t }

10



<a name="br18"></a> 

Percentage Window Approach

▶

▶

Divide text into windows based on percentage of words.

Adapts to diﬀerent document lengths.

Example (Percentage Window): For instance, if the window

percentage is 20%, then the window size is:

W<sub>size</sub> = 20% × len(D) = 2

and the generated windows are:





 {t , t }, 

1

2

4







 {t , t }, 

3

{t , t },

Windows =

5

6

 .



 .



.,





{t , t }

9

10



<a name="br19"></a> 

Union Graph and Node Weight Calculation

▶

▶

▶

Combined graph U from individual text graphs.

Sum of corresponding edge weights from all graphs.

Node weight NW<sub>k</sub> in ﬁnal graph U is calculated as:

ꢀ

ꢁ

ꢀ

ꢁ

W<sub>outk</sub>

1

NW<sub>k</sub> = log 1 + a ×

×log 1 + b ×

(3)

(W<sub>ink</sub> + 1) × (ng<sub>k</sub> + 1)

ng<sub>k</sub> + 1

▶

a, b: Parameters to quantify the eﬀect of GSB vs. Set-Based

W<sub>outk</sub> : Out-edge weight of node k

W<sub>ink</sub> : In-edge weight of node k

▶

▶

▶

ng<sub>k</sub>: Number of neighbors of node k



<a name="br20"></a> 

Information Retrieval and Scoring

▶

GSB follows Set-Based Model for information retrieval.

Generating term sets S<sub>i</sub> by the Apriori algorithm.

▶

Equations:

Y

TN<sub>Si</sub>

\=

NW<sub>k</sub>

(4)

k∈S

i

N

W<sub>Sij</sub> = (1 + log Sf<sub>ij</sub> ) × log(1 +

) × TN<sub>Si</sub>

(5)

(6)

dS<sub>i</sub>

N

W<sub>Siq</sub> = log(1 +

)

dS<sub>i</sub>

Documents and queries are represented as vectors:

⃗

d = (W , W<sub>S2j</sub> , . . . , W<sub>S</sub>

)

j

S<sub>1j</sub>

2

n

j

⃗

Q = (W , W<sub>S2q</sub> , . . . , W<sub>S</sub>

)

S<sub>1q</sub>

n

2

q

Cosine similarity as ranking function:

⃗

⃗

d ~~·~~ Q

sim(Q⃗, d⃗ ) = ~~ꢂ~~

~~ꢂ~~

j

~~ꢂ~~

~~ꢂ~~

j

ꢂ

ꢂ

ꢂ

ꢂ

ꢂd⃗ ꢂ × ꢂQ⃗ꢂ

j



<a name="br21"></a> 

Proposed Models

▶

Pruned Models:

▶

PGSB: Pruned Graphical Set-Based Model

▶

▶

PGSBW constant: Pruned Graphical Set-Based Window

PGSBW percentage: Pruned Graphical Set-Based Window

▶

▶

▶

▶

CGSBW constant: Conceptualized Graphical Set-Based

Window

CGSBW percentage: Conceptualized Graphical Set-Based

Window

CGSB-QE: Conceptualized Graphical Set-Based Model Query

Expansion



<a name="br22"></a> 

Overview

▶

Calibrated term weights using Spectral Clustering and

Conceptualization.

▶

▶

▶

Graph pruning based on cluster structures for manageable

clusters.

Graph sparsiﬁcation Out and In-Cluster pruning for more

eﬃcient execution.

Eﬃcient query expansion using term concepts via κ-Nearest

Neighbors.



<a name="br23"></a> 

Pruning and Spectral Clustering

▶

Spectral Clustering applied to the Union Graph.

▶

Cluster assignment for each node.

▶

Node embeddings representation for each node

▶

Pruning based on theorems:

▶

Theorem 1: Cluster-Based Pruning

▶

▶

Theorem 2: Edge Pruning based on Edge Weight

Theorem 3: Edge Pruning based on Node Similarity



<a name="br24"></a> 

Graph Pruning Theorems

▶

Theorem 1: Cluster-Based Pruning Given graph G = (V , E), clusters

C , C :

u

v

(

1

0

if C<sub>u</sub≯= C<sub>v</sub>

otherwise

P<sub>cluster</sub>(u, v) =

▶

Theorem 2: Edge Pruning based on Edge Weight Given graph

G = (V , E), edge weight w(u, v), clusters C , C :

u

v



1 if w(u, v) ≤ θ and C<sub>u</sub≯= C<sub>v</sub>

P<sub>edge</sub>(u, v) =

1

if w(u, v) ≤ 2θ and C = C<sub>v</sub>

u





0

otherwise

P

w(u, v)

v∈N(u)

θ<sub>u</sub>

\=

|N(u)|

▶

Theorem 3: Edge Pruning based on Node Similarity Given graph

G = (V , E), node embeddings E , E , clusters C , C :

u

v

u

v



1 if sim(E , E ) ≤ θ and C̸= C<sub>v</sub>

u

v

u

P<sub>sim</sub>(u, v) =

1

if sim(E , E ) ≤ 2|θ| and C = C<sub>v</sub>

u

v

u





0

otherwise

(Note: θ ∈ [−1, 1] for cosine similarity)



<a name="br25"></a> 

Information Retrieval and Scoring

▶

Pruned models use the same scoring Eq. (2) for termsets as GSB,

GSBW: TN<sub>Si</sub>

Q

\=

NW<sub>k</sub>

<sup>k∈S</sup>i

▶

Conceptualized models introduce a new metric:

P

NW<sub>t</sub>

<sup>t∈C</sup>k

C<sub>NWk</sub>

\=

|C<sub>k</sub> |

▶

▶

▶

C<sub>k</sub> : Cluster associated with term k

NW : Weight from Eq. (1) for vertex t ∈ C

t

k

|C |: Number of vertices in cluster C

k

k

deﬁned as the scalar centroid or scalar concept of each cluster.

The scoring equation for termsets for conceptualized models is

obtained by:

Y

TN<sub>Si</sub>

\=

C<sub>NWk</sub>

k∈S

Documents and queries again are represented by Eqs. (3, 4) and

retrieved by cosine similarity.

i

▶



<a name="br26"></a> 

Query Expansion in CGSB

▶

Utilizes precomputed embeddings for eﬃciency.

▶

▶

Input query is represented as the centroid of the query terms

Expands queries using the κ-nearest neighbors (K-Nearest

Neighbors) method.

▶

▶

▶

Identiﬁes semantically similar terms in the embedded space for

expansion.

Enhances retrieval by considering underlying geometric

relationships between terms.

Oﬀers signiﬁcant improvement with negligible additional time

cost.



<a name="br27"></a> 

Experimental Procedure and Analysis

Collection:

▶

Experiments on Cystic Fibrosis collection (Shaw et al., 1991):

▶

1209 texts, 100 questions.

Total size: 1.47 MB.

▶

Pre-processing:

▶

▶

▶

Punctuation removal.

Stop words not removed from collection.

Stop words removed from queries.

▶

▶

Apriori for query term sets:

▶

Minimum frequency = 1.

Experimental Context:

▶

Phase 1: Estimation metric is average accuracy.

▶

▶

▶

▶

Basis for comparison: GSB and optimal GSBW with window=7.

Base Accuracies: Set-Based (0.165), GSB (0.187), GSBW-7 (0.211)

Reference: Future models compared to y = 0.211 line.

Models and hyperparameters are outlined below.

Model

Cluster Sizes

Similarities(%) OR Edge Coeﬃcients Window Percentages

PGSB, CGSB, PGSBW<sub>const</sub> , CGSBW<sub>const</sub> 30, 50, 70, 90, 110, 130, 150, 170 0.1, 0.3, 0.5, 0.7, 1.1, 1.2, 1.3, 1.4

PGSBW<sub>per</sub> , CGSBW<sub>per</sub> 110, 130, 150, 170 0.5, 0.7, 1.3, 1.4

\-

0\.1, 0.3, 0.5, 0.7



<a name="br28"></a> 

CGSB vs. PGSB using Theorem 1

(b) Queries where one model

excels

(a) Avg. Precision & %Pruning

▶

CGSB consistently outperforms PGSB across all clusters.

▶

▶

CGSB-110 leads with 0.242 avg. accuracy (14.7% increase).

CGSB excels in more answered questions, enhancing its

trustworthiness.

▶

Some PGSB models underperform, falling below the GSBW-7

baseline.



<a name="br29"></a> 

CGSB vs. PGSB using Theorem 2

▶

▶

Top Accuracy: CGSB with 170 clusters.

CGSB superiority

▶

▶

Best Factor: Coeﬃcient 1.2 optimal for CGSB.

Pruning: Based on avg. edge weight of 8.2.



<a name="br30"></a> 

CGSB vs. PGSB using Theorem 3

▶

▶

▶

CGSB tops PGSB.

Top CGSB: 0.244 (15.7% increase).

Best with 150 clusters.

▶

▶

PGSB falters at high similarity.

Linear pruning vs similarity.



<a name="br31"></a> 

New Evaluation Metric

New Evaluation Metric:

P

` `P

!

ꢀ

ꢁ

ꢀ

ꢁ

<sup>positives</sup>ij

Q

k=1

Q

<sup>P</sup>k

<sup>E</sup>p

j̸=i

M<sub>evali</sub> = a

\+ b

\+ c

× 100

(n − 1) · 100

E

▶

▶

▶

▶

▶

Composite metric for performance comparison.

Incorporates model superiority, query precision, and graph pruning.

Balanced evaluation using weighted coeﬃcients.

Provides comprehensive insight into model performance.

Coeﬃcients: a = 50%, b = 40%, c = 10%.



<a name="br32"></a> 

New Metric Analysis of Best CGSB & PGSB Models

Figure: Evaluation values of the 3 best models per category

▶

Selection: Estimation values calculated for each model, top 3

chosen.

Superiority: CGSB dominates over PGSB.

Optimal Params: 50% similarity, 1.3 w[eight](#br31)[ ](#br31)[facto](#br33)[~~r~~](#br32)[,](#br32)[ ](#br32)[130](#br33)[ ](#br33)[clusters.](#br37)

▶

▶



<a name="br33"></a> 

Constant CGSBW vs. PGSBW using Theorem 1

(b) Queries where one model

excels

(a) Avg. Precision & %Pruning

▶

Fixed window size set to 7.

▶

▶

▶

▶

▶

PGSBW Outperforms CGSBW in all cases.

Top: PGSBW-170 (0.236), PGSBW-130 (0.233).

All models outperform GSBW-7 but no CGSBs, PGSBs.

Low pruning rates mostly close to zero.

PGSBWs answered > 95% queries better than CGSBWs in cluster

range 30 − 110.



<a name="br34"></a> 

Constant CGSBW vs. PGSBW using Theorem 2

▶

▶

Top PGSBW-150-10: 0.239.

PGSBWs tops CGSBWs.

▶

▶

Fewer clusters at 50%, 70%.

Less Pruning needed for window models.



<a name="br35"></a> 

Cosntant CGSBW vs. PGSBW using Theorem 3

▶

▶

Top Accuracy CGSBW-170-1.3: 0.234.

CGSBWs > PGSBWs for edge coeﬀ. > 1.1

▶

▶

Optimal Clusters: 170 clusters best for both

Pruning: Based on avg. edge weight of 2.5.

CGSBWs and PGSBWs



<a name="br36"></a> 

Constant Best CGSBW & PGSBW: New Metric

Figure: Evaluation values of the 3 best models per category

▶

▶

PGSBW Eﬃciency: Outperforms CGSBW in the majority.

Optimal Parameters: Similarity at 50%, edge weight factor at 1.3,

and 170 clusters.



<a name="br37"></a> 

Percentage Best CGSBW & PGSBW: New Metric

Figure: Evaluation values of the 3 best models per category

▶

PGSBW Eﬃciency: Outperforms CGSBW in the majority.

Theorem 2: Found to be ineﬀective for solving the problem.

Pruning: Unneeded in PGSBW.

Params: 50% sim, 50% percentage, 170 clusters.

Proven to be the least eﬃcient models.

▶

▶

▶

▶



<a name="br38"></a> 

Model Selection and Overview

Figure: Top 10 Models Based on New Metric

▶

▶

▶

69 Final Models: From 3 distinct classes.

Top 10: Best models selected for detailed analysis.

Eﬃciency-Accuracy Gap: Most eﬃcient model not the most

accurate.



<a name="br39"></a> 

Cumulative Accuracy

Figure: Cumulative Accuracy by Model

▶

▶

Visual Comparison: Stacked areas show model contributions to

each query accuracy.

CGSB-170-0.1: Dominates in most queries.



<a name="br40"></a> 

Deviation from Average & Eﬃciency

Figure: Deviation from Average Accuracy

▶

▶

▶

Pos/Neg: Above/below average accuracy indicated.

CGSB: Small deviations, consistent performance.

PGSBW: Larger deviations, varies by query.



<a name="br41"></a> 

Query Expansion: Setup and Context

▶

▶

▶

Algorithm: Query expansion using the κ-nearest neighbors method.

Target Model: Applied to most eﬃcient model CGSB-170-0.1.

Comparison: Diﬀerence plot to evaluate quality.

Figure: Performance Variations between CGSB-170-0.1 and

CGSB-170-0.1-QE



<a name="br42"></a> 

Query Expansion: Key Takeaways

▶

Improvement in accuracy for 60/100 queries, degrades for

40/100.

▶

▶

Magnitude of accuracy boost more than twice that of

degradation.

26 queries clustered in narrow range, indicating minimal

performance diﬀerences.

▶

▶

Overall average accuracy increased from 0.243 to 0.257.

Average accuracy improvement of 5.76% with query

expansion.

▶

Distribution skewed toward positive diﬀerences, suggesting

query expansion’s advantages outweigh disadvantages.



<a name="br43"></a> 

Bibliography/References I

[1] M. Belkin and P. Niyogi. “Laplacian eigenmaps for dimensionality

reduction and data representation”. In: Neural Computation (2003). url:

[https://www.mitpressjournals.org/doi/abs/10.1162/](https://www.mitpressjournals.org/doi/abs/10.1162/089976603321780317)

[089976603321780317](https://www.mitpressjournals.org/doi/abs/10.1162/089976603321780317).

[2] T. Bonald. “Spectral graph embedding”. Lecture notes, Institut

Polytechnique de Paris. 2019.

[3] F. R. Chung. Spectral graph theory. American Mathematical Society,

1997\.

[4] N.-R. Kalogeropoulos et al. “A graph-based extension for the set-based

model implementing algorithms based on important nodes”. In: Artiﬁcial

intelligence applications and innovations. Ed. by I. Maglogiannis,

L. Iliadis, and E. Pimenidis. Springer International Publishing, 2020,

pp. 143–154.

[5] U. von Luxburg. “A tutorial on spectral clustering”. In: Statistics and

Computing 17.4 (2007), pp. 395–416. doi:

[10.1007/s11222-007-9033-z](https://doi.org/10.1007/s11222-007-9033-z).



<a name="br44"></a> 

Bibliography/References II

[6] Andrew Y. Ng, M. I. Jordan, and Y. Weiss. “On spectral clustering:

Analysis and an algorithm”. In: Advances in neural information processing

systems. 2002, pp. 849–856.

[7] B. Possas et al. “Set-based model: A new approach for information

retrieval”. In: Proceedings of the 25th Annual International ACM SIGIR

Conference on Research and Development in Information Retrieval. 2002,

pp. 230–237. doi: [10.1145/564376.564417](https://doi.org/10.1145/564376.564417).



<a name="br45"></a> 

Thank You for Your

Attention!

Any Questions?

