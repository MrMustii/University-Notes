# Week 1
Given lots of data you should discover patterns and models that are:
- Valid
- Useful
- Unexpected
- Understandable
#### Data mining Tasks
- Descriptive Methods
	- Find human-interpretable patterns that describe the data
		- Clustering 
- Predictive Methods
	- Use some variables to predict unknown or future values of other variables 
		- Recommender systems (Netflix, Youtube to recommend )

This is a repository for my notes of the courses in my masters. They are written in Obisidean as markdown format . Beware that a lot of the text and images are taken from the slides provided by the lecturer or books. None of this is original work

#### Modeling 
- Create a model of your data that
	- Provides a good description of your data
	- Allows you to make predictions about new data
- **Example**Detecting phishing emails
	- The model could be weights on words
		- Phrases appearing unusally often in phishing emails receive positive weights other way around
	- Easy to use, hard to find the weights 

**Find an underlying probability distribution from which the data is drawn** 

#### Machine Learning
- Use data to train one of many types of algorithms used in ML
- The resulting parameters are the model of the data
- Best used when we do not fully understand what the data tells us about our problem
- Less effective when we better understand the data
- Often yield a model that we cannot fully explain

#### Approaches to Modeling
- Build a (random) process that could have generated the data
- Summarization
	- Summarizing the data succintly and approximately
- Feature Extractions 
	- Extracting the most prominent features of the data and ignoring the rest




#### TF.IDF measure

- TF.IDF measure of word importance
	- For each document it 
	- Often want to categorize documents by topic
	- Often want to categorize documents by topic
		- The topic(s) of a document will be identified by special words related to that topic
		- E.g. Articles about baseball would use “bat”, “pitch”, ”run”, etc. many times
	- Cannot come up with a list of words for *every* topic
		- Want to reverse engineer the topics from the words in the documents 
- Problem : How do we decide which words in a document are significant
- Most frequent words don’t work
	- "the","and","that", etc
	- Often remove the several hundred most common words (stop words)
- Often remove the several hundred most common words (stop words
	- Often remove the several hundred most common words (stop words
- Want words that appear fairly often in a document, but do not appear in too many documents

Let there be $N$ documents 
$f_{t,d}=$ the number of terms words t in document d
$TF_{t,d}=f_{t,d}/max_wf_{w,d}$ (term frequency )
$n_t=$ number of documents term 𝑡 appears in
$IDF_t=\log{n/n_t}$  (Inverse Document Frequency)
Large if term t appears in few documents $d=TF_{t,d}*IDF_t$
Terms with high 𝑇𝐹.𝐼𝐷𝐹 score are often the terms that best characterize the topic of a document
![[Pasted image 20240903191442.png]]


#### Bonferroni’s Principle
- Not all patterns are meaningful
- Certain patterns/events in your data that you are interested in might also occur randomly
- The principle:
	- Calculate the expected number of the events you are looking for, assuming the data is random
	- If this number is significantly higher than the number of “real” or “meaningful” instances you expect to find, then you should expect that almost anything you find is bogus
![[Pasted image 20240903192110.png]]
![[Pasted image 20240903192335.png]]


#### Power Laws
- Many phenomena relate two variables by a “power law”:
	- Many phenomena relate two variables by a “power law”:
	- $Log_{10} y =6-2log_{10} x$  
- General form: $Log y= b+a Log x$
	- thus $y=e^be^{alogx}=e^bx^a=cx^a$ 
- **The Matthew Effect, i.e., The rich get richer**
- Occurs when having a high value of some property causes that property to increase
- Example
	- Webpage has many links to it 
	- This increases traffic to the page
	- More people decide to link to it
- Leads to power laws with |a|>1
- **Things that obey power laws**
	- Node degrees in Web graph: 𝑎 ≈ 2.1
	- Sales of products: let 𝑦 be the # of sales of 𝑥 most popular book on Amazon
	- Sizes of website: order sites by # of pages, 𝑦 the # of pages at the 𝑥 site
	- Zipf’s Law: 𝑦 = # times 𝑥 most frequent word appears
		- $y=cx^{-1/2}$
		- Many other kinds of data obey this, e.g., populations of US states
----
# Week 3 - (MapReduce, Distributed File Systems, Cluster Computing)

- **Chunk servers**
	- Files split into “chunks"
	- Typically 16-64 MB
	- Each chunk replicated 2x or 3x, preferably on different racks
- **Master/Name node**
	- Stores metadata about where files are stored
	- Might be replicated
- **Client library for file access**
	- Talks to master to find chunks
	- Connects directly to chunk servers to access data
## MapReduce
### Dataflow
• Split: Partition data into chunks and distribute to different machines. 
• Map: Map data items to list of <key, value> pairs. 
• Shuffle: Group data with the same key and send to the same machine. 
• Reduce: Takes list of values with the same key <key, [value1, ..., valuek]> and outputs list of new data items.

You only write the Map and Reduce functions. Technically all inputs to Map tasks and outputs from Reduce tasks should have <key, value> form to allow for composition

### MapReduce: Word Counting Example

Input: Document of words
Output: Frequency of each word
Document: “Deer Bear River Car Car River Deer Car Bear
(Bear, 2), (Car, 3), (Deer, 2), (River, 2)
![](Pasted%20image%2020240917182749.png)
![](Pasted%20image%2020240917182852.png)
##### Example 2
•Input: Set of documents 
• Output: List of documents that contain each word 
• Document 1: “Deer Bear River Car Car River Deer Car Bear.” 
• Document 2: "Deer Antilope Stream River Stream" 
• (Bear, [1]), (Car, [1]), (Deer, [1,2]), (River, [1,2]), (Antilope, [2]), (Stream, [2

- map(word in document i) $\rightarrow$ <word,i>
- reduce(word,\[i<sub>1</sub>\,i,<sub>2</sub>,...i<sub>n</sub>])  $\rightarrow$ word,\[j<sub>1</sub>\,j<sub>2</sub>,...j<sub>n</sub>]
### Matrix Vector Multiplication
![](Pasted%20image%2020240917185009.png)
• map(𝑚 ) ➝ <𝑖,𝑚<sub>ij</sub>𝑣<sub>i</sub>> 
	• So all terms making up 𝑥 have the same key 
• reduce: simply sum up all values associated to a key 𝑖 to get <𝑖<sub>i</sub>,𝑥<sub>i</sub>>

**What if 𝑣 is too big for main memory?**

### Relations
- A relation is a table with column headers called attributes, e.g.
![](Pasted%20image%2020240917185222.png)
- Rows of the table are called tuples
- Write 𝑅(𝐴<sub>1</sub>,𝐴<sub>2</sub>,…,𝐴<sub>n</sub>) to say that the relation name is 𝑅 and its attributes are 𝐴<sub>1</sub>,𝐴<sub>2</sub>,…,𝐴<sub>n</sub>

Selection: Apply a condition 𝐶 to each tuple in the relation and produce as output only those tuple that satisfy 𝐶. Denoted 𝜎<sub>c</sub>(𝑅). 

• Projection: For some subset 𝑆 of attributes, produce from each tuple only the components for the attributes in 𝑆. The result of this projection is denoted 𝜋<sub>c</sub>(𝑅). 

• Union, Intersection, and Difference: Used for two relations with the same schema.

#### Operations on relations
• Natural Join: Given two relations 𝑅 and 𝑆, compare each pair of tuples, one from each relation. If the tuples agree on all the attributes that are common to the two schemas, then produce a tuple that has components for each of the attributes in either schema and agrees with the two tuples on each attribute. If the tuples disagree on one or more shared attributes, then produce nothing from this pair of tuples. Denoted 𝑅 ⋈ 𝑆.

• Grouping and Aggregation: 
	• Given a relation 𝑅, partition its tuples according to their values in one set of attributes 𝐺, called the grouping attributes. 
	• For each group, aggregate the values in certain other attributes. 
	• The normally permitted aggregations are SUM, COUNT, AVG, MIN, and MAX. 
	• The result of this operation is one tuple for each group. That tuple has a component for each of the grouping attributes, with the value common to tuples of that group. 
	• It also has a component for each aggregation, with the aggregated value for that group. 


##### Selections using MapReduce
• map: for each tuple 𝑡 in relation 𝑅, test if it satisfies 𝐶. 
	• If it does, produce <𝑡,𝑡> • Otherwise produce nothing 
• reduce: the identity function 
	• The values (or keys) form the relation 𝜎<sub>c</sub>(𝑅)

**Projections using MapReduce**

• **map**: For each tuple 𝑡 in 𝑅, construct a tuple 𝑡′ by eliminating from 𝑡 those components whose attributes are not in 𝑆. Output the key value pair <𝑡' ,𝑡′>. 
• **reduce**: <𝑡' ,\[𝑡' ,𝑡' ,…,𝑡'\]> ➝ <𝑡' ,𝑡′>

#### Operations using MapReduce
![](Pasted%20image%2020240917193031.png)
#### Natural Join using MapReduce
![](Pasted%20image%2020240917193115.png)
#### Group and Aggregation using MapReduce
![](Pasted%20image%2020240917193144.png)
![](Pasted%20image%2020240917193154.png)
![](Pasted%20image%2020240917193207.png)
![](Pasted%20image%2020240917193219.png)
# Week 4: Similar Items, Minhashing, Locality Sensitive Hashing 
### Hash Functions
- A hash function takes data of arbitrary size to fixed size values
	- Mapping integers to their remainder modulo 𝑚
	- Mapping strings to 32 bit integers
- Hash values are used as indices in arrays, or keys in dictionaries, where the data is stored – known as hash tables
- Example: strings of length 9, alphabet = {a,b,…,z,\_}
	- $27^9 ≈7.6×10^{12} ≈2^{43}$ possible strings (9 bytes each)
	- Hash to integer from 0 to 2<sup>32</sup>-1 (4 bytes)
- Want uniform coverage/few collisions
![]({03D1A8EE-B2FB-463E-A1CE-A710204FF5F9}.png)
## Shingles
- A 𝑞-shingle for a document is a sequence of 𝑞 consecutive “tokens” appearing in the document
	- Tokens can be characters, words, or something else depending on the application
	- For now assume tokens = characters
- Example: 𝒒 = 𝟐,document D= abcab
	- Set of 2-shingles: S(D) = {ab, bc, ca}
	- Option: shingles as multiset, count ab twice: S’(D) = {ab, bc, ca, ab}
**Whitespace**
-  Often makes sense to replace any sequence of one or more whitespace characters by a single blank
- Helps to distinguish shingles that cover one or more words from those that do not
- Example:
	- “The plane was ready for touch down” vs “The quarterback scored a touchdown”
	- Both contain ‘touchdown’ as a 9-shingle if whitespace is ignored
**Shingles and Similarity**
- Documents that are intuitively similar will have many shingles in common
- Changing a word only affects 𝑞-shingles within distance 𝑞 from that word
- Reordering paragraphs only affects the 2𝑞 shingles that cross paragraph boundaries
- Example 𝑞 = 3,“The dog which chased the cat” versus “The dog that chased the cat"
	- Only 3-shingles replaced are g\_w, \_wh, whi, hic, ich, ch_, and h\_c 

**Choosing the value of 𝑞**
- Too small:
	- Most documents will have most 𝑞-shingles
	- High similarity of documents even if they have none of the same sentences or phrases
- Too big:
	- Storing the shingles takes more space
- 𝑞 should be chosen so that the probability of any given shingle appearing in any given document is low.
	- Depends on how long the typical document is and how large the set of typical characters is
- Emails:
	- 𝑞 =5couldbegood
	- $27^5 =14,348,907$ possible shingle
	- Mostemails contain much fewer than 14 million characters
- More subtle than this
	- More than 27characters
	- Appear with different probability – some 5-shingles may be common
	- Rule of thumb: imagine there are only 20 characters – $20^q$ possible shingles
- For large documents (e.g., research articles) 𝑞 = 9 is considered safe

**Compressing Shingles**
- For large 𝑞 we might expect that most 𝑞-shingles do not appear in any of our documents
- Compress long shingles (e.g., q=10) by hashing them to (say) 4 bytes.
- Represent a document by the set of hash values of its shingles (still refer to them as shingles)
- Documents will still only have a small fraction of possible (hashed) shingles
- Two documents could (rarely) appear to have shingles in common, when in fact only the hash-values were shared

### Similarity of Sets
- The **Jaccard similarity** of two sets is the size of the intersection divided by the size of their union.
- $$𝑆𝑖𝑚(𝑆1,𝑆2) = \frac {|𝑆1⋂𝑆2|}  {|𝑆1⋃𝑆2|}$$
- ![]({8B04724F-0313-4383-AC4C-E30877188607}.png)
## Minhashing and Signatures of Sets

- If we have very many very large documents, we may not be able to store all of the sets of shingles in main memory
- Given a hash function ℎ, the minhash of a set 𝑆 with repsect to ℎ, denoted $\hat ℎ(𝑆)$, is
- $$ \hat ℎ(𝑆) = min \{ℎ(𝑠) :𝑠 ∈𝑆\} $$
- Use several (e.g., 100) independent hash functions to create signatures


- Set signature 
	- Pick 𝑘 hash functions $ℎ_1,ℎ_2,…,ℎ_𝑘$ independently 
	- These give 𝑘 minhashes $\hat ℎ_1 ,\hat ℎ_2 ,…, \hat ℎ_k$ 
	- 𝒔𝒊𝒈(𝑺) = \[$\hat ℎ_1 (S),\hat ℎ_2 (S),…, \hat ℎ_k (S)$ \]
- Jaccard similarity estimation
	- 𝑆𝑖𝑚 𝑆, 𝑇 ≈\[\#equal pairs in 𝑠𝑖𝑔(𝑆) and 𝑠𝑖𝑔(𝑇)\]/𝑘

![]({DD50D8AE-0541-4D66-8DB8-30B867611E2F}.png)
## Locality-Sensitive Hashing


# Week 5 Frequent itemsets

## Measures (Support)

Let 𝑰 be a set of items
$$
Support(I)=\frac{\text{\# baskets containing I}}{\text{totall number of baskets}}
$$

Meaning: Probability of finding all items of 𝐼 in one basket

Definition: We call an itemset 𝑰 frequent, if its support is greater than some initially fixed value $𝒔 ∈ ℝ$ $support(I) \geqq s$

## Measures for association rules: (Confidence)
Let 𝑰 be a set of items
Let 𝒋 be an item that is not in 𝑰.
Let 𝑰 → 𝒋 be an association rule
$$
confidence(i\rightarrow j) = \frac{\text{\#  baskets containing I $\cup$ \{j\}}}{\text{\#  baskets containing I}}
$$
Meaning: Probability of having all of 𝐼 ∪ {𝑗} in one basket, given that we already have all of 𝐼 in the basket
![]({3C7CBC73-4B92-4C8C-9777-9D31BB87665B}.png)
![]({1F41C49D-B80B-4F8E-B1BA-C64FE98B166F}.png)
## Measures for association rules: (Lift)
Let 𝑰 be a set of items
Let 𝒋 be an item that is not in 𝑰.
Let 𝑰 → 𝒋 be an association rule
$$
Lift(𝑰 → 𝒋) = \frac{confidence(𝑰 → 𝒋)}{Support(\{𝒋\})}
$$
Meaning: Probability of having 𝑗 in a basket given that we already have all of 𝐼 in the basket divided by the probability of having 𝑗 in a basket.

**Note**
- If lift (𝐼 →𝑗) is close to 1, then “not really interesting”
- If lift (𝐼 →𝑗)>>1  then 𝐼 encourages to buy 𝑗.
- If lift (𝐼 →𝑗)<<1  then 𝐼 discourages to buy 𝑗.
## Measures for association rules: (Interest)
Let 𝑰 be a set of items
Let 𝒋 be an item that is not in 𝑰.
Let 𝑰 → 𝒋 be an association rule
$$
interest (𝐼 →𝑗) = confidence(𝑰 → 𝒋)-Support(\{𝒋\})
$$
**Note**
- If Interest (𝐼 →𝑗) is close to 0, then “not really interesting”
- If Interest (𝐼 →𝑗) is highly positive then 𝐼 encourages to buy 𝑗.
- If Interest (𝐼 →𝑗 is highly negative then 𝐼 discourages to buy 𝑗.
## Finding frequent itemsets
Finding / determining interesting association rules is easy compared to finding frequent itemsets (what we have to do first anyway)

#### Store itemset count in main memory
For each frequent itemset algorithm we have to store the count for the itemsets
- (!) Use integers (4 byte) instead of item names / long IDs.
	- via a hash table.
- Limit on how many items we can deal with without thrashing(load the data more than once in the main memory)
	- Example: Say we have 𝑛 items and need to count all pairs of them. We must save about $n^2/2$ integers, each 4 bytes. So $2n^2$ bytes
	- If we have 2 GB main memory, say $2^{31}$ bytes, then  $n ≤ 2^{15} ≈ 33000$
### Triangular-Matrix Method
Let 𝑛 be the number of all items of which we want to count pairs
Use one-dimensional array a
a\[k\] contains count of the 𝑘-th pair of items
The ordering of the item pairs is as follows
{1,2},{1,3}, ... , {1,n}, {2,3}, ... , {n-1,n}
Formally: a\[k\] contains the count of the pair {i, 𝑗}  with 1 ≤ i < 𝑗 ≤ 𝑛, where $𝑘 = (i −1) (𝑛 −𝑖/2) +𝑗−i$

### Triples Method
Let 𝑛 be the number of all items of which we want to count pairs. Let $1 ≤ i < 𝑗 ≤𝑛$

Store counts as triples \[i,j,c\] where 𝑐 equals the count of {i,j}.
Advantage: No need to store pairs with count 𝑐 equal to 0
Disadvantage: For each pair we use 3 integers instead of just 1.
Hence: If fewer than 1/3 of all possible pairs appear in some basket,  then use the Triples Method

### Naïve algorithm for counting frequent itemsets
![]({B7980E80-C7AF-4D46-A732-F257CACC59E8}.png)
We measure the cost of computation by the number of passes an algorithm makes over the data. The time for reading data into main memory dominates the time for generating / counting the relevant pairs or small 𝑘𝑘-element itemsets.

### A-Priori Algorithm  (for item pairs)
**Observation**: If an itemset is frequent, then so is every subset of it
Definition: We call a frequent itemset **maximal** if no proper superset of it is frequent.
Saves space later: Only store maximal frequent itemsets

![]({957AB82E-4292-4895-99A5-346122F2AF1A}.png)
![]({0DECB59E-10E4-4266-8238-88F247D94B61}.png)
![]({5D20EC6E-550E-440B-B2CA-9D8F5CC3FF42}.png)
![]({963FE60C-D63A-4A4E-925D-F9E73A410F3B}.png)
#### For k- element itemsets
![]({9752B25A-F181-44BB-AF80-DF0AF54DC8D0}.png)
![]({08735E86-944A-4C76-861B-711292D4BEE2}.png)

### PCY-Algorithm
- A-Priori works fine unless generating $𝐶_2$ causes thrashing
- Utilise unused main memory in pass 1 for additional hash table
![]({8A9EF70E-DFF5-490B-94DE-B7177C406BA1}.png)
![]({B999AF71-C56B-4C2E-85C3-76BEFB460F18}.png)
![]({351C21F8-BB0A-48B4-A4CF-6C68E4AE6CC0}.png)
![]({50FBAD6B-256A-43F3-AEAD-CFC4A1FD0A0A}.png)

#### Advantages and disadvantages

- (+) We save memory if we have (many) buckets that are infrequent
- (-) We save memory if we have (many) buckets that are infrequent
	- In A-Priori: renumbered frequent singleton itemsets (form 1 to m instead to 𝑛) to save space
	- Now pairs in infrequent buckets could be rather randomly distributed within the lexicographic order.
- Hence: PCY only saves memory compared to A-Priori if 2 3of all candidate pairs can be discarded


# Week 6 Clustering
## Hierarchical Clustering
### Agglomerative (bottom up):
- Initialise each data point as a (singleton) cluster
- Repeatedly combine two “most similar / closest” clusters to a new one
- Stop w.r.t. certain criterion (e.g. # of clusters, density of clusters, …)
### Divisive (top down
- Initialise the whole data set as one big cluster.
- Recursively split up cluster which is least “dense / connected”.
- Stop w.r.t. certain criterion (e.g. # of clusters, density of clusters, …)

### Representation of clusters
**Euclidean case**
![]({7A4FC1F8-E9DE-49D6-BECC-56862544C8AA}.png)
**Non-Euclidean case**:
![]({8E70F003-92B6-4C07-BD51-1E8016EA0534}.png)
### Similarity / closeness of clusters
![]({5CF0D80B-CCE6-4DBE-A40E-CEE6B49DD399}.png)
- Define distance between two clusters 𝐶𝐶 and 𝐶𝐶𝐶 as the minimum distance between any data point from 𝑪𝑪 and any data point in 𝑪𝑪𝐶
- Set a notion of “cohesion” (or connectivity, density, …)
	- Merge two cluster whose union is most cohesive
	- Possible notions of “cohesion"
		- Minimum diameter: For a cluster 𝐶𝐶define its diameter diam(𝐶) as ![]({D03F9FCB-D560-47DD-AAE1-BFFB8757B0F0}.png)
		- Minimum average distance (between two points in the cluster): 
		- ![]({24EE1B97-3990-4564-BA5B-844B90BEF2D5}.png)
		- Minimum density
			- ![]({3E0A3879-1710-4E8B-8CFC-408640E741D5}.png)



### Evaluating the clustering (Davies-Bouldin index)
![]({0921033B-2E45-4BF5-908B-DF68107BA1B6}.png)


### (Naive) Hierarchical clustering algorithm
#### Agglomerative (bottom up)
- Initialise each data point as a (singleton) cluster. AND initialise representatives for clusters (centroid / clustroid)
- Repeatedly combine two “most similar / closest” clusters to a new one. BY computing in each turn distances between all pairs of clusters
-  w.r.t. certain criterion (e.g. # of clusters, density of clusters, …) BY computing abort parameter each turn. DTU Compute
If we have $𝑛$ data points in total. This part of the algorithm takes $𝑂(n^3)$ much time.
### (Naive) 𝑘-means algorithm
- Sometimes also called Lloyd’s algorithm, named after Stuart P. Lloyd
- During the algorithm, the number of clusters is fixed and equal to k
- **Point assignment method** no hierarchical approach
	- Every data point will be assigned to precisely one cluster.
- Iterative algorithm that assumes Euclidean setting.
- The algorithm alternates between two steps
	- Assignment step
	- Update step
**Step 0**  
- Choose 𝑘 data points as representatives (centroids), one for each (singleton) cluster
	- e.g. all at random
	- one at random and each next as far away from former(s) as possible
**Assignment step**
(Re)assign each data point 𝑝𝑝 to the cluster $C_i$ for which the distance between p and the centroid of $C_i$ is minimal, i.e. to the cluster $𝐶_i$ that minimises: $𝑑(𝑝,cent(𝐶_𝑖))$.
**Update step**
Recalculate the centroids for all clusters
**Termination rule**
- E.g. run until the process converges / stabilises, or
- stop if only a few / tiny proportion of data points were reassigned.
Then the running time is: $O(KknI)$.  Let 𝒏 be the total number of data points, Let 𝑰 denote the number of iteration.  Worst case $I=2^{\Omega (\sqrt{n})}$. it can be considered constant. 
#### Find k
![]({0504C992-4CD5-4D47-AD7C-6A6D084E0D99}.png)
#### Problem with 𝑘𝑘-means
![]({D1A51524-DFD2-411D-A82F-E2E50CCAD2FA}.png)
### CURE (Clustering Using REpresentatives) algorithm

- **Step 0** Pick random sample of data points fitting into main memory.
- **Step 1** Pick random sample of data points fitting into main memory.
- **Step 2**
	- Within each cluster $C_i$, pick a sample set $S_i$ of data points which is as “dispersed” as possible
	- Define set of representatives  $R_i$ of  $C_i$ as those points we get by  moving each point of  $S_i$ by some fraction towards cent( $C_i$).
	- Note: Points in  $R_i$ are not necessarily actual data points.
- **Next:** Merge two clusters if there are two representatives, one from each cluster, whose distance from each other is below some fixed threshold
	- pick new scattered representatives
- Now start clustering of the whole data by the following rule
	- **Assignment rule**: Place data point 𝑝 into cluster $𝐶_𝑖$ if 𝑝 is closest to a representative of $𝐶_𝑖$ among representatives of all clusters
- **Advantage**: Robust for data that is not well-distributed
- **Disadvantage**: Relatively costly with respect to running time
### DBSCAN algorithm
![]({E86E25DE-A63A-4966-864D-96F9CC6C6B5D}.png)
- **Initialise** Mark each point as unvisited
- Check each point 𝑝
	- if unvisited:
		- if 𝑝 is not core, mark as visited (rim point or outlier)
		- if 𝑝 is core, define cluster:
			- $C:=\{x|x\text{ is reachable from p and x is not already in a cluster}\}$
Running time (worst-case): $O(n^2)$
Advantage: Can identify clusters whose shape is more irregular
