## Entity-Relationship (E-R) modelling
### Entities and Relationships 
The data of an enterprise can be modelled as entities and relationships.
- An entity e
	- Is a thing/object, e.g. a specific student 
- An entity set E
	- Is a set of entities representing objects of the same type (having the same attributes) 
- Entity attributes
	- Each entity has properties called entity attributes. (column)
- A relationship
	- Is an association (e1, ..., en) among several entities ei.
- A relationship set among entity sets E1, ..., En
	- Is a set of relationships of the same type, i.e. a subset {(e1, ..., en) | ei ∈ Ei}.
- Relationship attributes
	- Each relationship can also have properties called relationship attributes

![[Pasted image 20240528122821.png]]



### Attribute Types
- Attributes
	- are descriptive properties possessed by all elements of an entity set or a relationship set
- Domain of an Attribute
	- is the set of permitted values for the attribute.
- Attribute Types
	- Simple (atomic) attributes
	- Composite attributes, which consist of component attributes.
	![[Pasted image 20240528123735.png]]
	- Multivalued attributes, which is a set of attribute values. E.g. phone_numbers
	- Derived attributes, which are computed from other attributes


### Order, Cardinality and Participation
- Order/Degree for relationship set
	- Is the number of entities (entity sets) being associated in a relationship (set)
	- If two entities (entity sets) are associated then the relationship (set) is Binary or if three is Ternary
- Cardinality for binary relationship set between entity sets E1 and E2:
	- **One-to-one**: An element in E1 can be associated with at most one (0..1) element in E2, and vice versa
	- **one-to-many:**  An element in E1 can be associated with many (0..\*) elements in E2, and an element in E2 can be associated with at most one (0..1) element in E.
	- **many-to-one:** An element in E1 can be associated with at most one (0..1) element in E2, and  an element in E2 can be associated with many (0..\*) elements in E1.
	- **many-to-many:** An element in E1 can be associated with many (0..\*) elements in E2 , and an element in E2 can be associated with many (0..\*) elements in E1.
- Participation in binary relationship set between entity sets E1 and E2:
	- Total Participation of E1 in a relationship set between E1 and E2:
		   All elements of E1 participate in at least one association with elements in E2. (So associations with 0 elements is not allowed.)
	- Partial Participation of E1 in a relationship set between E1 and E2 :  
			Some elements in E1 may not participate in any association with elements in E2. (So association with 0 elements is allowed.)
![[Pasted image 20240528125320.png]]


### Key for an Entity Set
- A (Super) Key of an Entity Set
	- A set of one or more attributes whose values uniquely determine each Entity element in an Entity Set
- A Candidate Key of an Entity Set
	- Is a Super Key which is minimal (i.e. has no other super key as a proper subset)
	- Example: ID is candidate key of instructor
- A Candidate Key is selected as Primary Key
	- Although several candidate keys may exist, one of the candidate keys is selected by to be the Primary Key
#### Key for a Relationship Set
- A Super Key of a Relationship Set
	- The combination of primary keys of the participating entity sets and the attributes of the relationship set forms a super key of a relationship set
- Deciding Candidate Keys of a Relationship Set
	- Must consider the semantics of the relationship set to decide whether attributes of the relationship set should contribute to candidate keys. (Usually they do not contribute, but if it is multivalued it contributes)
	- Must consider the cardinality of the relationship set when deciding what are the candidate keys. 
		- (student.ID, instructor.ID) is a candidate key, if the advisor relation is many-many.
		- student.ID is a candidate key, if the advisor relation is many-one (with student on the many side)
- Selecting the Primary Key
	- Remember to consider the semantics of the relationship set when selecting the primary key, since there can be more than one candidate key.
### Entity-Relationship Diagrams
- Entity sets are represented by rectangles with a name in the blue top
	- Entity attributes are listed inside the entity rectangle
	- Primary keys are underlined
- Relationship sets are represented by a diamond with a name listed inside, and lines connecting to the participating entities.
	- Relationship attributes are listed in a box, linked with a dashed line
	![[Pasted image 20240528132314.png]]
- The entity sets of a relationship set need not be distinct
	- Each occurrence of an entity set plays a “role” in the relationship set
	- The labels “course_id” and “prereq_id” are called roles![[Pasted image 20240528134221.png]]
###   Drawing Entity-Relationship
- We express cardinality constraints by drawing either a directed line  (→), signifying “one,” or an undirected line (—), signifying “many,”  between the relationship set and the entity set.![[Pasted image 20240528134543.png]]
- Total Participation
	- Every entity element participates in at least one relationship. This is shown by a double line
- Partial Participation
	- Some entity elements may not participate in any relationship. This is shown by a single line
![[Pasted image 20240528134820.png]]
#### E-R Diagram with a Ternary Relationship Set
- A ternary relationship (set) is a relationship (set) between three entities (entity sets).
- We allow at most one arrow out of a ternary (or greater degree) relationship set. E.g. an arrow from proj_guide to instructor indicates each student has at most one guide for a given project
- If there is more than one arrow, then there are several ways of defining the meaning. To avoid confusion at most one arrow is allowed
- To avoid handling the complexities of a ternary relationship set it can be substituted by three binary relationship sets!
![[Pasted image 20240528135118.png]]



### Convert Non-Binary Relationships to Binary
- In general, any non-binary relationship can be represented using binary relationships by creating an artificial entity set
	- Replace R between the entity sets A, B and C by an entity set E and three relationship sets:
		- RA, relating E and A
		- RB, relating E and B
		- RC, relating E and C
	- Create a special identifying attribute for E
	- Add any attributes of R to E
	- For each relationship (ai , bi , ci) in R, create:
		- A new entity ei in the entity set E
		- Add (ei , ai ) to RA
		- Add (ei , bi ) to RB
		- Add (ei , ci ) to RC
![[Pasted image 20240528135704.png]] to ![[Pasted image 20240528135716.png]]

### Strong and weak entity sets
- A strong entity set
	- is one that can form its primary key from its own attributes.
- A weak entity set
	- is one that cannot form its primary key from its own attributes alone.
- Identifying entity set
	- The existence of a weak entity set E_weak depends on the existence of an identifying entity set E_ident. There must be an identifying relationship set R between E_ident and E_weak: a one-to-many relationship set R from E_ident to E_weak, with total participation of E_weak. Identifying relationship sets are shown in double diamonds:
	![[Pasted image 20240528140244.png]]
	- Discriminator or partial key of E_weak
		- is a (sub)set of its attributes that can be used to distinguish an entity that depends on a particular strong entity.
	- Primary key of E_weak
		- is formed by the primary key of E_ident plus the partial key of E_weak.
![[Pasted image 20240528140534.png]]

### Summary 
![[Pasted image 20240528140646.png]]
![[Pasted image 20240528140723.png]]
![[Pasted image 20240528140752.png]]



### Converting E-R Diagrams to Relation Schemas
- Convert entity sets to relation schemas
	- special treatment of complex attributes
- Convert relationship sets to relation schemas
#### Convert Entity Sets to Relation Schemas
- First convert strong entity sets, then weak entity sets. If a weak entity set E1 has another weak entity set E2 as identifying entity set, then E2 should be converted before E1.
- A strong entity set (with atomic attributes)
	- Becomes a relation schema with the same attributes and primary key
	- Example: course(<ins>course_id</ins>, title, credits)
- A weak entity set (with atomic attributes)
	- Becomes a relation schema that additionally includes the primary key of the identifying entity set as a foreign key attribute.
	- Example: section(<ins>course_id</ins>, <ins>sec_id</ins>, <ins>semester</ins>, <ins>year</ins>) 
	![[Pasted image 20240529172152.png]]
#### Conversion of Complex Attributes
- Composite attributes
	- Example: name, address
	- Fix: Store leaf attributes like street_name only
- Multivalued attributes A
	- Example: { phone_number }, a set of phone numbers.
	- Fix: Store it instead in a separate relation together with the primary key: phones(ID, phone_number) foreign key ID references instructor
- Derived attributes
	- Example: age(), which can be computed from date_of_birth
	- Fix: Store only date_of_birth, not age – it can be calculated from data_of_birth when needed
- Conversion of the instructor entity set results hence in:
	- instructor(<ins>ID</ins>, first_name, ..., street_number, ..., zip, date_of_birth), 
	- phones(<ins>ID</ins>, <ins>phone_number</ins>)
	![[Pasted image 20240529172921.png]]
#### Convert Binary Relationship Sets to Relation Schemas
##### Convert Relationship Sets to Relation Schemas Many-to-Many relationship
- Many-to-Many relationship set R between E1 and E2 Is converted to a relation schema R with attributes from the primary keys of E1 and E2 and with any attributes of the relationship set.
- The primary key of R is the union of the primary keys of E1 and E2 (and in very, very rare cases some of the relationship attributes of R, cf. slide 12
- The primary keys of E1 and E2 become foreign keys in R referencing E1 and E2, respectively.
- Example: Relation Schema for the relationship advisor advisor(<ins>student.ID</ins>, <ins>instructor.ID</ins>, date)
![[Pasted image 20240529174337.png]]


##### Convert Relationship Sets to Relation Schemas One-to-Many relationship
- **Approach 1:** Can be converted to a relation schema R in the same way as for Many-to-Many relationship sets R, but the primary key of the resulting relation schema is different: it is the primary key of E1.
- Approach 2: It is possible to avoid making the new schema R, by instead modifying the schema of E1. This is explained on the following page. <small>(For simplicity, it is assumed that the primary key of relationship set R does not include relationship attributes.)</small> 
- **Approach 2 is usually preferred:** unless E1 has partial participation with only few entities in E1 taking part in R, as in
![[Pasted image 20240529175356.png]]

**APPROACH2** 
- Add the attributes A of R and primary key K2 of the E2 (one-side) schema to the E1 (many-side) schema. Hence, K2 becomes a foreign key in E1:
	- $E_1(\underline K_1,...,A,K_2),E_2(\underline K_2,...)$
- Example: Instead of creating a schema for the relationship inst_dept, add an attribute dept_name to the instructor schema:
	- department(<ins>dept_name</ins>, building, budget)
	- instructor(<ins>ID</ins>, name, dept_name, salary)
- instructor.dept_name is a foreign key of instructor referencing department.
- Total participation can be defined by a ”Not Null” constraint for instructor.dept_name


##### Convert Relationship Sets to Relation Schemas One-to-One relationship
- Approach 2 (from many-to-many) is preferred unless both both E1 and E2. have partial participation with only few entities taking part in R.
![[Pasted image 20240529180526.png]]
- The primary key from one of the entity schemas can together with the attributes A of R be added to the other entity schema as a foreign key. This gives two possible solutions:
	- $E_1(\underline K_1,..,A,K_2),E_2(\underline K_2,...)$
	- $E_1(\underline K_1,...),E_2(\underline K_2,...,A,K_1)$ 
- Choose the solution, which gives the least number of Null Values:
	- For solution 1 it holds that: only if participation of E1 is partial, then for some rows in E1, K2 will be Null.
	- For solution 2 it holds that: only if participation of E2 is partial, then for some rows in E2, K1 will be Null.
	- So if e.g. the participation of E1 is total and E2 is partial, then go for solution 1.
## Formal Query Languages
- Simple Relation Schema and SQL Query
	- Relation Schema R(A,B,C,D) 
	- SELECT B, A FROM R WHERE D>10
- Relational Algebra: use Relation Variables like R
	- $\Pi_{B,A}(\sigma_{D>100}(R))$ 
- Domain Calculus: use Domain Variables like a,b,c,d
	- $\{<b,a>|\exists_{c,d} (<a,b,c,d>\in R \land d> 100)\}$ 

### Relational Algebra 
- basic operation ![[Pasted image 20240528170250.png]]
- ![[Pasted image 20240528170323.png]]
- Examples
	- 10.1.1 Selection Operation  Find Instructor rows for instructors in the physics department earning more than 90000
		- $\sigma_{Deptnam='Physics'\land Salart > 9000}(Instructor)$ 
	- 10.1.2 Projection Operation Find name and IDs of all instructors
		- $\Pi_{InstName, InstID}(Instructor)$
	- 10.1.3 Set Union Find name and IDs of all instructors and students.
		- $\Pi_{InstName, InstID}(Instructor) \cup \Pi_{StudName, StudID}(Student)$ 
	- 10.1.4 Set Difference Find instructor IDs for instructors that have not taught any courses yet
		- $\Pi_{InstID}(Instructor) – \Pi_{InstID}(Teaches)$ 
	- 10.1.5 Cartesian Product Combine each instructor with each student
		- $Instructor \times Student$ 
	- 10.1.6 Rename Find salary of the instructor with the highest salary. (Use T as a temporary relation).
		- $$\Pi_{Salary}(Instructor) –  \Pi_{Instructor.Salary}  (\sigma_{Instructor.Salary < T.Salary} (Instructor \times \rho_T(Instructor)))$$
### Derived Operations in Relational Algebra
- Below R and S are relations, and {A1,..., An} is the intersection of their attributes.
- Θ is a predicate over attributes in the union of attributes of R and S
- ![[Pasted image 20240528171516.png]]
### Deriving Operations from Basic Operations
- Natural Join can be derived from Cartesian Product, Selection and Projection
	- Schemas R(A,B,C,D), S(B,D,E) and Natural Join Schema T(A,B,C,D,E)
	- $T \equiv R \bowtie S \equiv \Pi_{R.A,R.B,R.C,R.D,S.E}(\sigma_{R.B=S.B \land R.D=S.D}(R \times S))$
- Left Outer Join can be derived from Cartesian Product, Selection, Projection, Set Difference and Set Union
	- Schemas R(A,B,C), S(A,C,D,E) and Left Outer Join Schema T(A,B,C,D,E)
	- $T ≡ R \bowtie S ≡ (R \bowtie S) \cup ((R – ∏_{R.A,R.B,R.C} (R \bowtie S)) \times \{(Null, Null)\}$)
	- (Here $R – ∏_{R.A,R.B,R.C} (R ⋈ S)$ contains those tuples of R that did not match any tuple in S.) R ⋈ S can be further decomposed using Cartesian Product, Selection & Projection
- Set Intersection can be derived from Set Difference
	- $R \cap S \equiv R - (R - S)$ 
![[Pasted image 20240528172616.png]]
- Example
	- ![[Pasted image 20240528172713.png]]
### Extended Operations Relation Algebra
![[Pasted image 20240528172848.png]]
- Examples 
	- ![[Pasted image 20240528172932.png]]

### Using Relational Algebra for Language Semantics and Query Optimizing
- Relational Algebra can be used to specify the meaning of relational languages
	- SELECT A1,..., An FROM R1,..., Rn WHERE P
	- $\Pi_{A_1, ..., A_n}(\sigma_P(R_1 × ... × R_n ))$
#### Query Optimization 
- Queries consist of several atomic operations
	- Selection, projection, join, etc....
- Operations can be executed in different order, as long as the result remains the same
	- E.g., selection then join vs join then selection
- The order in which operations must be performed is called execution plan
	- Although leading to the same result, execution plans can have substantial differences in processing time and memory usage
- Optimization finds the best execution plan
	- Completely carried out by the DBMS
	- Users do not have to optimize their queries
**Example**
- Find the names of instructors in the Comp. Sci. department together with the course titles of the courses that the instructors teach
	- SELECT InstName, Title FROM Instructor NATURAL JOIN Teaches NATURAL JOIN Course WHERE deptname = ‘Comp. Sci‘
- This SQL query can be rewritten in relational algebra
	- $\Pi_{InstName, Title} (\sigma_{DeptName=‘Comp. Sci.’} (Instructor ⋈ Teaches ⋈ Course))$
- From this expression, the DBMS applies transformation rules to calculate equivalent expressions
- The DBMS then selects the one that minimizes the processing time
	- Some transformation rules always produce more efficient expressions
	- In the other cases, the DBMS estimates the cost of each operation, and selects the expression that minimizes the total cost
#### Syntax tree
- Relational algebra expressions can be represented as a syntax tree
	- Leaf nodes represent the relations
	- All other nodes represent the operations
	- The closest a node is from the leaves, the sooner the corresponding operation will be performed
**Example** 
- Find the names of instructors in the Comp. Sci. department together with the course titles of the courses that the instructors teach.
	- $\Pi_{InstName, Title} (\sigma_{DeptName=‘Comp. Sci.’} (Instructor ⋈ Teaches ⋈ Course))$ 
	- ![[Pasted image 20240529114739.png]]
#### Equivalence of expressions
- Making selection operations atomic
	- Can be done only if θ1 and θ2 are in conjunction (^).
	![[Pasted image 20240529114918.png]]

	- Expanding projection operations
		- Can be done only if L ⊆ L2.
			![[Pasted image 20240529115133.png]]

	- Switching natural joins
		- Can be done only if θ1 does not refer to attributes of r
		![[Pasted image 20240529115224.png]]
	- Removing cartesian product operations
		![[Pasted image 20240529115250.png]]
			
#### Query optimizations
- Pushing selection operations before joins
	- Can be done only if θ2 refers only to attributes of r.
		![[Pasted image 20240529115611.png]]
- Pushing selection operations before unions
		![[Pasted image 20240529115634.png]]
- Pushing selection operations before differences
	- ![[Pasted image 20240529115652.png]]
- Pushing projection operations before joins
	- LR = L ∩ Schema(r)          JR = Attributes(θ) ∩ Schema(r)
	- LS = L ∩ Schema(s)          JS = Attributes(θ) ∩ Schema(s)
		![[Pasted image 20240529120027.png]]
- Query optimizations
	- Pushing projection operations before unions
		![[Pasted image 20240529120056.png]]
	
**Example** 
- Find the names of instructors in the Comp. Sci. department together with the course titles of the courses that the instructors teach.
	- $\Pi_{InstName, Title} (\sigma_{DeptName=‘Comp. Sci.’} (Instructor ⋈ Teaches ⋈ Course))$ 
	- ![[Pasted image 20240529114739.png]]
	AFTER optimizations
	![[Pasted image 20240529120311.png]]
	
### Domain Calculus- Basic operations
- Domain Calculus restricted to Safe Expressions (i.e. expressions with a finite number of tuples in the result), is equivalent to Relational Algebra in expressive power with regard to the Basic operations and derived operations.
- Domain Calculus can be extended to handle arithmetic expressions as well as aggregation operations. However, that is outside the scope of this lecture
![[Pasted image 20240528173319.png]]
![[Pasted image 20240528173341.png]]
Examples
- ![[Pasted image 20240528173414.png]]


### Domain Calculus- Derived Operations
![[Pasted image 20240528173518.png]]
examples 
![[Pasted image 20240528173531.png]]

## Normalization
- **Normalization:**  is a simple, practical technique used to  minimize data redundancy and avoid modification anomalies.
	- **What:** Normalization involves decomposing a table into smaller tables without losing information and by defining foreign keys.
	- **The objective:** is to isolate data so that additions, updates and deletions can be made in just one place, and then the changes can be propagated through the rest of the database using the defined foreign keys. The **goal** is both to ensure data consistency and to avoid extensive searches.
- Redundancy is the Evil of Databases
	- Redundancy means that the same data is stored in more than one place
	- The problem with redundancy is the risk of modification anomalies: i.e. when data are modified (with SQL INSERT, UPDATE and DELETE commands), there is a risk that the data are not modified everywhere making the database inconsistent
	- The problem with an inconsistent database is that it might return different answers to the same question asked in different ways.
	- If the database is redundancy free, then the DBMS can modify the data efficiently, without having to search the entire database
- Features of Good Relational Design
	- To avoid modification anomalies, additional restrictions can be imposed on tables/relation schemas
	- Tables/relation schemas with such restrictions are said to be in a given normal form like “Third Normal Form” or 3NF.
	- Normalization is the process of bringing tables and their relation schemas to higher normal forms by avoiding redundancy. Normally it is done by projections of a table into two or more, smaller tables
	- Normalization is information preserving: it provides data lossless decompositions (usually projections), and is fully reversible, normally by joining the normalized tables back into the original table.
![[Pasted image 20240529182114.png]]
- Mathematics needed to define 1NF, 2NF, 3NF and BCNF
	- Functional dependencies: attribute associations within a relation (schema)
		- Trivial and nontrivial functional dependencies
		- Keys defined in terms of functional dependencies
		- Armstrong’s axioms and derived theorems: to find derived functional dependences
- Mathematics needed to define 4NF (later)
	- Multivalued dependencies
### Theoretical Foundations: Functional Dependencies

- Describe dependencies between attribute sets A and B of a relation schema R:
	- Basically a many-to-one relation of associations from one set A of attributes to another set B of attributes within a given relation
- Example: Consider Shipments(Vendor, Part, Qty):
![[Pasted image 20240529182511.png]]
{Vendor, Part} → {Qty} is a functional dependency holding for Shipments
The attribute set {Vendor, Part} functionally determines the attribute set {Qty}.
The attribute set {Qty} is functionally dependent of the attribute set {Vendor, Part}.
#### Formal Definition of Functional Dependency
Let X and Y be subsets of the set of attributes of a relation schema R .
- A functional dependency X → Y holds for R if and only if
	- in every legal instance of R, each X value has associated precisely one Y value (i.e. whenever two rows have the same X value, they also have the same Y values).
- When X → Y holds for R, we say
	- Y is functionally dependent of X and X functionally determines Y
	- X is said to be the determinant and Y the dependent
	- If $X = {A_1, ..., A_n}, Y = {B_1, ..., B_m}$ we sometimes write $A_1, ..., A_n -> B_1, ..., B_m or A_1 ... A_n -> B_1 ... B_m$ 
- Trivial Dependencies
	- X → Y is trivial, if Y ⊆ X.
	- Example 1: {Vendor, Part} → {Vendor}
	- Example 2: {Part} → {Part}
- Nontrivial Dependencies
	- Those FDs which are not trivial.
	- Nontrivial FDs lead to definitions of integrity constraints like keys.
- Example: Shipment1(Vendor, City, Part, Qty) (where a vendor only can be in one city)
![[Pasted image 20240529183232.png]]
- To decide whether a FD is valid <small>(legal for all relation instances)</small>, one has to consider the real world.
- Of a set of 4 attributes, it is possible to make  2^4= 16 subsets.
- Combining 16 determinants with 16 dependents gives 256 potential FDs
- Of the 256 potential FDs, some are valid and some are not
- **(Canonical) Cover Set:** A (irreducible/minimal) set of valid functional dependencies from which all valid functional dependencies can be determined. In the example: {{Vendor, Part} → {Qty}, {Vendor} → {City}} is a canonical cover set.
- **Closure Set F+ of a set F of functional dependencies**: The set of all valid functional dependencies that can be logically derived from the F
#### Super Key and Functional Dependencies
Let R be a relation schema and let K be a subset of the set A_R of attributes of R.
- Super key, original definition:
	- K is a *superkey* of R, if, in any legal instance of R, for all pairs t1 and t2 of tuples in the instance of R if t1 ≠ t2, then t1\[K\] ≠ t2\[K\]. Hence, a specific K value will uniquely define a specific tuple in R.
- Super key defined by functional dependencies:
	- K is a superkey of R, if K → X holds for every attribute X of R.
	- Super Key examples for Shipment1(Vendor, City, Part, Qty)
		- {Vendor, City, Part} Knowing the values of Vendor, City and Part, the value of Qty is defined
		- However, of interest is a Super Key with a minimum set of attributes:{Vendor, Part} Knowing the values of Vendor and Part, the values of City and Qty is defined.
#### Candidate Key and Functional Dependencies
- Candidate key is a minimal super key:
	- K is a candidate key for R if and only if, $K → A_R$ and for no $α  \subset K, α → AR$  
	- Candidate key example for Shipment1(Vendor, City, Part, Qty):  {Vendor, Part}
	- In some cases R can have several candidate keys!
- Primary key
	- A candidate key is selected by the DBA to be the primary key for a relation.
#### Armstrong’s Rules
Let R be a relation schema and let F be a set of functional dependencies on R.
- Armstrong’s Rules
	- Armstrong’s Rules are some Axioms and Derived Theorems for deriving functional dependences from F
	- Armstrong’s Rules are
		- **sound:** only valid FDs are derived from valid FDs.
		- **complete:** all valid FDs (the closure set of F) can be derived from a cover set F.
	- can e.g. be used for finding candidate keys
- Let X, Y, Z and V be sets of attributes.
- Armstrong’s axioms
	1) Reflexivity: If Y is a subset of X, then X → Y
	2) Augmentation: If X → Y, then XZ → YZ
		- Note: XZ is used as a shorthand for X U Z
		- Note: XY = YX and XX = X; Sets have no order and no repetitions
	3) Transitivity: If X → Y and Y → Z, then X → Z
- Derived theorems
	4) Self-determination: X → X
	5) Decomposition: If X → YZ, then X → Y and X → Z
	6) Union: If X → Y and X → Z, then X → YZ
	7) Composition: If X → Y and Z → V, then XZ → YV
	8) General Unification: If X → Y and Z → V, then X U (Z – Y) → YV (where ‘U’ is Set Union and ‘–’ is Set Difference)
	Example
		![[Pasted image 20240529191507.png]]
		
###  Normal Forms 1NF-3NF
#### Definitions
- Original Definitions???????????
	- 2NF and 3NF are defined using the notion of functional dependencies. The starting point for the definitions is:
		- a relational schema R
		- a cover set FD of functional dependencies for R
	- **The original definitions** (here) assume the tables have one candidate key which has been chosen as primary key, in the following called the key.
	- **The generalized definitions** of 2NF and 3NF (in the book) generalizes the original definitions by taking several candidate keys into account.
	- When there is only one candidate key, the original and the generalized definitions are equivalent.
- Defined Informally
	- 1st normal form
		- All attributes depend on **the key**.
	- 2nd normal form
		- All attributes depend on **the whole key.**
	- 3rd normal form
		- All attributes depend on **nothing but the key**


**General Definitions**
- When there is only one candidate key, the original and the generalized definitions of 2NF-3NF are equivalent.
- When there is only one candidate key: 3NF and BCNF are the same
- 3NF and BCNF are the same (according to Date) unless
	- there are several composite candidate keys CK1 and CK2
	- which are overlapping (CK1 ∩ CK2 ≠ {})
	This exception is very rare
- 2NF-3NF General Definitions
	- For a relation schema R to be in Second Normal Form 2NF
		1) It must be in 1NF.
		2) Each non candidate key attribute A must not depend on a strict subset $K_{part}$ of any candidate key attribute set K, i.e. $K_{part}$ -> A and $K_{part}$ ⊂ K must not be the case. A must depend on the entire K.
	- For a relation schema R(K, ..., A) to be in Third Normal Form 3NF
		1) It must be in 2NF.
		2) No non candidate key attribute A depends **transitively** on any candidate key K via other attributes B, like K -> B -> A, where B -/-> K and B -> A is non-trivial (i.e. A ⊈ B).
#### 1NF – First Normal Form
- For a relation to be in First Normal Form 1NF
	- Each attribute value must be a single value<small> (is atomic, not multivalued or composite).</small> 
- Example
	- OrdersTable below is not in 1NF as the values of the attribute ItemNos are <ins>not</ins> atomic
	![[Pasted image 20240529194102.png]]
	- Normalization to 1NF
	![[Pasted image 20240529194153.png]]

#### 2NF – Second Normal Form
- For a relation schema R to be in Second Normal Form 2NF
	1) It must be in 1NF.
	2) Each non primary key attribute A must not depend on a strict subset $K_{part}$ of the primary key attribute set K, i.e.  $K_{part}$-> A and  $K_{part}$ ⊂ K must not be the case. A must depend on the entire primary key K.
- Some special cases where a 1NF relation schema is in 2NF:
	- The primary key consists only of one attribute.
	- The primary key consists of all attributes in the relation.
- Normalisation 1NF to 2NF:
	- Move the set of all attributes A which depend on a  $K_{part}$ ⊂ K to a new relation R2 together with a copy of  $K_{part}$, which becomes the primary key as well as a foreign key. <small>If there is only one such attribute A we have:  </small> 
		$R(\underline K, ..., A) \Rightarrow R_1(\underline K, ...)$ <small>foreign key $K_{part}$ references R2</small>, $R2(\underline K_{part}, A)$ 
	- Repeat the step above, if R1 or R2 are not yet in 2 NF
	- Decomposition is data lossless:
		- `(select K, ... from R) natural join (select Kpart, A from R) = select * from R`
**EXAMPLES**
- Orders1NF is in 1NF, but not 2NF
![[Pasted image 20240529195703.png]]
- Normalization to (Orders2NF and Items in) 2NF
	- Orders2NF(<ins>OrderNo</ins>, <ins>ItemNo</ins>) foreign Key(ItemNo) references Items(ItemNo)
	- Items(<ins>ItemNo</ins>, ItemName)
	- Note that ItemNo is included in the key of Orders2NF, as OrderNo -/-> ItemNo
![[Pasted image 20240529195903.png]]

#### 3NF – Third Normal Form
- For a relation schema $R(\underline K, ..., A)$ to be in Third Normal Form 3NF
	1) It must be in 2NF.
	2) Each non primary key attribute A must depend **directly** on the entire primary key K. It must not depend **transitively** via other attributes B, like K -> B -> A, where B -/-> K and B -> A is non-trivial (i.e. A ⊈ B).
- Normalization 2NF to 3NF
	- Move attributes A which are transitively dependent on K via B, i.e. K -> B -> A for some attribute set B, to a new relation R2 together with a copy of the dependent attributes in B, which become the primary key and constitute a foreign key: $R(\underline K, ..., B, A)\Rightarrow R1(\underline K, ..., B)$ foreign key B references R2, $R2(\underline B, A)$
	- Repeat the step above, if R1 or R2 are not yet in 3 NF.
	- Decomposition is data lossless: `(select K, ..., B from R) natural join (select B, A from R) = select * from R`
Example
- Customer2NF in 2NF
![[Pasted image 20240529200906.png]]
- Normalization to 3NF
![[Pasted image 20240529200939.png]]

#### BCNF - Boyce-Codd Normal Form
- Boyce-Codd Normal Form BCNF
	- A relation schema is in Boyce-Codd Normal Form BCNF, if and only if, every nontrivial, left-irreducible FD α→β has a candidate key as its determinant α
	- Left-irreducible means that there is no proper subset s of α such that s → β .
- Normalization of $R(A_1, ..., A_n , B_1, ..., B_n , C)$ to BCNF:
	- Assume $B_1, ..., B_n → C$ is nontrivial, left-irreducible, but$B_1, ..., B_n$ is not a candidate key.
	- $R(A_1, ..., A_n , B_1, ..., B_n , C)\Rightarrow R_1(A_1,...,A_n,B_1,...,B_n)$ foreign key $B_1, ..., B_n$ references R2 and $R_2(B_1, ..., B_n , C)$ 
	- Repeat the step above, if R1 or R2 are not yet in BCNF.
- Normalization to BCNF, example:
	- $R(\underline A, B, C)$ with $FD = \{A→B,B→C\}$ is not in BCNF (B → C, but B is not candidate key)
	- Decomposition of R: R1(<ins>A</ins>,B) and R2(<ins>B</ins>, C).
**Example** 
![[Pasted image 20240529203358.png]]

#### Multivalued Dependencies
Let R be a relation schema and α and β be disjoint subsets of the attributes of R and let Υ be the remaining attributes of R
- A multivalued dependency α →→ β holds for R if and only if, in every legal instance of R, the set of β values matching a given α γ value pair depends only on the α value and is independent of the γ value.
- When  α →→ β we say  α  multivalue determines  β
- When α →→ β  holds for R, then α →→ γ also holds for R
- When α → β holds for R, then α →→ β also holds for R.
![[Pasted image 20240529204029.png]]
**Example**
![[Pasted image 20240529204052.png]]

#### 4NF - Fourth Normal Form
- Fourth Normal Form 4NF
	- A relation schema R is in Fourth Normal Form 4NF, if and only if, whenever
		- a non-trivially α →→ β holds, then α is a key of R (all attributes of R are also functionally dependent on α )
	- α →→ β is trivial means  β ⊆ α or α ∪  β = set of all attributes in R.
- Normalization from BCNF to 4NF
	- Assume α →→ β non-trivially holds and α is NOT a key of R.
	- This usually arises from many-to-many relationship sets or multivalued entity sets
	- $R(α, β, γ) R1(α, β), R2(α, γ)$
- Decomposition is data lossless: `(select ,  from R) natural join (select , Υ from R) = select * from R`
Example
![[Pasted image 20240529204728.png]]

## Indexing and Hashing
### File Structure and Organization
- File Structure
	- A file is a sequence of records.
	- A record is a sequence of fields.
- A record is a sequence of fields
	- A database is stored in files on a disk:
		- Each table (relation) is stored in a file:
			- Each row is stored in a record of the file:
				- Each attribute value is stored in a field in the record.
	- The files are partitioned into fixed-length storage units called blocks

- File organization refers to the way records are organized in a file.
	- Heap File
		- A record can be placed anywhere in the file where there is space
	- Sequential File
		- Records are stored in sequential order, based on the value of the search key of each record
	- Hash File
		- A hash function computed on some attribute of each record; The result specifies in which block of the file the record should be placed
- <big>How to Find a Record Fast?</big>
- SELECT statements give rise to searching for Records
	- Example: SELECT InstName FROM Instructor WHERE InstID = 15151;
	- The Instructor file is in 2 Blocks on the Disk.
- 2 ways 
1) Use an Index File of Search Keys
	- 1st field: <ins>Search Key</ins>
	- 2nd field: Pointer
	- The Pointer points to the start of the Block where the Record is stored, or directly to the record.
2) Use a Hashing Algorithm on the Search Key
	- Hash Algorithm: Search Key % 3 +1
	- Block Address: 15151 % 3 + 1 = 2
	- The x % y modulus operation is the remainder of x/y Using hashing saves to have the index file.


### Indexing
#### SQL 
- Create an index
	- `CREATE [UNIQUE] INDEX <index-name> ON <table-name>(<attribute-list>)`
	- Like: `CREATE INDEX NameIndex ON Instructor(InstName);`
	- Most database systems allow a specification of the type of index
	- Like: `CREATE INDEX NameIndex ON Instructor(InstName) USING BTREE;`
	- `CREATE INDEX NameIndex ON Instructor(InstName) USING HASH;`
	- MariaDB uses BTREE as default, if no type is specified.
	- MySQL and MariaDB will automatically for each table define a Primary Index on the Primary Key and Secondary Indexes for all Foreign Keys
- To drop an index
	- `DROP INDEX <index-name> ON <table-name>;`
	- Like: `DROP INDEX NameIndex ON Instructor;`
- To show the index set for a table
	- `SHOW INDEX FROM <table-name>:`
	- Like: `SHOW INDEX FROM Instructor;`
#### Ordered Index Files
- (Sequentially) Ordered Index
	- Index File <ins>is sorted on the Search Key</ins> which is one or several attributes from the data file
	- Since it is sorted we can <ins>use binary searching </ins>to find an index
- Primary Index
	- Index File and Data File are both sorted on the same Search Key
	- It is the Primary Index that <ins>determines the order</ins> of the records in the Data File. The Index is also called a Clustering Index, since it determines the clustering of records in blocks
	- A relation has <ins>at most one</ins> Primary Index. It is usually (but not necessarily) defined on the primary key attribute. In many DBMS the primary key automatically gets an index.
- Secondary Index = Index which is not Primary
	- Index File and the Data File are sorted on <ins>different</ins> Search Keys
	- A Secondary Index <ins>does not determine any order</ins> on the records in the Data File
	- Exists only, if there is a Primary Index.
	- A relation can have <ins>several</ins> Secondary Indexes.

#### Dense Versus Sparse Index Files
- Dense Index
	- The Index File <ins>has a record for every Search Key</ins> value in the Data File
- Sparse Index = Non-Dense Index
	- The Index File <ins>has not a record for every Search Key</ins> value in the Data File
	- <ins>Only possible for Primary Indexes</ins> (Index File and Data File sorted on the same Search Key), as the same ordering is needed to locate records
- Which Search Keys to include in the index:
	- A good choice is to include each key of the 1st record in every block in the file:
		- 1st record in index points to the 1st record in the 1st block of the data file
		- 2nd record in index points to the 1st record in the 2nd block of the data file


- Substantial time saving benefits in searching for records
	- Searching using Primary, Ordered Index File is efficient.
	- Searching using Secondary, Ordered Index File is less efficient:
		- Each record access may fetch a new Block from disk!
##### Index File Update After an Insertion in the Data File
- Single-level Index File record insertion
	- Search for the Search Key value of the record to be inserted
	- Dense Index: If the Search Key value does not appear in the index, then insert it
	- Sparse Index:
		- If the Index File stores an entry for each block of the Data File, no change needs to be made to the Index File unless a new block is created
		- If a new block is created, the first search key value appearing in the new block is inserted into the Index File.
- Multilevel insertion and deletion
	- Algorithms are simple extensions of the single-level algorithms
##### Index File Update After a Deletion in the Data File
![[Pasted image 20240530114554.png]]

##### Sparse Index Compared to Dense Index
- Sparse Index
	- +: Less space and less maintenance overhead for INSERT and DELETE
	- -: Generally slower than Dense Index for locating records.
- Good tradeoff solution
	- Sparse Index File with one index entry per block in the Data File:
		- 1st record in index points to the 1st record in the 1st block of the data file
		- 2nd record in index points to the 1st record in the 2nd block of the data file
		- ... 
#### Primary indexes: Examples 
##### Dense index on (primary) key attribute
	![[Pasted image 20240530112046.png]]
	<big>Queries Using Dense Index</big>
- To locate records with Search Key value = K (e.g. Instructor.InstID = 45565)
- Find the record with Search Key value V = K in the Index File.
- Follow the associated pointer to the record in the Data File and read the record (block containing it) into main memory.




##### Dense index on non primary key attribute
![[Pasted image 20240530112634.png]]
	<big>Queries Using Dense Index</big>
- To locate records with Search Key value = K (e.g. Instructor.DeptID = History)
- Find the record with Search Key value V = K in the Index File.
- Follow the associated pointer to the Data File and search sequentially from there
![[Pasted image 20240530112853.png]]

##### Sparse index on (primary) key attribute
![[Pasted image 20240530113622.png]]
	<big>Queries Using a Sparse Index</big>
- To locate a record with Search Key value K (e.g. Instructor.InstID = 45565)
- Find the record with largest Search Key value V <= K (V = 32343) in the Index File.
- Follow the pointer to a record in the data File and start sequential searching


#### Secondary indexes
- Secondary Index Files
	- Help finding all records whose values in a certain field satisfy some condition.
	- The Instructor table is stored sequentially by InstID, finding all instructors in a particular department is easy with a Secondary Index on DeptName.
	- A Secondary Index on Salary is helpful finding instructors with a specified salary or with a salary in a specified range of values.
- Secondary Index Files and MySQL/MariaDB
	- MySQL/MariaDB will automatically make a Primary Index for the Primary Key and Secondary Indexes for all Foreign Keys.

##### Secondary indexes:
- Dense index on key attribute: no example (similar to Primary Dense index on (primary) key attribute, but with arrows crossing)
- Sparse index is not an option
- Dense index on non primary key attribute
![[Pasted image 20240530120220.png]]
##### Querying Using a Secondary Index
- To locate a record with Search Key value K (e.g. Salary = 80000)
	- Find the record with Search Key value V = K in the Index File.
	- Follow the associated pointer to the associated bucket and follow each of the bucket pointers to a record in the Data File and read the record (block containing it) into main memory.
	- ![[Pasted image 20240530121903.png]]
#### Multi Column Index
- has Composite search keys containing more than one attribute
	- E.g. a secondary index on instructor(DeptName, Salary) or a primary index on Classroom(Building, Room).
- Index file is lexicographically ordered: (a1, a2) < (b1, b2) if either
	- a1 < b1, or a1=b1 and a2 < b2
##### Searching Using a Multi Column Index
- If the table has a multiple-column index on (A1, A2, ..., An), it provides indexed search capabilities on:
	- (A1), (A1, A2), ..., and (A1, A2, ..., An).
- Example: having a secondary index on (DeptName, Salary) for Instructor facilitates queries like:
	- `SELECT ... FROM Instructor WHERE DeptName = “Finance” AND Salary  = 80000`
	- `SELECT ... FROM Instructor WHERE DeptName = “Finance”`
- but not a query like
	- `SELECT ... FROM Instructor WHERE Salary = 80000`





#### Balanced Tree
- A B+ Tree Index File is a dynamic multi-level index structured as a B+ tree.
- B+ Trees are special cases of tree data structures

- Disadvantage of Index-Sequential Files
	- Performance degrades as file grows (e.g. due to overflow blocks)
	- Periodic reorganization of the entire file is required
- Advantage of a B+-Tree Index File
	- Automatically reorganizes itself with small, local, changes after insertions and deletions.
	- Reorganization of the entire file is never required.
- Minor disadvantage of B+-Trees
	- Extra insertion and deletion overhead, space overhead
- Advantages of B+-Trees outweigh disadvantages
	- B+-Trees are used extensively in most database systems although they call them B-Trees!

<big>Observations about B+-Trees</big>
- Major Observations
	- The leaf nodes form a Dense Index.
	- The
	- non-leaf nodes form a Sparse Index to the leaf nodes.
	- The B+-Tree is 
	- effective for queries and modification.
- Detailed Observations
	- Contains a relatively small number of levels.
	- This means that searches are conducted efficiently
	- Insertions and deletions to the data file can be handled efficiently, with a minimum overhead of index manipulation

#####  B+-Tree of Order n 
- The B+-Tree constraints and properties
	- Each node has one more pointer than search key values, except last leaf node
	- The tree must be sorted:
		- Keys must be in sorted order in each node: K1 < ... < Kq-1. (Assuming no duplicate search keys.)
		- The keys X in subtrees must satisfy:
		![[Pasted image 20240530155731.png]]
		- The leaf nodes must ordered, so Ki < Kj for ![[Pasted image 20240530155801.png]] 
	- The tree must be balanced (“B” in the B+-Tree stands for Balanced), i.e. all paths from the root to a leaf are of the same length
	- Filling of non-root nodes:
		- Each internal node has min ⌈n/2⌉ and max n pointers to subtrees / (children) and min ⌈n/2⌉ - 1 and max n - 1 search key values
		- A leaf node has min ⌈(n – 1)/2⌉ and max (n–1) search key values and pointers to records in the datafile
		- Above ⌈x⌉ denotes the least integer i for which x <= i.
	- The basic idea is to leave room in the nodes for new search keys.
##### Examples 
![[Pasted image 20240530160438.png]]
![[Pasted image 20240530160452.png]]

##### Operations on B+-Trees
###### Queries
![[Pasted image 20240530161738.png]]


- Queries on B+-Trees of order n
	- If there are K search-key values in the file, the height of the tree is no more than $⌈log_{⌈n/2⌉}(K)⌉$
	- A node is generally the same size B as a disk block.
	- Example:
		- If B is 4 kilobytes and the number bi of bytes per index entry is around 40 then n is typically around B/bi = 100
		- With K = 1 million search key values and n = 100 at most $log_{50}(1,000,000) = 4$ nodes are accessed in a lookup
		- Contrast this with a balanced binary tree with 1 million search key values — around log2(1,000,000) ~ 20 nodes are accessed in a lookup.
		- Above difference is significant since every node access may need a disk I/O, costing typically around 10-20 milliseconds.


###### Insertion
![[Pasted image 20240530161759.png]]


### Hashing
#### Static Hashing
- Principles
	- A bucket is a unit of storage containing one or more records, a bucket is typically a disk block.
	- In hashing the bucket of a record is found directly: it is calculated from its search key value using a hash function.
	- A hash function is a function mapping each search key value (also called a hash key) to a bucket number.
	- The hash function is used to locate records for access, insertion and deletion:
		- Record with search key k is placed in bucket with number hash(k).
	- When hashing is used, the data file is said to be a hash organized file.
	- Records with different search key values may be mapped to the same bucket; Thus the entire bucket has to be searched sequentially to locate a record.
##### Deficiencies of Static Hashing
- Static hashing
	- The hash function maps the search key values to a fixed set of bucket addresses. However, databases grow or shrink with time!
	- If the number of buckets is too small, and the file grows, performance will degrade due to too much overflow
	- If the number of buckets is too large, space will be wasted
- One solution
	- Periodic re-organization of the file with a new hash function.
	- Expensive and disrupts normal operations.
- Better solution
	- Use Dynamic Hashing allowing the number of buckets to be modified dynamically
#### Examples
- Search key: DeptName (a text).
- Number of buckets: 8, numbered 0, 1, ..., 7.
- Hash function: `h(text) = ( int(letter1(text)) + ... + int(lettern(text)) ) %8`. where int maps a letter to its alphabetic number
![[Pasted image 20240530162622.png]]

#### Hash Functions
- Good and bad hash functions:
	- Worst case of hashing: all records are assigned to the same bucket.
	- Ideal case of hashing: each bucket will have the same number of records assigned (record distribution to buckets is uniform). <small> Hard to achieve as the distribution of search keys in the records may change over time</small>
	- An ideal hash function h
		- is uniform, i.e. each bucket is assigned the same number of search key values from the set of all possible key values
		- is random, i.e. independent actual distribution of search key values in the file
- Typical hash functions
	- Perform computation on the internal binary representation of the search key.
	- For example, for a string search key, the binary representations of all the characters in the string could be added and the remainder of the sum modulo the number of buckets could be returned.

#### Hash Index Files
- Hash Index File
	- Hashing can also be used to make a stored hash index file, i.e. the index file is organized as a hash file: HashIndexFile(<ins>Search Key</ins>, Pointer)
	- Example: see next page.
	- Hash Index Files are always Secondary Index Files.
![[Pasted image 20240530163442.png]]

#### Hashing in SQL
- Can declare a table to be stored in a hash organized file, like
```MYSQL
CREATE TABLE employees (  
id INT NOT NULL,  
fname VARCHAR(30),  
lname VARCHAR(30),  
job_code INT,  
store_id INT )  
PARTITION BY HASH(store_id) PARTITIONS 4;
```
A record with store_id = 2005, will be placed in partition number 2005 % 4 = 1
- Can declare a hash index on a table, like:
	- `CREATE INDEX NameIndex ON Instructor(InstName) USING HASH;`

### Choosing Indexes/Hashing
- List Search Key Candidates (to potentially define indexes/hashing on)
	- Primary keys (many DBMS automatically choose primary indexes on these)
	- Foreign keys (many DBMS automatically choose secondary indexes on these)
	- Attributes Ai that are often used for searching and sorting (appear e.g. in join conditions, WHERE conditions, ORDER BY, ...)
- Select which of the candidates should have an index/hash access
	- Use database statistics about frequencies of queries and data modifications, and table sizes
	- Discuss with domain experts
	- Never define indexes for small tables or for attributes that are often updated
	- Trade off between
		- shorter time for queries (SELECT
		- longer time for data modifications (INSERT, DELETE, UPDATE)
- Hashing versus indexing on A
	- If SELECT ..., Ai ... FROM table WHERE Ai = c is common hashing on Ai is better.
	- If range queries like SELECT ... FROM table WHERE Ai <= c2 AND Ai >= c1 are common, then sequentially ordered indexes and B+ trees are better.
- It is all also a matter of your DBMS
![[Pasted image 20240530164115.png]]
