# Propositional Logic

| not ...                | ¬                 | negation       |
| ---------------------- | ----------------- | -------------- |
| ... and ...            | $\land$           | conjunction    |
| ... or ...             | $\lor$            | disjunction    |
| if ... then ...        | $\rightarrow$     | implication    |
| ... if and only if ... | $\leftrightarrow$ | bi-implication |

## Definition (Language of Propositional logic
1. Every proposition letter (p,q,r,..) is a formula
2. If φ is a formula, then $¬φ$ is also a formula
3. If $φ_1$and $φ_2$ are formulas, then $(φ_1∧φ_2), (φ_1∨φ_2), (φ_1→φ_2)$ and $(φ_1↔φ_2)$ are also formulas
4. Nothing else is a formula

Let $Φ$ be a set of propositional letters, and let $p∈Φ$ (that is: $p$ is an element of the set $Φ$).  Then the language of propositional logic is defined as follows:
$$
\phi := p|\lnot\phi|(\phi\land\phi)|(\phi\lor\phi)|(\phi\rightarrow\phi)|(\phi\leftrightarrow\phi)
$$
## Syntactic Derivation
A formula of propositional logic if it can argue for it by providing its syntactic derivation
**Example**
$(p↔(q∨(r→(p∨¬q))))$
$φ:=$
$φ \leftrightarrow φ:=$
$p \leftrightarrow φ:=$
$p \leftrightarrow (φ \lor φ):=$
$p \leftrightarrow (q \lor φ):=$
$... p \leftrightarrow (q \lor (r \rightarrow φ)):=$
$... p \leftrightarrow (q \lor (r \rightarrow (p \lor φ))):=$
$p \leftrightarrow (q \lor (r \rightarrow (p \lor \neg φ))):=$
$p \leftrightarrow (q \lor (r \rightarrow (p \lor \neg q)))$
![](Pasted%20image%2020240919135458.png)
## Valuations
Propositions generate different possibilities, ways the actual world might be. The set $\{p,q,r\}$ will generate $2^3= 8$ possibilities
![](Pasted%20image%2020240912084601.png)

Each row is described by a function that assigns to each proposition a truth value:  either 1 (=“true”) or 0 (=“false”).  We call such functions _v_, **valuations**. For any proposition _p_, _v(p)_ = 1 means that _p_ is true, and _v(p) = 0_ means that _p_ is false, in the situation represented by _v_.
![](Pasted%20image%2020240912084827.png)

A **tautology** is a statement that is always true, i.e., it is true under any valuation consider $(p\rightarrow q)\lor(q\rightarrow p)$ 

A formula is **satisfiable** if there exists a valuation which makes the formula true. Consider:$¬(p∧q)∨(¬r)$ 
## Logical Consequence

A formula $ψ$ is a logical consequence of the set of formulas ${φ1,...,φn}$ if $ψ$ is always true when all of $φ_1,...,φ_n$ are true, 
In other words :every valuation that makes all of $φ_1,...,φ_n$ true, also makes _ψ_ true

When _ψ_ is a logical consequence of $φ_1,...,φ_n$ we write
$$
φ_1,...,φ_n \vDash ψ
$$
You can determine if the equation above is true using the following method:
1. Construct a truth table for all of the formulas
2. Check that in every row where all $φ_1,...,φ_n$  et the value 1, ψ also gets the value 1.
![](Pasted%20image%2020240912090734.png)
## Proof system
A proof system is a set of formulas called axioms and a set of rules of inference. 

A proof of a formula $ψ$ is a sequence of formulas $φ_1,...,φ_n$, with $φ_n=ψ$, such that each $φ_k$ is either an axiom or it is derived from previous formulas by rules of inference.

When such a proof exists, we say that $ψ$ is a theorem (of the system) and that $ψ$ is provable (in the system), denoted by $\vdash ψ$
### Hilbert-style system
![](Pasted%20image%2020240912093332.png)
**Example**
![](Pasted%20image%2020240919141408.png)


# Epistemic Logic
## Multi-agent Systems: Agents 

- We call the individuals whose knowledge we describe agents
- Agents: human beings, computer agents, etc
- Epistemic logic is used to talk about knowledge in multi-agent systems
- A multi-agent system is a system containing agents that have the capacity of goal-directed and autonomous interaction
- The study of multi-agent systems is a research area of its own.
## Epistemic Logic

![](Pasted%20image%2020240919142744.png)
### Models of Epistemic Logic
Definition (Possible world model or epistemic model
A possible world model _**M**_ for _**n**_ agents over $Φ \text{is} (S,π,K_1,...,K_n)$, where:

1) a non-empty set states (or worlds) S
2) an interpretation π which associates with every state a truth assignment assignment to the propositions i.e.:
	- for each state $s∈S,π(s) : Φ→\{0,1\}$
3) for each agent _i_, $K_i$ is a binary relation on S.
A pointed possible world model is a pair $(M,s), \text{ where} M= (S,π,K_1,...,K_n)\text{ and }s∈S$.
![](Pasted%20image%2020240919144306.png)
### Equivalence Possibility Relation
$K_i$ is an equivalence relation on S, i.e., it is a binary relation that is
1) Reflexive:  for all $s∈S$, we have $(s,s)∈K_i$
2) Symmetric:  for all $s,t∈S$, we have $(s,t)∈K_i \text{ iff } (t,s)∈K_i$ 
3) Transitive:  for all $s,t,u∈S$, we have that if $(s,t)∈K_i$ and $(t,u)∈K_i$, then $(s,u)∈K_i$.


### When is a formula true in a situation?

We write $(M,s)\vDash φ$ to say that $φ$ is true at $s$ in $M$
**Definition**
- $(M,s)\vDash \top$   always
- $(M,s)\vDash p$  **iff** $π(s)(p) = 1$
- $(M,s)\vDash \neg φ$ **iff** it is not the case that:  $(M,s)\vDash p$
- $(M,s)\vDash φ \land ψ$ **iff** $(M,s)\vDash φ$ and $(M,s)\vDash ψ$ 
- $(M,s)\vDash φ \lor ψ$ **iff** $(M,s)\vDash φ$ or $(M,s)\vDash ψ$ 
- $(M,s)\vDash K_i φ$  **iff** for all v with $(s,v)∈K_i,(M,v)\vDash φ$ 
$K_i φ$ is false at states when there at such that $(s,t)∈K_i$ and $φ$ is false at v.

# Group Knowledge

### Everybody Knows
Let $G \subseteq A$ be a non-empty subset of agents
**Syntax** If $φ$ is a formula, then so is $E_Gφ$ 
**Semantics** If $(M,s)\vDash E_Gφ$ iff $(M,s)\vDash K_iφ$ for every $i∈G$.
Definition
We define inductively
$$
\begin{align}
E^0_Gφ:=φ\\
E^{k+1}_Gφ:=E_gE^k_Gφ
\end{align}
$$
### Common Knowledge

Let $G⊆A$ be a non-empty subset of agents
**Syntax** If $φ$ is a formula, then so is $C_Gφ$ 
**Semantics** If $(M,s)\vDash C_Gφ$ iff $(M,s)\vDash E^k_iφ$ for every $k=1,2,3,...$
### Distributed Knowledge
Let $G⊆A$ be a non-empty subset of agents
**Syntax** If $φ$ is a formula, then so is $D_Gφ$ 
**Semantics** If $(M,s)\vDash D_Gφ$ iff $(M,t)\vDash φ$ for all t such that $(s,t)∈⋂_{i∈G}K_i$ 
Example
It is distributed knowledge among humankind how the computer works

## Syntax and Semantics of Epistemic Logic with Group Knowledge
![](Pasted%20image%2020240919155225.png)
### G-reachability
![](Pasted%20image%2020240919160149.png)

# Validities
### Validity and Satisfiability 

Given a model $M= (S,π,K_1,...,K_n)$, we say that:
![](Pasted%20image%2020240919160953.png)
### Distribution of Knowledge
**Each agent knows all the logical consequences of her knowledge**
**Proposition**
$\vDash (K_iφ∧K_i(φ→ψ))→_Kiψ$
![](Pasted%20image%2020240919162335.png)
![](Pasted%20image%2020240919162346.png)
**Proof**
We need to show that the above formula is valid, i.e., it is valid in every possible-world model, i.e., it is true in every possible world of every possible-world model.

In order to show that we take
1. an arbitrary possible-worlds model M over n agents $M= (S,K_1,...,K_n,π),$
2. an arbitrary agent $i∈\{1,...,n\}$
3. an arbitrary world $s ∈ S$
We need to show that $(M,s)\vDash (K_iφ∧K_i(φ→ψ))→K_iψ$ 

![](Pasted%20image%2020240919163138.png)
b
### Knowledge Generalisation
**Each agent knows all the formulas that are valid in a given model.**
**Proposition**
For any possible-world model M, if $M\vDash φ \text{ then } M\vDash K_iφ$
![](Pasted%20image%2020240919163445.png)
![](Pasted%20image%2020240919163456.png)
![](Pasted%20image%2020240919163515.png)
**Proof**
Let us take an arbitrary model M.  To prove this proposition we need to only concern ourselves with models M such that $M\vDashφ$ (the case where $M6\not \vDashφ$ is irrelevant). 
Assume then that $M\vDashφ$ , and so for all $s∈S,M,s\vDash φ$.  In particular, for any fixed state $s∈S$, we get that $M,t\vDashφ$ at all $t∈S$, such that $(s,t)∈Ki$ .Hence $M,s\vDash K_iφ$, and, since s was chosen arbitrarily $,M\vDash K_iφ$
### Truthfulness of knowledge
**Agents can only know facts**
**Proposition**
$\vDash Kiφ→φ$ in the class of models with equivalence possibility relations.
![](Pasted%20image%2020240919164045.png)
![](Pasted%20image%2020240919164116.png)
**Proof**

As in the proof of Proposition 1, we take an arbitrary model M, this time requiring that all possibility relations are equivalences, an agent i, and a states in the model M.
We assume that $M,s\vDash K_iφ$.Then $M,t\vDash φ$ for all t, such that $(s,t)∈K_i$. Then, by the fact that $(s,s)∈K_i$(since $K_i$ is reflexive, because it is an equivalence), we obtain $M,s\vDash φ$.
Since M and s were chosen arbitrarily, we conclude $\vDash K_iφ→φ$.

### Positive and Negative Introspection
**Agents know what they know and what they do not know.**
**Proposition** for Positive Introspection
$\vDash K_iφ→K_iK_iφ$ in the class of models with equivalence possibility relations
![](Pasted%20image%2020240919164537.png)
![](Pasted%20image%2020240919164610.png)
![](Pasted%20image%2020240919164637.png)
**Proof**
![](Pasted%20image%2020240919164751.png)
**Proposition** for negative introspection 
$\vDash ¬K_iφ→K_i¬K_iφ$ in the class of models with equivalence possibility relations.
![](Pasted%20image%2020240919164957.png)
![](Pasted%20image%2020240919165013.png)
![](Pasted%20image%2020240919165023.png)
![](Pasted%20image%2020240919165032.png)
**Proof**
![](Pasted%20image%2020240919165046.png)
### Summary
![](Pasted%20image%2020240919165125.png)
# Axiomatic Systems
## Axiomatic Systems
An axiomatic system consists of:
a) set of formulas called axioms and
b) set of rules of inference.

Together they are used to infer (derive) theorems.
### Proof in an Axiomatic System
A proof of a formula ψ is a sequence of formulas $φ_1$,...,$φ_n$, with $φ_n=ψ$, such that each $φ_k$ is either an axiom or it is derived from previous formulas by rules of inference.

When such a proof exists, we say that $ψ$ is a theorem (of the system) and that $ψ$ is provable (in the system), denoted by:
$$
\vdash ψ
$$
### System H for Propositional Logic

H is a proof system with three axiom schemes and one rule of inference. For any formulas, $φ$, $ψ$ and $χ$, the following are axioms
$$
\begin{align}
&\tag{H1} (φ→(ψ→φ)\\
&\tag{H2} ((φ→(ψ→χ))→((φ→ψ)→(φ→χ)))\\
&\tag{H3} ((¬ψ→¬φ)→(φ→ψ))
\end{align}
$$
The single inference rule of H is modus ponens (MP for short):
$$
\frac{\vdash φ \qquad  \vdash (φ→ψ)}{ψ} \text{MP}
$$
##### Example
<u>Theorem:</u> $H\vdash p \rightarrow p$ 
<u>Proof.</u>
$$
\begin{align}
&(p→((p→p)→p))→((p→(p→p))→(p→p)) \tag{H2}\\
&p→((p→p)→p) \tag{H1}\\
&(p→(p→p))→(p→p) \tag{MP 1,2} \\
&p→(p→p) \tag{H1}\\
&p→p \tag{MP 3,4}
\end{align}
$$
### System $K_n$ for Epistemic Logic
A1. All tautologies of propositional logic
A2. $(K_iφ∧K_i(φ→ψ))→K_iψ,i∈{1,...,n}$ 
R1 $\frac{\vdash φ \qquad  \vdash (φ→ψ)}{ψ}$
R2 $\frac{\vdash φ)}{K_i \text{ for each } i∈{1,...,n}}$
##### Example
<u>Theorem:</u> $K_n\vdash K_i(p∧q)→K_ip$ 
<u>Proof.</u>
$$
\begin{align}
&\text{1. }(p∧q)→p \tag{A1}\\
&\text{2. }K_i((p∧q)→p) \tag{1,R2}\\
&\text{3. }(K_i(p∧q)∧K_i((p∧q)→p)→K_ip \tag{A2} \\
&\text{4. }((Ki(p∧q)∧Ki((p∧q)→p)→Kip)→\\
&(Ki((p∧q)→p)→(Ki(p∧q)→Kip)) \tag{A1}\\
&\text{5. }(K_i((p∧q)→p)→(K_i(p∧q)→K_ip)) \tag{3,4,R1}\\
&\text{6. } K_i(p∧q)→K_ip \tag{2,5 R1}
\end{align}
$$
$K_n$ is sound and complete with respect to $M_n$ for the language $L_n$.
### System $S5_n$ for Epistemic Logic
A1. All tautologies of propositional logic
A2. $(K_iφ∧K_i(φ→ψ))→K_iψ,i∈{1,...,n}$
A3 $K_iφ→φ,i∈{1,...,n}$
A4 $K_iφ→K_iK_iφ,i∈{1,...,n}$
A5 $¬K_iφ→K_i¬K_iφ,i∈{1,...,n}$
R1 $\frac{\vdash φ \qquad  \vdash (φ→ψ)}{ψ}$
R2 $\frac{\vdash φ)}{K_i \text{ for each } i∈{1,...,n}}$

$S5_n$ is sound and complete with respect to $M^{rst}_n$ for the language $L_n$
## Languages and Models
### Languages
Take $Φ$ to be a set of propositions.
- Let $L_n(Φ)$ be the set of formulas that can be built up starting from the primitive propositions in $Φ$, using $∧,¬,$ and $K_1,...,K_n.$
- Let $L^D_n(Φ)$ (resp.. $,L^C_n(Φ)$) be the language with $D_G$ (resp., $E_G$ and $C_G$), where $G⊆{1,...,n}$ is non-empty.
- Let $L^{CD}_n(Φ)$ be the language with $D_G$, $C_G$, and $E_G$.
### Models 
Let $M_n(Φ)$ be the class of all possible world structures for n agents over Φ (with no restrictions on the $K_i$ relations).
$M_n(Φ)$ can be restricted by specifying the $K_i$ relations, e.g.: for $M^{rst}_n(Φ)$, $K_i$ relations are reflexive, symmetric, and transitive.
Note:  Φ is fixed from now on and we suppress it from the notation.

### Validity with respect to a class of models
We say that $φ$ is valid with respect to $M_n$, and write $Mn\vDashφ$, if $φ$ is valid in all the structures in $M_n$.
- If $M$ is some subclass of $M_n,φ$ is valid with respect to $M,M\vDash φ$, if $φ$ is valid in all the structures in $M$
- If $M$ is some subclass of $M_n,φ$ is satisfiable with respect to $M$, if $φ$ is satisfied in some structure in $M$.
#### Example: Show that $φ$ is valid in $M_n$
let $φ:=K_i(p∧q)→K_ip$
Let us take:
- An arbitrary possible-world model $M= (S,π,K_1,...,K_n)$ over a set $A$ of n agents and a set of propositions $Φ$, and
- arbitrary $s∈S$ and $i∈A$
Let us assume that $(M,s)\vDash K_i(p∧q)$.  By the semantics of K, this means that for all t such that $(s,t)∈K_i$ it is the case that $(M,t)\vDash p∧q$.  By the semantics of∧, we know that then for all t such that $(s,t)∈K_i$ it is the case that $(M,t)\vDash p$.  Then, again by the semantics of K we get that $(M,s)\vDash K_ip$. We conclude that $Mn\vDash φ$. 

## AX-consistency

Take an axiom system AX
1. $φ$ is AX-consistent if $¬φ$ is not provable in AX.
2. A finite set $\{φ_1,...,φ_k\}$ of formulas is AX-consistent if $φ_1∧...∧φ_k$ is AX-consistent.
3. An infinite set of formulas is AX-consistent if all of its finite subsets are AX-consistent.

A set F of formulas is a maximal AX-consistent set wrt a language $L$ if:
- it is AX-consistent, and
- for all $φ$ in $L$ but not in $F$, the set $F∪{φ}$ is not AX-consistent.
### Lemma (Lindenbaum)
Suppose the language $L$ is a countable set of formulas closed wrt propositional connectives (so that if $φ$ and $ψ$ are in $L$, then so are $φ∧ψ$ and$¬φ$).
In any axiom system AX for $L$ that includes A1 and R1, every AX-consistent set $F⊆L$ can be extended to a maximal AX-consistent set wrt $L$.
#### Proof of lemma
![]({EC4AECCC-07A9-4C99-B166-16AFDDF7E6DF}.png)
![]({0C2BBA95-602F-4C79-AB27-77EDD7320D17}.png)

### Lemma 2
If $F$ is a maximal AX-consistent set, then it satisfies the following properties:
1. For every formula $φ∈L$, exactly one of $φ$ and $¬φ$ is in $F$;
2. $φ∧ψ∈F$ iff $φ∈F$ and $ψ∈F$
3. if $φ$ and $φ→ψ$ are both in $F$, then $ψ$ is in $F$;
4. if $φ$ is provable in AX, then $φ∈F$

#### Proof of lemma
![]({A33E4B1E-ADF8-435A-86D3-025011B2A391}.png)

## Soundness and Completeness

- An axiom system AX is sound for a language $L$ wrt a class $M$ of structures if every formula in $L$ provable in AX is valid wrt $M$.
- An axiom system AX is complete for a language $L$ wrt a class $M$ of structures if every formula in $L$ that is valid wrt $M$ is provable in AX

AX characterizes the class $M$ if it is sound and complete axiomatization of $M$

for any $φ,AX\vdash φ$ if and only if $M\vDash φ$
![]({81755ABE-AFDD-436E-A8CA-338EA6FA7F76}.png)
### Soundness of $K_n$
$K_n$ is a sound with respect to $M_n$ for $L_n$.
#### Proof of Soundness of $K_n$
![]({B292F038-8664-4585-8B7E-5953BD6F4381}.png)
![]({6DE3B513-F850-4A08-9549-7F2DCE2494C8}.png)
![]({BF1AC494-D854-4220-A8AC-01CE1D1C1D1A}.png)

### Completeness
$K_n$ is complete with respect to $M_n$ for $L_n$.

We want to show:
Every formula $φ$ in $Ln$ that is valid wrt $M_n$ is provable in $K_n$.
It suffices to show that:
Every $K_n$-consistent formula in $L_n$ is satisfiable with respect to $Mn(*)$
Why is it enough?
Because if we knew that (\*) is true we would get the theorem: 
Assume that $φ$ is valid.  Assume for contraction that $φ$ is not provable. Then, of course,$¬¬φ$ is also not provable.  This, by definition, makes$¬φKn$-consistent. But then by (\*) $¬φ$ is satisfiable, so $φ$ is not valid. Contradiction.
#### Proof
![]({56D5FE76-6A41-49CD-8D8F-7E00EDBFA9C9}.png)
![]({7B872EEA-1C7B-430E-B790-65F3A3D9E725}.png)
![]({A96AE77F-AF81-4DFF-B4D0-E5A399D70181}.png)
![]({EE06DF7E-40A4-4E69-8F26-BE2300EE4010}.png)
# Dynamic Epistemic Logic

## Logics of Public Announcements
- PAL (Public Announcement Logic)
- PAC (Public Announcement logic with common knowledge) is PAL +C
- PAL and PAC are examples of dynamic epistemic logic

## Dynamic Modalities
Dynamic Modalities To express informational changes, dynamic epistemic logics use a new kind of operators, called dynamic modalities:
$$
[a]φ
$$
where α is the name of some action involving communication,  such actions are called epistemic actions (as opposed to ontic actions)since they affect only the knowledge/beliefs of the agents. The intended meaning of $[α]φ$ is:  if action α is performed, then φ will become true.

An example is the truthful public announcement of some sentence $φ: [!φ]$The intended meaning of $[!φ]ψ$ is: :if a truthful public announcement of $φ$ is performed, then $ψ$ will become true.

## Public Announcement Logic
Definition (Syntax) 
Φ is a set of propositions, with $p∈Φ$, and $A ={1,...,n}$ is a set of agents. $φ:=\top |p|...|Kiφ|[!φ]φ$$ where $\top$ abbreviates a tautology and $i∈A$ is the name of some agent. As before, these formulas are interpreted in possible world models.

### Public Announcement as joint update
As we saw in many examples, this can be done by deleting worlds. LEARNING = ELIMINATING POSSIBILITIES 
From now on, we denote by $!φ$ the operation of deleting the non-$φ$ worlds, and call it public announcement with $φ$,  or joint update with $φ$.
### Semantics of Public Announcement Logic
![]({2A4548FE-0D98-4438-AAF1-F305312162E1}.png)
![]({5C80D83A-7131-4FB4-8E5B-EE86951BBAA8}.png)
$[!φ]φ = ¬<!φ>¬φ$ 

## Announcements about announcements

In PAL we can iterate and combine announcements.
We can announce not only facts:  $[!u]$
or combinations of facts (Boolean formulas):  $[!(u∨¬q)]$
but also epistemic formulas:  $[!(¬Kbu)]$
and make announcements about other announcements:  $[!([!u]Kbu)]$.

### Closure of public announcement under composition
![]({DF38675F-BBDA-4CDB-91E1-170E4EAD1FB1}.png)
## Reduction Laws for PAL

The following formulas are valid in $M_n$:
Atomic Permanence $[!φ]p↔(φ→p)$, for atomic propositions p
Announcement-Negation $[!φ]¬ψ↔(φ→¬[!φ]ψ)$
Announcement-Conjunction $[!φ](ψ∧θ)↔([!φ]ψ∧[!φ]θ)$
Announcement-Knowledge $[!φ]Kiψ↔(φ→Ki[!φ]ψ)$

Using those reduction axioms, one can translate formulas with announcement modalities into ones without.
This shows PAL can be reduced to EL

## Dynamic Epistemic Logic in General
DEL comprises a family of logics.
Each has syntax and semantics.
DEL concerns explicit informational actions.
Corresponding knowledge and belief changes in agents.
Often uses special action models.
![]({FC5E1A91-F6CF-4CAF-8602-412CE403DF51}.png)
![]({9637E0BB-BA4C-4C3E-9914-9B2397AE0CF8}.png)
![]({9A230D99-E553-459E-AF9B-ABA9181AF716}.png)
![]({B5679655-FA9B-41B7-B4A5-452F9E0B657E}.png)
![]({CFDBAD5A-1144-4142-9717-F456D99B2E3C}.png)
