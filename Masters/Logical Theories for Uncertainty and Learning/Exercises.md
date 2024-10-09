# Week 3a
Which of the following are correct:
1) $\top \vDash \bot$  NOT CORRECT
2) $\bot \vDash \top$ ???
3) $p\land q \vDash p \leftrightarrow q$ CORRECT
4) $p \leftrightarrow q \vDash p\land q$ NOT CORRECT 
5) $p \leftrightarrow q \vDash \neg p\land q$  NOT CORRECT
For each of the following decide if it is a well-formed formula of propositional logic. For those which are, provide a syntactic derivation.
1) P
	- p:=
2) $\neg \neg (p\rightarrow (q \lor p))$ 
	- $\phi:=$
	- $\neg \phi:=$
	- $\neg \neg \phi$
	- $\neg \neg (\phi \rightarrow \phi)$
	- $\neg \neg p \rightarrow \phi):=$
	- $\neg \neg p \rightarrow (\phi \lor \phi)):=$
	- $\neg \neg p \rightarrow (p \lor \phi)):=$
	- $\neg \neg(( p \rightarrow (p \lor q)):=$

For each of the following formulas decide: are they tautologies? are they satisfiable?
1) $p \lor \neg p$ is tautology
2) $p \land \neg p$ not satisfiable
3) $(p \land q)\rightarrow p$ satisfiable for p is 1 and q is 1 
4) $(p \land q)\rightarrow \neg p$  satisfiable for p is 0
5) $((p \land q) \rightarrow r) \rightarrow (q \rightarrow (p \rightarrow r))$  satisfiable for all false not a tautology for r and q =1 and p=0
6) $(((p∧q)→s)∧((p∧q)→t))→((p∧q)→(s∧t))$ satisfiable  for all true but not a tautology for p and q are false but s and t are true
# Week 3b

Ex.1 Give a scenario that could be described by the following formula: $K_1K_2p∧¬K_2K_1K_2p$ 
Agent 1 knows that agent 2 knows p but agent 2 does know that agent 1 knows that he knows

Ex. 2 Write the following sentence in epistemic logic:

Trump doesn’t know whether Al-Assad knows that Obama knows that Al-Assad knows that there was a chemical-weapon attack in Syria

let 
$K_A$ Assad
$K_T$ trump
$K_O$ obama
c chemical-weapon usages 
$\neg K_T (K_A K_O K_Ac)$

Ex. 3 Consider the following scenario

Alice, Bob and Carol each draw a card from a stack of three cards. The cards are J♠, Q♠ and A♠ with backsides indistinguishable. Players can only see their own card, but not the cards of other players. They do see that other players only hold a single card, and that this cannot be their own card, and they do know that all the players know that, etc.

(a)  Let JQA stand for the deal: Alice holds J, Bob holds Q, and Carol holds A, etc, and propositions like q<sub>a</sub> stand for “Alice holds the queen”. Represent the epistemic situation as a possible world mode

![](459675197_447036095153319_2728929346993351220_n.jpg)
(b) Answer the following questions.
1) In JAQ, what does Alice know about Carol’s card? 
	-   its either an A or Q
2)  And is it the case here that $K_a(j_c∨j_b)$?
	- No Alice  holds the jack
3) Describe in epistemic logic what Alice knows in situation JAQ
	- Alice knows she holds a jack and she knows that the other 2 have the other cards
4) In the world JQA, does Alice consider it possible that Alice has Q?
	- No no worlds Alice imagines  where she holds a Q
5) In AJQ, is it the case that Bob knows that Carol considers it possible that he has J?
	-  Yes, bob considers QJA to be possible thus Carol consider JQA to be possible


Ex. 4 Consider the following situations
_A.  A coin gets tossed under the cup. It lands heads up. Alice and Bob are present, but neither of them can see the outcome, and they both know this._

_B. Now Bob leaves the room for an instant. After he comes back, the cup is still over the coin. But Bob realises that Alice might have taken a look (in fact she didn’t look). Alice also realizes that Bob considers this possible_

Represent the situations A and B as epistemic models.
({Alice looked, Alice did not look},(($\pi(looked$ ) = false,($\pi(¬looked$ ) = true,) ,A,B)

Ex. 5 Three children are playing outside together, lets call them a ,b and c. Now it happens that during their play all of them get mud on their foreheads. Each can see mud on others but not on his own forehead. Along comes the father, who says, “At least one of you have mud on your forehead”. The father then asks the following question, over and over: “Does any of you know whether you have mud on your own forehead?” Assuming that all the children are perceptive,
intelligent, truthful, and they answer simultaneously, what will happen? Surprisingly, after the father asks the question for the third time all muddy children will say “yes”

