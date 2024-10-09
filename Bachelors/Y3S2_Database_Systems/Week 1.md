##### Relation Systems 
- Defines the overall design of a relation.  
-  Consists of  
	- the name R of the relation, e.g. Instructor  
	- the attribute names, A1, ...., An , e.g. InstID  
	- a domain Di for each attribute Ai  
	- attributes constituting primary key are underlined

##### Domains
Is a Set of allowed Attribute Values


NULL
- All Domains have a NULL value
	- A NULL value signifies a not existing value or a unknown value
	- Arithmetic Expressions containing NULL return NULL
	- The predicate `IS NULL` can be used to check for Null.
	- The predicate IS NOT NULL can be used to check for not Null
	- Comparison Expressions containing NULL return UNKNOWN
	- A `WHERE` or `HAVING` clause predicate is treated as FALSE, if it evaluates to  UNKNOWN
	- When comparing two tuples, it treats NULL as being equal to NULL (although  NULL = NULL returns UNKNOWN).(Examples on show distinct)
		- Example 1: If there are two rows (1, NULL) and (1, NULL), one of them are removed.
		- Example 2: If there are two rows (1, 2) and (1, NULL), both are kept
#### Keys
- Is the Attribute, or set of Attributes, that makes relation tuples unique
- No two tuples have the same Key
- Sometimes it takes more than one attribute to make a Key.
Let $R(A_1,A_2,...,A_n)$ be a relation schema and $K\subseteq \{A_1,A_2,...,A_n\}$

###### Super Key
- -K is a superkey of R, if values for K are sufficient to identify a unique tuple of  each permissible relation instance r of R. In other terms: no two rows have the same superkey.
###### Candidate Key
- A Superkey K is a candidate key if K is minimal

###### Primary Key (Constraints)
- One of the candidate keys is selected by the DB designer to be the primary key
- In the relational schema, attributes constituting the primary key are underlined
- This implies a constraint on the allowed relation instances:
	- no two rows may have the same value for the primary key
	- the primary key value must not be NULL.
###### Foreign Key
- K can be specified to be a foreign key referencing   another relation R’, if K is a primary key of R’
- This implies a referential integrity constraint on the allowed relation instances r of R and r’ of R’:   for any tuple in r, there must exist a tuple in r’ having the same values for K
###### String Operations
- =, <>, <, >: are case sensitive according to the SQL standard, but not in  MySQL and MariaDB, where e.g. 'anne' = 'Anne' evaluates to 1 (true)
- Concatenation, using CONCAT
- Finding string length, extracting substrings, etc, see the DBMS manual.
- Pattern matching, using LIKE
- LIKE
	- LIKE is a string-matching operator for comparisons of character strings `string-expr LIKE string pattern`.
	- Can be used where a Boolean expression is expected.
	- Returns 1 (true) if string-expr matches string-pattern, otherwise 0 (false).
	- The string-pattern can use two special characters:  
		- The % character matches any substring (of 0 - n characters).  
		- The _ character matches any (single) character.
	- Patterns are case sensitive according to the SQL standard. but not in MySQL and MariaDB
	- Examples
		- 'Anne' matches 'Anne' , but not 'Hanne'  
		- 'Intro%' matches any string beginning with “Intro”, e.g. 'Introduction'  
		- '%duc%' matches any string containing “duc” as a substring”, e.g.  'Introduction'  
		- '_ _ _ ' matches any string of exactly three characters, e.g. 'Ann'  
		- '_ _ _ %' matches any string of at least three characters, e.g. 'Hanne'


###### Aggregate Functions 
These functions operate on the set of values in a column of a given attribute A and return a value:

- AVG(A): Average of values in the A-column
- MIN(A): Minimum of values in the A-column
- MAX(A): Maximum of values in the A-column
- SUM(A): Sum of values in the A-column
- COUNT(A): Number of values in the A-column
- AVG(A), MIN(A), MAX(A), SUM(A), and COUNT(A) ignore rows where A is NULL 
- What if an A column only has NULL values or is empty? 
	- COUNT(A) returns 0,  
	- MIN(A), MAX(A), SUM(A), and AVG(A) return NULL
- COUNT(\*) does not ignore rows with NULL values