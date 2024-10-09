## SQL Data Definition Language (DDL)
### Create
```MYSQL
CREATE DATABASE University 
```
Create a database with the name 'University'

Create a new table, like the old table, and insert values
```MYSQL
CREATE TABLE Specialist1 LIKE Instructor; # Create an empty table  
INSERT Specialist1 SELECT * FROM Instructor WHERE DeptName='Comp. Sci.';
```
Create a table with contents from an old table
```MYSQL
CREATE TABLE Specialist2 SELECT * FROM Instructor WHERE DeptName='Physics';
```
- However, Specialist1 has a Primary Key(InstID), while Specialist2 has none!  `ALTER TABLE Specialist2 ADD PRIMARY KEY(InstID); # To define Primary Key`
- Neither Specialist1 nor Specialist2 has Foreign Keys defined! `ALTER TABLE Specialist1 ADD FOREIGN KEY(DeptName) REFERENCES Department(DeptName); # To define a Foreign Key
#### Create Table
Form
```MySQL
CREATE TABLE R (A1 D1,A2 D2,...,An Dn,(integrity_constraintk1)...)
```
- R is the name of the relation
- Each $A_i$ is an attribute name in the schema 
- $D_i$ is the data type (Domain) of values $A_i$ 
- Possible integrity constraints include: primary key, foreign key, not null
Example
```Mysql
CREATE TABLE Instructor  
(InstID VARCHAR(5),  
InstName VARCHAR(20),  
DeptName VARCHAR(20),  
Salary DECIMAL(8,2),  
PRIMARY KEY (InstID),  
FOREIGN KEY(DeptName) REFERENCES Department(DeptName));
```

The foreign key will ensure that the DeptName  in any Instructor row will also appear in Department. In particular this means that the DBMS should as a default
- Disallow
	- insert violation he insertion of a row into Instructor, if its DeptName is not in the Department table
	- delete violation: the deletion of a row from Department  table, if its DeptName is used in a row in Instructor
	- update violations: update of a DeptName in Department,  if its DeptName is used in a row in Instructor

###### Referential actions
```MySql
On Delete Set Null
On Delete Cascade
```
Now it is allowed to delete a Department row having a DeptName used in some rows in the Instructor table. In that case the DeptName in those Instructor rows are set to NULL. or deleted too. (Place at the end of Foreign key). also you can add `Not NULL` at the end of a attribute to make ensure that each row has said attribute.


#### Domain Types
- String types 
	- CHAR(N) fixed length of character strings, with user-specified length N
	- VARCHAR(N) Variable length character strings, with a user-specified length 
- Data and Time
	- A date in the format of \`YYYY-MM-DD\` 
- Numeric types
	- INT Integers, a finite subset of the integers that is machine- dependent
	- SMALLINT Small integers (a machine-dependent subset of INT).
	- DECIMAL(p,d) Fixed point numbers, with user-specified precision of p digits, with d digits to the right of decimal point.
	- FLOAT(n) Floating point numbers (like 1.23 and 1.23E5) with user-specified   precision of at least n digits.
### USE
```Mysql
USE University
```
Instruct the DBMS to use the University as the default (current) database for subsequent statements. The database remains the default until the end of the session or another USE statement is issued
#### Drop
```Mysql
Drop DATABASE University
```
#### DROP TABLE and ALTER TABLE
- To delete all rows in a table and the table schema
	- `Drop Table Student;`
- To delete all rows in a table, but retain the table schema
	- `DELETE FROM Student;`
- To add an attribute with assigned NULL values
	- `ALTER TABLE Student ADD Shoesize DECIMAL(2,0);`
- To drop an attribute
	- `ALTER TABLE Student DROP Shoesize;` 
`
### Constraints
- Integrity constraints
	- Express rules about allowed data in database tables.
	- Can be specified in CREATE TABLE (or ALTER TABLE ADD) statements.
	- If a data changing operation would lead to a violation of a specified constraint, the DBMS will reject it (if the feature is implemented by the DBMS)
	- Purpose: to guard against wrong and inconsistent data in a database.
- Integrity constraints on a single table
	- `Data types`
	- `NOT NULL`
	- `PRIMARY KEY`
	- `...`
- Integrity constraints between two tables
	- `FOREIGN KEY` Referential Integrity
#### Data Types as Attribute Value Constraints
The attribute must only be assigned values of the given type
- Data Types in MariaDB1
	- Numeric data types, e.g. `INT`, `DECIMAL(i,j)`,`...`
	- String data types, e.g. `VARCHAR(n)`, `ENUM(s1, ..., sn)`
	- Date and Time datatypes, e.g. `DATE`, `TIME`, `YEAR`
	- Also others
```MYSQL
CREATE TABLE Section (  
CourseID VARCHAR(8),  
SectionID VARCHAR(8),  
Semester ENUM('Fall', 'Winter', 'Spring', 'Summer'),  
StudyYear DECIMAL(4,0),  
Building VARCHAR(15),  
Room VARCHAR(7),  
TimeSlotID VARCHAR(4),  
PRIMARY KEY(CourseID, SectionID, Semester, StudyYear) );  
INSERT Section VALUES('BIO-101','1','Summer','2009','Painter','514','B'); -- is ok  
INSERT Section VALUES('BIO-101','1','Sommer','2009','Painter','514','B'); -- is not ok
```
#### NOT NULL and PRIMARY KEY Constraints
- NOT NULL
	- Insertion of a new row demands that Atribute X and Y are given valid attribute values from the data type specified. NULL values are not allowed
	- `X VARCHAR(20) NOT NULL;  Y DECIMAL(12,2) NOT NULL;`
- PRIMARY KEY( A1, A2, ..., An)
	- For each row the PRIMARY KEY will be unique: no two rows have the same values for the primary key.
	- Primary keys are automatically required to be NOT NULL.
- Foreign Key Referential Integrity Constraints
	- Let R be a table, and K a subset of its attributes
		- K can be specified to be a foreign key referencing another table R’, if K is a primary key of R’.
		- This implies a referential integrity constraint  on the allowed instances r of R and r’ of R’: for any row in r, there must exist a row in r’ having the same values for K, unless an attribute of K is NULL
	- This implies a referential integrity constraint on the allowed instances of Instructor and   Department: for any row in Instructor, DeptName must be NULL, or there must exist a row in Department having the same values for DeptName
		```MYSQL
		CREATE TABLE Instructor  
		(InstID VARCHAR(5),  
		InstName VARCHAR(20) NOT NULL,  
		DeptName VARCHAR(20),  
		Salary DECIMAL(8,2),  
		PRIMARY KEY (InstID),  
		FOREIGN KEY(DeptName) REFERENCES  
		Department(DeptName))```

- Foreign Key Referential Integrity Violations
	- SQL rejects operations that would  lead to a violation of the specified  referential integrity 
	- constraints
		- Insertions to or updates in the referencing table R (Instructor), e.g.  `INSERT INTO Instructor VALUES  (99999, 'Anne', 'Mathematics', 90000.00);` (There is no Mathematics department in the  Department table )
		- deletion or updates of rows in the referenced table R’ (Department),  (unless referential actions are specified, `DELETE FROM Department WHERE DeptName = 'Physics';`(There is an instructor that has theDeptName = Physics )
- Foreign Keys: Specification of Referential Actions
	- In the CREATE TABLE command for R one can write 
		- ON DELETE referential-actions
		- ON UPDATE  referential-actions
		- where referential-actions specify modifications to make in R
	- if a DELETE/UPDATE in R’ leads to a violation, where rows in R have K values not existing in R’:
	```MYSQL
 ON DELETE SET NULL -- set the K attributes that do not exist in R’ to NULL in R. 
 ON DELETE CASCADE -- delete the problematic rows of R.  
 ON UPDATE CASCADE-- make the same updates of K values in R.
	```
### Views
Provide a mechanism to hide certain data from the view of certain  users. Specific columns can be hidden as well as specific rows. It is a virtual table. can be used as a table, but it is a stored SQL expression, that is executed each time it is used!

- Command to define a view with name id:`CREATE VIEW id AS query-expression`
- id can be used in queries as if it had been a table name
- id is a short for writing query-expression
- A view definition is not the same as creating a new relation by evaluating the query expression
```MYSQL
CREATE VIEW Faculty AS  
SELECT InstID, InstName, DeptName FROM Instructor  
WHERE DeptName NOT IN ('Finance', 'Music');
```
- Add a new row to the Faculty view `INSERT Faculty VALUES ('30765', 'Green', 'Physics');`
- This insertion will be executed by the DBMS as in the statement:  `INSERT Instructor VALUES ('30765', 'Green', ‘Physics', NULL)`
- Some view updates can be ambiguous and some can be impossible
- Most SQL systems only allow updates of simple views satisfying conditions like
	- Any attribute not listed in the view’s SELECT clause can be set to NULL.
	- The FROM clause has only one table, no JOIN operations!
	- The SELECT clause contains only attribute names of the base table, and does not have any expressions, aggregates or DISTINCT specifications
	- The query does not have a GROUP BY or HAVING clause
	- 

## SQL Data Manipulation
### INSERT
Simplest form 
```MYSQL
INSERT INTO R (...,A,...) VALUES (...,val,...);
```
- R represents a relation
- A is an attributes of R, val is a value expression 
- into is not necessary 
Example 
```MYSQL
INSERT Course VALUES  
('CS-437', 'Database Systems', 'Comp. Sci.', 4),  
('CS-528', "Big Data Systems", 'Comp. Sci.', 5),  
('CS-530', "Data Warehouse", "Comp. Sci.", 4);
```

General form
```MYSQL
INSERT R (..,A,...) SELECT ... FROM ... WHERE ...;
```
### DELETE
Typical form
```MySQL
DELETE FROM R WHERE P;
```
- R is the relation
- P is a row predicate over attribute names 
- removes those rows for which P is true
- `DELETE FROM R` Deletes all rows
### UPDATE
Typical form
```MYSQL
UPDATE R SET ...,A = val, ... WHERE P;
```
- R is the relation
- P is a row predicate over attribute names 
- A is an attribute of R, val is a value expression
- changes the value of A_i to val_i for those rows for which P is true
With cases 
```MYSQL
UPDATE Instructor SET Salary =  
CASE  
WHEN Salary BETWEEN 80000 AND 89999 THEN Salary+10000  
WHEN Salary BETWEEN 70000 AND 79999 THEN Salary+5000  
WHEN Salary BETWEEN 0 AND 69999 THEN Salary+2500  
ELSE Salary  
END;  
```


## Data Query Language 
### SELECT
Basic form
```MYSQL
SELECT A1,A2,A3
FROM r1,r2,r3
where P;
```
- Ai represents an attribute
- ri represents a relation
- P is a (row) predicate over attribute names.

- The select clause can contain arithmetic expressions involving  the operation, +, –, \*, and /, and operating on constants or attributes of rows. `SELECT InstID, Salary/12 AS Monthly FROM Instructor;`
- The select clause can contain built-in functions associated with the built-in datatypes `SELECT CURDATE();`
- DISTINCT/ALL
	- To force the elimination of duplicates, insert the keyword DISTINCT after SELECT `SELECT DISTINCT DeptName FROM Instructor` and 
	- Keyword ALL specifies that duplicates are not removed `SELECT ALL DeptName FROM Instructor;`
- The '\*' 
	- A '\*' in the SELECT clause denotes “all attributes” `SELECT * FROM Instructor;`
	- You can use '\*' to get all attributes of one relation even if getting 2 `SELECT Instructor.* FROM Instructor, Teaches  WHERE Instructor.InstID=Teaches.InstID;`
- AS
	- SQL allows renaming tables and attributes using the AS clause: OldName AS NewName `SELECT InstID,InstName, Salary/12 AS Monthly  FROM Instructor;` or `FROM Instructor AS T, Instructor AS S WHERE T.Salary > S.Salary AND S.DeptName = 'Comp. Sci.';`
- ORDER BY
	- List result in order `SELECT a,... From ... Order BY a DESC;`
	- use DESC for descending order or ASC for ascending order,  for each attribute; Ascending order is the default.
	- can sort on multiple attributes `ORDER BY a,b;` orders by a first if the same then b.

#### WHERE 
Where clause specifies conditions for the rows to be included in the result
- Comparisons with <, <=, >, >=, =, <> can be applied to arithmetic expressions and strings
- Comparison results can be combined using the logical connectives AND, OR, and NOT.

#### FROM
FROM clause lists the tables involved in the query
- FROM r
- FROM r1, ... , rm corresponds to Cartesian Product operation r1 x ... x rm of Relational Algebra, also called a join.
Example 
` SELECT * FROM Instructor, Teaches;`  
- Generates every possible Instructor-Teaches pair
- The result table has as attributes the union of Instructor&Teaches attr.s
- Cartesian Product is not very useful directly, but useful combined with the  WHERE clause condition.
#### GROUP BY
Duplicates and Aggregates
	- Aggregate functions take duplicate values of the aggregate attribute into account`SELECT AVG(Salary) FROM Instructor WHERE DeptName='Comp. Sci.';`
	- To ignore duplicate values, use the DISTINCT keyword. `SELECT COUNT(DISTINCT InstID) FROM Teaches WHERE Semester='Spring' AND StudyYear=2010;`
	- The GROUP BY clause is used to group rows that have the same values into summary rows. - It divides the result set into groups based on one or more columns.
	- Aggregate functions can then be applied to each group separately. `SELECT DeptName, AVG(Salary) FROM Instructor GROUP BY DeptName;`
	-  **Notes** 
		- When using GROUP BY, all columns in the SELECT statement must either be part of the GROUP BY clause or be used with an aggregate function
		- It's often used in combination with aggregate functions like COUNT(), SUM(), AVG(), etc.
		-It's useful for generating summary reports or performing calculations on subsets of data.
 **HAVING**
-   The HAVING clause is used in conjunction with the GROUP BY clause to filter groups based on specified conditions.
- It is similar to the WHERE clause but operates on groups rather than individual rows.
- HAVING is applied after the GROUP BY operation, allowing filtering based on aggregated values. `SELECT DeptName, AVG(Salary) FROM Instructor GROUP BY DeptName HAVING AVG(Salary) > 60000;` 
- HAVING is typically used to filter groups based on aggregate functions (e.g., AVG(), SUM(), COUNT(), etc.).
- It allows for more specific filtering criteria compared to the WHERE clause when dealing with aggregated data.
- HAVING can only be used in queries that include a GROUP BY clause.
#### More General SQL Queries

- **A more general form of SELECT queries:**
    ```MYsql
    SELECT attributes 
    FROM r1, r2, ..., rm [WHERE P1] 
    [GROUP BY group-spec [HAVING P2]] 
    [ORDER BY order-spec];
    ```
- **Meaning:**
  1. Calculate the relation `r` represented by `r1`, `r2`, ..., `rm` and remove rows from `r` not satisfying `P1`.
  2. Arrange the selected rows into groups having the same values for group-spec and remove groups not satisfying `P2`.
  3. For each group calculate the attributes; this gives one tuple/row for each group.
  4. Order the rows according to the order-spec.

---
### Subqueries 

- A SELECT statement is said to be a subquery, if it occurs nested  inside another statement.
- A SELECT statement can be used to represent a relation in the WHERE, FROM and HAVING clauses of another (outer) SELECT statement.
- They provide alternative ways to perform operations that would otherwise require complex joins and unions.

#### **Scaler Subqueries** 
- If a subquery returns a 1x1 relation then it is said to be a scalar subquery
- A scalar subquery can be used where a single value is expected (in the SELECT, WHERE, FROM and HAVING clauses). Example: 
```MYSQL
SELECT InstName FROM Instructor  
WHERE Salary > (SELECT AVG(Salary) FROM Instructor);
```

#### **Set Membership Conditions:** *IN and NOT IN*
- IN and NOT IN can be used in a WHERE/HAVING clause to form a condition
```MySQL
SELECT..., Ai,... FROM ...  
WHERE Ai [NOT] IN (value1, value2, ...);

SELECT..., Ai,... FROM ...  
WHERE Ai [NOT] IN (SELECT Bj FROM ... WHERE ...); #often Ai = Bj'


SELECT DISTINCT InstName FROM Instructor  
WHERE InstName NOT IN ('Mozart', 'Einstein') ;

SELECT DISTINCT CourseID FROM Section  
WHERE  Semester='Fall' AND StudyYear=2009 AND CourseID IN  
(SELECT CourseID FROM Section WHERE Semester='Spring' AND StudyYear=2010);
```


#### Conditions: *ALL and SOME* 
- Can be used in a WHERE/HAVING clause to form a condition
- It checks whether A op v is true for all/some of the values v in the column specified by the SELECT. 
```Mysql
SELECT InstName FROM Instructor  
WHERE Salary > SOME (SELECT Salary FROM Instructor  
WHERE DeptName = 'Finance')
```


#### Conditions: *EXISTS and NOT EXISTS*
- Can be used in a WHERE/HAVING clause to form a condition: `[NOT] EXISTS (SELECT ... FROM ... WHERE ...);`
- It checks whether the relation is (not) non-empty
- Example: Find all courses taught in both the Fall 2009 semester and in the Spring 2010 semester.
```MYSQL
SELECT CourseID
FROM Section AS S
WHERE Semester = 'Fall' AND StudyYear = 2009
AND EXISTS (
    SELECT *
    FROM Section AS T
    WHERE Semester = 'Spring' AND StudyYear = 2010
    AND S.CourseId = T.CourseId);
```






#### Compound Queries: *UNION, INTERSECT & EXCEPT*
Set Operations UNION, INTERSECT & EXCEPT have
- Arguments: two SELECT statements (denoting two relations R and S).
- Result: is the relation consisting of the union/intersection and set difference of the tuples in R and S.
	- UNION and INTERSECT removes duplicates in the result
	- EXCEPT removes duplicates in its arguments before the operation is done.
- They can’t appear inside a SELECT statement
- Set INTERSECT and EXCEPT are implemented in MariaDB, but not in MYSQL, where they instead be expressed by nested subqueries

```MYSQL
(SELECT CourseID FROM Section WHERE Semester='Fall' AND StudyYear=2009)  
UNION  
(SELECT CourseID FROM Section WHERE Semester='Spring' AND StudyYear = 2010)


(SELECT CourseID FROM Section WHERE Semester='Fall' AND StudyYear=2009)  
INTERSECT  
(SELECT CourseID FROM Section WHERE Semester='Spring' AND StudyYear = 2010);  
#can be expressed by:  
SELECT DISTINCT CourseID FROM Section  
WHERE Semester='Fall' AND StudyYear=2009 AND CourseID  
IN (SELECT CourseID FROM Section WHERE Semester='Spring' AND StudyYear=2010);

  
SELECT CourseID FROM Section WHERE Semester='Fall' AND StudyYear=2009)  
EXCEPT  
(SELECT CourseID FROM Section WHERE Semester='Spring' AND StudyYear = 2010);  
#can be expressed by:  
SELECT DISTINCT CourseID FROM Section  
WHERE Semester='Fall' AND StudyYear=2009 AND CourseID  
NOT IN (SELECT CourseID FROM Section WHERE Semester='Spring' AND StudyYear=2010);
```

### JOIN 
- They take two tables as arguments and return a table as result
- The result table is a subset of the Cartesian Product of the argument tables which
	- combines rows of the two tables, if they match some *join condition*
	- may or may not retain rows that do not match rows in the other table (by padding NULLs for missing data in the other row)
- JOIN operations are typically used in the FROM clause of a SELECT command.
```MYSQL
SELECT * FROM Courses, PreReqs; #Cartesian Product  
SELECT * FROM Courses JOIN PreReqs; #Equivalent expression  
```
In general:  
- All attributes are included.  
- All row combinations are included.  
- Attributes(R1 , R2) =  Attributes(R1) union Attributes(R2)  
\#Rows(R1, R2) = \#Rows(R1) * \#Rows(R2) 
![[Pasted image 20240418170800.png]]
#### NATURAL (INNER) JOIN
- **Included attributes**: All, but only one occurrence of common attributes
- **Matching**: Only rows that have the same value of the common attributes are joined
- `SELECT * FROM Courses NATURAL JOIN PreReqs;`  
Matches rows with the same values for all common attributes, and retains only one copy of each common attribute value
![[Pasted image 20240418170824.png]]
#### OUTER JOINs
- Three kinds of OUTER JOINs to avoid loosing info:
	- LEFT OUTER JOIN preservers the rows in the left table 
	- RIGHT OUTER JOIN preserves the rows in the right table  
	- FULL OUTER JOIN preserves the rows in both tables
- Keyword OUTER can be left out without changing the meaning.
- Each outer join first computes the corresponding inner join and then adds tuples where NULL is padded for missing data
- `SELECT * FROM Courses NATURAL LEFT OUTER JOIN PreReqs`
![[Pasted image 20240418170931.png]]
- `SELECT * FROM Courses NATURAL RIGHT OUTER JOIN PreReqs`
![[Pasted image 20240418171426.png]]
- `SELECT * FROM Courses NATURAL FULL OUTER JOIN PreReqs;`
![[Pasted image 20240418171510.png]]
is not supported by MariaDB and MySQL, instead use:  
```MYSQL
(SELECT * FROM Courses NATURAL LEFT OUTER JOIN PreReqs)  
UNION  
(SELECT CourseID, Title, DeptName, Credits, PreReqID  
FROM Courses NATURAL RIGHT OUTER JOIN PreReqs);
```


#### JOIN conditions 
Define which rows in the two argument tables T1 and T2 match, and which attributes are present in the result of the JOIN:
- T1 <font color="crimson"> NATURAL</font> join-op T2 
	- Two rows match, if T1.A = T2.A for each common attribute name A
	- Common attribute names only occur ones in the result
- T1 join-op T2 <font color="crimson"> using (A1, ..., An)</font> 
	- Where A1, ..., An are (a subset of) common attributes
	- Two rows match, if T1.A1 = T2.A1 and ... and T1.An = T2.An
	- A1, ..., An only occur ones in the result
	- compared to NATURAL, it allows to not join on all common attributes
- T1 join-op T2 <font color="crimson">ON predicate</font>:
	- Two rows match, if predicate is true
	- All attributes are included in the result
	- Allows e.g. for ‘joining’ attributes named differently in the two tables
where join-op can be INNER JOIN, LEFT OUTER JOIN, RIGHT OUTER JOIN, FULL OUTER JOIN,  
where INNER and OUTER can be left out (without changing the meaning).
- **NOTE** :T1 join-op T2 USING (A1, ..., An) = T1 NATURAL join-op T2,  if A1, ..., An include all common attributes
- **Note** : T1 JOIN T2 is short for T1 JOIN T2 ON TRUE

```MYSQL
SELECT * FROM Courses JOIN PreReqs USING (CourseID);  
SELECT * FROM Courses NATURAL JOIN PreReqs; # equivalent expression
```
![[Pasted image 20240418174021.png]]

```MYSQL
SELECT * FROM Courses RIGHT OUTER JOIN PreReqs USING (CourseID);  
SELECT * FROM Courses NATURAL RIGHT OUTER JOIN PreReqs; # equivalent expression  
```
![[Pasted image 20240418174040.png]]
```MYSQL
SELECT * FROM Courses JOIN PreReqs ON Courses.CourseID = PreReqs.CourseID;  
SELECT * FROM Courses JOIN PreReqs WHERE Courses.CourseID = PreReqs.CourseID;
```
![[Pasted image 20240418174328.png]]
```Mysql
SELECT * FROM Courses LEFT OUTER JOIN PreReqs ON Courses.CourseID = PreReqs.CourseID;
```
![[Pasted image 20240418174343.png]]
```MYSQL
SELECT * FROM Courses LEFT OUTER JOIN PreReqs ON TRUE  
WHERE Courses.CourseID = PreReqs.CourseID;
```
![[Pasted image 20240418174404.png]]
##### IMPORTANT: ON is a part of the join operation, while WHERE is calculated afterwards!















## Data Control Language
### Users
#### CREATE USER
- As Database Administrator (i.e. when connected/logged in as root) you can create a user userid with a password password by issuing the command:`CREATE USER userid IDENTIFIED BY password;` example `CREATE USER 'Thomas'@'localhost' IDENTIFIED BY '1984';`
- Thomas can now connect/login to the database server, but they cannot access any databases or tables before privileges are granted `SHOW GRANTS FOR ‘Thomas'@'localhost`
#### Rename User
- `RENAME USER 'Bill'@'localhost' TO 'William'@'localhost';`
#### DROP USER
- `DROP USER 'William'@'localhost';`
#### SHOW created users (in MySQL and MariaDB):
- `SELECT user FROM mysql.user;`
#### Changing Password
- If DBA is connected as root `SET PASSWORD FOR 'Thomas'@'localhost' = Password ('Sec4525');`
- If connected as Thomas:  `SET PASSWORD = Password ('Sec4525');`


#### GRANT and REVOKE Privileges  Authorization 334. Intermediate SQL
![[chrome_NPHQmL3hei.png]]

- When users are created they can be granted privileges
- `GRANT SELECT (InstID, InstName, DeptName) ON University.Instructor TO 'Thomas'@'localhost';`Note: Thomas can only read from the named database, table and attributes
- `GRANT SELECT ON University.* TO 'Thomas'@'localhost';` Note: Thomas can read all tables and attributes from the University Database!
- `GRANT SELECT ON *.* TO 'Thomas'@'localhost';` Note: Thomas can read all databases, tables and attributes on the localhost!
- `GRANT ALL ON University.* TO 'Thomas'@'localhost'; SHOW GRANTS FOR 'Thomas'@'localhost';` Note: Thomas can do whatever he likes to the University database, since he has all privileges!
- The creator of an object holds all privileges on that object, incl. the privileges to grant privileges to other users
- If `WITH GRANT OPTION `is appended at the end of a `GRANT` command, the user will be allowed to grant the same privileges to other users


### Roles
- One can create roles `CREATE ROLE TeachingAssistant` 
- Roles can (just like users) be granted privileges, `GRANT SELECT ON University.Takes TO TeachingAssistant;`
- Roles can be granted to users (as well as to other roles) `GRANT TeachingAssistant TO 'Thomas'@'localhost';` by this Thomas is granted the privileges granted to TeachingAssistant
- Roles can be dropped `DROP ROLE TeachingAssistant;`
- NOTE: these commands are not supported by all DBMS (e.g. not for some MySQL versions), but are supported by MariaDB

## Procedural Statements
### Procedural Statements
#### BEGIN-END blocks
```MYSQL
[label_name:] BEGIN  
local-variable-declarations #scope is within the begin-end block  
statements  
END [label_name];
DECLARE var_name type [DEFAULT value] ;
```
local-variable-declarations are of the form: `DECLARE var_name type [DEFAULT value] ;`
#### Variable assignments 
- `SET var_name = expr;`
- Where var_name can be the name of
	- a local variable declared in an enclosed BEGIN-END statement
	- a formal parameter of an enclosed procedure (or function)
	- a user-defined variable (name is of the form @id, no specific declaration, is session specific),  e.g.` SET @x = 1; SET @x = @x +1;`
	- `SET [GLOBAL] system_var_name = value`

in SELECT clauses 
`SELECT ..., expr INTO var_name , `
#### IF ... THEN
```MYSQL
IF condition1 THEN statements1  
[ ELSEIF condition2 THEN statements2 ]  
...  
[ ELSE statementsn ]  
END IF;
```
#### Cases
```MYSQL
CASE expression  
WHEN value1 THEN  
statements1 #to execute when expression equals value_1  
...  
WHEN valuen THEN  
statementsn #to execute when expression equals value_n  
[ ELSE  
statements #to execute when no values matched ]  
END CASE;  
CASE  
WHEN condition1 THEN  
statements1 #to execute when condition_1 is TRUE  
...  
WHEN conditionn THEN  
statementsn #to execute when condition_n is TRUE  
[ ELSE  
statements #to execute when all conditions were FALSE ]  
END CASE;
```
#### LOOP 
- Loop
- `[ label_name: ] LOOP  statements #can be terminated by a LEAVE or RETURN statement  
`END LOOP [ label_name ];`
- While loop
	- `[ label_name: ] WHILE condition DO  statements END WHILE [ label_name ]`
- REPEAT UNTIL:
	- 
```MYSQL
[ label_name: ] REPEAT  
statements  
UNTIL condition  
END REPEAT [ label_name ]
```
- Leave: `LEAVE label_name # to exit a labelled loop`
- `ITERATE: ITERATE label_name # to start a labelled loop again`
- 
### Programming Objects
#### Functions
- Is a stored program for which parameters can be passed in, and then it will return a value:
- 
``` Mysql 
CREATE FUNCTION function_name (parameter1 datatype1, ... , parametern datatypen) 
RETURNS return_datatype
routine_body #must include a ‘RETURN value’ statement

##Example 
DELIMITER //  
CREATE FUNCTION DeptInstCount (vDeptName VARCHAR(20)) RETURNS INT  
BEGIN  
DECLARE vDeptInstCount INT;  
SELECT COUNT(*) INTO vDeptInstCount FROM Instructor  
WHERE DeptName = vDeptName;  
RETURN vDeptInstCount;  
END//  
DELIMITER ;

## Calculates age from date of birth
CREATE FUNCTION Age (vDate DATE) RETURNS INTEGER  
RETURN TIMESTAMPDIFF(YEAR, vDate, CURDATE());
```
- Call a function 
	- `function_name(e1, ..., en)`
	- `e1, ..., en `are expressions matching the formal parameters
	- Is syntactically a value expression
- Drop a function
	- `DROP FUNCTION function_name`
- 
#### Procedures
- Is a stored program for which parameters can be passed in and out
```MYSQL
CREATE PROCEDURE procedure_name  
([IN|OUT|INOUT] parameter1 datatype1 , ..., [IN|OUT|INOUT] parametern datatypen)  
routine_body

##EXAMPLE

DELIMITER $$  
CREATE PROCEDURE LuckyNumber()  
BEGIN  
DECLARE vNumber INTEGER DEFAULT 0;  
IF DAY(CURRENT_DATE())%2 = 0 THEN SET vNumber = 4; ELSE SET vNumber = 7; END IF;  
SELECT vNumber AS 'Todays Lucky Number';  
END $$  
DELIMITER;

##Example

DELIMITER //  
CREATE PROCEDURE AddInstructor  
(IN vInstID VARCHAR(5), IN vInstName VARCHAR(20), IN vDeptName VARCHAR(20))  
BEGIN  
INSERT Instructor(InstID, InstName, DeptName, Salary)  
VALUES (vInstID, vInstName, vDeptName, 29000); #As a side effect, a table is changed  
END //  
DELIMITER ;
##Example
DELIMITER //  
CREATE PROCEDURE BuildingCapacity  
(IN vBuilding VARCHAR(20), OUT vMaxCapacity INT)  
BEGIN  
SELECT SUM(Capacity) INTO vMaxCapacity FROM Classroom  
WHERE Building = vBuilding;  
END //  
DELIMITER ;

```

- A routine_body is a statement (typically a block).
- An` IN `parameter passes a value into a procedure.
- An `OUT` parameter passes a value from the procedure back to the caller
- An `INOUT` parameter passes a value into the procedure and back to the caller.
- Note that a procedure can have several output parameters!
- Call a procedure
	- `CALL procedure_name(e1, ..., en);`
	- `e1, ..., en` are expressions matching the formal parameters wrt types.
	- When parameter_i is OUT and INOUT parameters, ei must be a variable name.
	- Is syntactically a statement.
	- `CALL p();`is equivalent to `CALL p;`
- Drop a procedure  
	- `DROP PROCEDURE procedure_name`

#### Trigger  
- A set of SQL statements that are executed automatically by the DBMS system as a side effect of a table modification (i.e. an SQL INSERT, UPDATE or DELETE).
- Triggers can be used to preprocess and post process table changes
	- Example: In order to validate values of a new row, a trigger can be  executed BEFORE a row is inserted in a table, and it might reject insertion
	- Example: In order to update log tables, a trigger can be executed AFTER a row has been inserted in a table
	- Example: In order to enforce integrity constraints triggers may process tables changes.
- Create a trigger
	- 
	```MYSQL
	CREATE TRIGGER trigger_name #Use naming standard like: table_name_Before_Insert
	{ BEFORE | AFTER } { INSERT | UPDATE | DELETE } ON table_name  
	FOR EACH ROW actions # trigger actions


	#Example
	DELIMITER //  
	CREATE TRIGGER Instructor_Before_Insert  
	BEFORE INSERT ON Instructor FOR EACH ROW  
	BEGIN  
	IF NEW.Salary < 29000 THEN SET NEW.Salary = 29000;  
	ELSEIF NEW.Salary > 150000 THEN SET NEW.Salary = 150000;  
	END IF;  
	END//  
	DELIMITER;

	```
	- actions is a statement
	- `OLD.A` and `NEW.A` can be used in actions to refer to attribute A before an (UPDATE or DELETE) event and after an (UPDATE or INSERT) event, respectively.
	- In a BEFORE trigger, you can make assignments SET NEW.A = ... . This means you can use a trigger to modify the values to be inserted into a new row or used to update a row.
- The trigger definition specifies
	- The actions to be taken for inserted/updated/deleted rows when the trigger executes
		- **IMPORTANT:** Note that the "FOR EACH ROW" refers only to each row inserted/updated/deleted by the original query, not every row of the table! So if you only inserted one row, your trigger will run once in the context of that row.
	- the event for which a trigger is to be executed: an INSERT, UPDATE or DELETE
	- the time at which the trigger should be executed: BEFORE or AFTER the event
- Drop a trigger:
	- `DROP TRIGGER table_name.trigger_name; # table_name. only needed if trigger_name is defined on several tables`
- Show triggers in a database named db:
	- `SHOW TRIGGERS IN db;`
- When to NOT use triggers
	- Triggers were used earlier for tasks such as
		- Maintaining summary data tables (e.g., total salary of each department).
		- Replicating databases by recording changes to special relations (called change or delta relations) and having a separate process that applies the changes over to a replica
	- There are better ways of doing these now:
		- Databases today provide built-in View facilities to maintain summary data.
		- Databases provide built-in support for replication.

#### A Transaction
- SQL Transaction
	- Is a group of SQL statements that executes as one entity. 
	- If one or more statements fails to execute successfully, then the successfully executed statements are undone, and the database is returned to the initial state before the transaction was started. In practice, all changes are stored in temporary storage, until they are either committed and stored permanently, or discarded (i.e. rolled back) and deleted from temporary storage.
	- Transactions can be executed concurrently by the DBMS without disturbing each other.
- Nature or Transactions
	- All transactions have a beginning and an end.
	- Transaction results are either saved (committed) or aborted (rollback).
	- If a transaction fails, then no part of the transaction is saved.
- Controlling Transactions
	- `START TRANSACTION`
		- Will ensure that all row inserts, updates and deletes are stored in a Temporary   Buffer and NOT written to disk and permanently stored
	- `COMMIT;`.
		- Will ensure that the changes stored in the Temporary Buffer are written to disk and permanently stored
	- `ROLLBACK;`
		- Will ensure that all changes made since the START TRANSACTION will be deleted in the Temporary Buffer and NOT written to disk and permanently stored.
Example
```MYSQL
##Create a Procedure that includes a Transaction  
DELIMITER //  
CREATE PROCEDURE Transfer (  
IN vTransfer DECIMAL(9,2), vID1 INT, vID2 INT, OUT vStatus VARCHAR(45))  
BEGIN  
DECLARE OldAmountID1, NewAmountID1, OldAmountID2, NewAmountID2 INT DEFAULT 0;  
START TRANSACTION;  
SET SQL_SAFE_UPDATES = 0;  
SET OldAmountID1 = (SELECT Amount FROM Accounts WHERE ID = vID1);  
SET NewAmountID1 = OldAmountID1 - vTransfer;  
UPDATE Accounts SET Amount = NewAmountID1 WHERE ID = vID1;  
SET OldAmountID2 = (SELECT Amount FROM Accounts WHERE ID = vID2);  
SET NewAmountID2 = OldAmountID2 + vTransfer;  
UPDATE Accounts SET Amount = NewAmountID2 WHERE ID = vID2;  
IF (OldAmountID1 + OldAmountID2) = (NewAmountID1 + NewAmountID2)  
THEN SET vStatus = 'Transaction Transfer committed!'; COMMIT;  
ELSE SET vStatus = 'Transaction Transfer rollback'; ROLLBACK;  
END IF;  
END //  
DELIMITER ;
```

#### An Event
- Is a stored program that is executed at a given time or at given time intervals:  
- Create Event
	- 
		```MYSQL
		CREATE EVENT event_name  
		ON SCHEDULE schedule  
		DO statement
		```
	- schedule can e.g. be of the form
		- EVERY n timeunit
		- \[STARTS timestamp]
		- \[ENDS timestamp]
		- timeunit can be: YEAR | QUARTER | MONTH | DAY | HOUR | MINUTE | WEEK | SECOND | YEAR_MONTH | DAY_HOUR | DAY_MINUTE | DAY_SECOND |HOUR_MINUTE | HOUR_SECOND | MINUTE_SECOND
	- `SET GLOBAL event_scheduler = 1; # in order for MariaDB/MySQL to execute events`
	- Events can e.g. be used to schedule database period backups at regular time intervals.
```MYSQL
SHOW VARIABLES LIKE 'event_scheduler'; # Initially it is set OFF
SET GLOBAL event_scheduler = 1; # To set it ON. MySQL will look for/execute events

CREATE TABLE MarkLog(  
TS TIMESTAMP,  
Message VARCHAR(100));

CREATE EVENT MarkInsert  
ON SCHEDULE EVERY 2 MINUTE  
DO INSERT MarkLog VALUES (CURRENT_TIMESTAMP, "-- It's time again--");


```
#### User-defined Error Signalling
- It is possible to raise signals for user-defined conditions indicating errors and warnings using the SIGNAL statement. Typical pattern for this:
```MYSQL
IF condition THEN SIGNAL SQLSTATE state  
SET MYSQL_ERRNO = ..., MESSAGE_TEXT = ...;
#Example
IF NOT (NEW.Credits BETWEEN 0 AND 5)  
THEN SIGNAL SQLSTATE 'HY000'  
SET MYSQL_ERRNO = 1525, MESSAGE_TEXT = 'Invalid Credits; Range is [0,5]!';  
END IF;
```
- SQLSTATE HY000 means that a general error has occurred
- MYSQL_ERRNO 1525 means wrong value occurred
- When the signal is raised, the trigger program is aborted and the related INSERT is not executed!
#### User-defined Error Handling
- To raise a signal for a user-defined condition, use the SIGNAL statement.
- To declare a handler, use the DECLARE ... HANDLER statement.
```MYSQL
DECLARE handler_action HANDLER  
FOR condition-value statement
```
- Condition-value = mysql_or_mariadb_error_code | SQLSTATE sqlstate_value | SQLWARNING | NOT FOUND | SQLEXCEPTION | ...
- If the condition-value occurs, the specified statement executes
- The handler_action indicates what action the handler takes after execution of the statement:
	- CONTINUE: Execution of the current program continues
	- EXIT: Execution terminates for the BEGIN ... END compound statement in which the handler is  declared
#### SQL Access from a Programming Language
- Application Programs
	- Normally the application user interface and application logic are programmed using a common programming language like C, Java or Cobol
- Application Programs can interact with a database
	- Using an Application Program Interface (API) providing functionality for.
		- connecting with the database server
		- sending SQL commands to the database server
		- fetching tuples of a query result one-by-one into program variables
- Java DataBase Connectivity (JDBC)
	- API for Java
- Open DataBase Connectivity (ODBC)
	- Standard API. Works with a wide range of programming languages like C, C++, C#, Cobol and Visual Basic
- Embedded SQL
	- The SQL standard also defines embedding's of SQL in a variety of programming languages such as C, Java, and Cobol.
	- A language to which SQL queries are embedded is referred to as a host language, and the SQL structures permitted in the host language constitute embedded SQL
	- Special statements are used to identify embedded SQL, like
		- EXEC SQL \<embedded SQL statement > END_EXEC for embedded Cobol
		- \# SQL { embedded SQL statement } for embedded Java
	- A precompiler replaces the SQL statements in the application program with appropriate host language declarations and procedure calls that allow runtime execution of the database access
	- This approach differs from JDBC and ODBC that use a dynamic SQL  
approach which does not use a precompiler, but allows the application  
program to construct and submit SQL queries as strings at runtime
- 

# <font color="crimson"> color</font>

LIKE in create 
