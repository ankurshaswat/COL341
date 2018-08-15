# COL341: Assignment 1
##Linear Regression

**Name**

linreg - Run the executable program for linear regression

**Synopsis**

`./linreg <part> <tr> <ts> <out>`

**Description**

This program will train linear regression model using given code on train data, make predictions on test data and write final predictions
in given output file.

**Options**

-part  
    Part as per question i.e. a/b/c.  
-tr  
    File containing training data in csv format where 1st entry is the target  
-ts  
    File containing test data in csv format where 1st entry is the target  
-out  
    Output file for predictions. One value in each line.

**Example**
    
`./linreg a train.csv test.csv output`
    
**Data**

- msd_train.csv: Train data  
- msd_test.csv: Test data
    
**Marking scheme**

Marks will be given based on following categories:

1. Part-a/b/c: Run/format error (0 points) 
2. Part-a/b: Runs fine but predictions are incorrect within some threshold (6.25 points each) 
3. Part-a/b: Works as expected with correct predictions (12.5 points each)
4. Part-c: Relative marking will be done based on specified error.


**Checking Program**

Normalized mean squared error will be used as an evaluation criterion:
    error = \frac{\sum_{i=1}^{m} (y_i - \hat{y_i})^2}{\sum_{i=1}^{m} (y_i - min\_val)^2}

**Submission**

1. Your submission should be "ENTRY_NO.zip".
2. Make sure you clean up extra files/directories such as "__MACOSX"
3. Command "unzip ENTRY_NO.zip", should result in a single directory "ENTRY_NO".

-----------------
##Logistic Regression

**Name**

logreg - Run the executable program for linear regression

**Synopsis**

`./logreg <part> <method> <lr> <niter> <bs> <tr> <vocab> <ts> <out>`

**Description**

This program will train logistic regression model using given code on train data, make predictions on test data and write final predictions
in given output file.

**Options**

-part  
    Part as per question i.e. a or b.  
-method  
    1 (fixed step size), 2 (Adaptive learning rate), 3 (Exact line search)  
-lr  
    inital learning rate  
-niter  
    Run your algorithm for these many iterations.  
-bs  
    Batch size for SGD.  
-tr  
    File containing training data in csv format where 1st entry is the target  
-vocab  
    Vocabulary file
-ts  
    File containing test data in csv format where 1st entry is the target  
-out  
    Output file (write your predictions in this file) 

**Example**
    
`./logreg a 1 0.1 100 128 train.csv vocab.txt test.csv output`

**Data**

- imdb_train.csv: Train data
- imdb_test.csv: Test data
- imdb_vocab: Vocabulary; Use word count as features as per this vocabulary.
    
**Marking scheme**

Marks will be given based on following categories:

1. Report: 15 points, 35 points code
2. Run/format error (0 points in both)
3. Runs fine but predictions are incorrect within some threshold (50% in code) 
4. Works as expected with correct predictions (100% in code)
5. Report marks will be given manually.


**Checking Program**

Fraction of correct test samples and training time will be used as evaluation criterion.

**Submission**

1. Your submission should be "ENTRY_NO.zip".
2. Make sure you clean up extra files/directories such as "__MACOSX"
3. Command "unzip ENTRY_NO.zip", should result in a single directory "ENTRY_NO".
