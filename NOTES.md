# South American Borrowing - Database and Software
## Installation and Workflow for sabor

0. [Optional] PRE-INSTALLATION

Create virtual environment.  
We use <cite>conda</cite> for managing virtual environments.

<code>% conda create -n testsabor python=3  
% conda activate testsabor</code>

1. INSTALL SABOR ===
- Copy zip file of sabor to local directory.
- [Optional] Rename unzipped folder to sabor.  

Make sabor your working directory.  
<code>% cd sabor</code>  

Install requirements.
<code>% pip install -r requirements.txt</code>  

Install sabor. Developer version for commands access. 
<code>% pip install -e .</code>  

Install the datafiles for IDS and WOLD.  
<code>% cldfbench download lexibank_sabor.py</code>

Construct the Sabor datafile and activate Sabor commands.  
<code>% cldfbench lexibank.makecldf lexibank_sabor.py</code>

Check for access to commands.  
<code>% cldfbench --help  </code>  
It should show the available sabor commands with prefix <code>sabor.</code>


2. [Optional] SPLIT DATA ===

Optionally run Splits to create train and test datasets for cross validation.
Note, that if you do not change the folder to other than splits, 
this will overwrite cross-validation datasets used in our analysis and 
you will not be able to precisely duplicate paper results.

You can get help on the arguments for each command with the —help argument.  
<code>% cldfbench sabor.splitdata --help</code>  
<code>usage: cldfbench sabor.splitdata [-h] [--k K] [--folder FOLDER] [--test]</code>  
<code>Partition wordlists into k-fold train and test files for use in cross-validations.  
options:  
  -h, --help       show this help message and exit  
  --k K            Partition factor k to use for cross-validation. (default: 10)  
  --folder FOLDER  Folder to store train-test splits for k-fold cross-validations. (default: splits)  
  --test           Demonstrate loading of train-test datasets in sequence. (default: False)</code>  

Reproduce the 10-fold split from the paper, but store into a different folder. Save the output!  
<code>% cldfbench sabor.splitdata --k 10 --folder splits-new >> output/Split-data-10-fold.txt</code>


3. PERFORM CLOSEST MATCH BORROWING DETECTION ===

Analyze complete sabor dataset for closest match using SCA alignment method.  
Add --label argument to qualify the stored wordlist filename.  
<code>% cldfbench sabor.closest --function sca --label "All-data"  
INFO    Construct closest from SaBor database.  
INFO    Trained with donors ['Spanish'], function closest_match_sca_global  
INFO    Best: threshold 0.40, F1 score 0.789  
[INFO] Data has been written to file <store/CM-sp-predict-closest_match_sca_global-0.40-All-data-train.tsv>.</code>  

Analyze train dataset and infer to test dataset for closest match using SCA alignment method.  
<code>% cldfbench sabor.closest --function sca --label "Fold-00-data"\  
--file splits/CV10-fold-00-train.tsv --testfile splits/CV10-fold-00-test.tsv  
INFO    Construct closest from splits/CV10-fold-00-train.tsv.  
INFO    Trained with donors ['Spanish'], function closest_match_sca_global  
INFO    Best: threshold 0.40, F1 score 0.790  
[INFO] Data has been written to file </store/CM-sp-predict-closest_match_sca_global-0.40-Fold-00-data-train.tsv>.  
INFO    Test closest from splits/CV10-fold-00-test.tsv.  
INFO    Trained with donors ['Spanish'], function closest_match_sca_global                           
INFO    Best: threshold 0.40, F1 score 0.790  
INFO    Test: threshold 0.40, F1 score 0.788  
[INFO] Data has been written to file   <store/CM-sp-predict-closest_match_sca_global-0.40-Fold-00-data-test.tsv>.</code>  

We can evaluate language recipient details and overall performance as well as write out wordlist for diagnosis of errors for just the test dataset.  
<code>% cldfbench sabor.evaluate --file store/CM-sp-predict-closest_match_sca_global-0.40-Fold-00-data-test.tsv >>   output/CM-evaluate-sca_global_fold-00-test.txt  
[INFO] Data has been written to file <store/CM-sp-predict-closest_match_sca_global-0.40-Fold-00-data-test-evaluate.tsv>.</code>  


Analyze complete sabor dataset for closest match using edit distance method.  
<code>% cldfbench sabor.closest --function ned --label "All-data"  
INFO    Construct closest from SaBor database.  
INFO    Trained with donors ['Spanish'], function edit_distance  
INFO    Best: threshold 0.65, F1 score 0.765  
[INFO] Data has been written to file <store/CM-sp-predict-edit_distance-0.65-All-data-train.tsv>.</code>  


4. PERFORM COGNATE BASED BORROWING DETECTION ===

All previous options to label output, perform train and test steps, evaluate language details are still available.  

With SCA distance function.  
<code>% cldfbench sabor.cognate --method sca --label "All-data"  
INFO    Trained with donors ['Spanish'], function cognate_based_cognate_sca_global                   
INFO    Best: threshold 0.46, F1 score 0.786  
[INFO] Data has been written to file </sabor/store/CB-sp-predict-cognate_based_cognate_sca_global-0.46-All-data-train.tsv>.</code>  

With edit distance function.  
<code>% cldfbench sabor.cognate --method ned --label "All-data"  
INFO    Trained with donors ['Spanish'], function cognate_based_cognate_ned                           
INFO    Best: threshold 0.66, F1 score 0.774  
[INFO] Data has been written to file </sabor/store/CB-sp-predict-cognate_based_cognate_ned-0.66-All-data-train.tsv></code>  


5. PERFORM CLASSIFIER BASED BORROWING DETECTION ===

All previous options to label output, perform train and test steps, evaluate language details are still available.  

Default is both edit distance and SCA alignment methods together with 1-hot encoding of recipient language.  
<code>% cldfbench sabor.classifier  --label "All-data"   
INFO    Construct classifier from SaBor database.  
INFO    Trained with donors ['Spanish'], functions ['sca', 'ned']  
INFO    Train: F1 score 0.810  
[INFO] Data has been written to file </store/CL-sp-predict-linear_svm-All-data-train.tsv>.</code>  

Access to turn of 1-hot coding for language is not available from the command line.  


6. EVALUATE LANGUAGE DETAILS ===

Evaluate writes results to output as well as an evaluation wordlist to the store folder.  
Here we redirect output to the output folder.  
<code>% cldfbench sabor.evaluate --file store/CL-sp-predict-linear_svm-All-data-train.tsv >> output/CL-evaluate-All-data-train.txt
[INFO] Data has been written to file <store/CL-sp-predict-linear_svm-All-data-train-evaluate.tsv>.</code>

We could of course output directly to the terminal.  
<code>% cldfbench sabor.evaluate --file store/CL-sp-predict-linear_svm-All-data-train.tsv</code>

<pre>
Detection results for: store/CL-sp-predict-linear_svm-All-data-train.tsv.
Language             tp    tn    fp    fn    precision    recall    F1 score    accuracy
-----------------  ----  ----  ----  ----  -----------  --------  ----------  ----------
Mapudungun          136  1044     8    54        0.944     0.716       0.814       0.950
Yaqui               242  1106    16    69        0.938     0.778       0.851       0.941
ImbaburaQuechua     232   836    20    68        0.921     0.773       0.841       0.924
Otomi               137  2033    10    61        0.932     0.692       0.794       0.968
ZinacantanTzotzil   109  1093     8    56        0.932     0.661       0.773       0.949
Wichi               100  1062     5    52        0.952     0.658       0.778       0.953
Qeqchi               99  1605     7    62        0.934     0.615       0.742       0.961
Overall            1055  8779    74   422        0.934     0.714       0.810       0.952

Overall detection results for: store/CL-sp-predict-linear_svm-All-data-train.tsv
                  borrowed    not borrowed    total
--------------  ----------  --------------  -------
identified            1055              74     1129
not identified         422            8779     9201
total                 1477            8853    10330
</pre>
<code>[INFO] Data has been written to file <store/CL-sp-predict-linear_svm-All-data-train-evaluate.tsv>.</code>


This evaluation file may be inspected to help diagnose what are the errors in borrowing detection.


7. CROSS-VALIDATION ===

Perform 10 fold cross-validations for closest match, cognate based and classifier systems.  
Redirect output to the output folder.  

Closest match:  
<code>% cldfbench sabor.crossvalidate 10 --method cm_sca >> output/Closest-cross-validate-10fold.txt  
INFO    10-fold cross-validation on splits directory using closest_match_sca_global.

% cldfbench sabor.crossvalidate 10 --method cm_ned >> output/Closest-cross-validate-10fold.txt  
INFO    10-fold cross-validation on splits directory using edit_distance.</code>  

Cognate based:  
<code>% cldfbench sabor.crossvalidate 10 --method cb_sca >> output/Cognate-cross-validate-10fold.txt  

% cldfbench sabor.crossvalidate 10 --method cb_ned >> output/Cognate-cross-validate-10fold.txt</code>  

Classifier:  
Cl_simple includes both edit distance and SCA methods.  
<code>% cldfbench sabor.crossvalidate 10 --method cl_simple >> output/Classifier-cross-validate-10fold.txt  
INFO    10-fold cross-validation on splits directory using classifier_based_linear_svm_simple.</code>  

No_props drops the 1-hot encoding of recipient language for the same analysis.  
<code>% cldfbench sabor.crossvalidate 10 --method cl_simple_no_props >> output/Classifier-cross-validate-10fold.txt  
INFO    10-fold cross-validation on splits directory using classifier_based_linear_svm_simple_no_props.</code>  


