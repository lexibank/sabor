# Analysis of SABOR dataset.

# ----------------------------------------------
# Split data into fixed train and test datasets.
# ----------------------------------------------

cldfbench sabor.splitdata --k 4

# * Constructed 4 train-test splits *
#   k-fold    partition    # test concepts    # test entries    # train concepts    # train entries
# --------  -----------  -----------------  ----------------  ------------------  -----------------
#        4            0                327              3044                 981               9056
#        4            1                327              3004                 981               9096
#        4            2                327              3082                 981               9018
#        4            3                327              2970                 981               9130

cldfbench sabor.splitdata --k 10

# * Constructed 10 train-test splits *
#   k-fold    partition    # test concepts    # test entries    # train concepts    # train entries
# --------  -----------  -----------------  ----------------  ------------------  -----------------
#       10            0                130              1200                1178              10900
#       10            1                131              1210                1177              10890
#       10            2                131              1217                1177              10883
#       10            3                131              1187                1177              10913
#       10            4                131              1189                1177              10911
#       10            5                130              1174                1178              10926
#       10            6                131              1206                1177              10894
#       10            7                131              1183                1177              10917
#       10            8                131              1271                1177              10829
#       10            9                131              1263                1177              10837

# --------------------------------------------
# Detect borrowings by method of closest match
# --------------------------------------------

cldfbench sabor.closest --label all
# INFO    Construct closest from SaBor database.
# INFO    Trained with donors ['Spanish'], function sca_distance
# INFO    Best: threshold 0.40, F1 score 0.789
# INFO    Predict with  threshold 0.4.
# [INFO] Data has been written to file <store/CM-sp-predict-sca_distance-0.40-all-train.tsv>.

cldfbench sabor.closest --function NED --label all
# INFO    Construct closest from SaBor database.
# INFO    Trained with donors ['Spanish'], function edit_distance
# INFO    Best: threshold 0.65, F1 score 0.765
# INFO    Predict with  threshold 0.65.
# [INFO] Data has been written to file <store/CM-sp-predict-edit_distance-0.65-all-train.tsv>.

cldfbench sabor.closest --file splits/CV10-fold-00-train.tsv \
    --testfile splits/CV10-fold-00-test.tsv  --label cv10_fold_0
# INFO    Construct closest from splits/CV10-fold-00-train.tsv.
# INFO    Trained with donors ['Spanish'], function sca_distance
# INFO    Best: threshold 0.40, F1 score 0.790
# INFO    Predict with  threshold 0.4.
# [INFO] Data has been written to file <store/CM-sp-predict-sca_distance-0.40-cv10_fold_0-train.tsv>.
# INFO    Test closest from splits/CV10-fold-00-test.tsv.
# INFO    Evaluation:0.788
# [INFO] Data has been written to file <store/CM-sp-predict-sca_distance-0.40-cv10_fold_0-test.tsv>.

cldfbench sabor.closest --function NED --file splits/CV10-fold-00-train.tsv \
    --testfile splits/CV10-fold-00-test.tsv  --label cv10_fold_0
# INFO    Construct closest from splits/CV10-fold-00-train.tsv.
# INFO    Trained with donors ['Spanish'], function edit_distance
# INFO    Best: threshold 0.65, F1 score 0.767
# INFO    Predict with  threshold 0.65.
# [INFO] Data has been written to file <store/CM-sp-predict-edit_distance-0.65-cv10_fold_0-train.tsv>.
# INFO    Test closest from splits/CV10-fold-00-test.tsv.
# INFO    Evaluation:0.748
# [INFO] Data has been written to file <store/CM-sp-predict-edit_distance-0.65-cv10_fold_0-test.tsv>.

# ----------------------------------------------------------------
# Evaluate borrowing detection by target language for test dataset
# ----------------------------------------------------------------

cldfbench sabor.evaluate --file store/CM-sp-predict-edit_distance-0.65-cv10_fold_0-test.tsv

# Detection results for: store/CM-sp-predict-edit_distance-0.65-cv10_fold_0-test.tsv.
# Language             tp    tn    fp    fn    precision    recall    F1 score    accuracy
# -----------------  ----  ----  ----  ----  -----------  --------  ----------  ----------
# Qeqchi                9   174     2     8        0.818     0.529       0.643       0.948
# Mapudungun            8   100     5     4        0.615     0.667       0.640       0.923
# Yaqui                19   110     4     6        0.826     0.760       0.792       0.928
# ZinacantanTzotzil    15   102     5     3        0.750     0.833       0.789       0.936
# Otomi                11   205     1     4        0.917     0.733       0.815       0.977
# Wichi                 7   102     0     4        1.000     0.636       0.778       0.965
# ImbaburaQuechua      20    80     7     7        0.741     0.741       0.741       0.877
# Overall              89   873    24    36        0.788     0.712       0.748       0.941
#
# Overall detection results for: store/CM-sp-predict-edit_distance-0.65-cv10_fold_0-test.tsv
#                   borrowed    not borrowed    total
# --------------  ----------  --------------  -------
# identified              89              24      113
# not identified          36             873      909
# total                  125             897     1022
# [INFO] Data has been written to file <store/CM-sp-predict-edit_distance-0.65-cv10_fold_0-test-evaluate.tsv>.

cldfbench sabor.evaluate --file store/CM-sp-predict-sca_distance-0.40-cv10_fold_0-test.tsv

# Detection results for: store/CM-sp-predict-sca_distance-0.40-cv10_fold_0-test.tsv.
# Language             tp    tn    fp    fn    precision    recall    F1 score    accuracy
# -----------------  ----  ----  ----  ----  -----------  --------  ----------  ----------
# Qeqchi               11   175     1     6        0.917     0.647       0.759       0.964
# Yaqui                18   113     1     7        0.947     0.720       0.818       0.942
# Wichi                10   101     1     1        0.909     0.909       0.909       0.982
# ZinacantanTzotzil    14   105     2     4        0.875     0.778       0.824       0.952
# ImbaburaQuechua      21    82     5     6        0.808     0.778       0.792       0.904
# Mapudungun            7   103     2     5        0.778     0.583       0.667       0.940
# Otomi                10   203     3     5        0.769     0.667       0.714       0.964
# Overall              91   882    15    34        0.858     0.728       0.788       0.952
#
# Overall detection results for: store/CM-sp-predict-sca_distance-0.40-cv10_fold_0-test.tsv
#                   borrowed    not borrowed    total
# --------------  ----------  --------------  -------
# identified              91              15      106
# not identified          34             882      916
# total                  125             897     1022
# [INFO] Data has been written to file <store/CM-sp-predict-sca_distance-0.40-cv10_fold_0-test-evaluate.tsv>.


# --------------------------------------------
# Detect borrowings by cognate based method
# --------------------------------------------

cldfbench sabor.cognate --label all

# INFO    Construct cognate from SaBor database.
# INFO    Trained with donors ['Spanish'], function cognate_based_cognate_lex
# INFO    Best: threshold 0.46, F1 score 0.786
# INFO    Predict with  threshold 0.46.
# [INFO] Data has been written to file <store/CB-sp-predict-cognate_based_cognate_lex-0.46-all-train.tsv>.

cldfbench sabor.cognate --file splits/CV10-fold-00-train.tsv \
    --testfile splits/CV10-fold-00-test.tsv  --label cv10_fold_0

# INFO    Construct cognate from splits/CV10-fold-00-train.tsv.
# INFO    Trained with donors ['Spanish'], function cognate_based_cognate_sca
# INFO    Best: threshold 0.46, F1 score 0.786
# INFO    Predict with  threshold 0.46.
# INFO    Evaluation:0.761
# [INFO] Data has been written to file <store/CB-sp-predict-cognate_based_cognate_sca-0.46-cv10_fold_0-test.tsv>

# ----------------------------------------------------------------
# Evaluate borrowing detection by target language for test dataset
# ----------------------------------------------------------------

cldfbench sabor.evaluate --file store/CB-sp-predict-cognate_based_cognate_sca-0.46-cv10_fold_0-test.tsv

# Detection results for: store/CB-sp-predict-cognate_based_cognate_sca-0.46-cv10_fold_0-test.tsv.
# Language             tp    tn    fp    fn    precision    recall    F1 score    accuracy
# -----------------  ----  ----  ----  ----  -----------  --------  ----------  ----------
# ImbaburaQuechua      22    80     7     5        0.759     0.815       0.786       0.895
# Otomi                11   200     6     4        0.647     0.733       0.688       0.955
# Wichi                10    99     3     1        0.769     0.909       0.833       0.965
# ZinacantanTzotzil    15   105     2     3        0.882     0.833       0.857       0.960
# Mapudungun            7   100     5     5        0.583     0.583       0.583       0.915
# Qeqchi               11   174     2     6        0.846     0.647       0.733       0.959
# Yaqui                18   111     3     7        0.857     0.720       0.783       0.928
# Overall              94   869    28    31        0.770     0.752       0.761       0.942
#
# Overall detection results for: store/CB-sp-predict-cognate_based_cognate_sca-0.46-cv10_fold_0-test.tsv
#                   borrowed    not borrowed    total
# --------------  ----------  --------------  -------
# identified              94              28      122
# not identified          31             869      900
# total                  125             897     1022
# [INFO] Data has been written to file <store/CB-sp-predict-cognate_based_cognate_sca-0.46-cv10_fold_0-test-evaluate.tsv>.


# --------------------------------------------
# Detect borrowings by classifier based method
# --------------------------------------------

cldfbench sabor.classifier --method SCA --label all
# INFO    Construct classifier from SaBor database.
# INFO    Trained with donors ['Spanish'], functions ['SCA', 'NED'], methods ['SCA']
# INFO    Predict
# INFO    Evaluation:0.81
# [INFO] Data has been written to file <store/CL-sp-predict-SVC_linear-all-train.tsv>.


cldfbench sabor.classifier --method LEX --label all-lex

# INFO    Construct classifier from SaBor database.
# INFO    Trained with donors ['Spanish'], functions ['SCA', 'NED'], methods ['LEX']
# INFO    Predict
# INFO    Evaluation:0.82

cldfbench sabor.classifier --function SCA --method LEX --file splits/CV10-fold-00-train.tsv \
    --testfile splits/CV10-fold-00-test.tsv  --label cv10_fold_0
# INFO    Construct classifier from splits/CV10-fold-00-train.tsv.
# INFO    Trained with donors ['Spanish'], functions ['SCA'], methods ['LEX']
# INFO    Predict
# INFO    Evaluation:0.809
# [INFO] Data has been written to file <store/CL-sp-predict-SVC_linear-cv10_fold_0-train.tsv>.
# INFO    Test classifier from splits/CV10-fold-00-test.tsv.
# Evaluation:0.782
# [INFO] Data has been written to file <store/CL-sp-predict-SVC_linear-cv10_fold_0-test.tsv>.

# ----------------------------------------------------------------
# Evaluate borrowing detection by target language for test dataset
# ----------------------------------------------------------------

cldfbench sabor.evaluate --file store/CL-sp-predict-SVC_linear-cv10_fold_0-test.tsv

# Detection results for: store/CL-sp-predict-SVC_linear-cv10_fold_0-test.tsv.
# Language             tp    tn    fp    fn    precision    recall    F1 score    accuracy
# -----------------  ----  ----  ----  ----  -----------  --------  ----------  ----------
# ImbaburaQuechua      20    83     4     7        0.833     0.741       0.784       0.904
# Yaqui                19   113     1     6        0.950     0.760       0.844       0.950
# Otomi                 8   203     3     7        0.727     0.533       0.615       0.955
# ZinacantanTzotzil    14   107     0     4        1.000     0.778       0.875       0.968
# Qeqchi               10   176     0     7        1.000     0.588       0.741       0.964
# Wichi                 8   102     0     3        1.000     0.727       0.842       0.973
# Mapudungun            7   104     1     5        0.875     0.583       0.700       0.949
# Overall              86   888     9    39        0.905     0.688       0.782       0.953
#
# Overall detection results for: store/CL-sp-predict-SVC_linear-cv10_fold_0-test.tsv
#                   borrowed    not borrowed    total
# --------------  ----------  --------------  -------
# identified              86               9       95
# not identified          39             888      927
# total                  125             897     1022
# [INFO] Data has been written to file <store/CL-sp-predict-SVC_linear-cv10_fold_0-test-evaluate.tsv>.


# ---------------------------------------------------------------------------
# Cross-validation of borrowing detection by target language for test dataset
# ---------------------------------------------------------------------------

cldfbench sabor.crossvalidate 10

# INFO    10-fold cross-validation on splits directory using sca_distance.
#
# 10-fold cross-validation on splits directory using sca_distance.
#   fn    fp     tn     tp    precision    recall     f1     fb    accuracy    threshold  fold
# ----  ----  -----  -----  -----------  --------  -----  -----  ----------  -----------  ------
# 34.0  15.0  882.0   91.0        0.858     0.728  0.788  0.788       0.952         0.40  0
# 43.0  13.0  893.0   97.0        0.882     0.693  0.776  0.776       0.946         0.40  1
# 34.0  13.0  908.0   77.0        0.856     0.694  0.766  0.766       0.954         0.40  2
# 63.0  13.0  797.0  136.0        0.913     0.683  0.782  0.782       0.925         0.40  3
# 48.0  19.0  828.0  120.0        0.863     0.714  0.782  0.782       0.934         0.40  4
# 35.0  18.0  841.0  102.0        0.850     0.745  0.794  0.794       0.947         0.40  5
# 33.0  13.0  860.0  115.0        0.898     0.777  0.833  0.833       0.955         0.40  6
# 46.0  20.0  857.0   86.0        0.811     0.652  0.723  0.723       0.935         0.40  7
# 36.0  18.0  950.0   98.0        0.845     0.731  0.784  0.784       0.951         0.40  8
# 40.0  14.0  881.0  143.0        0.911     0.781  0.841  0.841       0.950         0.40  9
# 41.2  15.6  869.7  106.5        0.869     0.720  0.787  0.787       0.945         0.40  mean
#  9.3   2.8   43.3   21.5        0.032     0.041  0.033  0.033       0.010         0.00  stdev

cldfbench sabor.crossvalidate 10 --method cmned

# INFO    10-fold cross-validation on splits directory using edit_distance.
#
#   fn    fp     tn     tp    precision    recall     f1     fb    accuracy    threshold  fold
# ----  ----  -----  -----  -----------  --------  -----  -----  ----------  -----------  ------
# 39.0  21.0  876.0   86.0        0.804     0.688  0.741  0.741       0.941         0.60  0
# 43.0  13.0  893.0   97.0        0.882     0.693  0.776  0.776       0.946         0.60  1
# 40.0  24.0  897.0   71.0        0.747     0.640  0.689  0.689       0.938         0.65  2
# 71.0   9.0  801.0  128.0        0.934     0.643  0.762  0.762       0.921         0.60  3
# 48.0  20.0  827.0  120.0        0.857     0.714  0.779  0.779       0.933         0.60  4
# 41.0  24.0  835.0   96.0        0.800     0.701  0.747  0.747       0.935         0.65  5
# 37.0  26.0  847.0  111.0        0.810     0.750  0.779  0.779       0.938         0.65  6
# 51.0  11.0  866.0   81.0        0.880     0.614  0.723  0.723       0.939         0.60  7
# 38.0  18.0  950.0   96.0        0.842     0.716  0.774  0.774       0.949         0.60  8
# 49.0  11.0  884.0  134.0        0.924     0.732  0.817  0.817       0.944         0.60  9
# 45.7  17.7  867.6  102.0        0.848     0.689  0.759  0.759       0.938         0.61  mean
# 10.1   6.3   42.4   20.7        0.059     0.044  0.036  0.036       0.008         0.02  stdev


cldfbench sabor.crossvalidate 10 --method cbsca

# INFO 10-fold cross-validation on splits directory using cognate_based_cognate_sca.
#
# 10-fold cross-validation on splits directory using cognate_based_cognate_sca.
#   fn    fp     tn     tp    precision    recall     f1     fb    accuracy    threshold  fold
# ----  ----  -----  -----  -----------  --------  -----  -----  ----------  -----------  ------
# ...
# 37.6  28.8  856.5  110.1        0.787     0.743  0.763  0.763       0.936         0.46  mean
#  8.4   5.8   40.9   23.1        0.055     0.047  0.040  0.040       0.006         0.01  stdev


cldfbench sabor.crossvalidate 10 --method clsvcfull

# INFO    10-fold cross-validation on splits directory using classifier_based_SVM_linear_full.
#
# 10-fold cross-validation on splits directory using classifier_based_SVM_linear_full.
#   fn    fp     tn     tp    precision    recall     f1     fb    accuracy  fold
# ----  ----  -----  -----  -----------  --------  -----  -----  ----------  ------
# ...
# 41.8   7.4  877.9  105.9        0.931     0.713  0.807  0.807       0.952  mean
#  8.8   2.3   43.6   24.2        0.028     0.055  0.042  0.042       0.009  stdev


% cldfbench sabor.crossvalidate 10 --method clsvclean

# INFO    10-fold cross-validation on splits directory using classifier_based_SVM_linear_lean.
#
# 10-fold cross-validation on splits directory using classifier_based_SVM_linear_lean.
#   fn    fp     tn     tp    precision    recall     f1     fb    accuracy  fold
# ----  ----  -----  -----  -----------  --------  -----  -----  ----------  ------
# ...
# 44.3  10.5  874.8  103.4        0.904     0.695  0.785  0.785       0.947  mean
#  8.8   2.5   43.2   24.8        0.032     0.062  0.048  0.048       0.010  stdev


cldfbench sabor.crossvalidate 10 --method clsvcfast

# INFO    10-fold cross-validation on splits directory using classifier_based_SVM_linear_fast.
#
# 10-fold cross-validation on splits directory using classifier_based_SVM_linear_fast.
#   fn    fp     tn     tp    precision    recall     f1     fb    accuracy  fold
# ----  ----  -----  -----  -----------  --------  -----  -----  ----------  ------
# ...
# 43.1   7.0  878.3  104.6        0.936     0.705  0.803  0.803       0.952  mean
#  9.1   3.1   42.7   23.1        0.027     0.053  0.035  0.035       0.008  stdev