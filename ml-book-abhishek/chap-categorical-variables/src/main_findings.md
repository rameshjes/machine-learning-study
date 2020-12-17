
#### Take Aways

* No need to scale/normalize the features for tree-based algos. (but for Linear algos. you need normalization)
* Generally tree-based algos has more training and inference time. 
* It is not necessary that one algo. performs better on all the datasets
* When you want to compare different algos. on single imbalanced dataset, then AUC score is best metric.

#### Different ways of handling categorical variables
* Label Encoding
* One hot Encoding
* Categorical Encoding
* Target Encoding
* Entity Embedding

### Results on categorical(cat-in-the-dat) dataset (target is binary)


##### Logistic Regression
python3 -W ignore ohe_logreg_cat_dataset.py
fold : 0 auc : 0.7866697914351795
fold : 1 auc : 0.7877065137115273
fold : 2 auc : 0.7864918776667466
fold : 3 auc : 0.7850842470092773
fold : 4 auc : 0.785170370852281
Average auc score 0.7862245601350024

##### Random Forest (with and without SVD)

python3 lbl_rf_cat_dataset.py              
fold : 0 auc : 0.7156096900710428
fold : 1 auc : 0.7183006015004585
fold : 2 auc : 0.7163518427889818
fold : 3 auc : 0.7145920610988761
fold : 4 auc : 0.7155328379383771
Average auc score 0.7160774066795472

Note that: folds are take much longer than as compared to the LRE
Also, inference time of random forest is much longer and it takes much larger space

lets try to reduce total no. of features by using singular value decomposition
fold : 0 auc : 0.7162020657665855
fold : 1 auc : 0.7185562347955913
fold : 2 auc : 0.7165761696871734
fold : 3 auc : 0.7143878042672419
fold : 4 auc : 0.7156900353564006

##### XGBoost (Famous gradient boosting algorithm)

python3 lbl_xgb_cat_dataset.py 
fold : 0 auc : 0.7617999104505528
fold : 1 auc : 0.7647876245018422
fold : 2 auc : 0.7619810489710925
fold : 3 auc : 0.7597561951971425
fold : 4 auc : 0.760745133852605
Average auc score 0.761813982594647

**Note: All results were (without tuning params)**

#### Lets Take Another Dataset (US Adult dataset)


##### After removing Numerical features   (without tuning params)

python3 -W ignore ohe_logreg_us_adult_dataset.py
fold : 0 auc : 0.8763998619060397
fold : 1 auc : 0.8779107394530967
fold : 2 auc : 0.8811708137054598
fold : 3 auc : 0.880219197619382
fold : 4 auc : 0.879276420518492
Average auc score 0.8789954066404941

python3 lbl_xgb_adult_dataset.py
fold : 0 auc : 0.8745685858803185
fold : 1 auc : 0.8737985704581379
fold : 2 auc : 0.8798176140187342
fold : 3 auc : 0.8767134315203509
fold : 4 auc : 0.8753361333626084
Average auc score 0.87604686704803


##### Include Numerical Features (without tuning params)

python3 lbl_xgb_adult_dataset.py
fold : 0 auc : 0.9235766600486092
fold : 1 auc : 0.9258149598490314
fold : 2 auc : 0.9285493002604421
fold : 3 auc : 0.9255189967066575
fold : 4 auc : 0.92688604509656
Average auc score 0.92606919239226

**Wow, we have got much better results.**
**Note that: Tree-based algos. can handle mix features(categorical + numerical) easily**
**Also, there is no need for data normalization in tree-based algos.**

##### Do Feature Engineering (Adding combination of categorical features)

python3 lbl_xgb_adult_dataset.py
fold : 0 auc : 0.9267517793296272
fold : 1 auc : 0.926579852579287
fold : 2 auc : 0.9281367758963694
fold : 3 auc : 0.9275957263535407
fold : 4 auc : 0.9298427864120555
Average auc score 0.9277813841141759

**Naive way is used for combine features.**
**Key function to combine features: list(itertools.combinations([1,2,3], 2)) returns:
            [(1,2), (1,3), (2,3)]**


##### Encode categorical variables based on target values i.e. target encoding
python3 target_encoding.py
fold : 0 auc : 0.9233755512217967
fold : 1 auc : 0.92724321240442
fold : 2 auc : 0.9271905150260733
fold : 3 auc : 0.9263546119089517
fold : 4 auc : 0.9277974970642766
Average auc score 0.9263922775251038

** In other words, categories are replaced by the mean/median of the target value
** There is not much improvement


#### Entity Embeddings
