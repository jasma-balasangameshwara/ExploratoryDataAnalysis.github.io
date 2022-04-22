# Exploratory Data Analysis - Basic Steps

Step 1 - Get the number of data points and features

Step 2 - Delete columns from the dataset which are fully empty

Step 3 - Delete duplicate rows from the dataset

Step 4 - Delete columns that have only single-value & columns with blanks and single-values.

Step 5 - Delete static columns

Step 6 - Check the data types of the columns and correct if there is datatype mismatch

Step 7 - Impute missing values

         Step 7.1 - Plot of missing values
         
         import missingno as msno
         msno.bar(df)

Step 8 - Plot a correlation heatmap

         plt.figure(figsize = (20, 12))
         corr = df.corr()
         mask = np.triu(np.ones_like(corr, dtype = bool))
         sns.heatmap(corr, mask = mask, linewidths = 1, annot = True, fmt = ".2f")
         plt.show()
         
         Step 8.1 - Choose to retain or delete highly correlated columns. In the below code snippet, all columns with correlation < 0.85 are dropped. 
         
         corr_matrix = df.corr().abs() 
         mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
         tri_df = corr_matrix.mask(mask)
         to_drop = [x for x in tri_df.columns if any(tri_df[x] < 0.85)]
         df = df.drop(to_drop, axis = 1)
         
Step 9 - Drop target label from the dataset

         X = df.drop('label', axis = 1)
         y = df['label']
         
Step 10 - Get description of the dataset

Step 11 - Plot categorical columns

Step 12 - Analyze outliers

Step 13 - Extracting only useful and relevant features

         #apply SelectKBest to extract top 5 best features from a df
         bestfeatures = SelectKBest(score_func=chi2, k=5)
         fit = bestfeatures.fit(X_train,y_train)
         dfscores = pd.DataFrame(fit.scores_)
         dfcolumns = pd.DataFrame(X_train.columns)
         #concat two dataframes for better visualization 
         featureScores = pd.concat([dfcolumns,dfscores],axis=1)
         featureScores.columns = ['Features','Score']  #naming the dataframe columns
         print(featureScores.nlargest(5,'Score'))  #print 5best features
         
         Step 13.1 - 
         In SelectKclass you need to specify that you want top k features. But sometimes, you do not know how many features you need. So you simply use Boruta plot.            Also it will remove the highly correlated features.
         
         from sklearn.ensemble import RandomForestClassifier
         from boruta import BorutaPy
         # NOTE BorutaPy accepts numpy arrays only, if X_train and y_train #are pandas dataframes, then add .values attribute X_train.values in #that case
         X_train = X_train.values
         y_train = y_train.values
         # define random forest classifier, with utilising all cores and
         # sampling in proportion to y labels
         rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
         # define Boruta feature selection method
         feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
         # find all relevant features - 5 features should be selected
         feat_selector.fit(X_train, y_train)
         # check selected features
         feat_selector.support_
         # check ranking of features
         feat_selector.ranking_
         # call transform() on X to filter it down to selected features
         X_filtered = feat_selector.transform(X_train)
         #To get the new X_train now with selected features
         X_train.columns[feat_selector.support_]

Step 14 - Check imbalance class and correct it
    
    fig= px.histogram(df, x='label',color='label', barmode='group')
    fig.show()
    
    Step 14.1 - Class weights in the models
    
    Most of the machine learning models provide a parameter called class_weights. For example, in a random forest classifier using, class_weights we can specify higher     weight for the minority class using a dictionary.
    
    from sklearn.linear_model import LogisticRegressionclf
    LogisticRegression(class_weight={0:1,1:10})
    
Step 15 - Search for features with different names but similar data.

Step 16 - Use np.where for feature engineering. 
    There is no one method to do this, and this should be constructed as a hyperparameter search problem for your particular problem. 
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    
                        (or)
    Step 14.2 - Treat the problem as anomaly detection
    
    Anomaly detection is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. 
    

