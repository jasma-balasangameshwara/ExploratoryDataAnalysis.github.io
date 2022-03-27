# Exploratory Data Analysis - Basic Steps

Step 1 - Get the number of data points and features

Step 2 - Delete columns from the dataset which are fully empty

Step 3 - Delete duplicate rows from the dataset

Step 4 - Delete columns that have more than 50% of missing values. The recommended missing values is 20 - 30%

Step 5 - Delete columns that are of type 'ID'

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
         
Step 9 - Drop the label from the dataset

         X = df.drop('label', axis = 1)
         y = df['label']
         
Step 10 - Get description of the dataset

Step 11 - Plot categorical columns

Step 12 - Analyze outliers

Step 13 - Check imbalance class and correct it
    
    fig= px.histogram(df, x='label',color='label', barmode='group')
    fig.show()
    
    Step 13.1 - Class weights in the models
    
    Most of the machine learning models provide a parameter called class_weights. For example, in a random forest classifier using, class_weights we can specify higher     weight for the minority class using a dictionary.
    
    from sklearn.linear_model import LogisticRegressionclf
    LogisticRegression(class_weight={0:1,1:10})
    
    There is no one method to do this, and this should be constructed as a hyperparameter search problem for your particular problem. 
    
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', np.unique(y), y)
    
                        (or)
    Step 13.2 - Treat the problem as anomaly detection
    
    Anomaly detection is the identification of rare items, events or observations which raise suspicions by differing significantly from the majority of the data. 
    

