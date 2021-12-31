#Importation des bibliotheques
import pandas as pd
import numpy as np
import pickle
import xlrd



#Importer notre dataset
alldata=pd.read_excel('default of credit card clients.xls',na_values='?')

# Changer les noms du chaque colonne
alldata.columns=alldata.iloc[0]
newdata=alldata.drop(0)


#Transformer le type de chaque colonne de object vers numerique
newdata=newdata.apply(pd.to_numeric)

#Indexer la colonne ID et supprimer la numérotation
newdata.set_index(newdata.columns[0],inplace=True)

# Renomer la colonne PAY_0
newdata.rename(columns={'PAY_0':'PAY_1'}, inplace=True)

# Suppression des observations dupliquées
newdata.drop_duplicates(inplace = True )

#  Standardisation des colonnes numériques
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Features_Numerique = ["LIMIT_BAL" ,"AGE" ,  "BILL_AMT1" , "BILL_AMT2" , "BILL_AMT3" , "BILL_AMT4" , "BILL_AMT5" ,"BILL_AMT6",
                      "PAY_AMT1" , "PAY_AMT2" , "PAY_AMT3" ,"PAY_AMT4" ,"PAY_AMT5" , "PAY_AMT6"]
transformed_numerical_features  = scaler.fit_transform(newdata[Features_Numerique])

#Extraire les valeurs en utilisant la méthode de Inter Quantile Range IQR
Quantile1 = newdata[Features_Numerique].quantile(0.25)
Quantile3 = newdata[Features_Numerique].quantile(0.75)
IQR = Quantile3 - Quantile1


outliers_dataframe = pd.DataFrame((newdata[Features_Numerique] <  Quantile1 - 1.5 * IQR ) | (newdata[Features_Numerique] > Quantile3  + 1.5 * IQR))
outliers_dataframe_int  = outliers_dataframe.astype(int)

#Créer une fonction qui détecte les lignes qui ont plus que 4 valeurs aberrantes
#et retourne une nouvelle dataframe sans ces lignes et afficher le nombre de lignes effacées
def drop_outliers(data):
    dropped_rows = 0
    for i in data.index:
        outlier_count = 0
        outlier_count = data.loc[i , : ].sum()
        if outlier_count >= 4:
            dropped_rows += 1
            data.drop(i , axis = 0 ,inplace = True)
    return (data,dropped_rows)

# Extraction les indices des lignes restantes après le nettoyage des données

data, dropped_rows = drop_outliers(outliers_dataframe_int)
data_index = np.array(data.index)
df_with_no_outlier_rows = newdata.loc[data_index]


# import relevant libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# features extraction

x = df_with_no_outlier_rows[[ 'PAY_1', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
y = df_with_no_outlier_rows['default payment next month']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)  #splitting data with test size of 25%

logreg = LogisticRegression()   #build our logistic model
logreg.fit(x_train, y_train)  #fitting training data
y_pred  = logreg.predict(x_test)    #testing model’s performance
print("Accuracy={:.2f}".format(logreg.score(x_test, y_test)))
pickle.dump(logreg,open('model.pkl','wb'))









