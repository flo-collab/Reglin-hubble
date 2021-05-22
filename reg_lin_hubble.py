import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# import seaborn as sns
plt.style.use('seaborn-darkgrid')

# On charge le dataset
hubble_df = pd.read_csv('hubble_data.csv')


# Aperçu du dataset
print(hubble_df.head())
print(hubble_df.describe())
print(hubble_df.shape)

#
X = np.ones((len(hubble_df), 2))
X[:, 1] = hubble_df['distance']
y = hubble_df['recession_velocity']

'''
print("voila X :", X, "\n voila y :", y)
print(X.shape, y.shape)
'''

#  on divise notre jeu de données en 2 parties
# 80%, pour l’apprentissage et les 20% restant pour le test.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


'''
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(type(X_train))
'''

# entrainement du modèle
model_regLin = LinearRegression()
model_regLin.fit(X_train, y_train)

# on regarde les resultats : Les coefficients
a = model_regLin.coef_
a = a[1]
b = model_regLin.intercept_
print('Les coefficients trouves sont: \n', 'a =', a, ' et b = ', b)


# Evaluation du training set
y_train_predict = model_regLin.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

# affichage des resultats de performance
print('La performance du modele sur la base de test')
print('--------------------------------------')
print('Lerreur quadratique moyenne est {}'.format(rmse))
print('le score R2 est {}'.format(r2))


plt.title("Relation entre distance et vitesse radiale des nebuleuses extra-galactiques")
plt.xlabel('Distance')
plt.ylabel('vitesse radiale')
# parametrage de l'affichage du nuage de points :
plt.plot(hubble_df['distance'], hubble_df['recession_velocity'],'ro', color = '#FF9933', markersize=7  )

# parametrage de l'affichage de la droite de regression linéaire de 0 à 2 :
plt.plot([0, 2], [ b, b + 2 * a], linestyle='--', c='#d00000' , label="y = {} * x + {}".format(a, b))

plt.legend(loc='lower right')

plt.show()
