import sqlite3
import prince
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sqlite3 import OperationalError


connection = sqlite3.connect("data.db")
cur = connection.cursor()
# ["movie_id", "title", "budget", "homepage", "overview", "popularity", "release_date", "revenue", "runtime", "movie_status", "tagline", "vote_average", "vote_count"]
cur.execute("SELECT budget, popularity, runtime, revenue, vote_average FROM movies WHERE vote_count>=100 AND budget>999 AND runtime>0 AND revenue>999;")
fetch = cur.fetchall()

# cur.execute("SELECT title FROM movies WHERE vote_count>=100 AND budget>999 AND runtime>0 AND revenue>999;")
# fetchtitle = pd.DataFrame(cur.fetchall())
# print(fetch)
donnees = pd.DataFrame(data=fetch, columns=["budget", "popularity", "runtime", "revenue", "vote_average"])
# y =pd.Series(donnees).map({0})
print(donnees.head(3000))
connection.close()
donnees=np.squeeze(np.asarray(donnees))
X=donnees[:,0]
Y=donnees[:,1]
Z=donnees[:,2]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X, Y, Z, c=Z, cmap='RdYlGn', linewidth=0.5)
plt.title("avant")
pca = prince.PCA(
  n_components=5,
  n_iter=10,
  rescale_with_mean=False,  # Centrer
  rescale_with_std=True,  # Réduire
  copy=False,
  check_input=True,
  engine='auto',
  random_state=234)

pca = pca.fit(donnees)
pca.transform(donnees).head()  # Composantes principales
pca.plot_row_coordinates(donnees)

#print(donnees)
donnees=np.squeeze(np.asarray(donnees))
print(donnees[:,0])
X=donnees[:,0]
Y=donnees[:,1]
Z=donnees[:,2]


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter(X, Y, Z, c=Z, cmap='RdYlGn', linewidth=0.5)
plt.title("apres")
"""
ax = pca.plot_row_coordinates( #Projection sur les composantes principales
  donnees,
  ax=None,
  figsize=(6, 6,6),
  x_component=0,
  y_component=1,
  z_component=2,
  labels=None,
  color_labels={0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'},
  ellipse_outline=False,
  ellipse_fill=True,
  show_points=True
)
ax.legend()
"""
plt.show()

print("\nValeurs propres :")
print(pca.eigenvalues_)  # Valeurs propres
print("\nInertie totale")
print(pca.total_inertia_)
print("\nInerties :")
print(pca.explained_inertia_)  # Inerties
print("\nCorrélations :")
print(pca.column_correlations(donnees))  # Correlations
print("\nContributions :")
print(pca.row_contributions(donnees).head())  # Contributions