import sqlite3
import prince
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlite3 import OperationalError
from sklearn.decomposition import PCA

connection = sqlite3.connect("data.db")
cur = connection.cursor()
# ["movie_id", "title", "budget", "homepage", "overview", "popularity", "release_date", "revenue", "runtime", "movie_status", "tagline", "vote_average", "vote_count"]
cur.execute("SELECT budget, popularity, runtime, revenue, vote_average, vote_count, genre_name FROM movies JOIN movie_genres ON movie_genres.movie_id = movies.movie_id JOIN genre ON genre.genre_id = movie_genres.genre_id WHERE vote_count>=100 AND budget>999 AND runtime>0 AND revenue>999;")
fetch = cur.fetchall()
#cur.execute("SELECT genre_name FROM movies JOIN movie_genres ON movie_genres.movie_id = movies.movie_id JOIN genre ON genre.genre_id = movie_genres.genre_id WHERE vote_count>=100 AND budget>999 AND runtime>0 AND revenue>999;")
#fetch2 = cur.fetchall()
connection.close()
# cur.execute("SELECT title FROM movies WHERE vote_count>=100 AND budget>999 AND runtime>0 AND revenue>999;")
# fetchtitle = pd.DataFrame(cur.fetchall())
# print(fetch)

features = ["budget", "popularity", "runtime", "revenue", "vote_average", "vote_count", "genre"]
df = pd.DataFrame(data=fetch, columns=features)
print(df.head())
donnees = df
# Separating out the features
x = df.loc[:, features[:-1]].values
# Separating out the target
y = df.loc[:,['genre']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['genre']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Adventure', 'Fantasy', 'Animation', 'Drama', 'Horror', 'Action', 'Comedy', 'History', 'Western', 'Thriller', 'Crime', 'Documentary', 'Science Fiction', 'Mystery', 'Music', 'Romance', 'Family', 'War', 'Foreign', 'TV Movie']
NUM_COLORS = len(targets)
cm = plt.get_cmap('gist_rainbow')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['genre'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
"""
# y =pd.Series(donnees).map({0})
print(donnees.head(3000))
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
  rescale_with_mean=True,  # Centrer
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
#\"""
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
# \"""
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

"""