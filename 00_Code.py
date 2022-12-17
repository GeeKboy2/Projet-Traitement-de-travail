import sqlite3
import prince
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlite3 import OperationalError
from sklearn.decomposition import PCA
import sys

np.set_printoptions(threshold=sys.maxsize)

connection = sqlite3.connect("data.db")
cur = connection.cursor()
# cur.execute("SELECT budget, popularity, runtime, revenue, vote_average, vote_count, genre_name FROM movies JOIN movie_genres ON movie_genres.movie_id = movies.movie_id JOIN genre ON genre.genre_id = movie_genres.genre_id WHERE vote_count>=100 AND budget>999 AND runtime>0 AND revenue>999;")
cur.execute("SELECT budget, popularity, runtime, revenue, vote_average, vote_count, genre_name FROM movies JOIN movie_genres ON movie_genres.movie_id = movies.movie_id JOIN genre ON genre.genre_id = movie_genres.genre_id WHERE vote_count>=50 AND budget>999 AND runtime>0 AND revenue>999;")
fetch = cur.fetchall()
connection.close()
# print(fetch)

features = ["budget", "popularity", "runtime", "revenue", "vote_average", "vote_count", "genre"]
df = pd.DataFrame(data=fetch, columns=features)
print(df.head())
donnees = df

x = df.loc[:, features[:-1]].values  # Separating out the features
y = df.loc[:,['genre']].values  # Separating out the target
x = StandardScaler().fit_transform(x)  # Standardiser
nb_composantes = 3  # 3D
pca = PCA(n_components=nb_composantes)  #
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,
                           columns=['principal component 1', 'principal component 2', 'principal component 3'][:nb_composantes])

# print(np.corrcoef(x))
finalDf = pd.concat([principalDf, df[['genre']]], axis=1)

targets = ['Adventure', 'Fantasy', 'Animation', 'Drama', 'Horror', 'Action', 'Comedy', 'History', 'Western', 'Thriller', 'Crime', 'Documentary', 'Science Fiction', 'Mystery', 'Music', 'Romance', 'Family', 'War', 'Foreign', 'TV Movie']
NUM_COLORS = len(targets)
cm = plt.get_cmap('gist_rainbow')

for i in range(len(targets)):
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    # ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize=15)

    ax.set_title('{} component PCA {}'.format(nb_composantes, targets[i]), fontsize = 20)
    colors = ['k' for _ in range(NUM_COLORS)]
    colors[i] = cm(1.*i/NUM_COLORS)
    # colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
    k=0
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['genre'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   finalDf.loc[indicesToKeep, 'principal component 3'],
                   c = color,
                   s = 50,
                   alpha = .05+.95*int(k==i))
        k+=1
    ax.legend(targets)
    ax.grid()
    plt.axis('equal')
print(pca.explained_variance_)  # Valeurs propres
print(pca.explained_variance_ratio_)  # Inerties
# Plot a variable factor map for the first two dimensions. (Cercle des corrélations)
(fig, ax) = plt.subplots(figsize=(8, 8))
for i in range(0, len(features)-1):  # len(pca.components_)):
    ax.arrow(0, 0,  # Start the arrow at the origin
             pca.components_[0, i], pca.components_[1, i],  # 0 and 1 correspond to dimension 1 and 2
             head_width=0.1, head_length=0.1)
    plt.text(pca.components_[0, i] + 0.05, pca.components_[1, i] + 0.05, df.columns.values[i])

an = np.linspace(0, 2 * np.pi, 100)  # Add a unit circle for scale
plt.plot(np.cos(an), np.sin(an))
plt.axis('equal')
ax.set_title('Variable factor map (Cercle des corrélations)')
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