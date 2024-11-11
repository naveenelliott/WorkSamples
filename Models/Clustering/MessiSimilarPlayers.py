import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from highlight_text import fig_text, ax_text
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import patheffects

# Finding clusters for similar players to Messi
just_values = pd.read_csv("Messi KMeans.csv")
pukki = pd.read_csv("Messi.csv")

names = just_values.pop('Player')
pukki_name = pukki.pop('Player')
names = names.append(pukki_name)
names.reset_index(drop=True, inplace=True)
just_values = pd.concat([just_values, pukki])

# Define the font properties
font_props = FontProperties(family='Roboto', weight='light')
font_path = 'CookbookNormalRegular-6YmjD.ttf'
cook = FontProperties(fname=font_path)

x = just_values.values
# scaling and transforming the data
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
X_norm = pd.DataFrame(x_scaled)

# Transforming the statistics and metrics to two dimensions with PCA
pca = PCA(n_components=2)
reduced = pd.DataFrame(pca.fit_transform(X_norm))

# specify the number of clusters
kmeans = KMeans(n_clusters=4)
# fit the input data
kmeans = kmeans.fit(reduced)
# get the cluster labels
labels = kmeans.predict(reduced)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()

reduced['cluster'] = clusters
reduced['name'] = names
reduced.columns = ['x', 'y', 'cluster', 'name']
reduced.head()

# Creating a plot to visualize results
legend_labels = ['Creative 10\'s', 'Target Men', 'Creative Wide Players', 'Creative Strikers']
legend_markers = ['o', 'o', 'o', 'o']
custom = ['#F7B5CD', '#3f383a', '#7f7f7f', 'white']

# Create the legend
legend_elements = [
    plt.Line2D([0], [0], marker=legend_markers[0], color=custom[0], label=legend_labels[0], markersize=10),
    plt.Line2D([0], [0], marker=legend_markers[1], color=custom[1], label=legend_labels[1], markersize=10),
    plt.Line2D([0], [0], marker=legend_markers[2], color=custom[2], label=legend_labels[2], markersize=10),
    plt.Line2D([0], [0], marker=legend_markers[3], color=custom[3], label=legend_labels[3], markersize=10)
]

sns.set_style("whitegrid", {'axes.facecolor': '#f3ece6', 'grid.color': '#f3ece6'})
ax = sns.lmplot(x="x", y="y", hue='cluster', data=reduced, legend=False,
                fit_reg=False, size=15, scatter_kws={"s": 250}, palette=custom)

texts = []

ax.set(ylim=(-1.5, 2))


# Customize the legend
plt.legend(handles=legend_elements, loc='upper right', fontsize=25)
plt.tick_params(labelsize=15)

fig = plt.gcf()
fig.patch.set_facecolor('#f3ece6')

font_path = 'Caprasimo-Regular.ttf'
caprasimo = FontProperties(fname=font_path)

# add title
fig_text(
    0.55, 1.08, "Similar Players to <Lionel Messi> in MLS", size=38, transform=ax.fig.transFigure,
    ha="center", va='center', fontproperties=caprasimo, color="#000000", highlight_textprops = [{'weight':'bold', 'color':'#F7B5CD'}]
)

plt.text(0.23, 1.03, 'A look at similar players in MLS to new Inter Miami signing\nLionel Messi, based on statistics from @WyScout',
         font='Karla', va='center', fontsize=22, transform=ax.fig.transFigure)

ax = plt.gca()
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Plot the five closest players to T. Pukki
import numpy as np

# Calculate the Euclidean distance between each point and T. Pukki
reduced['distance'] = np.sqrt((reduced['x'] - reduced.loc[reduced['name'] == 'L. Messi', 'x'].values[0]) ** 2 +
                              (reduced['y'] - reduced.loc[reduced['name'] == 'L. Messi', 'y'].values[0]) ** 2)

# Find the indices of the five closest points to T. Pukki
closest_players = reduced.iloc[reduced['distance'].argsort()[:6]]

# Plot the scatterplot
#sns.scatterplot(x="x", y="y", data=closest_players, color='red', marker='*', s=250)

outline_effect = [patheffects.withStroke(linewidth=1.5, foreground='white')]
# Add player names as text annotations
for index, player in closest_players.iterrows():
    if player['name'] == 'L. Messi':
        image_path = 'MessiGoat.png'
        image = plt.imread(image_path)
        imagebox = OffsetImage(image, zoom=0.02)
        ab = AnnotationBbox(imagebox, (player['x'], player['y']-.02), frameon=False)
        ax.add_artist(ab)
        texts.append(plt.text(player['x'], player['y']+.06, player['name'], ha='center', va='bottom', fontproperties=cook, 
                              fontsize=30, path_effects=outline_effect))
    else:
        texts.append(plt.text(player['x'], player['y']+.02, player['name'], ha='center', va='bottom', fontproperties=cook, 
                              fontsize=30, path_effects=outline_effect))  

other_texts = []
for cluster in clusters:
    if cluster == 0:  # Skip printing contents of Cluster 1
        continue
    players_in_cluster = reduced[reduced['cluster'] == cluster].head(2)
    for index, player in players_in_cluster.iterrows():
        other_texts.append(plt.text(player['x'], player['y'], player['name'], ha='center', va='bottom', fontproperties=cook, 
                                    fontsize=27.5, path_effects=outline_effect))

ax.annotate(
    xy=(.75, 1.15),
    xytext=(-250,150),
    text='Players comparable to Lionel Messi',
    ha='center', 
    va='center',
    font='Karla', fontsize=22,
    textcoords='offset points',
    arrowprops=dict(
            arrowstyle="->, head_length=0.8, head_width=0.8", shrinkA=10, shrinkB=0, color="black", linewidth=3,
            connectionstyle="arc3,rad=0.05"
    )
)


image_path = 'TeamLogos/Inter Miami.png'  # Replace with the actual path to your image
image = plt.imread(image_path)
imagebox = OffsetImage(image, zoom=0.1)
ab = AnnotationBbox(imagebox, (0.05, 1.07), xycoords='axes fraction', frameon=False)
ax.add_artist(ab)

plt.show()





