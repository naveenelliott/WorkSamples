import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from tabulate import tabulate
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties
from highlight_text import fig_text, ax_text

# Example data
data = [
    ["Thiago Almada", "Team1", .2635],
    ["Carles Gil", "Team2", .5446],
    ["Riqui Puig", "Team3", .5708],
    ["Luciano Acosta", 'Team4', .7576],
    ["Mauricio Pereyra", "Team5", .7805]
]
df = pd.DataFrame(
    data, columns=['Player',  'Team', 'Distance Away'])
df = df.sort_values(by='Distance Away', ascending=False)
fig = plt.figure(figsize=(8, 8), dpi=300)
ax = plt.subplot()

ncols = 3
nrows = 5

ax.set_xlim(0, ncols + 1)
ax.set_ylim(0, nrows + 0.5)

positions = [0.25, 1.9, 3.35]
columns = ['Player', 'Team', 'Distance Away']  # Swapped columns

# Define image paths for each team
image_paths = {
    "Team2": "TeamLogos/Revs.png",
    "Team1": "TeamLogos/Atlanta United.png",
    "Team3": "TeamLogos/LA Galaxy.png",
    "Team4": "TeamLogos/Cincy.png",
    "Team5": "TeamLogos/OrlandoCity.png"
}

# Battery symbol parameters
battery_width = 0.3
battery_height = 0.3

# Add table's main text and bar chart
for i in range(nrows):
    for j, column in enumerate(columns):
        if j == 0:
            ha = 'left'
        else:
            ha = 'center'
        if column == 'Player':
            text_label = df[column].iloc[i]
            weight = 'bold'
            ax.annotate(
                xy=(positions[j]-.15, i + .5),
                text=str(text_label),
                ha=ha,
                va='center',
                weight=weight
            )
        elif column == 'Team':
            team_name = df[column].iloc[i]
            image_path = image_paths.get(team_name, "")
            if image_path:
                img = mpimg.imread(image_path)
                ax.imshow(img, extent=[positions[j] - 0.19,
                          positions[j] + 0.19, i + 0.3, i + 0.675])
            text_label = ""  # Empty string since image is displayed
            weight = 'normal'
        else:
            text_label = df[column].iloc[i]
            ax.annotate(
                xy=(positions[j], i + .49),
                text=str(text_label),
                ha=ha,
                va='center',
                fontsize=10,
                weight=weight
            )

column_names = ['Player', 'Team', 'Distance Away']
for index, c in enumerate(column_names):
    if index == 0:
        ha = 'left'
    else:
        ha = 'center'
    ax.annotate(
        xy=(positions[index], nrows + .15),
        text=column_names[index],
        ha=ha,
        va='bottom',
        weight='bold', size='large'
    )

# Add dividing lines
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [nrows, nrows],
        lw=1.5, color='black', marker='', zorder=4)
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0],
        lw=1.5, color='black', marker='', zorder=4)
for x in range(1, nrows):
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x],
            lw=1.15, color='gray', ls=':', zorder=3, marker='')

fig.set_facecolor('#f3ece6')
ax.set_axis_off()


font_path = 'Belanosima-SemiBold.ttf'
belanosima = FontProperties(fname=font_path)

fig_text(
    0.51, .9, "Similar MLS Players to Lionel Messi", size=23,
    ha="center", va='center', fontproperties=belanosima, color="#000000"
)

plt.show()


