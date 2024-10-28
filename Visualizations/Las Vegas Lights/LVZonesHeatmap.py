import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import norm
import seaborn as sns
from PIL import Image
import matplotlib.colors as colors
from highlight_text import fig_text
from matplotlib.font_manager import FontProperties
import os
import glob

color_list = ['#00BDEF', '#FFEF3C', '#EE378B']
cmap = colors.LinearSegmentedColormap.from_list('CustomColormap', color_list)
font_path = 'CookbookNormalRegular-6YmjD.ttf'
cook = FontProperties(fname=font_path)
font_path = 'RussoOne-Regular.ttf'
title = FontProperties(fname=font_path)

folder_path = 'LV_Field'

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# List to hold individual idp_reports
df_list = []

lv = pd.DataFrame()

for file in csv_files:
    df = pd.read_csv(file)
    df['Zone #'] = os.path.basename(file).split('zone')[1].split('.csv')[0]
    df['Rank'] = df['Goals Added diff'].rank(ascending=False)
    df_list.append(df)
    
lv = pd.concat(df_list, ignore_index=True)
lv = lv[['Team', 'Zone #', 'Goals Added diff', 'Rank']]

max_teams = lv['Rank'].max()

lv = lv.loc[lv['Team'] == 'LV']

lv['Percentile'] = (lv['Rank']/max_teams)
lv['Percentile'] = 1 - (lv['Rank']/max_teams)

lv['Zone #'] = lv['Zone #'].astype(float)

lv = lv.sort_values('Zone #', ascending=False).reset_index(drop=True)

counts = lv['Percentile']
num_rows, num_columns = 6, 5
# Ensure the number of rows and columns match the size of the counts data
assert num_rows * num_columns == len(counts), "The number of rows and columns do not match the size of the counts data."

# Reshape the counts data into a matrix
counts_matrix = np.array(counts).reshape(num_rows, num_columns)
soccer_field_image = Image.open("SoccerPitchTransparent.png").convert("L")
# Create the heatmap using seaborn
fig, ax = plt.subplots(figsize=(4,4), dpi=600)
heatmap = sns.heatmap(counts_matrix, annot=True, fmt=".2f", cmap=cmap, cbar=True, alpha = 0.9, xticklabels=False, yticklabels=False, 
            annot_kws={'fontproperties':cook})
plt.imshow(soccer_field_image, extent=[-0.13, 5.13, -0.25, 6.25], cmap='gray')

ax.set_ylabel('Attacking ⟶', fontsize=11)
fig_text(
    x = 0.13, y = .93, 
    s = "Where Does Las Vegas Rank Compared\nto Other Teams in the USLC?",
    va = "bottom", ha = "left",
    color = "black", fontproperties=title, fontsize = 11
)

cbar = heatmap.collections[0].colorbar
cbar.set_label('Goals Added Percentile', fontsize=8)

plt.show()