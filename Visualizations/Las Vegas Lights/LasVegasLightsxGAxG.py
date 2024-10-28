import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.patches import Circle
from highlight_text import fig_text, ax_text
import matplotlib.patheffects as path_effects

font_path = 'Cushy.ttf'
comic = FontProperties(fname=font_path)
font_path = 'Belanosima-SemiBold.ttf'
belanosima = FontProperties(fname=font_path)

leagues_cup = pd.read_csv('USLC_xG_xGA.csv')

leagues_cup['sort_order'] = leagues_cup['Team'].apply(lambda x: 1 if x == 'LV' else 0)

# Sort by sort_order and then by another column (e.g., 'Pts') for other teams
leagues_cup = leagues_cup.sort_values(by=['sort_order', 'Pts'], ascending=[True, False]).drop(columns=['sort_order'])

fig = plt.figure(figsize=(8,8), dpi=600)
ax = plt.subplot()
ax.grid(visible=False, ls='--', color='lightgray')

scatter = ax.scatter(leagues_cup['xGF'], leagues_cup['xGA'])


mean_xG = leagues_cup['xGF'].mean()
mean_xGA = leagues_cup['xGA'].mean()
# Plot average lines
plt.axhline(mean_xG, color='black', linestyle='--', label='Average xG')
plt.axvline(mean_xGA, color='black', linestyle='--', label='Average xGA')

for index, row in leagues_cup.iterrows():
    image_path = f'USL Logos/{row["Team"]}.png'
    image = plt.imread(image_path)
    imagebox = OffsetImage(image, zoom=0.022)
    ab = AnnotationBbox(imagebox, (row['xGF'], row['xGA']), frameon=False)
    ax.add_artist(ab)
    
    
lv_data = leagues_cup[leagues_cup['Team'] == 'LV']
lv_x, lv_y = lv_data['xGF'].values[0], lv_data['xGA'].values[0]    
# Pink circle around Las Vegas point
circle = Circle((lv_x, lv_y), 0.05, color='pink', fill=False, lw=4)
ax.add_artist(circle)

# Arrow pointing to Las Vegas
annotation = ax.annotate('Las Vegas', xy=(lv_x+0.025, lv_y+0.025), xytext=(lv_x + .2, lv_y + 0.2),
            arrowprops=dict(facecolor='pink', shrink=0.05),
            fontsize=12, color='pink', fontproperties='Arial')

annotation.set_path_effects([
    path_effects.Stroke(linewidth=.25, foreground='black'),
    path_effects.Normal()
])
        
        
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Expected Goals Against (xGA)', fontsize=15)
ax.set_xlabel('Expected Goals (xG)', fontsize=15) 


fig_text(x=0.18, y=.775, s='Low Scoring, Poor Defense', va='bottom', ha='left', color='black', fontproperties=comic, size=9)
fig_text(x=0.15, y=.2, s='Low Scoring, Good Defense', va='bottom', ha='left', color='black', fontproperties=comic, size=9)
fig_text(x=0.54, y=.225, s='High Scoring, Good Defense', va='bottom', ha='left', color='black', fontproperties=comic, size=9)
fig_text(x=0.57, y=.775, s='High Scoring, Poor Defense', va='bottom', ha='left', color='black', fontproperties=comic, size=9)

fig_text(
    x = 0.12, y = .94, 
    s = "xG and xGA for <Las Vegas Lights> and\nUSL Championship Teams",  # Use <> around the text to be styled
    va = "bottom", ha = "left",
    highlight_textprops=[
        {"color": "pink", "fontproperties": belanosima, "weight": "bold"}  # Style for Las Vegas Lights
    ],
    color = "black", fontproperties = belanosima, weight = "bold", size=22
)