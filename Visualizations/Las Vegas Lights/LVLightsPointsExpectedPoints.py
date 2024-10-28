import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from highlight_text import fig_text, ax_text
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties
import seaborn as sns

font_path = 'Belanosima-SemiBold.ttf'
belanosima = FontProperties(fname=font_path)

folder_path = 'LV_Running_Avg'

# Find all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# List to hold individual idp_reports
df_list = []

lv = pd.DataFrame()

for file in csv_files:
    df = pd.read_csv(file)
    
    # Filter for LV rows
    lv_df = df[df['Team'] == 'LV']
    
    # Check if any rows in lv_df already exist in lv_combined
    # Use `merge` to identify any matching rows between lv_combined and lv_df
    if not lv.empty:
        duplicate_check = pd.merge(lv, lv_df, how='inner')
    else:
        duplicate_check = pd.DataFrame()  # No duplicates if lv_combined is empty

    # If there are no duplicates, append the new LV data
    if duplicate_check.empty:
        lv = pd.concat([lv, df], ignore_index=True)

team_rank = pd.read_csv('LV_Table_Rank.csv')

team_rank['Rank'] = team_rank.groupby('Games')['Pts'].rank(ascending=False, method='min')

# Filter Las Vegas rows and extract its rank for each unique week
lv_ranks = team_rank[team_rank['Team'] == 'Las Vegas Lights'][['Games', 'Team', 'Rank']]

lv = lv.loc[lv['Team'] == 'LV']

lv = lv.sort_values('Games', ascending=True).reset_index(drop=True)
#lv = lv.drop_duplicates().reset_index(drop=True)

lv = lv[['Team', 'Games', 'Pts', 'xPts']]

lv['Pts Change'] = lv['Pts'].diff()
lv['xPts Change'] = lv['xPts'].diff()

lv['Diff'] = abs(lv['Pts Change'] - lv['xPts Change'])

# Fill the first change as NaN (since there's no previous game) and replace it with 0 for clarity
lv['Pts Change'] = lv['Pts Change'].fillna(0)
lv['xPts Change'] = lv['xPts Change'].fillna(0)

# Plotting Pts and xPts as a running line chart
plt.figure(figsize=(10, 6), dpi=600)
ax = plt.subplot()

# Plot Pts
plt.plot(lv['Games'], lv['Pts'], label='Pts', color='#00BDEF', marker='o')

# Plot xPts
plt.plot(lv['Games'], lv['xPts'], label='xPts', color='#EE378B', marker='o')


last_game = lv.iloc[-1]['Games']  # The 'Games' value of the last data point
last_pts = lv.iloc[-1]['Pts'] 
last_xpts = lv.iloc[-1]['xPts']
image_path = 'USL Logos/LV.png'
image = plt.imread(image_path)
imagebox = OffsetImage(image, zoom=0.015)
ab = AnnotationBbox(imagebox, (last_game, last_pts), frameon=False)
ax.add_artist(ab)
ab = AnnotationBbox(imagebox, (last_game, last_xpts), frameon=False)
ax.add_artist(ab)

# Adding labels and title
plt.xlabel('Games', fontsize=15)
plt.ylabel('Points', fontsize=15)
plt.title('Running Line Chart of Pts and xPts Over Games')
plt.legend()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig_text(
    x = 0.12, y = .94, 
    s = "Las Vegas Lights Points and Expected Points 2024 Season",  # Use <> around the text to be styled
    va = "bottom", ha = "left",
    color = "black", fontproperties = belanosima, weight = "bold", size=22
)

game_5_x = 5  # Game 5
game_5_y = lv.loc[lv['Games'] == game_5_x, 'Pts'].values[0]
plt.annotate('v San Antonio\n(W 1-0)', 
             xy=(game_5_x, game_5_y), 
             xytext=(game_5_x-2, game_5_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

# red card for LV Lights
game_x = 29  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'Pts'].values[0]
plt.annotate('v Sac Republic\n(W 2-1)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

game_x = 19  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'Pts'].values[0]
plt.annotate('v Sac Republic\n(W 1-0)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

game_x = 16  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'xPts'].values[0]
plt.annotate('v Colorado\n(D 3-3)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

# red card for San Antonio
game_x = 10  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'xPts'].values[0]
plt.annotate('v San Antonio\n(D 1-1)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

plt.show()

# Plotting Pts and xPts as a running line chart
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=600)

# Plot Pts and xPts on ax1 (primary axis)
line1, = ax1.plot(lv['Games'], lv['Pts'], label='Pts', color='#00BDEF', marker='o')
line2, = ax1.plot(lv['Games'], lv['xPts'], label='xPts', color='#EE378B', marker='o')

# Adding labels and title
ax1.set_xlabel('Games', fontsize=15)
ax1.set_ylabel('Points', fontsize=15)

# Hide right and top spines for cleaner look
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

fig_text(
    x = 0.12, y = .94, 
    s = "Las Vegas Lights Points, Expected Points, and Table Rank: 2024 Season", 
    va = "bottom", ha = "left",
    color = "black", fontproperties = belanosima, weight = "bold", size=22
)

# Add game annotations
game_5_x = 5
game_5_y = lv.loc[lv['Games'] == game_5_x, 'Pts'].values[0]
ax1.annotate('v San Antonio\n(W 1-0)', 
             xy=(game_5_x, game_5_y), 
             xytext=(game_5_x-2, game_5_y + 5),  
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

# red card for LV Lights
game_x = 29  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'Pts'].values[0]
plt.annotate('v Sac Republic\n(W 2-1)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

game_x = 19  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'Pts'].values[0]
plt.annotate('v Sac Republic\n(W 1-0)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

game_x = 16  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'xPts'].values[0]
plt.annotate('v Colorado\n(D 3-3)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

# red card for San Antonio
game_x = 10  # Game 5
game_y = lv.loc[lv['Games'] == game_x, 'xPts'].values[0]
plt.annotate('v San Antonio\n(D 1-1)', 
             xy=(game_x, game_y), 
             xytext=(game_x-2, game_y + 5),  # Adjust text position for better visibility
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, color='black')

# More annotations for other games...

# Create secondary axis for the histogram (team rank)
ax2 = ax1.twinx()

lv_ranks = lv_ranks.sort_values('Games', ascending=True).reset_index(drop=True)

# Plot a histogram of the rank of the team
line3, = ax2.plot(lv_ranks['Games'], lv_ranks['Rank'], color='#FFEF3C', marker='o')

# Label for the histogram axis (Rank)
ax2.set_ylabel('Team Rank', fontsize=15)
ax2.set_ylim(0, 24)  # Set limits for rank values, if needed

legend_handles = [
    plt.Line2D([0], [0], color='#00BDEF', marker='o', label='Pts'),
    plt.Line2D([0], [0], color='#EE378B', marker='o', label='xPts'),
    plt.Line2D([0], [0], color='#FFEF3C', marker='o', label='Rank')
]

# Add custom legend
ax1.legend(handles=legend_handles, loc='lower right')

plt.title('Running Line Chart of Pts, xPts, and Table Rank Over Games')

# Show the plot
plt.show()