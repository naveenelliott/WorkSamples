import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.font_manager import FontProperties
from highlight_text import fig_text
import matplotlib.cm as cm
import matplotlib.colors as mcolors

font_path = 'AccidentalPresidency.ttf'
title = FontProperties(fname=font_path)
font_path = 'Karla-Light.ttf'
karla = FontProperties(fname=font_path)

running = pd.read_csv('Mens 800 - 800.csv')

running['Time'] = pd.to_timedelta('00:' + running['Mark'])

# Convert to string format "minutes:seconds.milliseconds"
running['Time'] = running['Time'].dt.components.apply(
    lambda x: f"{int(x['minutes'])}:{int(x['seconds']):02d}.{int(x['milliseconds']):03d}", axis=1
)

running['Date'] = pd.to_datetime(running['Date'], format='%d %b %Y')

# Extract and create a new column with the month and year
running['Month and Year'] = running['Date'].dt.strftime('%b %Y')
running['Year'] = running['Date'].dt.strftime('%Y')

del running['Mark']

running['Seconds'] = pd.to_timedelta('00:' + running['Time']).dt.total_seconds()

# Filter for performances under 1:43.00 (which is 103 seconds)
filtered_df = running[running['Seconds'] < 103]

# Group by Month_Year and count the number of performances
sub_143 = filtered_df.groupby('Year').size().reset_index(name='Count')

all_years = pd.DataFrame({'Year': pd.Series(running['Year'].unique()).sort_values()})
sub_143 = all_years.merge(sub_143, on='Year', how='left').fillna(0)

# Convert the Count column to integers
sub_143['Count'] = sub_143['Count'].astype(int)


no_duplicates = filtered_df.drop_duplicates(subset=['Competitor', 'Year'])

# Group by Month_Year and count the number of performances
sub_143_unique = no_duplicates.groupby('Year').size().reset_index(name='Count')

all_years = pd.DataFrame({'Year': pd.Series(running['Year'].unique()).sort_values()})
sub_143_unique = all_years.merge(sub_143_unique, on='Year', how='left').fillna(0)

# Convert the Count column to integers
sub_143_unique['Count'] = sub_143_unique['Count'].astype(int)

fig = plt.figure(figsize=(10,7), dpi=600)
ax = plt.subplot(111)

norm = mcolors.Normalize(vmin=sub_143_unique['Count'].min(), vmax=sub_143_unique['Count'].max())
cmap = cm.get_cmap('coolwarm')

colors = cmap(norm(sub_143_unique['Count']))

bars = plt.bar(sub_143['Year'], sub_143['Count'], color=colors)

for i, bar in enumerate(bars):
    yval = bar.get_height()
    unique_yval = sub_143_unique['Count'].iloc[i]
    ax.text(bar.get_x() + bar.get_width() / 2, yval+.25, f"{int(yval)} ({int(unique_yval)})", 
            ha='center', va='bottom', fontsize=13)

# Add title and labels
fig_text(
    x = 0.12, y = 1.0, 
    s = "Number of Under 1:43 Performances in the Men's 800m",
    va = "bottom", ha = "left",
    color = "black", fontproperties = title, size=28
)

fig_text(
    x = 0.135, y = .91, 
    s = "This graphic depicts the count of sub 1:43 performances since the WR Holder\nDavid Rudisha's retirement in 2017. As you can see, there is a massive jump this season.",
    va = "bottom", ha = "left",
    color = "#222222", fontproperties = karla, size=14,
)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for the colorbar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Count of Unique Athletes Under 1:43', fontproperties=karla, fontsize=15)

plt.xlabel('Season', fontsize=15)
plt.ylabel('Count of Sub 1:43 Performances', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

fig.set_facecolor('#f3ece6')
plt.gca().set_facecolor('#f3ece6')

image_path = 'olympics.png'  # Replace with the actual path to your image
image = plt.imread(image_path)
imagebox = OffsetImage(image, zoom=0.0225)
ab = AnnotationBbox(imagebox, (0.79, 0.97), xycoords='figure fraction', frameon=False)
ax.add_artist(ab)

plt.show()