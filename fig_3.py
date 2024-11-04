##########################################
# Fig. 3                                 #
##########################################
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the dataset
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Calculate the percentage change in coral cover (compared to 2020 in 2100)
data['coral_cover_change'] = (data['coral_cover_2100'] - data['coral_cover_2020']) / data['coral_cover_2020'] * 100

# Filter out unreasonable change rates (assuming reasonable range is -100% to 100%)
data = data[(data['coral_cover_change'] >= -100) & (data['coral_cover_change'] <= 100)]

# Define a function to remove outliers
def remove_outliers(df, columns, threshold=1.5):
    for column in columns:
        Q1 = df[column].quantile(0.25)  # First quartile
        Q3 = df[column].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Calculate interquartile range
        # Use IQR to define lower and upper bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        #  Filter out data within the lower and upper bounds
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Remove outliers from 'coral_cover_change' 
data = remove_outliers(data, ['coral_cover_change'])

# Group latitude into bins of 1 intervals
data['latitude_bin'] = (data['latitude'] // 1) * 1

# Calculate the average coral cover change rate for each 1-degree latitude interval and each model
latitude_change = data.groupby(['latitude_bin', 'model'])['coral_cover_change'].mean().reset_index()

# Calculate the average for all models
mean_change = latitude_change.groupby('latitude_bin')['coral_cover_change'].mean().reset_index()

# Set figure size
plt.figure(figsize=(8,6), constrained_layout=True)
# Use color mapping to make each model more distinguishable
colors = plt.get_cmap('tab20', 12)  # use 12 colors

#Plot line graphs for different models, using different colors for each model
for model_num in sorted(latitude_change['model'].unique()):
    model_data = latitude_change[latitude_change['model'] == model_num]
    plt.plot(model_data['latitude_bin'], model_data['coral_cover_change'], 
             label=f'Model {model_num}', linewidth= 2, color=colors(model_num))

# Plot average change rate line with a thicker black line
plt.plot(mean_change['latitude_bin'], mean_change['coral_cover_change'], 
         label='Mean', linewidth= 3, color='black')

# Add title and axis labels
plt.title('Average Coral Cover Change Rate by Latitude',fontweight='bold', fontsize=16)
plt.xlabel('Latitude', fontsize=14)
plt.ylabel('Average Coral Cover Change Rate (%)', fontsize=14)

# Display the legend outside the plot, sorted from Model 0 to Model 11
plt.legend(bbox_to_anchor=(0.88, 1), loc='upper right', frameon=False)

output_dir = "output"
# Set output path
output_path = os.path.join(output_dir, "Fig3.svg")

# output the plot as svg
plt.savefig(output_path, dpi=600, format='svg', bbox_inches='tight')
