##########################################
# Fig. 2                                 #
##########################################
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Read the dataset
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Create a new column representing unique (longitude, latitude) combinations to identify different locations
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# Group by location (lon_lat) and calculate the mean for each variable at each location
data_mean = data.groupby('lon_lat').mean().reset_index()

# Calculate the percentage change in coral cover (compared to 2020 in 2100)
data_mean['coral_cover_change'] = ((data_mean['coral_cover_2100'] - data_mean['coral_cover_2020']) / 
                                   data_mean['coral_cover_2020']) * 100

# Calculate changes in SST and pH
data_mean['SST_change'] = data_mean['SST_2100'] - data_mean['SST_2020']
data_mean['pH_change'] = data_mean['pH_2100'] - data_mean['pH_2020']


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

# Remove outliers from 'coral_cover_change', 'SST_2100', 'SST_seasonal', 'pH_2100', 'PAR' 
data_mean_clean = remove_outliers(data_mean, ['coral_cover_change', 'SST_change', 'SST_seasonal', 'pH_change', 'PAR'])

# Plot the cleaned data
plt.figure(figsize=(8, 6), constrained_layout=True)  # 使用 constrained_layout=True 自動調整佈局

# Set main title
plt.suptitle("Relationships between Variables and Coral Cover Change", fontsize=14, fontweight='bold')

##########################################
# Fig. 2                                 #
##########################################
# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

# Read the dataset
data = pd.read_csv('data/coral_forecast.csv', skiprows=[1])

# Create a new column representing unique (longitude, latitude) combinations to identify different locations
data['lon_lat'] = list(zip(data.longitude, data.latitude))

# Group by location (lon_lat) and calculate the mean for each variable at each location
data_mean = data.groupby('lon_lat').mean().reset_index()

# Calculate the percentage change in coral cover (compared to 2020 in 2100)
data_mean['coral_cover_change'] = ((data_mean['coral_cover_2100'] - data_mean['coral_cover_2020']) / 
                                   data_mean['coral_cover_2020']) * 100

# Calculate changes in SST and pH
data_mean['SST_change'] = data_mean['SST_2100'] - data_mean['SST_2020']
data_mean['pH_change'] = data_mean['pH_2100'] - data_mean['pH_2020']

# Define a function to remove outliers
def remove_outliers(df, columns, threshold=1.5):
    for column in columns:
        Q1 = df[column].quantile(0.25)  # First quartile
        Q3 = df[column].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Calculate interquartile range
        # Use IQR to define lower and upper bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # Filter out data within the lower and upper bounds
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

# Remove outliers from 'coral_cover_change', 'SST_2100', 'SST_seasonal', 'pH_2100', 'PAR' columns
data_mean_clean = remove_outliers(data_mean, ['coral_cover_change', 'SST_change', 'SST_seasonal', 'pH_change', 'PAR'])

# Plot the cleaned data
plt.figure(figsize=(8, 6), constrained_layout=True)  # Use constrained_layout=True for automatic layout adjustment

# Set main title
plt.suptitle("Relationships between Variables and Coral Cover Change", fontsize=14, fontweight='bold')

# Define the list of variables to plot
variables = ['SST_change', 'SST_seasonal', 'pH_change', 'PAR']
# Use subplots to plot the relationship between each variable and coral cover change
for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)  # 2x2 subplot layout
    x = data_mean_clean[var]
    y = data_mean_clean['coral_cover_change']
    
    # Plot gray scatter points
    plt.scatter(x, y, color='gray', alpha=0.6, s=0.5, edgecolor='none')
    
    # Add correlation line using linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    plt.plot(x, slope * x + intercept, color='black', linestyle='--', linewidth=1)
    
    # Display R^2 value inside the subplot
    plt.text(0.05, 0.9, f'$R^2 = {r_value**2:.2f}$', transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    # Show correlation coefficient
    plt.xlabel(var)  # Set X-axis label
    plt.ylabel('Coral Cover Change (%)')  # Set Y-axis label


# Save the image to the output folder
output_dir = "output"
output_path = os.path.join(output_dir, "Fig2.png")
plt.savefig(output_path, dpi=180)