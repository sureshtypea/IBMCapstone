# Import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:/Users/sures/IBM Workspace/EDA_Visual/dataset_part_2.csv"  # Update with the correct file path
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"File not found at {file_path}. Please check the file path and try again.")
    exit()

# TASK 1: Flight Number vs. Launch Site
sns.catplot(x="FlightNumber", y="LaunchSite", hue="Class", data=df, aspect=2, height=6)
plt.title("Flight Number vs. Launch Site", fontsize=16, pad=30)
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Launch Site", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().subplots_adjust(top=0.85)
plt.show()

# TASK 2: Payload Mass vs. Launch Site
sns.catplot(x="PayloadMass", y="LaunchSite", hue="Class", data=df, aspect=2, height=6)
plt.title("Payload Mass vs. Launch Site", fontsize=16, pad=30)
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Launch Site", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().subplots_adjust(top=0.85)
plt.show()

# TASK 3: Success Rate by Orbit Type
success_rate = df.groupby('Orbit')['Class'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='Orbit', y='Class', data=success_rate)
plt.title("Success Rate by Orbit Type", fontsize=16, pad=30)
plt.xlabel("Orbit Type", fontsize=14)
plt.ylabel("Success Rate", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().subplots_adjust(top=0.85)
plt.show()

# TASK 4: Flight Number vs. Orbit Type
sns.catplot(x="FlightNumber", y="Orbit", hue="Class", data=df, aspect=2, height=6)
plt.title("Flight Number vs. Orbit Type", fontsize=16, pad=30)
plt.xlabel("Flight Number", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().subplots_adjust(top=0.85)
plt.show()

# TASK 5: Payload Mass vs. Orbit Type
sns.catplot(x="PayloadMass", y="Orbit", hue="Class", data=df, aspect=2, height=6)
plt.title("Payload Mass vs. Orbit Type", fontsize=16, pad=30)
plt.xlabel("Payload Mass (kg)", fontsize=14)
plt.ylabel("Orbit Type", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().subplots_adjust(top=0.85)
plt.show()

# TASK 6: Launch Success Yearly Trend
def Extract_year():
    year = []
    for i in df["Date"]:
        year.append(i.split("-")[0])
    return year

# Extract the year and add it as a new column
df['Year'] = Extract_year()

# Calculate the average success rate per year
yearly_success = df.groupby('Year')['Class'].mean().reset_index()

# Plot a line chart for the yearly success trend
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Class', data=yearly_success, marker='o')
plt.title("Launch Success Yearly Trend", fontsize=16, pad=30)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Average Success Rate", fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.gcf().subplots_adjust(top=0.85)
plt.show()

# TASK 7: Create dummy variables for categorical columns
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 
               'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

# Create dummy variables for the categorical columns
features_one_hot = pd.get_dummies(features, columns=['Orbit', 'LaunchSite', 'LandingPad', 'Serial'])

# Display the resulting dataframe with encoded features
print("Encoded Features DataFrame:")
print(features_one_hot.head())

# TASK 8: Cast all numeric columns to float64 and export
features_one_hot = features_one_hot.astype('float64')

# Export the dataframe to a CSV file for the next section
output_file_path = "dataset_part_3.csv"
features_one_hot.to_csv(output_file_path, index=False)

print(f"All numeric columns cast to float64 and exported to {output_file_path}.")
# Display the first few rows of the updated dataframe
print("Processed DataFrame:")
print(features_one_hot.head())
