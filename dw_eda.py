# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/sures/IBM Workspace/Data_Wrangling/dataset_part_1.csv'# Update this path
df = pd.read_csv(file_path)

# Task 1: Analyze the number of launches on each site
# Calculate the number of launches at each site
launch_site_counts = df['LaunchSite'].value_counts()

# Plotting the number of launches per site
plt.figure(figsize=(10, 6))
launch_site_counts.plot(kind='bar', color='coral')
plt.title('Number of Launches at Each Site', fontsize=16)
plt.xlabel('Launch Site', fontsize=14)
plt.ylabel('Number of Launches', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Displaying the calculated data for Task 1
print("Launch Site Counts:")
print(launch_site_counts)

# Task 2: Analyze orbit types used in launches
# Calculate the number of launches for each orbit type
orbit_counts = df['Orbit'].value_counts()

# Plotting the number of launches per orbit type
plt.figure(figsize=(12, 7))
orbit_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Launches by Orbit Type', fontsize=16)
plt.xlabel('Orbit Type', fontsize=14)
plt.ylabel('Number of Launches', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Displaying the calculated data for Task 2
print("\nOrbit Counts:")
print(orbit_counts)
# Task 3: Analyze mission outcomes of the orbits

# Calculate the number and occurrence of each mission outcome
landing_outcomes = df['Outcome'].value_counts()

# Display landing outcomes
print("Mission Outcomes (Landing):")
print(landing_outcomes)

# Enumerate the outcomes
print("\nEnumerating Mission Outcomes:")
for i, outcome in enumerate(landing_outcomes.keys()):
    print(i, outcome)

# Define bad outcomes where the second stage did not land successfully
bad_outcomes = set(landing_outcomes.keys()[[1, 3, 5, 6, 7]])
print("\nBad Outcomes (Unsuccessful Landings):")
print(bad_outcomes)

# Plotting the landing outcomes
plt.figure(figsize=(12, 7))
landing_outcomes.plot(kind='bar', color='lightgreen')
plt.title('Mission Outcomes by Count', fontsize=16)
plt.xlabel('Mission Outcome', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Task 4: Create a landing outcome label from the Outcome column

# Create the landing_class based on bad_outcomes
landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]
df['Class'] = landing_class

# Display the first 8 rows of the Class column
print("First 8 values of 'Class' column:")
print(df[['Class']].head(8))

# Display the first 5 rows of the updated DataFrame
print("\nFirst 5 rows of the DataFrame:")
print(df.head(5))

# Calculate the success rate
success_rate = df["Class"].mean()
print(f"\nSuccess Rate of Landings: {success_rate:.2%}")

# Export the updated DataFrame to a CSV file
output_file_path = "dataset_part_2.csv"
df.to_csv(output_file_path, index=False)
print(f"\nUpdated DataFrame has been saved to: {output_file_path}")
