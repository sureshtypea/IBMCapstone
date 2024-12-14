# Import necessary libraries
import requests
import pandas as pd
import numpy as np
from datetime import datetime, date

# Global variables
BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

# Fetch static JSON data
static_json_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json"
response = requests.get(static_json_url)

# Check if request was successful
if response.status_code == 200:
    print("Data request was successful!")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

# Parse the JSON content into a Pandas DataFrame
data = pd.json_normalize(response.json())

# Subset of the dataframe with required columns
data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

# Remove rows with multiple cores or payloads
data = data[data['cores'].map(len) == 1]
data = data[data['payloads'].map(len) == 1]

# Extract the single value from the lists
data['cores'] = data['cores'].map(lambda x: x[0])
data['payloads'] = data['payloads'].map(lambda x: x[0])

# Convert date_utc to a datetime object and extract the date
data['date'] = pd.to_datetime(data['date_utc']).dt.date

# Correctly filter data for dates up to November 13, 2020
data = data[data['date'] <= date(2020, 11, 13)]

# Helper functions
def getBoosterVersion(data):
    for rocket_id in data['rocket']:
        if rocket_id:
            response = requests.get(f"https://api.spacexdata.com/v4/rockets/{rocket_id}").json()
            BoosterVersion.append(response.get('name', None))

def getLaunchSite(data):
    for launchpad_id in data['launchpad']:
        if launchpad_id:
            response = requests.get(f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}").json()
            Longitude.append(response.get('longitude', None))
            Latitude.append(response.get('latitude', None))
            LaunchSite.append(response.get('name', None))

def getPayloadData(data):
    for payload_id in data['payloads']:
        if payload_id:
            response = requests.get(f"https://api.spacexdata.com/v4/payloads/{payload_id}").json()
            PayloadMass.append(response.get('mass_kg', None))
            Orbit.append(response.get('orbit', None))

def getCoreData(data):
    for core_id in data['cores']:
        if core_id and core_id.get('core'):
            response = requests.get(f"https://api.spacexdata.com/v4/cores/{core_id['core']}").json()
            Block.append(response.get('block', None))
            ReusedCount.append(response.get('reuse_count', None))
            Serial.append(response.get('serial', None))
        else:
            Block.append(None)
            ReusedCount.append(None)
            Serial.append(None)
        Outcome.append(str(core_id.get('landing_success', None)) + ' ' + str(core_id.get('landing_type', None)))
        Flights.append(core_id.get('flight', None))
        GridFins.append(core_id.get('gridfins', None))
        Reused.append(core_id.get('reused', None))
        Legs.append(core_id.get('legs', None))
        LandingPad.append(core_id.get('landpad', None))

# Apply the helper functions
getBoosterVersion(data)
getLaunchSite(data)
getPayloadData(data)
getCoreData(data)

# Construct the dataset
launch_dict = {
    'FlightNumber': list(data['flight_number']),
    'Date': list(data['date']),
    'BoosterVersion': BoosterVersion,
    'PayloadMass': PayloadMass,
    'Orbit': Orbit,
    'LaunchSite': LaunchSite,
    'Outcome': Outcome,
    'Flights': Flights,
    'GridFins': GridFins,
    'Reused': Reused,
    'Legs': Legs,
    'LandingPad': LandingPad,
    'Block': Block,
    'ReusedCount': ReusedCount,
    'Serial': Serial,
    'Longitude': Longitude,
    'Latitude': Latitude
}

# Create a Pandas DataFrame
final_df = pd.DataFrame(launch_dict)

# Task 2: Filter for Falcon 9 launches
data_falcon9 = final_df[final_df['BoosterVersion'] != 'Falcon 1'].copy()

# Reset the FlightNumber column
data_falcon9.loc[:, 'FlightNumber'] = list(range(1, data_falcon9.shape[0] + 1))

# Check for missing values
print("\nMissing Values Before Handling:")
print(data_falcon9.isnull().sum())

# Task 3: Dealing with Missing Values
# Calculate the mean value of the PayloadMass column
payload_mass_mean = data_falcon9['PayloadMass'].mean()
print(f"\nMean PayloadMass: {payload_mass_mean}")

# Replace np.nan values in PayloadMass with the calculated mean
data_falcon9['PayloadMass'].replace(np.nan, payload_mass_mean, inplace=True)

# Retain np.nan values in LandingPad for missing landing pad information
data_falcon9['LandingPad'].fillna(value=np.nan, inplace=True)

# Verify missing values after updates
print("\nMissing Values After Handling:")
print(data_falcon9.isnull().sum())

# Export the cleaned dataset to the specified directory
output_csv_path = r'C:\Users\sures\IBM Workspace\Data_Collection\dataset_part_1.csv'
data_falcon9.to_csv(output_csv_path, index=False)
print(f"\nDataset exported to {output_csv_path}.")
