import csv
from datetime import datetime
import supplemental_english as sup
import numpy as np
import re
import pickle as pkl
# Initialize a dictionary to hold columns
columns = {}

with open('test.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    # Initialize lists for each column
    for field in reader.fieldnames[1:4]:  # Assuming you want to skip the first field   
        columns[field] = []
    # Populate the lists
    for row in reader:
        for field in reader.fieldnames[1:4]:
            columns[field].append(row[field])

# Now, each column is a list in the 'columns' dictionary
# For example, to access the list for column 'A': print(columns['A'])

# Convert date strings to datetime objects and find their difference from the minimum date
min_date = datetime.strptime("2021-02-17 21:21:56", "%Y-%m-%d %H:%M:%S")
columns["date"] = [(datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") - min_date).days for date_str in columns['date']]





#approach1# Convert license plate strings to hot encoded vectors
features=np.zeros((len(columns['plate']),420))
chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
regions = list(set(sup.REGION_CODES.values()))

for n,plate in enumerate(columns['plate']):
    
    for i in range(len(plate)-1):
        features[n][36*i + chars.index(plate[i])]=1

    features[n][36*9 + regions.index(sup.REGION_CODES[plate[6:]])] = 1

    if len(plate) == 9: features[n][36*9+91]==1 # to signify that the plate is 9 characters long as a feature

    features[n][-3:] = sup.find_code(plate) # putting the data for special governement plates in the last 3 columns

keys = columns['price']


# Store the features and keys in pickle files
pkl.dump(features, open('features.pkl', 'wb'))
pkl.dump(keys, open('keys.pkl', 'wb'))
