import csv
import os
import shutil

# CSV file details
csv_file = r"C:\Users\moini\OneDrive\Desktop\major project\cifar-10\trainLabels.csv"
image_column = 'id'  # Column containing the image paths
group_column = 'label'  # Column containing the group or folder names

# Destination folder for grouped images
destination_folder = r'C:\Users\moini\OneDrive\Desktop\major project\benchmarked\dataset\train'

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    
    # Iterate through each row in the CSV file
    for row in reader:
        image_path = row[image_column]  # Extract image path
        group_name = row[group_column]  # Extract group or folder name

        x=".png"
        image_path = "C:\\Users\\moini\\OneDrive\\Desktop\\major project\\cifar-10\\train\\train\\" + image_path + x
                         
        # Create folder for the group if it doesn't exist
        group_folder = os.path.join(destination_folder, group_name)
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        # Move or copy the image to the appropriate folder
        shutil.copy(image_path, group_folder)
