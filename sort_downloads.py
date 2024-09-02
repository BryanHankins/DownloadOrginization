import os
import shutil

# Define the path to the Downloads folder
DOWNLOADS_FOLDER = os.path.expanduser('~/Downloads')

# Define file extensions and their corresponding folder names
EXTENSIONS = {
    'Images': ['.jpg', '.jpeg', '.png', '.gif', '.webp'],
    'Documents': ['.pdf', '.docx', '.txt', '.csv', '.html'],
    'Videos': ['.mp4', '.mov', '.avi', '.webm'],
    'Music': ['.mp3', '.wav'],
    'Archives': ['.zip', '.tar.gz', '.rar'],
    'Executables': ['.exe', '.msi', '.appinstaller'],
    'Others': ['.ini']  # Add any other extensions you want to categorize
}

# Create folders for each file type category if they don't exist
for folder_name in EXTENSIONS.keys():
    target_folder = os.path.join(DOWNLOADS_FOLDER, folder_name)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

# Sort files into the corresponding folders
for filename in os.listdir(DOWNLOADS_FOLDER):
    file_path = os.path.join(DOWNLOADS_FOLDER, filename)
    if os.path.isfile(file_path):
        file_ext = os.path.splitext(filename)[1].lower()
        moved = False
        for folder, extensions in EXTENSIONS.items():
            if file_ext in extensions:
                target_folder = os.path.join(DOWNLOADS_FOLDER, folder)
                shutil.move(file_path, target_folder)
                moved = True
                break
        if not moved:
            # Move uncategorized files to an "Unsorted" folder
            unsorted_folder = os.path.join(DOWNLOADS_FOLDER, 'Unsorted')
            if not os.path.exists(unsorted_folder):
                os.makedirs(unsorted_folder)
            shutil.move(file_path, unsorted_folder)
            print(f'No specific match found for {filename}, moved to Unsorted.')

print("Downloads folder sorted.")
