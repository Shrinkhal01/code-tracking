import os
import shutil

downloads_folder = os.path.expanduser("~/Downloads")  # this is the location part which can be edited as per the requirements

file_categories = {
    "Documents": [".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt"],
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
    "Music": [".mp3", ".wav", ".aac", ".flac"],
    "Videos": [".mp4", ".mkv", ".mov", ".avi", ".flv"],
    "Archives": [".zip", ".tar", ".gz", ".7z", ".rar"],
    "Code": [".py", ".java", ".c", ".cpp", ".js", ".html", ".css", ".sh", ".bat"],
    "Applications": [".exe", ".msi", ".dmg", ".pkg", ".app"],  # Added .app for macOS applications
    "Others": []
}

for category in file_categories.keys():
    category_path = os.path.join(downloads_folder, category)
    os.makedirs(category_path, exist_ok=True)

def move_file(file_name, category_folder):
    src = os.path.join(downloads_folder, file_name)
    dest = os.path.join(category_folder, file_name)
    shutil.move(src, dest)

for file_name in os.listdir(downloads_folder):
    file_path = os.path.join(downloads_folder, file_name)
    if os.path.isfile(file_path):
        file_extension = os.path.splitext(file_name)[1].lower()
        moved = False
        for category, extensions in file_categories.items():
            if file_extension in extensions:
                move_file(file_name, os.path.join(downloads_folder, category))
                moved = True
                break
        if not moved:
            move_file(file_name, os.path.join(downloads_folder, "Others"))
print("Downloads folder organized successfully!")