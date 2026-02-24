#!/usr/bin/env python3
import os
import subprocess
import csv
import datetime
import pwd

gdata_path = "/g/data/xe2"
categories = ["Top-level"] + ["datasets", "projects", "users", "references", "GIS", "shared"]
output_file = "/scratch/xe2/cb8590/tmp/gdata_xe2_storage_report.csv"
min_size_gb = 100

def get_folder_info(path):
    try:
        result = subprocess.run(['du', '-sb', path], capture_output=True, timeout=300, text=True)
        size_gb = int(result.stdout.split()[0]) / (1024**3)

        # Use os.stat to get modification time and owner reliably
        st = os.stat(path)
        try:
            owner = pwd.getpwuid(st.st_uid).pw_name
        except KeyError:
            owner = str(st.st_uid)

        mtime = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        return size_gb, mtime, owner
    except Exception:
        return None, None, None

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Folder", "Size_GB", "Last_Modified_Date", "Owner"])
    
    for category in categories:
        if category == "Top-level":
            path = gdata_path
            items = os.listdir(path) if os.path.isdir(path) else []
        else:
            path = os.path.join(gdata_path, category)
            items = os.listdir(path) if os.path.isdir(path) else []
        
        print(f"Processing {category}...")
        folders = []
        for i, item in enumerate(sorted(items), 1):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                size_gb, atime, owner = get_folder_info(item_path)
                if size_gb and size_gb >= min_size_gb:
                    folders.append((item, size_gb, atime, owner))
                print(f"  [{i}/{len(items)}] {item}")
        
        folders.sort(key=lambda x: -x[1])
        for name, size, atime, owner in folders:
            writer.writerow([category, name, f"{size:.2f}", atime, owner])

print(f"Report saved to {output_file}")
