#!/usr/bin/python3
import sys,os

rico_dir_path = sys.argv[1]
rico_dir = os.fsencode(rico_dir_path)

json_file_names = set()
img_file_names = set()

for file in os.listdir(rico_dir):
    file_name = os.fsdecode(file)
    if file_name.endswith('.jpg'):
        img_file_names.add(file_name.replace('.jpg', ''))
    elif file_name.endswith('.json'):
        json_file_names.add(file_name.replace('.json', ''))

both_file_names = json_file_names.intersection(img_file_names)
print('Found %d JSON files' % len(json_file_names))
print('Found %d img files' % len(img_file_names))
print('Found %d entries with both JSON and img' % len(both_file_names))




