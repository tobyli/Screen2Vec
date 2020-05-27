#!/usr/bin/python3
import sys,os
from shutil import copyfile

rico_dir_path = sys.argv[1]

output_path = sys.argv[2]
if not os.path.exists(output_path):
    os.makedirs(output_path)

start_index = int(sys.argv[3])
end_index = int(sys.argv[4])

# copy the sample files to output_path
for i in range(start_index, end_index + 1):
    try:
        copyfile(rico_dir_path + '/' + str(i) + '.jpg', output_path + '/' + str(i) + '.jpg')
        copyfile(rico_dir_path + '/' + str(i) + '.json', output_path + '/' + str(i) + '.json')
    except FileNotFoundError as e:
        print(e)