from rico_utils import get_all_texts_from_rico_screen, ScreenInfo
from rico_dao import load_rico_screen, read_embedding_from_file, write_embedding_to_file
import json
import os
from rico_dao import load_rico_screen_dict

rico_dir = './datasets/filtered_traces'
embedding_dir = './embeddings/filtered_traces'

rico_screen_id_text_label_list_dict = {}
# rico_screen_id_rico_screen_dict = {}
rico_screen_id_screen_info_dict = {}
rico_screen_id_screenshot_path_dict = {}
rico_trace_id_trace_dict = {}

def get_rico_screen_id (package_dir, trace_dir, request_id):
    return package_dir + '-' + trace_dir + '-' + request_id

def get_trace_id_from_rico_screen_id (rico_screen_id):
    return '-'.join(rico_screen_id.split('-')[:2])

json_count = 0
screenshot_count = 0
app_count = 0
trace_count = 0

progress_count = 0

for package_dir in os.listdir(rico_dir):
    if os.path.isdir(rico_dir + '/' + package_dir):
        # for each package directory
        for trace_dir in os.listdir(rico_dir + '/' + package_dir):
            # for each trace directory
            if os.path.isdir(rico_dir + '/' + package_dir + '/' + trace_dir) and (not trace_dir.startswith('.')):
                if os.path.isdir(rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                    for view_hierarchy_json in os.listdir(rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies'):
                        if view_hierarchy_json.endswith('.json') and (not view_hierarchy_json.startswith('.')):
                            json_file_path = rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'view_hierarchies' + '/' + view_hierarchy_json
                            # populate rico_screen_id_screen_info_dict
                            with open(json_file_path) as f:
                                try:
                                    rico_screen = load_rico_screen_dict(json.load(f))
                                    package_name = rico_screen.activity_name.split('/')[0]
                                    if not view_hierarchy_json.replace('.json', '') == rico_screen.request_id:
                                        rico_screen.request_id = view_hierarchy_json.replace('.json', '')
                                    rico_screen_id = get_rico_screen_id(package_dir, trace_dir, rico_screen.request_id)
                                    screen_info = ScreenInfo(rico_screen_id, package_name, rico_screen.activity_name)
                                    # rico_screen_id_rico_screen_dict[rico_screen_id] = rico_screen
                                    rico_screen_id_screen_info_dict[rico_screen_id] = screen_info.toDict()
                                    rico_screen_id_text_label_list_dict[rico_screen_id] = get_all_texts_from_rico_screen(rico_screen)

                                    # print (json_file_path)
                                    json_count += 1
                                    view_screenshot_file_path = rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'screenshots' + '/' + rico_screen.request_id + '.jpg'
                                    if os.path.exists(view_screenshot_file_path):
                                        rico_screen_id_screenshot_path_dict[rico_screen_id] = view_screenshot_file_path
                                        screenshot_count += 1
                                    else:
                                        print ('can\'t find %s' % view_screenshot_file_path)
                                    progress_count += 1
                                    if progress_count % 100 == 0:
                                        print('Processd %d JSON files' % progress_count)
                                except TypeError as e:
                                    print(str(e) + ': ' + json_file_path)
                gestures_file_path = rico_dir + '/' + package_dir + '/' + trace_dir + '/' + 'gestures.json'
                # load the sequence path
                if os.path.exists(gestures_file_path):
                    with open(gestures_file_path, 'r') as gestures_file:
                        gesture_data = json.load(gestures_file)
                        gesture_sequence = gesture_data.keys()
                        rico_screen_id_sequence = []
                        for screen_id in gesture_sequence:
                            rico_screen_id_sequence.append(get_rico_screen_id(package_dir, trace_dir, screen_id))
                        rico_trace_id_trace_dict[package_dir + '-' + trace_dir] = rico_screen_id_sequence
                    trace_count += 1
                else:
                    print('can\'t find the trace: %s' % gestures_file_path)
        app_count += 1

print('Found %d JSON files and %d screenshots from %d traces in %d apps' %(json_count, screenshot_count, trace_count, app_count))


rico_screen_id_text_label_list_dict_json = json.dumps(rico_screen_id_text_label_list_dict)
with open(embedding_dir + '/' + 'rico_screen_id_text_label_list_dict' + '.json', 'w') as f:
    f.write(rico_screen_id_text_label_list_dict_json)

rico_screen_id_screen_info_dict_json = json.dumps(rico_screen_id_screen_info_dict)
with open(embedding_dir + '/' + 'rico_screen_id_screen_info_dict' + '.json', 'w') as f:
    f.write(rico_screen_id_screen_info_dict_json)

rico_screen_id_screenshot_path_dict_json = json.dumps(rico_screen_id_screenshot_path_dict)
with open(embedding_dir + '/' + 'rico_screen_id_screenshot_path_dict' + '.json', 'w') as f:
    f.write(rico_screen_id_screenshot_path_dict_json)

rico_trace_id_trace_dict_json = json.dumps(rico_trace_id_trace_dict)
with open(embedding_dir + '/' + 'rico_trace_id_trace_dict' + '.json', 'w') as f:
    f.write(rico_trace_id_trace_dict_json)




