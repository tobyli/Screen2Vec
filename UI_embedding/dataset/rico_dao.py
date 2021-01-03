import json
import numpy as np
from .rico_models import RicoActivity, RicoScreen, ScreenInfo

# methods for parsing rico dataset files

def rico_node_decoder(rico_node_dict: dict):
    return rico_node_dict

def load_rico_activity_dict(rico_activity_dict: dict):
    root_node = rico_node_decoder(rico_activity_dict['root'])
    added_fragments = rico_activity_dict['added_fragments']
    active_fragments = rico_activity_dict['active_fragments']
    return RicoActivity(root_node, added_fragments, active_fragments)


def load_rico_screen_dict(rico_screen_dict: dict):
    activity_name = rico_screen_dict['activity_name']
    activity: RicoActivity = load_rico_activity_dict(rico_screen_dict['activity'])
    is_keyboard_deployed = rico_screen_dict['is_keyboard_deployed']
    request_id = rico_screen_dict['request_id']
    return RicoScreen(activity_name, activity, is_keyboard_deployed, request_id)


def load_rico_screen(rico_dir, rico_number):
    rico_screen_path = rico_dir + '/' + str(rico_number) + '.json'
    with open(rico_screen_path) as f:
        rico_screen = json.load(f)
        return load_rico_screen_dict(rico_screen)

def write_embedding_to_file(dir, rico_id, embedding):
    embedding_json = json.dumps(embedding.tolist())
    with open(dir + '/' + str(rico_id) + '.json', 'w') as f:
        f.write(embedding_json)

def read_embedding_from_file(embedding_dir, rico_id):
    try:
        with open(embedding_dir + '/' + rico_id + '.json', 'r') as f:
            json_string = f.read()
            if (not json_string):
                return None
            enc_list = json.loads(json_string)
            dataArray = np.array(enc_list)
            return dataArray
    except FileNotFoundError as e:
        # print(e)
        return None

def read_rico_id_text_label_list_dict(embedding_dir):
    with open(embedding_dir + '/' + 'rico_id_text_label_list_dict' + '.json', 'r') as f:
        rico_id_text_label_list_dict_json = f.read()
    return json.loads(rico_id_text_label_list_dict_json)

def read_rico_id_screen_info_dict(embedding_dir):
    with open(embedding_dir + '/' + 'rico_id_screen_info_dict' + '.json', 'r') as f:
        rico_id_screen_info_dict_json = f.read()
    rico_id_screen_info_dict_dict = json.loads(rico_id_screen_info_dict_json)
    rico_id_screen_info_dict = {}
    for rico_id, screen_info_dict in rico_id_screen_info_dict_dict.items():
        rico_id_screen_info_dict[rico_id] = ScreenInfo(**screen_info_dict)
    return rico_id_screen_info_dict
