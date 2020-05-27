from rico_utils import get_all_texts_from_rico_screen, ScreenInfo
from rico_dao import load_rico_screen, read_embedding_from_file, write_embedding_to_file
import json

rico_dir = './datasets/sample_rico'
embedding_dir = './embeddings/sample_rico'

rico_id_text_label_list_dict = {}
rico_id_screen_info_dict = {}

for rico_id in range(1, 1001):
    try:
        rico_screen = load_rico_screen(rico_dir, rico_id)
        package_name = rico_screen.activity_name.split('/')[0]
        screen_info = ScreenInfo(rico_id, package_name, rico_screen.activity_name)
        rico_id_text_label_list_dict[rico_id] = get_all_texts_from_rico_screen(rico_screen)
        rico_id_screen_info_dict[rico_id] = screen_info.toDict()
    except FileNotFoundError as e:
        print(e)

rico_id_text_label_list_dict_json = json.dumps(rico_id_text_label_list_dict)
with open(embedding_dir + '/' + 'rico_id_text_label_list_dict' + '.json', 'w') as f:
    f.write(rico_id_text_label_list_dict_json)

rico_id_screen_info_dict_json = json.dumps(rico_id_screen_info_dict)
with open(embedding_dir + '/' + 'rico_id_screen_info_dict' + '.json', 'w') as f:
    f.write(rico_id_screen_info_dict_json)

