from collections.abc import Iterable
from .rico_models import RicoScreen, RicoActivity, ScreenInfo
from .convert_class_to_label import convert_class_to_text_label

def get_all_texts_from_node_tree(node):
    results = []
    if 'text' in node and isinstance(node['text'], Iterable):
        if node['text'] and node['text'].strip():
            results.append(node['text'])
    if 'children' in node and isinstance(node['children'], Iterable):
        for child_node in node['children']:
            if (isinstance(child_node, dict)):
                results.extend(get_all_texts_from_node_tree(child_node))
    return results

def get_all_labeled_texts_from_node_tree(node, in_list: bool, in_drawer: bool):
    results = []
    if 'text' in node and isinstance(node['text'], Iterable):
        text_class = 0
        if node['text'] and node['text'].strip():
            text = node['text']
            if node['class'] and node['class'].strip():
                if node['class'] == 'TextView':
                    if node['clickable']:
                        text_class = 20
                    else:
                        text_class = 11
                else:
                    text_class = convert_class_to_text_label(node['class'])
            if text_class==0 and (in_drawer or in_list):
                if in_drawer:
                    text_class = 25
                if in_list:
                    text_class = 24
            if node["bounds"]:
                bounds = node["bounds"]
            results.append([text, text_class, bounds])
    if 'children' in node and isinstance(node['children'], Iterable):
        for child_node in node['children']:
            if (isinstance(child_node, dict)):
                if text_class == 12:
                    in_list = True
                if text_class == 7:
                    in_drawer = True
                results.extend(get_all_labeled_texts_from_node_tree(child_node, in_list, in_drawer))
    return results

def get_all_texts_from_rico_screen(rico_screen: RicoScreen):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_texts_from_node_tree(rico_screen.activity.root_node)

def get_all_labeled_texts_from_rico_screen(rico_screen: RicoScreen):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_labeled_texts_from_node_tree(rico_screen.activity.root_node, False, False)


