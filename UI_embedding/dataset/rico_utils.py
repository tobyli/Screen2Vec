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

def get_all_labeled_texts_from_node_tree(node):
    results = []
    if 'text' in node and isinstance(node['text'], Iterable):
        text_class = ''
        if node['text'] and node['text'].strip():
            text = node['text']
            if node['class'] and node['class'].strip():
                text_class = convert_class_to_text_label(node['class'])
            if node["bounds"]:
                bounds = node["bounds"]
            results.append([text, text_class, bounds])
    if 'children' in node and isinstance(node['children'], Iterable):
        for child_node in node['children']:
            if (isinstance(child_node, dict)):
                results.extend(get_all_labeled_texts_from_node_tree(child_node))
    return results

def get_all_texts_from_rico_screen(rico_screen: RicoScreen):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_texts_from_node_tree(rico_screen.activity.root_node)

def get_all_labeled_texts_from_rico_screen(rico_screen: RicoScreen):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_labeled_texts_from_node_tree(rico_screen.activity.root_node)


