from collections.abc import Iterable
from rico_models import RicoScreen, RicoActivity, ScreenInfo

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

def get_all_texts_from_rico_screen(rico_screen: RicoScreen):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_texts_from_node_tree(rico_screen.activity.root_node)




