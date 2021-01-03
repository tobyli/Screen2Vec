from collections.abc import Iterable
from .rico_models import RicoScreen, RicoActivity, ScreenInfo
from .convert_class_to_label import convert_class_to_text_label

import numpy as np

# contains methods for collecting UI elements

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

def get_all_labeled_texts_from_node_tree(node, in_list: bool, in_drawer: bool, testing):
    results = []
    text_class = 0
    if 'text' in node and isinstance(node['text'], Iterable):
        if node['text'] and node['text'].strip():
            text = node['text']
            if "class" in node:
                the_class = node["class"]
            elif "className" in node:
                the_class = node["className"]
            if the_class and the_class.strip():
                if the_class == 'TextView':
                    if node['clickable']:
                        text_class = 20
                    else:
                        text_class = 11
                else:
                    text_class = convert_class_to_text_label(the_class)
            if text_class==0 and (in_drawer or in_list):
                if in_drawer:
                    text_class = 25
                if in_list:
                    text_class = 24
            if node["bounds"]:
                bounds = node["bounds"]
            if testing and text_class==0:
                results.append([text, text_class, bounds, the_class])
            else:
                results.append([text, text_class, bounds])
    if 'children' in node and isinstance(node['children'], Iterable):
        for child_node in node['children']:
            if (isinstance(child_node, dict)):
                if text_class == 12:
                    in_list = True
                if text_class == 7:
                    in_drawer = True
                results.extend(get_all_labeled_texts_from_node_tree(child_node, in_list, in_drawer, testing))
    return results

def get_all_texts_from_rico_screen(rico_screen: RicoScreen):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_texts_from_node_tree(rico_screen.activity.root_node)

def get_all_labeled_texts_from_rico_screen(rico_screen: RicoScreen, testing=False):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_labeled_texts_from_node_tree(rico_screen.activity.root_node, False, False, testing)


def get_all_labeled_uis_from_node_tree(node, in_list: bool, in_drawer: bool, testing):
    results = []
    text_class = 0
    if 'text' in node and isinstance(node['text'], Iterable) and node['text'] and node['text'].strip():
        text = node['text']
    else: 
        text = ''
    if "class" in node:
        the_class = node["class"]
    elif "className" in node:
        the_class = node["className"]
    if the_class and the_class.strip():
        if the_class == 'TextView':
            if node['clickable']:
                text_class = 20
            else:
                text_class = 11
        else:
            text_class = convert_class_to_text_label(the_class)
    if text_class==0 and (in_drawer or in_list):
        if in_drawer:
            text_class = 25
        if in_list:
            text_class = 24
    if node["bounds"]:
        bounds = node["bounds"]
    if "visible-to-user" in node:
        visibility = node["visible-to-user"]
    elif "visible_to_user" in node:
        visibility = True #node["visible_to_user"]
    if visibility and testing and text_class==0:
        results.append([text, text_class, bounds, the_class])
    elif visibility:
        results.append([text, text_class, bounds])
    if 'children' in node and isinstance(node['children'], Iterable):
        for child_node in node['children']:
            if (isinstance(child_node, dict)):
                if text_class == 12:
                    in_list = True
                if text_class == 7:
                    in_drawer = True
                results.extend(get_all_labeled_uis_from_node_tree(child_node, in_list, in_drawer, testing))
    return results


def get_all_labeled_uis_from_rico_screen(rico_screen: RicoScreen, testing=False):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        return get_all_labeled_uis_from_node_tree(rico_screen.activity.root_node, False, False, testing)


def get_hierarchy_dist_from_node_tree(node, node_idx, node_parent_idx, parent_dif, distance_mtx):
    # go through parent and add one
    if "visible-to-user" in node and node["visible-to-user"]:
        for i in range(node_idx):
            #print(i, node_parent_idx, node_idx)
            distance_mtx[i,node_idx] = distance_mtx[node_parent_idx,i] + parent_dif
            distance_mtx[node_idx,i] = distance_mtx[i,node_idx]
        node_parent_idx = node_idx
        node_idx += 1
        parent_dif = 1
    else:
        parent_dif += 1
    if 'children' in node and isinstance(node['children'], Iterable):
        for child_node in node['children']:
            if (isinstance(child_node, dict)):
                distance_mtx, fin_idx = get_hierarchy_dist_from_node_tree(child_node, node_idx, node_parent_idx, parent_dif, distance_mtx)
                node_idx = fin_idx
    return distance_mtx, node_idx

def get_hierarchy_dist_from_rico_screen(rico_screen: RicoScreen, num_uis):
    if rico_screen.activity is not None and rico_screen.activity.root_node is not None:
        arr = np.zeros((num_uis, num_uis))
        distances, _ = get_hierarchy_dist_from_node_tree(rico_screen.activity.root_node,0, -1, 1, arr)
        return distances