import json

# class definitions for RICO dataset

class ScreenInfo:
    def __init__(self, rico_id, package_name, activity_name):
        self.rico_id = rico_id
        self.package_name = package_name
        self.activity_name = activity_name

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    def toDict(self):
        return self.__dict__


class RicoScreen:
    def __init__(self, activity_name, activity, is_keyboard_deployed, request_id):
        self.activity_name = activity_name
        self.activity = activity
        self.is_keyboard_deployed = is_keyboard_deployed
        self.request_id = request_id


class RicoActivity:
    def __init__(self, root_node, added_fragments, active_fragments):
        self.root_node = root_node
        self.added_fragments = added_fragments
        self.active_fragments = active_fragments