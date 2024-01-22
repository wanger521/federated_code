from collections import defaultdict


class ConfigOperateFather:
    def __init__(self, dicts=None):
        if dicts is None:
            dicts = defaultdict(list)
        self.dicts = dicts

    def get_config(self):
        return self.dicts

    def get_value(self, key):
        return self.dicts[key]

    def edit_key_value(self, key, value):
        self.dicts[key] = value

    def create_key_value(self, key, value):
        self.dicts[key] = value


class ConfigOperate(ConfigOperateFather):
    def __init__(self, dicts):
        if dicts is None:
            pass
        super(ConfigOperate, self).__init__(dicts)
