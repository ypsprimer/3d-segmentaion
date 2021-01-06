import os
import yaml


class Config(object):
    PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def load(cls, file_path):
        stream = open(file_path)
        data = yaml.load(stream)
        return data


if __name__ == "__main__":
    Config.load("Basic_configs/test.yml")
    print(Config.net)
