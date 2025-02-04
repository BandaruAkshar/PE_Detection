import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

LOGGED_IN_CONFIG = []


class Logger(object):
    def __int__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def create_logger(directory="", name="logs.txt"):
    """ Function that creates a log in the given file"""
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)
    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_path):
        os.mkdir(log_today)

    exp_id = len(os.listdir(log_today))
    log_folder = log_today + f"{exp_id}/"

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    else:
        print("Exp already present")

    return log_folder


def save_config(config, path):
    dictionary = config.__dict__.copy()
    del dictionary["__doc__"], dictionary["__module__"], dictionary["__dict__"], dictionary["__weakref__"]

    with open(path + ".json", "w") as f:
        json.dump(dictionary, f)

    dictionary["selected_folds"] = [", ".join(np.array(dictionary["selected_folds"]).astype(str))]
    config_df = pd.DataFrame.from_dict(dictionary)
    config_df.to_csv(path+".csv",index = False)
    return config_df