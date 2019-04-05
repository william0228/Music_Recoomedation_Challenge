# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def _Feature_Counting(csv, feature_name=""):
    csv[feature_name] = csv[feature_name].map(str) + " " + csv['target'].map(str)
    a = csv.groupby(feature_name)

    print("{0} : {1}".format(feature_name, a.size()))
    print("------------------------------------------------------------------------------------------")

    return


def _Process_CSV(csv, song, members, extra):
    _Feature_Counting(csv, "source_system_tab")
    _Feature_Counting(csv, "source_screen_name")
    _Feature_Counting(csv, "source_type")

    return


def _Analyzation():
    # read and deal with CSV
    Train_CSV = pd.read_csv("./data/train.csv", )
    Test_CSV = pd.read_csv("./data/test.csv")
    Song_CSV = pd.read_csv("./data/songs.csv")
    Member_CSV = pd.read_csv("./data/members.csv", parse_dates=['registration_init_time','expiration_date'])
    Extra_Song_CSV = pd.read_csv("./data/song_extra_info.csv")

    Train_CSV = _Process_CSV(Train_CSV, Song_CSV, Member_CSV, Extra_Song_CSV)
    

if __name__ == '__main__':
    _Analyzation()

