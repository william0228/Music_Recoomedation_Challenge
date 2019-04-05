# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import lightgbm as lgb
from collections import defaultdict

params = [{
        'objective': 'binary',
        'boosting': 'gbdt',
        'learning_rate': 0.2,
        'verbose': 0,
        'num_leaves': 2 ** 7,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 30,
        'num_rounds': 200,
        'metric': 'auc'
}, {
        'objective': 'binary',
        'boosting': 'dart',
        'learning_rate': 0.2,
        'verbose': 0,
        'num_leaves': 2 ** 7,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 20,
        'num_rounds': 200,
        'metric': 'auc'
}]


def _Merge_CSV(csv, song, members, extra):
    csv = pd.merge(csv, song, on='song_id', how='left')
    csv = pd.merge(csv, extra, on='song_id', how='left')
    csv = pd.merge(csv, members, on='msno', how='left')

    return csv


def _Fill_NAN(csv):
    csv['source_system_tab'].fillna('others', inplace=True)
    csv['source_screen_name'].fillna('others', inplace=True)
    csv['source_type'].fillna('nan', inplace=True)
    csv['song_length'].fillna(200000, inplace=True)
    # csv(['genre_ids'].fillna(0, inplace=True))
    csv['artist_name'].fillna('no_artist',inplace=True)
    csv['composer'].fillna('no_composer',inplace=True)
    csv['lyricist'].fillna('no_lyricist',inplace=True)
    csv['language'].fillna(0, inplace=True)
    csv['city'].fillna('Unknow', inplace=True)
    csv['bd'].fillna(0, inplace=True)
    csv['gender'].fillna('asexual', inplace=True)


def _Song_Counting(csv):
    song_values = csv['song_id'].value_counts().keys().tolist()
    song_counts = csv['song_id'].value_counts().tolist()

    count_csv = pd.DataFrame()
    count_csv['song_id'] = song_values
    count_csv['count_song_time'] = song_counts

    count_csv = pd.merge(csv, count_csv, on='song_id', how='left')

    return count_csv


def _Artist_Counting(csv):
    artist_values = csv['artist_name'].value_counts().keys().tolist()
    artist_counts = csv['artist_name'].value_counts().tolist()

    count_csv = pd.DataFrame()
    count_csv['artist_name'] = artist_values
    count_csv['count_artist_time'] = artist_counts

    count_csv = pd.merge(csv, count_csv, on='artist_name', how='left')

    return count_csv


def _Source_Tab(csv):
    a = csv['source_system_tab']
    adding = []
    for idx, val in enumerate(a):
        if (val == 'my library') or (val == 'settings'):
            adding.append(1)
        elif (val == 'listen with') or (val == 'radio'):
            adding.append(0)
        else:
            adding.append(0.5)

    csv['specific_tab'] = adding

    return csv


def _Source_Screen(csv):
    a = csv['source_screen_name']
    adding = []
    for idx, val in enumerate(a):
        if (val == 'Local playlist') or (val == 'My library'):
            adding.append(1)
        elif (val == 'Discover Genre') or (val == 'Others profile more') or (val == 'Radio'):
            adding.append(0)
        else:
            adding.append(0.5)

    csv['specific_screen'] = adding

    return csv


def _Source_Type(csv):
    a = csv['source_type']
    adding = []
    for idx, val in enumerate(a):
        if (val == 'local-library') or (val == 'local-playlist'):
            adding.append(1)
        elif (val == 'listen-with') or (val == 'radio'):
            adding.append(0)
        else:
            adding.append(0.5)

    csv['specific_type'] = adding

    return csv


def _Add_Feature(csv):
    new_csv = _Song_Counting(csv)
    new_csv = _Artist_Counting(new_csv)
    new_csv = _Source_Tab(new_csv)
    new_csv = _Source_Screen(new_csv)
    new_csv = _Source_Type(new_csv)

    print(new_csv)

    return new_csv


def _Process_CSV(csv, song, members, extra):
    merge_csv = _Merge_CSV(csv, song, members, extra)
    # merge_csv = _Fill_NAN(merge_csv)
    adding_csv = _Add_Feature(merge_csv)

    return adding_csv


def _Run():
    # read and deal with CSV
    Train_CSV = pd.read_csv("./data/train.csv", )
    Test_CSV = pd.read_csv("./data/test.csv")
    Song_CSV = pd.read_csv("./data/songs.csv")
    Member_CSV = pd.read_csv("./data/members.csv", parse_dates=['registration_init_time','expiration_date'])
    Extra_Song_CSV = pd.read_csv("./data/song_extra_info.csv")

    Train_CSV = _Process_CSV(Train_CSV, Song_CSV, Member_CSV, Extra_Song_CSV)
    Test_CSV = _Process_CSV(Test_CSV, Song_CSV, Member_CSV, Extra_Song_CSV)

    X_train = Train_CSV.drop(['target'], axis=1)
    y_train = Train_CSV[['target']]
    X_test = Test_CSV.drop(['id'], axis=1)
    ids = Test_CSV['id'].values
    # print("ids: ", ids)

    d = defaultdict(preprocessing.LabelEncoder)

    X_train = X_train.apply(lambda x: d[x.name].fit_transform(x.astype(str)))
    X_test = X_test.apply(lambda x: d[x.name].fit_transform(x.astype(str)))


    first = lgb.Dataset(X_train, y_train)
    second = lgb.Dataset(X_train, y_train)

    model1 = lgb.train(params[0], train_set=first,  valid_sets=second, verbose_eval=5)
    model2 = lgb.train(params[1], train_set=first,  valid_sets=second, verbose_eval=5)

    print('Making predictions')
    prediction1 = model1.predict(X_test)
    prediction2 = model2.predict(X_test)
    prediction = np.mean([prediction1, prediction2], axis=0)

    print('Done making predictions')

    print('Saving predictions Model model of gbdt')
    ########
    result1 = pd.DataFrame()
    result1['id'] = ids
    result1['target'] = prediction1

    result1.to_csv('d1.csv', index=False, float_format='%.5f')
    ########
    result2 = pd.DataFrame()
    result2['id'] = ids
    result2['target'] = prediction2

    result2.to_csv('d2.csv', index=False, float_format='%.5f')
    ########
    result3 = pd.DataFrame()
    result3['id'] = ids
    result3['target'] = prediction

    result3.to_csv('d3.csv', index=False, float_format='%.5f')
    ########


    print('Done!')
    


if __name__ == '__main__':
    _Run()

