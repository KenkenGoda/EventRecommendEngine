#!/Users/shohei/.pyenv/shims/python
# Main.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from Preprocessing import Preprocessing
from Learning import Learning
from Predicting import Predicting


def main():
    # データの読み出し
    log_data = read('log')              # ユーザーの行動ログデータ
    event_data = read('events')         # イベントの属性データ
    user_data = read('users')           # ユーザーのデモグラ情報
    user_test = read('test')            # 評価対象ユーザー
    sample_data = read('sample_submit') # 応募用サンプルファイル

    # 説明変数（特徴量）と目的変数を作成
    X_v_train, X_t_train, X_valid, X_test, y_v_train, y_t_train, y_valid = Preprocessing(log_data, event_data, user_data, user_test).preprocessing()
    
    # ベストモデルの作成
    model = Ridge(fit_intercept=False, normalize=True)
    parameters = {'alpha':[5.0, 6.0, 7.0, 8.0, 9.0]}
    best_model = Learning(X_v_train, X_valid, y_v_train, y_valid, model, parameters, cv=3).scoring()

    # 予測データの作成
    submit_data = Predicting(X_t_train, y_t_train, X_test, best_model).predicting()

    # 提出用ファイルの書き出し
    submit_data.to_csv('submit.tsv', sep='\t', index=False, header=False)

    print('All Done')

def read(name):
    # 各データの読み出し
    return pd.read_csv('\'{}.tsv\'.txt'.format(name), delimiter='\t')

if __name__=='__main__':
    main()
