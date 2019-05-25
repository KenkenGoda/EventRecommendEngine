#!/Users/shohei/.pyenv/shims/python
# Preprocessing.py

import numpy as np
from numpy import random as rd
from scipy import stats
import pandas as pd
from copy import copy

import Join
import MatrixFactorization


class Function:

    def __init__(self, log_data, event_data, user_data, user_test, r=0.8, alpha=0.1, beta=0.01, iterations=10):
        self.log_data = log_data        # ユーザーの行動ログデータ
        self.event_data = event_data    # イベントの属性データ
        self.user_data = user_data      # ユーザーのデモグラ情報
        self.user_test = user_test      # 評価対象ユーザー
        self.r = r                      # 訓練用過去データと訓練データの比率
        self.alpha = alpha              # スコアリングする際のパラメーター
        self.beta = beta                # スコアリングする際のパラメーター
        self.iterations = iterations    # スコアリングする際のイテレーションの数

    def area(self, data):
        # 都道府県を各地方に振り分ける
        data = data.apply(lambda x:'北海道' if x=='北海道'
                               else('東北' if (x=='青森県' or x=='岩手県' or x=='宮城県' or x=='秋田県' or x=='山形県' or x=='福島県')
                               else('関東' if (x=='茨城県' or x=='栃木県' or x=='群馬県' or x=='埼玉県' or x=='千葉県' or x=='東京都' or x=='神奈川県')
                               else('中部' if (x=='新潟県' or x=='富山県' or x=='石川県' or x=='福井県' or x=='山梨県' or x=='長野県' or x=='岐阜県' or x=='静岡県' or x=='愛知県')
                               else('近畿' if (x=='京都府' or x=='大阪府' or x=='兵庫県' or x=='滋賀県' or x=='奈良県' or x=='和歌山県' or x=='三重県')
                               else('中国' if (x=='鳥取県' or x=='島根県' or x=='岡山県' or x=='広島県' or x=='山口県')
                               else('四国' if (x=='高知県' or x=='香川県' or x=='徳島県' or x=='愛媛県')
                               else '九州' )))))))
        return data
    
    def training_hist(self, data):
        # 比率rで入力データを訓練用過去データと訓練データに分割
        hist_length = int(len(data)*self.r)
        train_length = len(data)-hist_length
        hist_data = data.iloc[:hist_length, :].reset_index().set_index(['user_id', 'event_id'])
        log_data = data.iloc[train_length-1:, :].sort_index()
        return hist_data, log_data

    def validation_hist(self, data):
        # 2017/9/17以前のデータを検証用過去データ、2017/9/18以降のデータを検証用データとして抽出
        hist_data = data[data.index<'2017-9-18'].reset_index().set_index(['user_id', 'event_id'])
        log_data = data[data.index>='2017-9-18'].reset_index()
        return hist_data, log_data

    def test_hist(self, data):
        # 全データを評価用過去データとして抽出
        return data.reset_index().set_index(['user_id', 'event_id'])

    def training_user(self, data):
        # 1週間に1回以上のペースで行動していたユーザーを選出
        d_week = ((data.iloc[-1].name-data.iloc[0].name)/7).days
        action_num = data.groupby('user_id')[['action_type']].count()
        return action_num[action_num.action_type>=d_week].index.values

    def validation_user(self, data, num):
        # '2017/9/18以降に行動したユーザーを選出
        user = data.user_id.unique()
        return rd.choice(user, num, replace=False)

    def training_event(self, user, data):
        # 選出されたユーザーが抽出された期間内に行動したイベントの中で、'action_type'の合計値が上位10%のイベントを訓練用の推薦イベントとして抽出
        df_user = pd.DataFrame(index=pd.Index(user, name='user_id'))
        df_event = df_user.join(data.reset_index().set_index('user_id')).groupby('event_id')[['action_type']].sum()
        famous_event_action = stats.scoreatpercentile(df_event, 90)
        return df_event[df_event.action_type>=famous_event_action].index.values

    def validation_event(self, data):
        # 2017/9/18~23に開催されるイベントを検証用の推薦イベントとして抽出
        data = data.reset_index().set_index('event_start_at')
        return data[(data.index>='2017-9-18')&(data.index<'2017-9-24')].event_id.values

    def test_event(self, data):
        # 2017/9/24~30に開催されるイベントを評価用の推薦イベントとして抽出
        data = data.reset_index().set_index('event_start_at')
        return data[data.index>='2017-9-24'].event_id.values

    def multi_index(self, user, event):
        # 'user_id'と'event_id'をインデックスに持つDFを作成
        return pd.DataFrame(index=pd.MultiIndex.from_product([user, event], names=['user_id', 'event_id']))

    def scoring(self, user, event, data):
        # 目的変数の不足部分をスコアリングしたDFを作成
        MI = pd.MultiIndex.from_product([user, event], names=['user_id', 'event_id'])
        df = pd.DataFrame(index=MI).join(data.set_index(['user_id', 'event_id']))[['action_type']]
        df = df.groupby(level=[0, 1]).max().reset_index()
        df = df.pivot(index='user_id', columns='event_id', values='action_type').fillna(0)
        R = df.as_matrix()
        mf = MatrixFactorization.MF(R, K=2, alpha=self.alpha, beta=self.beta, iterations=self.iterations)
        training_process = mf.train()
        R_pred = mf.full_matrix()
        df_pred = pd.DataFrame(R_pred, index=df.index, columns=df.columns).reset_index()
        df_pred = pd.melt(df_pred, id_vars='user_id', value_vars=list(df_pred.columns[1:]),
                          var_name='event_id', value_name='action_type').set_index(['user_id', 'event_id']).sort_index()
        return df_pred


class Preprocessing(Function):

    def preprocessing(self):
        # データをコピー
        log_data = copy(self.log_data)
        event_data = copy(self.event_data)
        user_data = copy(self.user_data)
        user_test = copy(self.user_test)
        
        # 時刻データを全てdatetime型に変換
        log_data.time_stamp = pd.to_datetime(log_data.time_stamp)
        event_data.event_start_at = pd.to_datetime(event_data.event_start_at)
        event_data.first_published_at = pd.to_datetime(event_data.first_published_at)
        user_data.created_on = pd.to_datetime(user_data.created_on)

        # ユーザーの性別を0(女性),1(男性)に変換
        user_data.gender = user_data.gender.apply(lambda x: 0 if x=='女性' else 1)

        # ユーザーの居住地を'prefecture'から'residence'に変更
        user_data = user_data.rename(columns={'prefecture': 'residence'})

        # user_testのインデックスに'user_id'を指定
        user_test = user_test.set_index('user_id')

        # 全データの'user_id'を含むユーザーのデモグラ情報データを作成し、インデックスに'user_id'を指定
        user_log = pd.DataFrame({'user_id':log_data.user_id.unique()}).set_index('user_id')
        user_data = user_data.set_index('user_id').join([user_log, user_test], how='outer')

        # user_dataのnullを補完
        user_data = user_data.fillna({'age':user_data.age.median(),
                                      'gender':user_data.gender.mode()[0],
                                      'residence':'東京都',
                                      'created_on':pd.to_datetime(user_data.created_on).max()})

        # event_dataの年齢制限が0の部分に中央値を代入
        event_data.female_age_lower[event_data.female_age_lower==0] = event_data.female_age_lower.median()
        event_data.female_age_upper[event_data.female_age_upper==0] = event_data.female_age_upper.median()
        event_data.male_age_lower[event_data.male_age_lower==0] = event_data.male_age_lower.median()
        event_data.male_age_upper[event_data.male_age_upper==0] = event_data.male_age_upper.median()

        # event_dataのnullを補完
        event_data = event_data.fillna({'female_age_upper':event_data.female_age_upper.median(),
                                        'male_age_upper':event_data.male_age_upper.median(),
                                        'female_price':event_data.female_price.median(),
                                        'male_price':event_data.male_price.median()})

        # log_dataのnullを補完
        log_data = log_data.fillna({'num_of_people':0, 'total_price':0})

        # event_dataのインデックスに'event_id'を指定
        event_data = event_data.set_index('event_id')

        #user_data.residence = area(user_data.residence)
        #event_data.prefecture = area(event_data.prefecture)

        # user_idとevent_idで紐づけたmarge_dataを作成
        marge_data = pd.merge(pd.merge(log_data, event_data.reset_index(), how='left'), user_data.reset_index(), how='left')

        # 'time_stamp'をインデックスに指定
        marge_data = marge_data.set_index('time_stamp')
    
        # 訓練用、検証用過去データと、訓練、検証、評価データの作成
        v_train_hist, v_train_data = super().training_hist(marge_data[marge_data.index<'2017-9-18'])
        t_train_hist, t_train_data = super().training_hist(marge_data)
        valid_hist, valid_data = super().validation_hist(marge_data)
        test_hist = super().test_hist(marge_data)

        # それぞれのデータで使用するユーザーを選出
        v_train_user = super().training_user(v_train_data)
        t_train_user = super().training_user(t_train_data)
        valid_user = super().validation_user(valid_data, len(user_test))
        test_user = user_test.index.values

        # それぞれのデータで使用するイベントを選出
        v_train_event = super().training_event(v_train_user, v_train_data)
        t_train_event = super().training_event(t_train_user, t_train_data)
        valid_event = super().validation_event(event_data)
        test_event = super().test_event(event_data)

        # それぞれのデータのインデックスを作成
        v_train_index = super().multi_index(v_train_user, v_train_event)
        t_train_index = super().multi_index(t_train_user, t_train_event)
        valid_index = super().multi_index(valid_user, valid_event)
        test_index = super().multi_index(test_user, test_event)

        # 説明変数（特徴量）のDFを作成
        X_v_train = Join.Join(v_train_index, user_data, event_data, v_train_hist).extract()
        X_t_train = Join.Join(t_train_index, user_data, event_data, t_train_hist).extract()
        X_valid = Join.Join(valid_index, user_data, event_data, valid_hist).extract()
        X_test = Join.Join(test_index, user_data, event_data, test_hist).extract()

        # 目的変数のDFを作成
        y_v_train = super().scoring(v_train_user, v_train_event, v_train_data)
        y_t_train = super().scoring(t_train_user, t_train_event, t_train_data)
        y_valid = super().scoring(valid_user, valid_event, valid_data)

        print('Preprocessing Done')
        return X_v_train, X_t_train, X_valid, X_test, y_v_train, y_t_train, y_valid
