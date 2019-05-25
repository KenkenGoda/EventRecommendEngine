#!/Users/shohei/.pyenv/shims/python
import pandas as pd
from copy import copy
from Feature import Feature


class UserFeature(Feature):

    def age(self):
        # ユーザーの年齢を返す
        return self.user_data.age

    def sex(self):
        # ユーザーの性別を返す
        return self.user_data.gender

    def residence(self):
        # ユーザーの居住地を返す
        return pd.get_dummies(self.user_data.residence, prefix='from', prefix_sep=' ')

    def action_hist_user_count(self):
        # ユーザーの過去の行動別回数を返す
        return pd.get_dummies(self.hist_data.action_type, prefix='action_hist_user').groupby('user_id').count()

    def action_hist_user_sum(self):
        # ユーザーの過去の行動の合計値を返す
        return self.hist_data.groupby('user_id').action_type.sum().rename('action_hist_user_sum')





