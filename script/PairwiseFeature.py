#!/Users/shohei/.pyenv/shims/python
import pandas as pd
from copy import copy
from Feature import Feature


class PairwiseFeature(Feature):

    def resistration_period(self):
        # ユーザーが登録してからイベントが開催されるまでの日数を返す
        df = self.index.join(self.user_data).join(self.event_data)
        df_created = df.reset_index().set_index('created_on')
        df_start = df.reset_index().set_index('event_start_at')
        period = (df_start.index-df_created.index).days.values
        return self.index.reset_index().join(pd.DataFrame({'resistration_period':period})).set_index(['user_id', 'event_id'])

    def from_at_prefecture(self):
        # 居住地と開催地が一致していれば1、異なれば0を返す
        df = self.index.join(self.user_data).join(self.event_data)
        return pd.DataFrame((df.residence==df.prefecture)*1, columns=['from_at_prefecture'])

    def at_user_hist(self):
        # 各ユーザが過去に行ったイベントで最も多かった開催地域と、ユーザーの居住地が一致していれば1、異なれば0を返す
        df = copy(self.user_data)
        df = df.join(self.hist_data.groupby('user_id').prefecture.agg(lambda x:x.value_counts().index[0]))
        return pd.DataFrame((df.prefecture==df.residence)*1, columns=['at_user_hist'])

    def action_hist_count(self):
        # 過去のログの行動別回数を返す
        return pd.get_dummies(self.hist_data.action_type, prefix='action_hist_pairwise').groupby(['user_id', 'event_id']).count()

    def action_hist_sum(self):
        # 過去のログの行動の合計値を返す
        return self.hist_data.groupby(['user_id', 'event_id']).action_type.sum().rename('action_hist_pairwise_sum')


