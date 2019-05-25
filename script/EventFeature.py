#!/Users/shohei/.pyenv/shims/python
import pandas as pd
from Feature import Feature


class EventFeature(Feature):

    def female_lower(self):
        # 女性の年齢の下限を返す
        return pd.get_dummies(self.event_data.female_age_lower, prefix='fl')

    def female_upper(self):
        # 女性の年齢の上限を返す
        return pd.get_dummies(self.event_data.female_age_upper, prefix='fu')

    def male_lower(self):
        # 男性の年齢の下限を返す
        return pd.get_dummies(self.event_data.male_age_lower, prefix='ml')

    def male_upper(self):
        # 男性の年齢の上限を返す
        return pd.get_dummies(self.event_data.male_age_upper, prefix='mu')

    def start_weekday(self):
        # イベントの開催日の曜日を返す
        index = self.event_data.index
        df = self.event_data.reset_index().set_index('event_start_at')
        weekday = df.index.weekday_name.values
        df_weekday = pd.DataFrame({'weekday':weekday}, index=index)
        return pd.get_dummies(df_weekday.weekday, prefix='on', prefix_sep=' ')

    def venue(self):
        # イベントの開催地を返す
        return pd.get_dummies(self.event_data.prefecture, prefix='at', prefix_sep=' ')

    def female_price(self):
        # 女性の参加費用を返す
        return self.event_data.female_price

    def male_price(self):
        # 男性の参加費用を返す
        return self.event_data.male_price

    def interest(self):
        # イベントの興味対象を返す
        return pd.get_dummies(self.event_data.interest, prefix='', prefix_sep='')

    def action_hist_event_count(self):
        # イベントの過去の行動別回数を返す
        return pd.get_dummies(self.hist_data.action_type, prefix='action_hist_event').groupby('event_id').count()

    def action_hist_event_sum(self):
        # イベントの過去の行動の合計値を返す
        return self.hist_data.groupby('event_id').action_type.sum().rename('action_hist_event_sum')
