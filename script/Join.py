#!/Users/shohei/.pyenv/shims/python
import pandas as pd
from Feature import Feature
from UserFeature import UserFeature
from EventFeature import EventFeature
from PairwiseFeature import PairwiseFeature


class Join(Feature):

    def extract(self):
        # 特徴量DFを作成
        user = UserFeature(self.index, self.user_data, self.event_data, self.hist_data)
        event = EventFeature(self.index, self.user_data, self.event_data, self.hist_data)
        pairwise = PairwiseFeature(self.index, self.user_data, self.event_data, self.hist_data)
        X = self.index
        X = X.join(user.age())
        X = X.join(user.sex())
        #X = X.join(user.residence())
        X = X.join(user.action_hist_user_count())
        X = X.join(user.action_hist_user_sum())
        X = X.join(event.female_lower())
        X = X.join(event.female_upper())
        X = X.join(event.male_lower())
        X = X.join(event.male_upper())
        X = X.join(event.start_weekday())
        #X = X.join(event.venue())
        X = X.join(event.female_price())
        X = X.join(event.male_price())
        X = X.join(event.interest())
        X = X.join(event.action_hist_event_count())
        X = X.join(event.action_hist_event_sum())
        X = X.join(pairwise.resistration_period())
        X = X.join(pairwise.from_at_prefecture())
        X = X.join(pairwise.at_user_hist())
        X = X.join(pairwise.action_hist_count())
        X = X.join(pairwise.action_hist_sum())
        return X.fillna(0)
