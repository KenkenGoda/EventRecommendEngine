#!/Users/shohei/.pyenv/shims/python
import pandas as pd


class Feature(object):

    def __init__(self, index, user_data, event_data, hist_data):
        self.index = index
        self.user_data = user_data
        self.event_data = event_data
        self.hist_data = hist_data
