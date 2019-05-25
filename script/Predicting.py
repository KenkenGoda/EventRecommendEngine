#!/Users/shohei/.pyenv/shims/python
# Predicting.py

import pandas as pd


class Function:

    def __init__(self, X_train, y_train, X_test, model):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model

    def convert(self, data):
        # 関連度を0,1,2,3のいずれかに変換
        return data.apply(lambda x:0 if x<0.5 else(1 if 0.5<=x<1.5 else(2 if 1.5<=x<2.5 else 3)))


class Predicting(Function):

    def predicting(self):
        # ベストモデルを用いて再学習し、予測データを計算
        self.model.fit(self.X_train, self.y_train)
        pred = self.model.predict(self.X_test)
        df_pred = pd.DataFrame(pred, index=self.X_test.index, columns=['action_type']).reset_index('event_id')
        user_id = self.X_test.index[:0].levels[0].values
        pred_data = pd.concat([df_pred.loc[i].sort_values(by='action_type', ascending=False).iloc[:20, :].reset_index() for i in user_id], ignore_index=True)
        pred_data.action_type = super().convert(pred_data.action_type)
        print('Predicting Done')
        return pred_data
