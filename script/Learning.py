#!/Users/shohei/.pyenv/shims/python
# Learning.py

from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV

import NDCG


class Function:

    def __init__(self, X_train, X_test, y_train, y_test, model, parameters, cv=3):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.parameters = parameters
        self.cv = cv

    def grid_search(self):
        # GridSearchCVを用いたハイパーパラメーターのチューニング
        mod_grid = GridSearchCV(self.model, self.parameters, cv=self.cv)
        mod_grid.fit(self.X_train, self.y_train)
        best_model = mod_grid.best_estimator_
        print('Best Model Parameters:', mod_grid.best_params_)
        return best_model


class Learning(Function):

    def scoring(self):
        # ベストモデルを使用した時のNDCGを計算
        mod = super().grid_search()
        pred = mod.predict(self.X_test)
        print('NDCG:', NDCG.ndcg2(self.y_test.action_type.values, pred, k=20))
        print('Learning Done')
        return mod 
