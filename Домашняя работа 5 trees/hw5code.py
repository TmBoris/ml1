import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    n = len(target_vector)

    # sort feature vector for iterate throw thresholds
    order = np.argsort(feature_vector)
    fv = np.array(feature_vector)[order]
    tv = np.array(target_vector)[order]

    # keep only unique values in feature_vector
    fv, counts = np.unique(fv, return_counts=True)

    # calculating ginis
    R_l = np.array(list(range(1, n)))
    tv_from_left_sum = np.cumsum(tv)
    p_1_l = tv_from_left_sum[:-1] / R_l
    left_gini = 1 - (p_1_l) ** 2 - (1 - p_1_l) ** 2
    R_r = n - R_l
    tv_from_right_sum = tv_from_left_sum[-1] - tv_from_left_sum
    p_1_r = tv_from_right_sum[:-1] / R_r
    right_gini = 1 - (p_1_r) ** 2 - (1 - p_1_r) ** 2
    pre_ginis = -R_l / n * left_gini - R_r / n * right_gini
    ginis = pre_ginis[np.cumsum(counts)[:-1] - 1]

    # calculating thresholds
    thresholds = (np.hstack((fv, np.array([0]))) + np.hstack((np.array([0]), fv)))[1:-1] / 2

    # searching for best gini and threshold
    ind = np.argmax(ginis)
    gini_best = ginis[ind]
    threshold_best = thresholds[ind]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    
    def set_params(self, **params):
        return super().set_params(**params)

    def _fit_node(self, sub_X, sub_y, node, depth):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            # add stop conditions here
            if len(feature_vector) < self.min_samples_split:
                continue

            if np.all(feature_vector == feature_vector[0]):
                continue

            if self.max_depth is not None and depth >= self.max_depth:
                continue


            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            possible_split = feature_vector < threshold
            if gini_best is None or \
              (gini > gini_best and self.min_samples_leaf <= np.sum(possible_split) <= len(feature_vector) - self.min_samples_leaf):
                feature_best = feature
                gini_best = gini
                split = possible_split
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                            filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        elif self.feature_types[node['feature_split']] == 'real':
            if x[node['feature_split']] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        elif self.feature_types[node['feature_split']] == 'categorical':
            if np.isin(x[node['feature_split']], node['categories_split']):
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
