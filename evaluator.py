from random import shuffle

class Evaluator(object):
    def __init__(self, recommender_class):
        self.recommender_class = recommender_class
    
    def computeMAP(self, dataset_handler, relevant_treshold=3.0, topN=5):
        k_cross = 5
        total_aps = 0.0
        total = 0
        set_ = dataset_handler.load_users_ratings()
        set_items = set_.items()
        shuffle(set_items)
        set_parts = [
            set_items[k*(len(set_items)/k_cross):(k+1)*(len(set_items)/k_cross) if k < k_cross-1 else len(set_items)]
            for k in range(k_cross)
        ]
        recommender = self.recommender_class(dataset_handler)
        for i in range(k_cross):
            test_set, train_set = dict(set_parts[i]), dict([user for part in set_parts[:i]+set_parts[i+1:] for user in part])
            recommender.train(train_set)
            for user_ratings in test_set.values():
                user_items = user_ratings.items()
                shuffle(user_items)
                parts = [
                    user_items[k*(len(user_items)/k_cross):(k+1)*(len(user_items)/k_cross) if k < k_cross-1 else len(user_items)]
                    for k in range(k_cross)
                ]
                for j in range(k_cross):
                    test, training = parts[j], [rat for part in parts[:j]+parts[j+1:] for rat in part]
                    relevant = [movieId for (movieId, rating) in test if rating >= relevant_treshold]
                    predicted = recommender.top(dict(training), topN=topN)
                    if relevant:
                        total_aps += self._computeAP(relevant, predicted)
                        total += 1
        return total_aps/total
    
    def _computeAP(self, relevant, predicted):
        ap = 0.0
        good_predictions = 0.0
        for i, item in enumerate(predicted):
            if item in relevant:
                good_predictions += 1
                ap += 1.0/(i+1) * good_predictions/(i+1)
        return ap