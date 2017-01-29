from random import shuffle

class Evaluator(object):
    def __init__(self, recommender):
        self.recommender = recommender
    
    def computeMAP(self, relevant_treshold=3.0, topN=5):
        k_cross = 5
        total_aps = 0.0
        total = 0
        users_ratings = self.recommender.dataset_handler.load_users_ratings()
        training_data = {user: user_ratings for user, user_ratings in users_ratings.items() if user < 0.8*len(users_ratings)}
        test_data = {user: user_ratings for user, user_ratings in users_ratings.items() if user not in training_data}
        self.recommender.train(training_data)
        for user_ratings in test_data.values():
            user_items = user_ratings.items()
            shuffle(user_items)
            parts = [
                user_items[k*(len(user_items)/k_cross):(k+1)*(len(user_items)/k_cross) if k < k_cross-1 else len(user_items)]
                for k in range(k_cross)
            ]
            for i in range(k_cross):
                test, training = parts[i], [rat for part in parts[:i]+parts[i+1:] for rat in part]
                relevant = [movieId for (movieId, rating) in test if rating >= relevant_treshold]
                user_profile = self.recommender.create_user_profile(dict(training))
                predicted = self.recommender.top(user_profile, topN=topN)
                if relevant:
                    total_aps += self._computeAP(relevant, predicted)
                    total += 1
        return total_aps/total
    
    def computeRMSE(self):
        k_cross = 5
        rse = 0.0
        total = 0
        users_ratings = self.recommender.dataset_handler.load_users_ratings()
        training_data = {user: user_ratings for user, user_ratings in users_ratings.items() if user < 0.8*len(users_ratings)}
        test_data = {user: user_ratings for user, user_ratings in users_ratings.items() if user not in training_data}
        self.recommender.train(training_data)
        for user_ratings in test_data.values():
            user_items = user_ratings.items()
            shuffle(user_items)
            parts = [
                user_items[k*(len(user_items)/k_cross):(k+1)*(len(user_items)/k_cross) if k < k_cross-1 else len(user_items)]
                for k in range(k_cross)
            ]
            for i in range(k_cross):
                test, training = parts[i], [rat for part in parts[:i]+parts[i+1:] for rat in part]
                user_profile = self.recommender.create_user_profile(dict(training))
                for (movieId, rating) in test:
                    predicted = self.recommender.predict_rating(user_profile, movieId)
                    if predicted > 0:
                        rse += (rating - predicted)**2
                        total += 1
        return rse/total
    
    def _computeAP(self, relevant, predicted):
        ap = 0.0
        good_predictions = 0.0
        for i, item in enumerate(predicted):
            if item in relevant:
                good_predictions += 1
                ap += 1.0/(i+1) * good_predictions/(i+1)
        return ap