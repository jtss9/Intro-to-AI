import numpy


class WeakClassifier:
    def __init__(self, feature, threshold=0, polarity=1):
        """
          Parameters:
            feature: The HaarFeature class.
            threshold: The threshold for the weak classifier.
            polarity: The polarity of the weak classifier.(1 or -1)
        """
        self.feature = feature
        self.threshold = threshold
        self.polarity = polarity
    
    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s" % (self.threshold, self.polarity, str(self.feature))
    
    def classify(self, x):
        """
        Classifies an integral image based on a feature f 
        and the classifiers threshold and polarity.
          Parameters:
            x: A numpy array with shape (m, n) representing the integral image.
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        if isinstance(self.feature.computeFeature(x), numpy.float64):
            return 1 if self.polarity * self.feature.computeFeature(x) < self.polarity * self.threshold else 0
        else:
            return 1 if self.polarity * self.feature.computeFeature(x)[0] < self.polarity * self.threshold else 0
    