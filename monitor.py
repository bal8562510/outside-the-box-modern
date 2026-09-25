import numpy as np


class Box:
    """
    A simple axis-aligned box.

    For every dimension we store:
        minimum value
        maximum value
    """

    def __init__(self, points):

        points = np.asarray(points)

        # Minimum value in every dimension
        self.lower = np.min(points, axis=0)

        # Maximum value in every dimension
        self.upper = np.max(points, axis=0)

    def contains(self, point):

        point = np.asarray(point)

        inside_lower = point >= self.lower
        inside_upper = point <= self.upper

        return np.all(inside_lower & inside_upper)


class BoxMonitor:
    """
    Creates one box for each class.
    """

    def __init__(self):

        self.boxes = {}

    def fit(self, representations, labels):

        representations = np.asarray(representations)
        labels = np.asarray(labels)

        classes = np.unique(labels)

        for class_id in classes:

            class_points = representations[labels == class_id]

            self.boxes[int(class_id)] = Box(class_points)

            print(
                f"Class {class_id}: "
                f"{len(class_points)} points"
            )

    def check(self, representation, predicted_class):

        if predicted_class not in self.boxes:
            return False

        box = self.boxes[predicted_class]

        return box.contains(representation)