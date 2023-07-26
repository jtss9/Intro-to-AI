import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """
    listdir function can get all files(images) in the given folder("face" or "non-face")
    os.path.join can get the path of each file and then read them respectively
    put the image and label to a tuple and append in the array(dataset)
    """
    dataset = []
    for i in os.listdir(os.path.join(dataPath, "face")):
        img = cv2.imread(os.path.join(dataPath, "face", i))
        data = (img, 1)
        dataset.append(data)
    for i in os.listdir(os.path.join(dataPath, "non-face")):
        img = cv2.imread(os.path.join(dataPath, "non-face", i))
        data = (img, 0)
        dataset.append(data)

    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)
    return dataset


# trainData = loadImages('data/train')
# cv2.waitKey(0)