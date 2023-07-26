import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def detect(dataPath, clf, img_dir=None):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    store the img path in tmp_dir
    copy the face image, resize it to 19*19, and convert into gray
    classify each face image if 1 draw green rectangle and 0 draw red rectangle
    image need to convert into rgb for ax.imshow    
    """
    with open(dataPath) as file:
        while True:
            d = file.readline()
            if d is None:
                break
            ds = d.split()
            if len(ds) != 2:
                break
            img_path = ds[0]
            num_of_faces = int(ds[1])

            cords = []
            for c in range(num_of_faces):
                tmp = file.readline()
                tmp = tmp.split()
                res = tuple(map(int, tmp))
                cords.append(res)
            imgs = []
            tmp_dir = 'data/detect/' + img_path
            image = cv2.imread(tmp_dir)
            for cord in cords:
                tmp_img = image[cord[1]:cord[1]+cord[3], cord[0]:cord[0]+cord[2]].copy()
                tmp_img = cv2.resize(tmp_img, (19, 19), interpolation=cv2.INTER_AREA)
                tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
                imgs.append(tmp_img)

            fig, ax = plt.subplots()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image_rgb)
            index = 0
            for img in imgs:
                if clf.classify(img):
                    rect = patches.Rectangle((cords[index][0], cords[index][1]), cords[index][2], cords[index][3], linewidth=1, edgecolor='g', facecolor='none')
                else:
                    rect = patches.Rectangle((cords[index][0], cords[index][1]), cords[index][2], cords[index][3], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                index += 1
            plt.show()
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
