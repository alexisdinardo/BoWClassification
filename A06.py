#cell 0
#cell 1
import cv2
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

### FUNCTIONS PROVIDED TO STUDENTS ###

def dist_sqr(x,y):
    d, n = x.shape
    d, m = y.shape

    z    = x.T @ y
    x_2  = np.sum(x**2,0)[:,np.newaxis]
    y_2  = np.sum(y**2,0)[np.newaxis,:]

    for i in range(m):
        r = (x_2 + y_2[:,i] - 2*z[:,i][:,np.newaxis])
        z[:,i] = r.reshape(-1)

    return z

def eff_dist_sqr (O00000000OO0OO0O0 ,O0000OO0OOO00OO00 ):#line:1
    OO00OO00000000OO0 ,O0OO000OO000OO00O =O00000000OO0OO0O0 .shape #line:2
    O0O0OO0O0OO00OOO0 ,O0OO000OO000OO00O =O0000OO0OOO00OO00 .shape #line:3
    OO00O0O0O0O0OOO0O =(np .ones ((O0O0OO0O0OO00OOO0 ,1 ))@np .sum ((O00000000OO0OO0O0 **2 ).T ,0 )[np .newaxis ,:]).T +np .ones ((OO00OO00000000OO0 ,1 ))@np .sum ((O0000OO0OOO00OO00 **2 ).T ,0 )[np .newaxis ,:]-2 *(O00000000OO0OO0O0 @O0000OO0OOO00OO00 .T )#line:5
    OO00O0O0O0O0OOO0O =np .where (OO00O0O0O0O0OOO0O <0 ,0 ,OO00O0O0O0O0OOO0O )#line:6
    return OO00O0O0O0O0OOO0O #line:8
    
def compute_BOW_repr (OO0OOOO0O00O000OO ,O0OOO0OO0O000OOO0 ):#line:10
    OO0OOOO0O00O000OO =np .array (OO0OOOO0O00O000OO )#line:11
    O0O0OOOO00OOO0000 =O0OOO0OO0O000OOO0 .shape [0 ]#line:13
    O0O00O0O0OO0000OO =np .zeros ((1 ,O0O0OOOO00OOO0000 ))#line:14
    if OO0OOOO0O00O000OO .shape [0 ]==0 :#line:16
        return O0O00O0O0OO0000OO #line:17
    if OO0OOOO0O00O000OO .shape [1 ]<O0OOO0OO0O000OOO0 .shape [1 ]:#line:19
        return O0O00O0O0OO0000OO #line:20
    OOO00OOO0OOO000OO =eff_dist_sqr (OO0OOOO0O00O000OO ,O0OOO0OO0O000OOO0 )#line:22
    O0OOO0O00000O0O0O =np .argmin (OOO00OOO0OOO000OO ,1 )#line:24
    for O00OO0O0000O0O0OO in range (O0O0OOOO00OOO0000 ):#line:26
        O0O00O0O0OO0000OO [:,O00OO0O0000O0O0OO ]=np .sum (O0OOO0O00000O0O0O ==O00OO0O0000O0O0OO )#line:27
    O0O00O0O0OO0000OO =O0O00O0O0OO0000OO /np .sum (O0O00O0O0OO0000OO )#line:29
    return O0O00O0O0OO0000OO 

def load_split_dataset():
    path = 'scenes_lazebnik/'

    sift = cv2.SIFT_create(nfeatures=200)
    all_scenes = os.listdir(path)

    all_images = []
    all_sift = []
    all_labels = []
    all_train_ids = []
    all_test_ids = []
    
    for i in range(len(all_scenes)):
        scene_entry = all_scenes[i]
        scene_dir   = path + scene_entry
        data_for_this_scene = [dirx for dirx in os.listdir(scene_dir) if dirx.endswith('.jpg') ]
        this_cat_sift   = []
        this_cat_images = []

        for j in range(len(data_for_this_scene)):
            this_file = data_for_this_scene[j]
            img_path  = scene_dir + '/' + this_file 
            img       = cv2.imread(img_path, 0)#.astype(np.float32)
            this_cat_images.append(img)

        r = np.random.permutation(len(this_cat_images))[0:75]
        this_cat_images = [this_cat_images[i] for i in r]
        this_cat_sift   = [sift.detectAndCompute(img, None) for img in this_cat_images]

        r = np.random.permutation(len(this_cat_images))

        train_ids = len(all_labels) + r[0:50]
        test_ids  = len(all_labels) + r[50:75]

        all_train_ids.extend(train_ids)
        all_test_ids.extend(test_ids)

        all_labels.extend([i] * len(this_cat_images))
        all_sift.extend(this_cat_sift)
        all_images.extend(this_cat_images)

    train_images = [ all_images[i] for i in all_train_ids]
    train_sift   = [ all_sift[i] for i in all_train_ids]
    train_labels = [ all_labels[i] for i in all_train_ids]
    test_images  = [ all_images[i] for i in all_test_ids]
    test_sift    = [ all_sift[i] for i in all_test_ids]
    test_labels  = [ all_labels[i] for i in all_test_ids]

    means = np.loadtxt('means.out')
    return train_images, train_sift, train_labels, test_images, test_sift, test_labels, means

#cell 2
#cell 3
#cell 4
#cell 5
def compute_SPM_repr(img, features, means):
    keypoints, descriptors = features

    level_0 = compute_BOW_repr(descriptors, means)

    level_1 = []

    width = img.shape[1]
    height = img.shape[0]

    quadrants = [
        (0, height // 2, 0, width // 2),  # top left
        (0, height // 2, width // 2, width),  # top right
        (height // 2, height, 0, width // 2),  # bottom left
        (height // 2, height, width // 2, width)  # bottom right
    ]

    for i, (y_min, y_max, x_min, x_max) in enumerate(quadrants):
        quadrant_descriptors = [
            descriptors[j] for j in range(len(keypoints))
            if y_min <= keypoints[j].pt[1] < y_max and x_min <= keypoints[j].pt[0] < x_max
        ]

        if quadrant_descriptors:
            quadrant_hist = compute_BOW_repr(np.array(quadrant_descriptors), means).flatten()
        else:
            quadrant_hist = np.zeros(50)
        level_1.append(quadrant_hist)

    level_1 = np.vstack(level_1)

    pyramid = np.vstack([level_0.reshape(1, 50), level_1])
    return pyramid, level_0, level_1


#cell 6
#cell 7
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def find_labels_KNN(pyramids_train, labels_train, pyramids_test, k):
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(pyramids_train, labels_train)
    predicted_labels_test = knn.predict(pyramids_test)
    
    return predicted_labels_test


def find_labels_SVM(pyramids_train, labels_train, pyramids_test):
    svm = SVC(kernel='linear')
    svm.fit(pyramids_train, labels_train)
    predicted_labels_test = svm.predict(pyramids_test)
    
    return predicted_labels_test

#cell 8
#cell 9
from sklearn.metrics import accuracy_score

def compare_representations(train_images, train_sift, train_labels, test_images, test_sift, test_labels, means):
    return 0



def compare_classifiers(train_images, train_sift, train_labels, test_images, test_sift, test_labels, means):
    return 0

#cell 10
#cell 11
#cell 12
train_images, train_sift, train_labels, test_images, test_sift, test_labels, means = load_split_dataset()
compare_representations(train_images, train_sift, train_labels, test_images, test_sift, test_labels, means)
compare_classifiers(train_images, train_sift, train_labels, test_images, test_sift, test_labels, means)

