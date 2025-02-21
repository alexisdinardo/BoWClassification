
import numpy as np
import cv2 
import A06


def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))

   return mse

def mse2(img1, img2):
   h, w , l = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w*l))

   return mse


def test01() :  #compute_SPM_repr  TEST01
   tolerance=1e-3
   sift = cv2.SIFT_create(nfeatures=200)
   img_path='images/autograd01.jpg'
   img= cv2.imread(img_path, 0)
   features = sift.detectAndCompute(img, None)
   means = np.loadtxt('means.out')
   pyr, lev0, lev1 = A06.compute_SPM_repr(img, features, means)


   correct_pyr = np.loadtxt('autograd/pyr01.out')
   correct_lev0 = np.loadtxt('autograd/lvl0_01.out')
   correct_lev1 = np.loadtxt('autograd/lvl1_01.out')

   print("level 0:")
   print(abs(lev0-correct_lev0))
   print('\n')

   print("level 1:")
   print(abs(lev1-correct_lev1))
   print('\n')

   print(abs(pyr-correct_pyr))
   print('\n')


   flag=True
   flag &= ((abs(pyr-correct_pyr) ).sum()<tolerance )
   flag &= ((abs(lev0-correct_lev0) ).sum()<tolerance )
   flag &= ((abs(lev1-correct_lev1) ).sum()<tolerance )


   return  flag

def test02() :  #compute_SPM_repr  TEST02
   tolerance=1e-3
   sift = cv2.SIFT_create(nfeatures=200)
   img_path='images/autograd02.jpg'
   img= cv2.imread(img_path, 0)
   features = sift.detectAndCompute(img, None)
   means = np.loadtxt('means.out')
   pyr, lev0, lev1 = A06.compute_SPM_repr(img, features, means)


   correct_pyr = np.loadtxt('autograd/pyr02.out')
   correct_lev0 = np.loadtxt('autograd/lvl0_02.out')
   correct_lev1 = np.loadtxt('autograd/lvl1_02.out')


   flag=True
   flag &= ((abs(pyr-correct_pyr) ).sum()<tolerance )
   flag &= ((abs(lev0-correct_lev0) ).sum()<tolerance )
   flag &= ((abs(lev1-correct_lev1) ).sum()<tolerance )


   return  flag

def test03() :  #compute_SPM_repr  TEST03
   tolerance=1e-3
   sift = cv2.SIFT_create(nfeatures=200)
   img_path='images/autograd03.jpg'
   img= cv2.imread(img_path, 0)
   features = sift.detectAndCompute(img, None)
   means = np.loadtxt('means.out')
   pyr, lev0, lev1 = A06.compute_SPM_repr(img, features, means)


   correct_pyr = np.loadtxt('autograd/pyr03.out')
   correct_lev0 = np.loadtxt('autograd/lvl0_03.out')
   correct_lev1 = np.loadtxt('autograd/lvl1_03.out')


   flag=True
   flag &= ((abs(pyr-correct_pyr) ).sum()<tolerance )
   flag &= ((abs(lev0-correct_lev0) ).sum()<tolerance )
   flag &= ((abs(lev1-correct_lev1) ).sum()<tolerance )


   return  flag

def test04() :  #compute_SPM_repr  TEST04
   tolerance=1e-3
   sift = cv2.SIFT_create(nfeatures=200)
   img_path='images/autograd04.jpg'
   img= cv2.imread(img_path, 0)
   features = sift.detectAndCompute(img, None)
   means = np.loadtxt('means.out')
   pyr, lev0, lev1 = A06.compute_SPM_repr(img, features, means)


   correct_pyr = np.loadtxt('autograd/pyr04.out')
   correct_lev0 = np.loadtxt('autograd/lvl0_04.out')
   correct_lev1 = np.loadtxt('autograd/lvl1_04.out')


   flag=True
   flag &= ((abs(pyr-correct_pyr) ).sum()<tolerance )
   flag &= ((abs(lev0-correct_lev0) ).sum()<tolerance )
   flag &= ((abs(lev1-correct_lev1) ).sum()<tolerance )


   return  flag

if __name__ == "__main__":
    results = [test01(), test02(), test03(), test04()]
    for i, result in enumerate(results, start=1):
        print(f"Test {i}: {'Passed' if result else 'Failed'}")