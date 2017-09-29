import numpy as np
import cv2 # terminal에서 설치 먼저
from matplotlib import pyplot as plt

for i in range(10):
    #if(cv2.imread('./Picture/Lee_inhyun/Inhyun{}.png'.format(i)))):
    if i%2 == 0:
        print('./Picture/Lee_inhyun/Inhyun{}.png'.format(i))
    else:
        continue