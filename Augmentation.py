import numpy as np
import cv2 # terminal에서 설치 먼저
from matplotlib import pyplot as plt
import os

for i in range(10000):
    print('./Picture/Seon_seokkyu/ssk{}.png'.format(i))

    if (os.path.isfile('./Picture/Seon_seokkyu/ssk{}.png'.format(i))):
        img = cv2.imread('./Picture/Seon_seokkyu/ssk{}.png'.format(i))
        num_rows, num_cols = img.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation30', img_rotation)
        cv2.imwrite('./Picture/Seon_seokkyu/ssk{}-30.png'.format(i),img_rotation)

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 60, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation60', img_rotation)
        cv2.imwrite('./Picture/Seon_seokkyu/ssk{}-60.png'.format(i),img_rotation)

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation90', img_rotation)
        cv2.imwrite('./Picture/Seon_seokkyu/ssk{}-90.png'.format(i),img_rotation)

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 120, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation120', img_rotation)
        cv2.imwrite('./Picture/Seon_seokkyu/ssk{}-120.png'.format(i),img_rotation)

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 150, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation150', img_rotation)
        cv2.imwrite('./Picture/Seon_seokkyu/ssk{}-150.png'.format(i),img_rotation)

        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 180, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        cv2.imshow('Rotation180', img_rotation)
        cv2.imwrite('./Picture/Seon_seokkyu/ssk{}-180.png'.format(i),img_rotation)
    else:
        continue