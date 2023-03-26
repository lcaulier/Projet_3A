import cv2
import re
import glob
import shutil
import numpy as np
import sys
import logging
from PIL import Image,ImageEnhance, ImageFilter
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Détection de zone de texte avec CV2
simage = r"D:/Users/leoca/Documents/Cours/Projet_3A/dataset/nutrition/nut_104.jpg"
img = cv2.imread(simage)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Binarisation de l'image
threshold=120
max_value=255 
threshold_stype=cv2.THRESH_BINARY 
#ret, img_binary = cv2.threshold(img_gray, threshold, max_value, threshold_stype)
#ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
img_binary = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,71,17)
img_binary2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,12)

#Détermination des zones de textes
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))
dilation = cv2.dilate(img_binary, rect_kernel, iterations = 1)

text = pytesseract.image_to_string(img_binary, lang='fra')
print(text)

#Affichage des images
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img_binary)
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2',img_binary2)
cv2.waitKey(0)
cv2.destroyAllWindows()



