#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import numpy as np
import glob, os
import matplotlib.pyplot as plt


# In[2]:


# Resize gambar Utama

img = cv.imread('aksarajawa.jpg')

scale_percent = 11
width = int(img.shape[0]*scale_percent / 100)
height = int(img.shape[1]*scale_percent / 100)
dimension = (height,width)

rsz = cv.resize(img,dimension)
print("Ukuran Asli :",img.shape)
print("Ukuran Resize :",rsz.shape)


# In[4]:


#Grayscale pada gambar utama

grayFrame = cv.cvtColor(rsz, cv.COLOR_RGB2GRAY)
img2 = grayFrame.copy()
plt.imshow(grayFrame, cmap='gray')


# In[5]:


# Resize Inputan Template menjadi 15x15

template = cv.imread('Input/inputha.jpg')

scale_percent1 = 14
scale_percent2 = 11
# [1] = height dan [0] = width
width = int(template.shape[0]*scale_percent1 / 100)
height = int(template.shape[1]*scale_percent2 / 100)
dimension = (height,width)

resized = cv.resize(template,dimension)

print("Ukuran asli :",template.shape)
print("Ukuran Resize :",resized.shape)


# In[6]:


#Grayscale pada Inputan Template

grayTempl = cv.cvtColor(resized, cv.COLOR_RGB2GRAY)
plt.imshow(grayTempl, cmap='gray')
cv.imwrite('Output/new_ha.jpg',resized)


# In[7]:


# Ekstraksi Fitur Operasi OR

masukan = cv.imread('Output/new_ha.jpg')
bawaan = cv.imread('aksara/baris1/rsz_ha.jpg')

bit_or = cv.bitwise_or(bawaan, masukan)


# In[8]:


plt.imshow(masukan)

cv.imshow("Input",masukan)

cv.waitKey(0)
cv.destroyAllWindows()


# In[9]:


plt.imshow(bawaan)

cv.imshow("Bawaan",bawaan)

cv.waitKey(0)
cv.destroyAllWindows()


# In[10]:


plt.imshow(bit_or)

cv.imshow("Image OR",bit_or)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('Output/ha_OR.jpg',bit_or)


# In[11]:


# Baris 1

t1 = cv.imread('Output/ha_OR.jpg',0)
w, h = t1.shape[::-1]
t2 = cv.imread('aksara/baris1/rsz_na.jpg',0)
w, h = t2.shape[::-1]
t3 = cv.imread('aksara/baris1/rsz_ca.jpg',0)
w, h = t3.shape[::-1]
t4 = cv.imread('aksara/baris1/rsz_ra.jpg',0)
w, h = t4.shape[::-1]
t5 = cv.imread('aksara/baris1/rsz_ka.jpg',0)
w, h = t5.shape[::-1]


# In[12]:


# Baris 2

t6 = cv.imread('aksara/baris2/rsz_da.jpg',0)
w, h = t6.shape[::-1]
t7 = cv.imread('aksara/baris2/rsz_ta.jpg',0)
w, h = t7.shape[::-1]
t8 = cv.imread('aksara/baris2/rsz_sa.jpg',0)
w, h = t8.shape[::-1]
t9 = cv.imread('aksara/baris2/rsz_wa.jpg',0)
w, h = t9.shape[::-1]
t10 = cv.imread('aksara/baris2/rsz_la.jpg',0)
w, h = t10.shape[::-1]


# In[13]:


# Baris 3

t11 = cv.imread('aksara/baris3/rsz_pa.jpg',0)
w, h = t11.shape[::-1]
t12 = cv.imread('aksara/baris3/rsz_dha.jpg',0)
w, h = t12.shape[::-1]
t13 = cv.imread('aksara/baris3/rsz_ja.jpg',0)
w, h = t13.shape[::-1]
t14 = cv.imread('aksara/baris3/rsz_ya.jpg',0)
w, h = t14.shape[::-1]
t15 = cv.imread('aksara/baris3/rsz_nya.jpg',0)
w, h = t15.shape[::-1]


# In[14]:


# Baris 4

t16 = cv.imread('aksara/baris4/rsz_ma.jpg',0)
w, h = t16.shape[::-1]
t17 = cv.imread('aksara/baris4/rsz_ga.jpg',0)
w, h = t17.shape[::-1]
t18 = cv.imread('aksara/baris4/rsz_ba.jpg',0)
w, h = t18.shape[::-1]
t19 = cv.imread('aksara/baris4/rsz_tha.jpg',0)
w, h = t19.shape[::-1]
t20 = cv.imread('aksara/baris4/rsz_nga.jpg',0)
w, h = t20.shape[::-1]


# In[15]:


# Cek Matching Hanya Menggunakan Inputan

templates = cv.imread('Output/ha_OR.jpg',0)
w, h = templates.shape[::-1]

methods = ['cv.TM_CCORR_NORMED']

for meth in methods:
      img = img2.copy()
      method = eval(meth)
      # Apply template Matching (Template Correlation)
      res = cv.matchTemplate(img,templates,method)
      min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    

      top_left = max_loc
      bottom_right = (top_left[0] + w, top_left[1] + h)
      cv.rectangle(img,top_left, bottom_right, 0, 1)
      plt.subplot(121),plt.imshow(res,cmap = 'gray')
      plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
      plt.subplot(122),plt.imshow(img,cmap = 'gray')
      plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
      plt.suptitle(meth)
      plt.show()


# In[16]:


# Cek Matching Keseluruhan

templates = [t1, t2, t3, t4, t5,
             t6, t7, t8, t9, t10,
             t11, t12, t13, t14, t15,
             t16, t17, t18, t19, t20]

methods = ['cv.TM_CCORR_NORMED']

for templ in templates :

    for meth in methods:
      img = img2.copy()
      method = eval(meth)
      # Apply template Matching (Template Correlation)
      res = cv.matchTemplate(img,templ,method)
      min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    

      top_left = max_loc
      bottom_right = (top_left[0] + w, top_left[1] + h)
      cv.rectangle(img,top_left, bottom_right, 0, 1)
      plt.subplot(121),plt.imshow(res,cmap = 'gray')
      plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
      plt.subplot(122),plt.imshow(img,cmap = 'gray')
      plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
      plt.suptitle(meth)
      plt.show()


# In[ ]:




