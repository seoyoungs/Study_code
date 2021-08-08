# 웰시코기 detector
# https://towardsdatascience.com/custom-object-detection-using-tensorflow-from-scratch-e61da2e10087

# =============== 웹 크롤링 100개 한정
from google_images_download import google_images_download
# response = google_images_download.googleimagesdownload()   #class instantiation
# arguments = {"keywords":"welsh corgi dog", "limit":100, "print_urls":True, "format":"jpg"}   #creating list of arguments
# response.download(arguments)   #passing the arguments to the function


# =============== 파일 이름 변환
import os
# path = 'C:/Users/USER/anaconda3/Lib/site-packages/tensorflow/models/images/'
i = 1

HOME_DIR = 'C:/Users/USER/anaconda3/'
file_path = os.path.join(HOME_DIR, r'Lib/site-packages/tensorflow/models/images/')
# file_path = 'C:/Users/USER/anaconda3/Lib/site-packages/tensorflow/models/images/'
file_names = os.listdir(file_path)
j = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    j += 1




