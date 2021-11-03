import os
# import glob
  
# data_path = './dataset/labels/final_coco_train/'
# files_list = os.listdir data_path)

import os
import glob

# path = "./dataset/labels/final_coco_train/"
files = glob.glob('./dataset/labels/final_coco_train/*')

for i, f in enumerate(files):
    ftitle, fext = os.path.splitext(f)
    os.rename(f, ftitle + '_' + '{0:01d}'.format(i) + fext)




