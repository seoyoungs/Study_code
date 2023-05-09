import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from PIL import Image


file_list=os.listdir('C:/dacon/230220/123_image/3_output_txt_')
file_list.sort()
file_na = 'C:/dacon/230220/123_image/3_output_txt_/'
label_save = 'C:/dacon/230220/123_image/3_output_txt/'


a,b,getnum=next(os.walk('C:/dacon/230220/123_image/3_output_txt'))
count = len(getnum)



img_path = 'C:/dacon/230220/119_output_image/out_frame127.jpg'
# label_path = 'C:/dacon/230220/119_output_txt/out_frame127.txt'
# label_save = 'C:/dacon/230220/119_output_txt_/out_frame127.txt'

image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # (500, 335)



for i in file_list:
    path=file_na+i
    bboxes = np.loadtxt(fname=path, delimiter=" ",ndmin=2)
    # print(bboxes)
    bboxes = np.roll(bboxes, 4, axis=1).tolist()  # [[0.641, 0.5705705705705706, 0.718, 0.8408408408408409, 6.0]]
    # print(bboxes)

    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=1),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
    )

    transformed = train_transforms(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    # print(transformed_bboxes)
    transformed_bboxes = np.roll(transformed_bboxes, 1, axis=1).tolist()
    # print(transformed_bboxes)

    out= label_save + i

    with open(out, "wb") as file:
        np.savetxt(out, transformed_bboxes, fmt = '%.3f', delimiter=' ') # fmt='%d'
    count += 1
    # for name in transformed_bboxes:
    #         file.write(name + '\n')

