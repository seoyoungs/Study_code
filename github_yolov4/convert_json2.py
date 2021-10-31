 

import json
import os
import cv2


def yolo2coco(image_dir_path, label_dir_path,save_file_name , is_normalized, image_coco_path):

    total = {}

    # make info
    info = {
            'description' : '',
            'url' : '',
            'version' : '',
            'year' : 2020,
            'contributor' : '',
            'data_created' : '2020-04-14 01:45:18.567988'
            }
    total['info'] = info

    # make licenses
    licenses_list = []
    licenses_0= {
            'id' : '1',
            'name' : 'your_name',
            'url' : 'your_name'
            }
    licenses_list.append(licenses_0)

    ''' if you want to add licenses, copy this code
    licenses_1 = {
        'id': '2',
        'name': 'your_name',
        'url': 'your_name'
    }
    licenses_list.append(licenses_1)
    '''

    total['licenses'] = licenses_list

    # make categories
    category_list = []
    class_0 = {
            'id':  1,
            'name' : 'defect',
            'supercategory' : 'None'
            }
    category_list.append(class_0)

    '''
    # if you want to add class
    class_1 = {
            'id':  2,
            'name' : 'defect',
            'supercategory' : 'None'
            }
    category_list.append(class_1)
    '''
    total['categories'] = category_list

    # make yolo to coco format

    # get images
    image_list = os.listdir(image_dir_path)
    print('image length : ', len(image_list))
    label_list = os.listdir(label_dir_path)
    print('label length : ',len(label_list))

    image_dict_list = []
    count = 0
    for image_name in image_list :
        img = cv2.imread(image_dir_path+image_name)
        
        image_dict = {
                'id' : count,
                'file_name' : image_name,
                'width' : img.shape[1],
                'height' : img.shape[0],
                'date_captured' : '2020-04-14 -1:45:18.567975',
                'license' : 1, # put correct license
                'coco_url' : '',
                'flickr_url' : ''
                }
        image_dict_list.append(image_dict)
        count += 1
    total['images'] = image_dict_list

    # make yolo annotation to coco format
    label_dict_list = []
    image_count = 0
    label_count = 0
    for image_name in image_list :
        img = cv2.imread(image_dir_path+image_name)
        label = open(label_dir_path+image_name[0:-4] + '.txt','r')
        
        if not os.path.isfile(label_dir_path + image_name[0:-4] + '.txt'): # debug code
            print('there is no label match with ',image_dir_path + image_name)
            return
        while True:
            line = label.readline()
            if not line:
                break
            # class_number, center_x,center_y,box_width,box_height = line.split()
            try: class_number, center_x,center_y,box_width,box_height = line.split()

            except ValueError as e: pass
            # should put bbox x,y,width,height
            # bbox x,y is top left

            if is_normalized :
                center_x =  int(float(center_x) * int(img.shape[1]))
                center_y = int(float(center_y) * int(img.shape[0]))
                box_width = int(float(box_width) * int(img.shape[1]))
                box_height = int(float(box_height) * int(img.shape[0]))
                top_left_x = center_x - int(box_width/2)
                top_left_y = center_y - int(box_height/2)

            if not is_normalized :
                center_x = float(center_x)
                center_y = float(center_y)
                box_width = float(box_width)
                box_height = float(box_height)
                top_left_x = center_x - int(box_width / 2)
                top_left_y = center_y - int(box_height / 2)

            bbox_dict = []
            bbox_dict.append(top_left_x)
            bbox_dict.append(top_left_y)
            bbox_dict.append(box_width)
            bbox_dict.append(box_height)

            # segmetation dict : 8 points to fill, x1,y1,x2,y2,x3,y3,x4,y4
            segmentation_list_list = []
            segmentation_list= []
            segmentation_list.append(bbox_dict[0])
            segmentation_list.append(bbox_dict[1])
            segmentation_list.append(bbox_dict[0] + bbox_dict[2])
            segmentation_list.append(bbox_dict[1])
            segmentation_list.append(bbox_dict[0]+bbox_dict[2])
            segmentation_list.append(bbox_dict[1]+bbox_dict[3])
            segmentation_list.append(bbox_dict[0])
            segmentation_list.append(bbox_dict[1] + bbox_dict[3])
            segmentation_list_list.append(segmentation_list)

            label_dict = {
                    'id' : label_count,
                    'image_id' : image_count,
                    'category_id' : int(class_number)+1,
                    'iscrowd' : 0,
                    'area' : int(bbox_dict[2] * bbox_dict[3]),
                    'bbox' : bbox_dict,
                    'segmentation' : segmentation_list_list
                    }
            label_dict_list.append(label_dict)
            label_count += 1
        label.close()
        # image_dir_coco= '/total_image/final_train_coco/'
        cv2.imwrite(image_coco_path + f'{str(image_count)}.jpg', img)  
        image_count += 1


    total['annotations'] = label_dict_list

    with open(save_file_name,'w',encoding='utf-8') as make_file :
        
        json.dump(total,make_file, ensure_ascii=False,indent='\t')
        


if __name__ == '__main__':
    image_dir_path = './total_image/total_val/'
    label_dir_path = './total_label/total_val/'
    save_file_name = './json/fianl_coco_val.json'
    is_normalized = True
    image_coco_path ='./total_image/fianl_coco_val/'
    # if you want to add more licenses or classes
    # add in code
    yolo2coco(image_dir_path, label_dir_path, save_file_name,is_normalized, image_coco_path)