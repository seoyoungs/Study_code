import os
path = 'D:/image/nohuman_fire/candle-flame-dataset-master/yolo-labels/txts/'
file_path = 'burning-candle-light-260nw-507962296.txt'
file_list = os.listdir(path)
file_list_py = [file for file in file_list if file.endswith('.txt')]
# print(file_list)

# file = open('D:/image/nohuman_fire/candle-flame-dataset-master/yolo-labels/txts/burning-candle-light-260nw-507962296.txt')
# contents = file.read()
# replaced_contents = contents.replace('0 ', '2 ')
def replace_in_file(file_path, old_str, new_str):
    # 파일 읽어들이기
    fr = open(file_path, 'r')
    lines = fr.readlines()
    fr.close()
    
    # old_str -> new_str 치환
    fw = open(file_path, 'w')
    for line in lines:
        fw.write(line.replace(old_str, new_str))
    fw.close()

# for i in file_list_py:
#     # 호출: file1.txt 파일에서 comma(,) 없애기
#     replace_in_file(file_list_py, "0 ", "2 ")
#     dict_list.append(json.loads(line))
# df = pd.DataFrame(dict_list)
# 현재 디렉토리내에 각각의 파일을 출력
for file in file_list:
    replace_in_file(file_list_py, "0 ", "2 ")
    filepath = path + '/' + file
    print(file)

