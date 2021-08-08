# 파이썬에는 인자를 입력받고, 파싱하고, 예외처리하고, 
# 심지어 사용법(usage) 작성까지 자동으로 해주는 매우 편리한 모듈 argparse!!

import argparse
import os

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--print-number', type = int, 
                    help= 'an integer for printing repeatably')

args = parser.parse_args()

for i in range(args.print_number):
    print('print number{}'.format(i+1))


# 일단 ArgumentParser에 원하는 description을 입력하여 parser 객체를 생성한다. description 외에도 usage, default value 등을 지정할 수 있다.
# 그리고 add_argument() method를 통해 원하는 만큼 인자 종류를 추가한다.
# parse_args() method로 명령창에서 주어진 인자를 파싱한다.
# args라는 이름으로 파싱을 성공했다면 args.parameter 형태로 주어진 인자 값을 받아 사용할 수 있다.
