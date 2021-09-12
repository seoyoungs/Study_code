# 실행방법 : 터미널 들어가서 cd seleniumenv 파일 접속 -> python google.py

# 모듈 가져오기
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import pathlib

# 크롬 열고 검색하기
driver = webdriver.Chrome("C:/Users/USER/Downloads/chromedriver_win32/chromedriver.exe")
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&ogbl")
elem = driver.find_element_by_name("q")
elem.send_keys("fire broke") #검색 
elem.send_keys(Keys.RETURN)

# 페이지 끝까지 스크롤 내리기
SCROLL_PAUSE_TIME = 1

# 스크롤 깊이 측정하기
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # 스크롤 끝까지 내리기
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 페이지 로딩 기다리기
    time.sleep(SCROLL_PAUSE_TIME)

    # 더 보기 요소 있을 경우 클릭하기
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
         driver.find_element_by_css_selector(".mye4qd").click()
        except:
            break
    last_height = new_height

#이미지 찾고, 다운받기
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd")
count = 1
for image in images:
    try:                
        image.click()
        time.sleep(1)
        #orgimage = driver.find_element_by_css_selector(".n3VNCb")
        #orgimage.screenshot_as_png("red_bollard/red_bollard" + str(count) + ".png")                
        imgUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img").get_attribute('src')
        path = pathlib.Path(imgUrl)
        ext = path.suffix
        ext = ext.lower()        
        print(imgUrl)
        print(ext)
        if '3m.com' in imgUrl :
            print("pass")
            continue
        if 'mouser' in imgUrl :
            print("pass")
            continue   
        if 'amazon' in imgUrl :
            print("pass")
            continue   
        if 'grainger' in imgUrl :
            print("pass")
            continue   
        if ext == '.jfif' :
            print("pass")
            continue           
        urllib.request.urlretrieve(imgUrl, "C:/tf2/image/fire broke/firebroke_" + str(count) + ".jpg")                
        count = count + 1
        print("save " ,imgUrl)
        print("count = ", count)
    except:
        pass

#프로그램 종료
driver.close()
