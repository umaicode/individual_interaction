import os
import sys
import cv2  # OpenCV 사용
import requests

# 네이버 API 클라이언트 정보
client_id = "LW9Mp0aTi71_VGx97bCw"
client_secret = "Fcd9Krs0yZ"
url = "https://openapi.naver.com/v1/vision/face"

# 웹캠을 통해 이미지 캡처
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    sys.exit()

ret, frame = cap.read()  # 웹캠에서 프레임 읽기
if ret:
    # 캡처한 이미지를 파일로 저장
    img_path = "captured_image.jpg"
    cv2.imwrite(img_path, frame)

    # 네이버 API로 이미지 전송
    files = {"image": open(img_path, "rb")}
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    response = requests.post(url, files=files, headers=headers)
    rescode = response.status_code

    # 결과 출력
    if rescode == 200:
        print("응답: ", response.text)
    else:
        print("Error Code:", rescode)

    # 웹캠 창 닫기
    cap.release()
    cv2.destroyAllWindows()
else:
    print("이미지를 캡처할 수 없습니다.")
