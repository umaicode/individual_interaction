import cv2
from deepface import DeepFace
import numpy as np

# 감정을 3단계로 단순화한 레이블 정의
emotion_labels = ["Negative", "Neutral", "Positive"]

# Haar Cascade 분류기 초기화
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 가면 이미지 불러오기 (4K 사이즈 가면 이미지)
mask_images = {
    "Negative": cv2.imread("disgust_face.png"),  # 4K 이미지
    "Neutral": cv2.imread("neutral_face.png"),  # 4K 이미지
    "Positive": cv2.imread("happy_face.png"),  # 4K 이미지
}

current_emotion = "Neutral"
alpha = 1.0  # 투명도 (1: 불투명, 0: 완전 투명)
fade_duration = 30  # 페이드 인/아웃 지속 시간 (프레임 수)
mask_alpha = 0.0  # 마스크 이미지의 초기 투명도


# 감정을 단계별로 매핑하는 함수
def map_emotions(emotion):
    mapping = {
        "angry": "Negative",
        "fear": "Negative",
        "disgust": "Negative",
        "neutral": "Neutral",
        "happy": "Positive",
    }
    return mapping.get(emotion, "Neutral")


# 비디오 캡처 초기화
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        face_roi = frame[y : y + h, x : x + w]
        try:
            analysis = DeepFace.analyze(
                face_roi, actions=["emotion"], enforce_detection=False
            )
            dominant_emotion = analysis[0]["dominant_emotion"]
            emotion_label = map_emotions(dominant_emotion)

            if current_emotion != emotion_label:
                current_emotion = emotion_label
                mask_alpha = 0.0  # 새로운 감정으로 바뀌면 투명도를 0으로 초기화

            color = {
                "Negative": (0, 0, 255),
                "Neutral": (0, 255, 0),
                "Positive": (0, 128, 0),
            }[emotion_label]

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                emotion_label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        except Exception as e:
            print(f"Error analyzing face: {e}")

    # 얼굴 감지 창 (기본 웹캠 크기)
    cv2.imshow("Real-time Emotion Detection", frame)

    # 감정에 맞는 마스크 이미지 페이드 인/아웃 처리 (4K 이미지)
    if current_emotion in mask_images and mask_images[current_emotion] is not None:
        mask_image = mask_images[current_emotion]

        # 알파값(투명도)을 증가시켜 마스크 이미지가 서서히 나타나는 효과
        if mask_alpha < 1.0:
            mask_alpha += 1.0 / fade_duration  # 서서히 나타나는 효과
        else:
            mask_alpha = 1.0

        # 빈 이미지 생성 (4K 사이즈로)
        empty_image = np.zeros_like(mask_image)

        # 투명도를 적용하여 마스크 이미지 블렌딩
        blended_mask = cv2.addWeighted(
            mask_image, mask_alpha, empty_image, 1 - mask_alpha, 0
        )

        # 별도의 창에서 4K 크기의 마스크 이미지 표시
        cv2.imshow("Mask Image", blended_mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
