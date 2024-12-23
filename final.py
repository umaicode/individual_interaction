import cv2
from deepface import DeepFace

# 감정을 3단계로 단순화한 레이블 정의
emotion_labels = ["Negative", "Neutral", "Positive"]

# Haar Cascade 분류기 초기화
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 가면 이미지 불러오기
mask_images = {
    "Negative": cv2.imread("negative_face.png"),
    "Neutral": cv2.imread("neutral_face.png"),
    "Positive": cv2.imread("happy_face.png"),
}

current_emotion = "Neutral"


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
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # 가장 큰 얼굴만 선택
    if len(faces) > 0:
        # 얼굴 크기에 따라 정렬하여 가장 큰 얼굴만 선택
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]  # 가장 큰 얼굴만 사용

        face_roi = frame[y : y + h, x : x + w]
        try:
            # 감정 분석 수행
            analysis = DeepFace.analyze(
                face_roi, actions=["emotion"], enforce_detection=False
            )
            dominant_emotion = analysis[0]["dominant_emotion"]
            emotion_label = map_emotions(dominant_emotion)

            if current_emotion != emotion_label:
                current_emotion = emotion_label

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

    # 감정에 맞는 이미지 표시
    if current_emotion in mask_images and mask_images[current_emotion] is not None:
        cv2.imshow("Mask Image", mask_images[current_emotion])

    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
