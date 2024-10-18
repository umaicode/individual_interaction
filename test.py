import cv2
from deepface import DeepFace

# 감정을 3단계로 단순화한 레이블 정의
emotion_labels = [
    "Negative3",
    "Negative2",
    "Negative1",
    "Positive1",
    "Positive2",
    "Positive3",
]

# Haar Cascade 분류기 초기화
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# 감정을 단계별로 매핑하는 함수
def map_emotions(emotion):
    mapping = {
        "angry": "Negative3",
        "disgust": "Negative3",
        "fear": "Negative2",
        "sad": "Negative1",
        "neutral": "Positive1",
        "happy": "Positive2",
        "surprise": "Positive3",
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

    for x, y, w, h in faces:
        face_roi = frame[y : y + h, x : x + w]
        try:
            analysis = DeepFace.analyze(
                face_roi, actions=["emotion"], enforce_detection=False
            )
            dominant_emotion = analysis[0]["dominant_emotion"]
            emotion_label = map_emotions(dominant_emotion)

            color = {
                "Negative3": (0, 0, 139),
                "Negative2": (0, 0, 255),
                "Negative1": (0, 165, 255),
                "Positive1": (0, 255, 255),
                "Positive2": (0, 255, 0),
                "Positive3": (0, 128, 0),
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

    cv2.imshow("Real-time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
