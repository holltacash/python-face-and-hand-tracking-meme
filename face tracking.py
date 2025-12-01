import cv2
import mediapipe as mp
import numpy as np
import os

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720

TONGUE_OUT_THRESHOLD = 0.03
FIST_CLOSED_THRESHOLD = 0.17

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1
)

# =========================
# DETECTION FUNCTIONS
# =========================
def is_tongue_out(face_landmarks):
    up = face_landmarks.landmark[13].y
    down = face_landmarks.landmark[14].y
    return abs(up - down) > TONGUE_OUT_THRESHOLD


def is_fist(hand_landmarks):
    tips = [4,8,12,16,20]  # thumb -> pinky
    avg_tip_y = sum(hand_landmarks[i].y for i in tips) / 5
    wrist_y = hand_landmarks[0].y
    return abs(avg_tip_y - wrist_y) < FIST_CLOSED_THRESHOLD


# =========================
# MAIN
# =========================
def main():
    # Load image assets
    apple = cv2.imread("apple.png")
    tongue_img = cv2.imread("appletongue.png")
    cringe_img = cv2.imread("im_crine.png")

    if apple is None or tongue_img is None or cringe_img is None:
        print("❌ Missing image files!")
        return

    apple = cv2.resize(apple, (WINDOW_WIDTH, WINDOW_HEIGHT))
    tongue_img = cv2.resize(tongue_img, (WINDOW_WIDTH, WINDOW_HEIGHT))
    cringe_img = cv2.resize(cringe_img, (WINDOW_WIDTH, WINDOW_HEIGHT))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed.")
        return

    current = apple.copy()  # DEFAULT IMAGE

    print("\n================ READY ================")
    print("😛 Tongue → AppleTongue")
    print("👊 Fist   → Cringe")
    print("🍎 No gesture → Apple Default")
    print("Press Q to quit.")
    print("=======================================\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = face_mesh.process(rgb)
        hand = hands.process(rgb)

        # ➤ Reset image EVERY frame (so blank = apple)
        current = apple.copy()

        # TONGUE
        if face.multi_face_landmarks:
            f = face.multi_face_landmarks[0]
            if is_tongue_out(f):
                current = tongue_img.copy()
                cv2.putText(frame, "TONGUE OUT!", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,(0,255,0),3)

        # FIST
        if hand.multi_hand_landmarks:
            h = hand.multi_hand_landmarks[0].landmark
            if is_fist(h):
                current = cringe_img.copy()
                cv2.putText(frame, "FIST DETECTED!", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3,(255,0,0),3)

        # DISPLAY
        cv2.imshow("Camera", frame)
        cv2.imshow("Meme", current)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
