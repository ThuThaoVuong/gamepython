import cv2
import mediapipe as mp
from pynput.keyboard import Key,Controller
class handDetector():
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.k=-1
        self.kb=Controller()
        self.tmp=''

    def findHands(self, img):
        # Chuyển từ BGR thành RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Đưa vào thư viện mediapipe
        results = self.hands.process(imgRGB)
        hand_lms = []

        if results.multi_hand_landmarks:
            # Vẽ landmark cho các bàn tay
            for handlm in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handlm, self.mpHands.HAND_CONNECTIONS)


            # Trích ra các toạ độ của khớp của các ngón tay
            firstHand = results.multi_hand_landmarks[0]
            h,w,_ = img.shape
            for id, lm in enumerate(firstHand.landmark):
                real_x, real_y = int(lm.x * w), int(lm.y * h)
                hand_lms.append([id, real_x, real_y])

            finger_start_index=[4,8,12,16,20]
            if len(hand_lms)>0:
                if hand_lms[4][1]>hand_lms[2][1]:
                    b=2
                elif hand_lms[8][2]>hand_lms[6][2]:
                    b=0
                elif hand_lms[20][2]>hand_lms[18][2]:
                    b=1
                else: b=3
                if b == 0:
                    self.kb.press(Key.left)
                    self.tmp=Key.left
                elif b == 1:
                    self.kb.press(Key.right)
                    self.tmp=Key.right
                elif b == 2:
                    self.kb.press(Key.up)
                    self.tmp=Key.up
                elif b == 3:
                    self.kb.press(Key.space)
                    self.tmp=Key.space

