import cv2
import mediapipe as mp
import pygame
from pygame import mixer
from fighter2 import Fighter

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --- Pygame setup ---
mixer.init()
pygame.init()

# Game window
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Brawler with Hand Control")

clock = pygame.time.Clock()
FPS = 60

# Colours
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# Game variables
intro_count = 3
last_count_update = pygame.time.get_ticks()
score = [0, 0]  # [P1, P2]
round_over = False
ROUND_OVER_COOLDOWN = 2000

# Fighter variables
WARRIOR_SIZE = 162
WARRIOR_SCALE = 4
WARRIOR_OFFSET = [72, 56]
WARRIOR_DATA = [WARRIOR_SIZE, WARRIOR_SCALE, WARRIOR_OFFSET]
WIZARD_SIZE = 250
WIZARD_SCALE = 3
WIZARD_OFFSET = [112, 107]
WIZARD_DATA = [WIZARD_SIZE, WIZARD_SCALE, WIZARD_OFFSET]

# Load assets
bg_image = pygame.image.load("assets/images/background/background.jpg").convert_alpha()
warrior_sheet = pygame.image.load("assets/images/warrior/Sprites/warrior.png").convert_alpha()
wizard_sheet = pygame.image.load("assets/images/wizard/Sprites/wizard.png").convert_alpha()
victory_img = pygame.image.load("assets/images/icons/victory.png").convert_alpha()

WARRIOR_ANIMATION_STEPS = [10, 8, 1, 7, 7, 3, 7]
WIZARD_ANIMATION_STEPS = [8, 8, 1, 8, 8, 3, 7]

# Load sounds
'''pygame.mixer.music.load("assets/audio/music.mp3")
pygame.mixer.music.set_volume(0.5)
pygame.mixer.music.play(-1, 0.0, 5000)
sword_fx = pygame.mixer.Sound("assets/audio/sword.wav")
magic_fx = pygame.mixer.Sound("assets/audio/magic.wav")'''

# Functions
def draw_bg():
    scaled_bg = pygame.transform.scale(bg_image, (SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.blit(scaled_bg, (0, 0))

def draw_health_bar(health, x, y):
    ratio = health / 100
    pygame.draw.rect(screen, WHITE, (x - 2, y - 2, 404, 34))
    pygame.draw.rect(screen, RED, (x, y, 400, 30))
    pygame.draw.rect(screen, YELLOW, (x, y, 400 * ratio, 30))

def process_hands():
    """Process hand landmarks using OpenCV."""
    ret, frame = cap.read()
    if not ret:
        return None

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    n1_finger=-1
    n2_finger=-1
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            lbl = results.multi_handedness[idx].classification[0].label
            if lbl == "Left":
                hand_lms=[]
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = rgb_frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_lms.append([id,cx,cy])
                n1_finger=0
                if hand_lms[4][1]>hand_lms[3][1]: n1_finger+=1
                for i in range(8,21,4):
                    if hand_lms[i][2]<hand_lms[i-2][2]: n1_finger+=1

            else:
                hand_lms = []
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = rgb_frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    hand_lms.append([id,cx,cy])
                n2_finger = 0
                if hand_lms[4][1] < hand_lms[3][1]: n2_finger += 1
                for i in range(8, 21, 4):
                    if hand_lms[i][2] < hand_lms[i - 2][2]: n2_finger += 1


    cv2.imshow("Hand Tracking", frame)
    cv2.waitKey(1)
    return n1_finger,n2_finger


# Create fighters
fighter_1 = Fighter(1, 200, 310, False, WARRIOR_DATA, warrior_sheet, WARRIOR_ANIMATION_STEPS)
fighter_2 = Fighter(2, 700, 310, True, WIZARD_DATA, wizard_sheet, WIZARD_ANIMATION_STEPS)

# Game loop
run = True
while run:
    clock.tick(FPS)
    draw_bg()

    # Process hands
    p1,p2 = process_hands()

    # Hand control logic
    fighter_1.move(SCREEN_WIDTH, SCREEN_HEIGHT, screen, fighter_2, round_over,p1)
    fighter_2.move(SCREEN_WIDTH, SCREEN_HEIGHT, screen, fighter_1, round_over,p2)
    # Update fighters
    fighter_1.update()
    fighter_2.update()

    # Draw health bars and fighters
    draw_health_bar(fighter_1.health, 20, 20)
    draw_health_bar(fighter_2.health, 580, 20)
    fighter_1.draw(screen)
    fighter_2.draw(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    pygame.display.update()

# Clean up
cap.release()
cv2.destroyAllWindows()
pygame.quit()
