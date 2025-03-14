import cv2
import pygame
import random
import numpy as np
import time
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_SIZE = 800
GRID_SIZE = 20
GRID_COUNT = WINDOW_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Set up display
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption('Gesture Controlled Snake Game')

class Snake:
    def __init__(self):
        self.reset()
        self.lives = 3
        self.score = 0
        self.high_score = 0
        self.speed = 5  # Initial speed (reduced by 50%)
        self.is_invincible = False
        self.invincible_timer = 0
        self.invincible_duration = 3  # seconds

    def reset(self):
        self.body = [(GRID_COUNT // 2, GRID_COUNT // 2)]
        self.direction = [1, 0]  # Start moving right
        self.grow = False
        self.score = 0

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
            
        self.body.insert(0, new_head)

    def check_collision(self):
        head = self.body[0]
        # Check wall collision
        if (head[0] < 0 or head[0] >= GRID_COUNT or 
            head[1] < 0 or head[1] >= GRID_COUNT):
            return True
        # Check self collision
        if head in self.body[1:]:
            return True
        return False

class Food:
    def __init__(self):
        self.position = self.generate_position()
        self.type = random.choice(['normal', 'bonus', 'powerup'])
        self.spawn_time = time.time()
        self.duration = 10  # seconds for bonus/powerup food

    def generate_position(self):
        x = random.randint(0, GRID_COUNT - 1)
        y = random.randint(0, GRID_COUNT - 1)
        return (x, y)

    def update(self):
        if self.type in ['bonus', 'powerup']:
            if time.time() - self.spawn_time > self.duration:
                self.type = 'normal'
                self.position = self.generate_position()

class Game:
    def __init__(self):
        self.snake = Snake()
        self.food = Food()
        self.game_over = False
        self.infinite_mode = False
        self.difficulty = 'normal'
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def reset(self):
        self.snake.reset()
        self.food = Food()
        self.game_over = False

    def draw_menu(self):
        screen.fill(BLACK)
        title = self.font.render('Snake Game Menu', True, WHITE)
        start = self.font.render('Press SPACE to Start', True, WHITE)
        infinite = self.font.render('Press I for Infinite Mode', True, WHITE)
        difficulty = self.font.render(f'Difficulty: {self.difficulty} (Press D to change)', True, WHITE)
        
        # Add gesture instructions
        gesture_title = self.small_font.render('Gesture Controls:', True, WHITE)
        gesture_up = self.small_font.render('Hand at top = Move Up', True, WHITE)
        gesture_right = self.small_font.render('Hand at right = Move Right', True, WHITE)
        gesture_down = self.small_font.render('Hand at bottom = Move Down', True, WHITE)
        gesture_left = self.small_font.render('Hand at left = Move Left', True, WHITE)
        
        screen.blit(title, (WINDOW_SIZE//2 - title.get_width()//2, 100))
        screen.blit(start, (WINDOW_SIZE//2 - start.get_width()//2, 200))
        screen.blit(infinite, (WINDOW_SIZE//2 - infinite.get_width()//2, 300))
        screen.blit(difficulty, (WINDOW_SIZE//2 - difficulty.get_width()//2, 400))
        
        # Draw gesture instructions
        screen.blit(gesture_title, (WINDOW_SIZE//2 - gesture_title.get_width()//2, 500))
        screen.blit(gesture_up, (WINDOW_SIZE//2 - gesture_up.get_width()//2, 550))
        screen.blit(gesture_right, (WINDOW_SIZE//2 - gesture_right.get_width()//2, 580))
        screen.blit(gesture_down, (WINDOW_SIZE//2 - gesture_down.get_width()//2, 610))
        screen.blit(gesture_left, (WINDOW_SIZE//2 - gesture_left.get_width()//2, 640))
        
        pygame.display.flip()

    def draw_game_over(self):
        screen.fill(BLACK)
        game_over = self.font.render('Game Over!', True, RED)
        score = self.font.render(f'Score: {self.snake.score}', True, WHITE)
        high_score = self.font.render(f'High Score: {self.snake.high_score}', True, WHITE)
        restart = self.font.render('Press R to Restart', True, WHITE)
        menu = self.font.render('Press M for Menu', True, WHITE)
        
        screen.blit(game_over, (WINDOW_SIZE//2 - game_over.get_width()//2, 200))
        screen.blit(score, (WINDOW_SIZE//2 - score.get_width()//2, 300))
        screen.blit(high_score, (WINDOW_SIZE//2 - high_score.get_width()//2, 400))
        screen.blit(restart, (WINDOW_SIZE//2 - restart.get_width()//2, 500))
        screen.blit(menu, (WINDOW_SIZE//2 - menu.get_width()//2, 600))
        
        pygame.display.flip()

    def draw_game(self):
        screen.fill(BLACK)
        
        # Draw snake with invincibility effect
        for i, segment in enumerate(self.snake.body):
            color = YELLOW if self.snake.is_invincible else GREEN
            pygame.draw.rect(screen, color,
                           (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                            GRID_SIZE - 1, GRID_SIZE - 1))
        
        # Draw food with different colors based on type
        food_color = RED if self.food.type == 'normal' else YELLOW if self.food.type == 'bonus' else BLUE
        pygame.draw.rect(screen, food_color,
                        (self.food.position[0] * GRID_SIZE, self.food.position[1] * GRID_SIZE,
                         GRID_SIZE - 1, GRID_SIZE - 1))
        
        # Draw HUD
        score_text = self.font.render(f'Score: {self.snake.score}', True, WHITE)
        lives_text = self.font.render(f'Lives: {self.snake.lives}', True, WHITE)
        mode_text = self.small_font.render('Infinite Mode' if self.infinite_mode else 'Normal Mode', True, WHITE)
        
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 50))
        screen.blit(mode_text, (10, 90))
        
        pygame.display.flip()

def detect_hand_gesture(frame):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks
    results = hands.process(rgb_frame)
    
    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hand detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return 0
    
    # Get hand landmarks
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Draw hand landmarks
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Get finger tip and palm coordinates
    palm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculate palm openness
    palm_openness = 0
    for tip in [index_tip, middle_tip, ring_tip, pinky_tip]:
        if tip.y < palm.y:  # If finger tip is above palm
            palm_openness += 1
    
    # Draw palm openness indicator
    cv2.putText(frame, f"Palm Openness: {palm_openness}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Determine gesture based on palm openness
    if palm_openness == 0:  # Closed palm
        cv2.putText(frame, "UP", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 1  # Up
    elif palm_openness >= 3:  # Open palm
        cv2.putText(frame, "DOWN", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 3  # Down
    
    # For left/right, use hand position
    frame_h, frame_w = frame.shape[:2]
    palm_x = int(palm.x * frame_w)
    
    if palm_x < frame_w//3:
        cv2.putText(frame, "LEFT", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 4  # Left
    elif palm_x > 2*frame_w//3:
        cv2.putText(frame, "RIGHT", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 2  # Right
    
    cv2.putText(frame, "No direction", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return 0  # No clear direction

def main():
    game = Game()
    game_state = 'menu'  # menu, playing, game_over
    cap = cv2.VideoCapture(0)
    
    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return
                elif event.key == pygame.K_r and game_state == 'game_over':
                    game.reset()
                    game_state = 'playing'
                elif event.key == pygame.K_m and game_state == 'game_over':
                    game_state = 'menu'
                elif event.key == pygame.K_SPACE and game_state == 'menu':
                    game.reset()
                    game_state = 'playing'
                elif event.key == pygame.K_i and game_state == 'menu':
                    game.infinite_mode = not game.infinite_mode
                elif event.key == pygame.K_d and game_state == 'menu':
                    difficulties = ['easy', 'normal', 'hard']
                    current_index = difficulties.index(game.difficulty)
                    game.difficulty = difficulties[(current_index + 1) % len(difficulties)]
                    # Adjust speed based on difficulty
                    if game.difficulty == 'easy':
                        game.snake.speed = 3
                    elif game.difficulty == 'normal':
                        game.snake.speed = 5
                    else:
                        game.snake.speed = 7

        if game_state == 'menu':
            game.draw_menu()
        elif game_state == 'playing':
            # Read webcam frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Detect hand gesture
            finger_count = detect_hand_gesture(frame)
            
            # Update snake direction based on finger count
            if finger_count == 1:
                game.snake.direction = [0, -1]  # Up
            elif finger_count == 2:
                game.snake.direction = [1, 0]   # Right
            elif finger_count == 3:
                game.snake.direction = [0, 1]   # Down
            elif finger_count == 4:
                game.snake.direction = [-1, 0]  # Left
            
            # Move snake
            game.snake.move()
            
            # Update food
            game.food.update()
            
            # Check for food collision
            if game.snake.body[0] == game.food.position:
                if game.food.type == 'normal':
                    game.snake.grow = True
                    game.snake.score += 1
                elif game.food.type == 'bonus':
                    game.snake.score += 5
                elif game.food.type == 'powerup':
                    game.snake.is_invincible = True
                    game.snake.invincible_timer = time.time()
                
                game.food = Food()
                if game.infinite_mode:
                    game.snake.speed = min(15, game.snake.speed + 0.5)  # Gradually increase speed
            
            # Update invincibility
            if game.snake.is_invincible:
                if time.time() - game.snake.invincible_timer > game.snake.invincible_duration:
                    game.snake.is_invincible = False
            
            # Check for game over
            if game.snake.check_collision():
                if not game.snake.is_invincible:
                    if game.infinite_mode:
                        game.snake.lives -= 1
                        if game.snake.lives <= 0:
                            game_state = 'game_over'
                            if game.snake.score > game.snake.high_score:
                                game.snake.high_score = game.snake.score
                        else:
                            game.snake.reset()
                    else:
                        game_state = 'game_over'
                        if game.snake.score > game.snake.high_score:
                            game.snake.high_score = game.snake.score
            
            # Draw game
            game.draw_game()
            
            # Show webcam feed
            cv2.imshow('Hand Gestures', frame)
            
            # Control game speed
            game.clock.tick(game.snake.speed)
            
        elif game_state == 'game_over':
            game.draw_game_over()
            game.clock.tick(30)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main() 