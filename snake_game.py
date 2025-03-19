import cv2
import pygame
import random
import numpy as np
import time
import mediapipe as mp

# ------------------------------
# Initialization of MediaPipe
# ------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ------------------------------
# Initialization of Pygame
# ------------------------------
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

# ------------------------------
# Game Classes
# ------------------------------
class Snake:
    def __init__(self):
        self.reset()
        self.lives = 3
        self.score = 0
        self.high_score = 0
        self.speed = 5  # Initial speed
        self.is_invincible = False
        self.invincible_timer = 0
        self.invincible_duration = 3  # seconds

    def reset(self):
        """Reset snake to initial position and state."""
        self.body = [(GRID_COUNT // 2, GRID_COUNT // 2)]
        self.direction = [1, 0]  # Moving right initially
        self.grow = False
        self.score = 0

    def move(self):
        """Move the snake by inserting a new head in the direction of travel."""
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        if not self.grow:
            self.body.pop()
        else:
            self.grow = False
        self.body.insert(0, new_head)

    def check_collision(self):
        """Check for collisions with walls or self."""
        head = self.body[0]
        # Wall collision check
        if head[0] < 0 or head[0] >= GRID_COUNT or head[1] < 0 or head[1] >= GRID_COUNT:
            return True
        # Self collision check
        if head in self.body[1:]:
            return True
        return False

class Food:
    def __init__(self):
        self.position = self.generate_position()
        self.type = random.choice(['normal', 'bonus', 'powerup'])
        self.spawn_time = time.time()
        self.duration = 10  # Seconds for bonus/powerup food

    def generate_position(self):
        """Generate a random position within the grid."""
        x = random.randint(0, GRID_COUNT - 1)
        y = random.randint(0, GRID_COUNT - 1)
        return (x, y)

    def update(self):
        """Update food type if bonus/powerup food has expired."""
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
        """Reset the game state."""
        self.snake.reset()
        self.food = Food()
        self.game_over = False

    def draw_menu(self):
        """Draw the game menu screen."""
        screen.fill(BLACK)
        title = self.font.render('Snake Game Menu', True, WHITE)
        start = self.font.render('Press SPACE to Start', True, WHITE)
        infinite = self.font.render('Press I for Infinite Mode', True, WHITE)
        difficulty = self.font.render(f'Difficulty: {self.difficulty} (Press D to change)', True, WHITE)

        # Gesture control instructions
        gesture_title = self.small_font.render('Gesture Controls:', True, WHITE)
        gesture_up = self.small_font.render('Hand at top = Move Up', True, WHITE)
        gesture_right = self.small_font.render('Hand at right = Move Right', True, WHITE)
        gesture_down = self.small_font.render('Hand at bottom = Move Down', True, WHITE)
        gesture_left = self.small_font.render('Hand at left = Move Left', True, WHITE)

        screen.blit(title, (WINDOW_SIZE // 2 - title.get_width() // 2, 100))
        screen.blit(start, (WINDOW_SIZE // 2 - start.get_width() // 2, 200))
        screen.blit(infinite, (WINDOW_SIZE // 2 - infinite.get_width() // 2, 300))
        screen.blit(difficulty, (WINDOW_SIZE // 2 - difficulty.get_width() // 2, 400))
        screen.blit(gesture_title, (WINDOW_SIZE // 2 - gesture_title.get_width() // 2, 500))
        screen.blit(gesture_up, (WINDOW_SIZE // 2 - gesture_up.get_width() // 2, 550))
        screen.blit(gesture_right, (WINDOW_SIZE // 2 - gesture_right.get_width() // 2, 580))
        screen.blit(gesture_down, (WINDOW_SIZE // 2 - gesture_down.get_width() // 2, 610))
        screen.blit(gesture_left, (WINDOW_SIZE // 2 - gesture_left.get_width() // 2, 640))
        pygame.display.flip()

    def draw_game_over(self):
        """Draw the game over screen."""
        screen.fill(BLACK)
        game_over_text = self.font.render('Game Over!', True, RED)
        score_text = self.font.render(f'Score: {self.snake.score}', True, WHITE)
        high_score_text = self.font.render(f'High Score: {self.snake.high_score}', True, WHITE)
        restart = self.font.render('Press R to Restart', True, WHITE)
        menu = self.font.render('Press M for Menu', True, WHITE)

        screen.blit(game_over_text, (WINDOW_SIZE // 2 - game_over_text.get_width() // 2, 200))
        screen.blit(score_text, (WINDOW_SIZE // 2 - score_text.get_width() // 2, 300))
        screen.blit(high_score_text, (WINDOW_SIZE // 2 - high_score_text.get_width() // 2, 400))
        screen.blit(restart, (WINDOW_SIZE // 2 - restart.get_width() // 2, 500))
        screen.blit(menu, (WINDOW_SIZE // 2 - menu.get_width() // 2, 600))
        pygame.display.flip()

    def draw_game(self):
        """Draw the active game state including snake, food, and HUD."""
        screen.fill(BLACK)
        # Draw snake with invincibility indicator
        for segment in self.snake.body:
            color = YELLOW if self.snake.is_invincible else GREEN
            pygame.draw.rect(screen, color,
                             (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE,
                              GRID_SIZE - 1, GRID_SIZE - 1))

        # Draw food with color based on type
        if self.food.type == 'normal':
            food_color = RED
        elif self.food.type == 'bonus':
            food_color = YELLOW
        else:  # powerup
            food_color = BLUE
        pygame.draw.rect(screen, food_color,
                         (self.food.position[0] * GRID_SIZE, self.food.position[1] * GRID_SIZE,
                          GRID_SIZE - 1, GRID_SIZE - 1))

        # Draw HUD information
        score_text = self.font.render(f'Score: {self.snake.score}', True, WHITE)
        lives_text = self.font.render(f'Lives: {self.snake.lives}', True, WHITE)
        mode_text = self.small_font.render('Infinite Mode' if self.infinite_mode else 'Normal Mode', True, WHITE)

        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 50))
        screen.blit(mode_text, (10, 90))
        pygame.display.flip()

# ------------------------------
# Gesture Detection Function
# ------------------------------
def detect_hand_gesture(frame):
    """
    Process the frame to detect hand landmarks and interpret the gesture.
    
    Returns:
        int: A number representing the direction:
             1 = Up, 2 = Right, 3 = Down, 4 = Left, 0 = No clear direction.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return 0

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Retrieve landmarks for palm and finger tips
    palm = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Calculate palm openness (number of fingers extended above the palm)
    palm_openness = sum(1 for tip in [index_tip, middle_tip, ring_tip, pinky_tip] if tip.y < palm.y)
    cv2.putText(frame, f"Palm Openness: {palm_openness}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Determine gesture: closed palm signals UP, open palm signals DOWN
    if palm_openness == 0:
        cv2.putText(frame, "UP", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 1
    elif palm_openness >= 3:
        cv2.putText(frame, "DOWN", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 3

    # For left/right gestures, use horizontal palm position
    frame_h, frame_w = frame.shape[:2]
    palm_x = int(palm.x * frame_w)
    if palm_x < frame_w // 3:
        cv2.putText(frame, "LEFT", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 4
    elif palm_x > 2 * frame_w // 3:
        cv2.putText(frame, "RIGHT", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return 2

    cv2.putText(frame, "No direction", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return 0

# ------------------------------
# Event Handling
# ------------------------------
def handle_events(game, game_state):
    """
    Handle pygame events and update game state accordingly.
    
    Args:
        game (Game): The game instance.
        game_state (str): The current game state.
    
    Returns:
        str: Updated game state.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return 'quit'
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return 'quit'
            elif event.key == pygame.K_r and game_state == 'game_over':
                game.reset()
                return 'playing'
            elif event.key == pygame.K_m and game_state == 'game_over':
                return 'menu'
            elif event.key == pygame.K_SPACE and game_state == 'menu':
                game.reset()
                return 'playing'
            elif event.key == pygame.K_i and game_state == 'menu':
                game.infinite_mode = not game.infinite_mode
            elif event.key == pygame.K_d and game_state == 'menu':
                difficulties = ['easy', 'normal', 'hard']
                current_index = difficulties.index(game.difficulty)
                game.difficulty = difficulties[(current_index + 1) % len(difficulties)]
                # Adjust snake speed based on difficulty
                if game.difficulty == 'easy':
                    game.snake.speed = 3
                elif game.difficulty == 'normal':
                    game.snake.speed = 5
                else:
                    game.snake.speed = 7
    return game_state

# ------------------------------
# Main Game Loop
# ------------------------------
def main():
    game = Game()
    game_state = 'menu'  # States: 'menu', 'playing', 'game_over'
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Handle events and update game state accordingly
        new_state = handle_events(game, game_state)
        if new_state == 'quit':
            break
        game_state = new_state

        if game_state == 'menu':
            game.draw_menu()

        elif game_state == 'playing':
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.flip(frame, 1)
            gesture = detect_hand_gesture(frame)

            # Map detected gesture to snake direction
            if gesture == 1:
                game.snake.direction = [0, -1]  # Up
            elif gesture == 2:
                game.snake.direction = [1, 0]   # Right
            elif gesture == 3:
                game.snake.direction = [0, 1]   # Down
            elif gesture == 4:
                game.snake.direction = [-1, 0]  # Left

            # Update game state
            game.snake.move()
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
                    game.snake.speed = min(15, game.snake.speed + 0.5)

            # Update invincibility status
            if game.snake.is_invincible and (time.time() - game.snake.invincible_timer > game.snake.invincible_duration):
                game.snake.is_invincible = False

            # Check for collisions and update game state
            if game.snake.check_collision():
                if not game.snake.is_invincible:
                    if game.infinite_mode:
                        game.snake.lives -= 1
                        if game.snake.lives <= 0:
                            game_state = 'game_over'
                            game.snake.high_score = max(game.snake.high_score, game.snake.score)
                        else:
                            game.snake.reset()
                    else:
                        game_state = 'game_over'
                        game.snake.high_score = max(game.snake.high_score, game.snake.score)

            game.draw_game()
            cv2.imshow('Hand Gestures', frame)
            game.clock.tick(game.snake.speed)

        elif game_state == 'game_over':
            game.draw_game_over()
            game.clock.tick(30)

        # Exit if 'q' is pressed in the OpenCV window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
