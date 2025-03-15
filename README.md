# Gesture Controlled Snake Game 🐍

A fun Snake game that you can control using hand gestures! The game uses your camera to detect hand gestures and control the snake's movement.

## Installation

1. Make sure you have Python 3.8 or higher installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Play

1. Run the game:
```bash
python snake_game.py
```

2. Use your hand gestures to control the snake:
- 1 finger up: Move Up
- 2 fingers up: Move Right
- 3 fingers up: Move Down
- 4 fingers up: Move Left
- No hand detected: Continue in current direction

3. Try to eat the food (red squares) to grow longer
4. Avoid hitting the walls or yourself
5. Press 'q' to quit the game

## Features

- Real-time hand gesture recognition using MediaPipe
- Smooth snake movement
- Score tracking
- Game over detection
- Visual feedback with hand landmarks 
