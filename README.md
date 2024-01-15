# Self-Driving Car Game using AI with PyTorch

This project features a self-driving car game developed in Python using the Deep Q Learning algorithm. The Car and Environment are created using the KIVY Python module, and PyTorch is employed for the AI. The goal of the game is for the car to reach the bottom right corner and then return to the top left corner of the interface.

## Features

- **Deep Q Learning Algorithm with PyTorch:** The self-driving car utilizes the Deep Q Learning algorithm with PyTorch for learning and decision-making within the game environment.

- **Interactive Environment:** The game allows the user to draw obstacles in the car's path using the left mouse key. The AI learns to navigate around these obstacles during round trips.

- **Reward System:** The AI receives positive rewards upon reaching its goal and negative rewards when moving away from the goal or colliding with obstacles.

- **Save and Load:** The game provides options to save the trained AI model using the save button and load a previously saved AI using the load button.

## How to Run

1. Install the required dependencies:

   ```bash
   pip install kivy torch

2. Run the game:
```bash
python self_driving_car_game.py
```
