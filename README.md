# Self-Driving Car Game using AI

This project features a self-driving car game developed in Python using the Deep Q Learning algorithm. The Car and Environment are created using the KIVY Python module. The objective of the game is for the car to reach the bottom right corner and then return to the top left corner of the interface.

## Features

- **Deep Q Learning Algorithm:** The self-driving car utilizes the Deep Q Learning algorithm for learning and decision-making within the game environment.

- **Interactive Environment:** The game allows the user to draw obstacles in the car's path using the left mouse key. The AI learns to navigate around these obstacles during round trips.

- **Reward System:** The AI receives positive rewards upon reaching its goal and negative rewards when moving away from the goal or colliding with obstacles.

- **Save and Load:** The game provides options to save the trained AI model using the save button and load a previously saved AI using the load button.

## How to Run

1. Install the required dependencies:

   ```bash
   pip install kivy
