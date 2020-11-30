# Importing the libraries
import numpy as np
from random import random, randint

import time
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

#Importing the deep_q_learning class.
from AI import Deep_Q_Learning

Config.set('input', 'mouse', 'mouse, multitouch_on_demand') #To avoid a red point on the screen with right mouse click.

last_obstacle_x = 0 #x point of the last obstacle we draw 
last_obstacle_y = 0 #y point of the last obstacle we draw
num_of_points_obstacle = 0 # number of points in  the last obstacle.
obstacle_len = 0 #length of the last obstacle.

#Creating the AI
our_ai = Deep_Q_Learning(5, 3, 0.9) #5 inputs and 3 outputs, 0.9 is the value of gamma
action_to_rotation = [0, 20, -20] # 0(no rotation), 20(go Right), -20(Go left)
reward = 0 # reward that ai will get after each state.

#Creating the map
map_update = True #initialize the map only once.

def init():
    
    global obstacle #array which has number of cells equal to the number of pixels of interface.
    global goal_x #x  coordinate of goal which ai has to reach
    global goal_y #y coordinate of the goal which ai has to reach.
    global map_update #map initializer
    
    obstacle = np.zeros((longueur, largeur)) #initializing the obstacle array to all zeros. Means blank screen with no obstacles..
    goal_x = 20
    goal_y = largeur - 20
    map_update = False
    
#Distance of the car to the goal.
distance_car_to_goal = 0

#Creating the Car.
class Car(Widget):
    
    angle = NumericProperty(0) #Angle between the x axis of the map and the axis of the car.
    rotation = NumericProperty(0) #Last rotation of the car.
    
    velocity_x = NumericProperty(0) # x_coordinate of the velocity vector.
    velocity_y = NumericProperty(0) #y_coordinate of the velocity vector.
    velocity = ReferenceListProperty(velocity_x, velocity_y) #velocity vector of the car.
    
    front_sensor_x = NumericProperty(0) # x_coordinate of the forward sensor.
    front_sensor_y = NumericProperty(0) # y_coordinate of the forward sensor.
    front_sensor = ReferenceListProperty(front_sensor_x, front_sensor_y) #Forward sensor vector.
    
    right_sensor_x = NumericProperty(0) # x_coordinate of the right sensor.
    right_sensor_y = NumericProperty(0) # y_coordinate of the right sensor.
    right_sensor = ReferenceListProperty(right_sensor_x, right_sensor_y) #Right sensor vector.
    
    left_sensor_x = NumericProperty(0) # x_coordinate of the left sensor.
    left_sensor_y = NumericProperty(0) # y_coordinate of the left sensor.
    left_sensor = ReferenceListProperty(left_sensor_x, left_sensor_y) #Left sensor vector.
    
    front_signal = NumericProperty(0) #Signal received by forward sensor.
    right_signal = NumericProperty(0) #Signal received by right sensor.
    left_signal = NumericProperty(0) #Signal received by left sensor.
    
    #Move method for the car.
    def move(self, rotation):
        
        self.pos = Vector(*self.velocity) + self.pos # Updating the position of the car according to its last postion and velocity.
        self.rotation = rotation # rotation of the car.
        self.angle = self.angle + self.rotation # updating the angle.
        
        self.front_sensor = Vector(30, 0).rotate(self.angle) + self.pos #Updating the position of the forward sensor.
        self.right_sensor = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos #Updating the position of right sensor.
        self.left_sensor = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos #Updating the position of the left sensor.
        
        self.front_signal = int(np.sum(obstacle[int(self.front_sensor_x)-10:int(self.front_sensor_x)+10, int(self.front_sensor_y)-10:int(self.front_sensor_y)+10]))/400. #getting the density of obstacle around the forward sensor.
        self.right_signal = int(np.sum(obstacle[int(self.right_sensor_x)-10:int(self.right_sensor_x)+10, int(self.right_sensor_y)-10:int(self.right_sensor_y)+10]))/400. #getting the density of obstacle around the right sensor.
        self.left_signal = int(np.sum(obstacle[int(self.left_sensor_x)-10:int(self.left_sensor_x)+10, int(self.left_sensor_y)-10:int(self.left_sensor_y)+10]))/400. #getting the density of obstacle around the left sensor.
        
        if self.front_sensor_x > longueur-10 or self.front_sensor_x <10 or self.front_sensor_y>largeur-10 or self.front_sensor_y < 10: # To check if the forward sensor is out of the map or not.
            self.front_signal = 1. # forward sensor detects obstacle
        
        if self.right_sensor_x > longueur-10 or self.right_sensor_x <10 or self.right_sensor_y>largeur-10 or self.right_sensor_y < 10: # To check if the right sensor is out of the map or not.
            self.right_signal = 1. # right sensor detects obstacle
        
        if self.left_sensor_x > longueur-10 or self.left_sensor_x <10 or self.left_sensor_y>largeur-10 or self.left_sensor_y < 10: # To check if the left sensor is out of the map or not.
            self.left_signal = 1. # left sensor detects obstacle

class Front_Sensor_Ball(Widget):
    pass
class Left_Sensor_Ball(Widget):
    pass
class Right_Sensor_Ball(Widget):
    pass

#Creating the game class.
class Game(Widget):
    
    car = ObjectProperty(None) #Getting the car object
    front_sensor_ball = ObjectProperty(None) #Getting the front_sensor object
    left_sensor_ball = ObjectProperty(None) #Getting the left_sensor object
    right_sensor_ball = ObjectProperty(None) #Getting the right_sensor object
    
    def start_car(self): #Starting the car.
        self.car.center = self.center #The car will start at the center
        self.car.velocity = Vector(6, 0) #The car will start with speed of 6.
    
    #update function to update everything after each state change of the car.
    def update(self, dt):
        
        global our_ai 
        global reward
        global distance_car_to_goal
        global goal_x
        global goal_y
        global longueur
        global largeur
        
        #Setting the value of the longueur
        longueur = self.width #Width of the map
        largeur = self.height #Height of the map
        
        if map_update: #Initializing the map
            init()
            
        #Calculating the orientation
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/ 180. #Direction of the car with respect to the goal.
        
        state = [orientation, -orientation, self.car.front_signal, self.car.right_signal, self.car.left_signal] #Input state vector.
        action = our_ai.update(state, reward) #Updating the weights of the neural network
        rotation = action_to_rotation[action] #Converting the action played to the rotation
        
        self.car.move(rotation) #Moving the car based on the  that our AI took.
        new_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2) #New Distance from the goal after car moved

        self.front_sensor_ball.pos = self.car.front_sensor #Updating the position of the front sensor
        self.left_sensor_ball.pos = self.car.left_sensor #Updating the position of the left sensor
        self.right_sensor_ball.pos = self.car.right_sensor #Updating the position of the right sensor
        
        if obstacle[int(self.car.x), int(self.car.y)] > 0: #To check if the car hits obstacle or not.
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) #Slows down the car.
            reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            reward = -0.2
            if new_distance < distance_car_to_goal:
                reward = 0.1
            
        if self.car.x < 10: # if the car is in the left edge of the frame
            self.car.x = 10 
            reward = -1 
            
        if self.car.x > self.width-10: # if the car is in the right edge of the frame
            self.car.x = self.width-10 
            reward = -1
            
        if self.car.y < 10: # if the car is in the bottom edge of the frame
            self.car.y = 10 
            reward = -1
            
        if self.car.y > self.height-10: # if the car is in the upper edge of the frame
            self.car.y = self.height-10
            reward = -1

        if new_distance < 100: # when the car reaches its goal
            goal_x = self.width - goal_x  # the goal becomes the bottom right corner of the map
            goal_y = self.height - goal_y  # the goal becomes the bottom right corner of the map

        distance_car_to_goal = new_distance
        
        
#Painting class
class MyPaintWidget(Widget):

    def on_touch_down(self, touch): # Draw the obstacle
        global obstacle_len
        global num_of_points_obstacle
        global last_obstacle_x
        global last_obstacle_y
        
        with self.canvas:
            Color(0.8,0.7,0)
            d=10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_obstacle_x = int(touch.x)
            last_obstacle_y = int(touch.y)
            num_of_points_obstacle = 0
            obstacle_len = 0
            obstacle[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        
        global obstacle_len
        global num_of_points_obstacle
        global last_obstacle_x
        global last_obstacle_y
        
        if touch.button=='left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            obstacle_len += np.sqrt(max((x - last_obstacle_x)**2 + (y - last_obstacle_y)**2, 2))
            num_of_points_obstacle += 1.
            density = num_of_points_obstacle/(obstacle_len)
            touch.ud['line'].width = int(20*density + 1)
            obstacle[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_obstacle_x = x
            last_obstacle_y = y

class CarApp(App):

    def build(self): # building the app
        parent = Game()
        parent.start_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save',pos=(parent.width,0))
        loadbtn = Button(text='load',pos=(2*parent.width,0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj): # clear button
        global obstacle
        self.painter.canvas.clear()
        obstacle = np.zeros((longueur,largeur))

    def save(self, obj): # save button
        print("Saving Our AI>>>>>")
        our_ai.save()

    def load(self, obj): # load button
        print("Loading OUR AI>>>>>>>>>")
        our_ai.load()

# Running the app
if __name__ == '__main__':
    CarApp().run()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
