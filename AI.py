#First step is to import the libraries
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#Here Neural Network Class will be created.
class Neural_Network(nn.Module): #This class will inherit all the methods of the nn.Module class.
    
    def __init__(self, number_of_inputs, number_of_actions):
        super(Neural_Network, self).__init__()
        self.number_of_inputs = number_of_inputs
        self.number_of_actions = number_of_actions
        
        #Here we will create the first full connection between the input layer and the hidden layer
        self.input_hidden_conn = nn.Linear(number_of_inputs, 30)
        
        #Here we will create the second full connection between the hidden layer and the output layer.
        self.hidden_output_conn = nn.Linear(30, number_of_actions)
        
    #Here we will create the forward function, which will forward the input from input layer neurons to hidden layer neurons.
    #And further from the hidden layer neurons to output layer neurons and return q_values
    def forward(self, state):
        from_input = F.relu(self.input_hidden_conn(state))
        q_values = self.hidden_output_conn(from_input)
        return q_values
    
#Further we will implement the experience replay.
#This is one of the important step of deep q learning.
class ExperienceReplay(object): #This will inherit all the methods of simple object class.
        
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] #Here we will set memory as an empty list.
        
    #Push function will be implemented to to append the events to the memory.
    #Event will include the last_state, last_action and last_reward.
    def push(self, event):
        self.memory.append(event) #Appending the event to the memory list.
        if len(self.memory) > self.capacity: #To have defined number of events.
            del self.memory[0] #Delete the first event from the memory list.
        
    #This function is to get the samples from the memory.
    def get_samples(self, size_of_batch):
        samples = zip(*random.sample(self.memory, size_of_batch)) #Zip is used to regroup the states, actions and rewards into separate batches of equal size or size_of_batch.
        #Map function is used to wrap the each sample into torch variable object, which will have both the tensor and the gradient.
        return map(lambda x: Variable(torch.cat(x, 0)), samples) #Torch.cat() is used to put the samples in right format.

#Deep Q_Learning Implementation.
class Deep_Q_Learning(object):
    
    def __init__(self, number_of_inputs, number_of_actions, gamma): #Gamma is the discount factor.
        self.gamma = gamma # Discount Factor
        self.network_model = Neural_Network(number_of_inputs, number_of_actions) #Instance of the Neural_Network Class
        self.memory = ExperienceReplay(capacity = 100000) #Instance of Experience memory class
        self.optimizer = optim.Adam(params = self.network_model.parameters()) #Adam optimizer to update the weights, for learning process.
        self.last_state = torch.Tensor(number_of_inputs).unsqueeze(0) #Getting the last_state and converting it into tensor and one fake dimension.
        self.last_action = 0 # Last Action took by car Or AI.
        self.last_reward = 0 #Last Reward AI got by performing a action.
        
    #Select the action
    def select_action(self, state):
        probabilities_of_actions = F.softmax(self.network_model(Variable(state)) * 100) #Getting the probabilites of all the three actions(Forward, left, right), and multiplying it with 100 to make sure it takes the action with high prob.
        action = probabilities_of_actions.multinomial(len(probabilities_of_actions)) #Taking the action which have high probability value.
        return action.data[0, 0] #returning the action
    
    #Learning process
    #Here is the main algorithm for AI.
    def learning_process(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        
        batch_outputs = self.network_model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1) #Getting the output values.
        batch_next_outputs = self.network_model(batch_next_states).detach().max(1)[0] #Getting the next output values.
        batch_targets = batch_rewards + self.gamma * batch_next_outputs #Markov Decision Process.
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets) #Calculating the temporal Difference between the outputs and the targets.
        self.optimizer.zero_grad()#Reinitializing the optimizer for each iteration.
        td_loss.backward() # backpropagating the temporal difference loss.
        self.optimizer.step() #Updating the weights.
    
    #Main method for updating
    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float().unsqueeze(0) #the new state of our AI or Car.
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state)) # Appending the event to the memory list, Event is in the form of tensors.
        new_action = self.select_action(new_state) #the new action that the AI will take.
        
        if len(self.memory.memory) > 100: #Taking the sample from the memory.
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.get_samples(100)
            self.learning_process(batch_states, batch_actions, batch_rewards, batch_next_states) #Initializing the learning process.
        self.last_state = new_state #Updating the last state of AI.
        self.last_action = new_action #Updating the last action of AI.
        self.last_reward = new_reward #Updating the last reward of AI.
        return new_action # Returning the new action to the map class.
    
    #Save Method to save AI.
    def save(self):
        torch.save({'state_dict' : self.network_model.state_dict(), #saving the last_state of the AI.
                    'optimizer' : self.optimizer.state_dict() # Saving the weights of synapses.
            }, 'last_brain.pth') # state and weights will be stored in the last_brain.pth file.
    
    #Load Method for AI.
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.network_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded Successfully.....................")
        else:
            print("Nothing is saved.........Sorry")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
            
            
        