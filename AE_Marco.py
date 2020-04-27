
"""
Stacked AutoEncoder algorithm
"""

# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the aataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
# first column: user id, second column: the movie's ids for each user, 
# third column: rating for each movie, fourth column:  data stamp, we don't need this one


# Preparing the training set and data set
# There are 5 random splits of training and test set from the original 100k dataset, we're only using the first split for now
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t') 
#separator is a tab, but it's better to use the argument delimiter this time
training_set = np.array(training_set, dtype = 'int64') #dtype converts the data ito integers
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t') 
test_set = np.array(test_set, dtype = 'int64')

# Getting the number of users and movies (both the training and test set)
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0]))) # obtaining total number of users 
nb_movies =  int(max(max(training_set[:,1]), max(test_set[:,1])))# obtaining total number of movies

# Converting the data into an array with users in lines (observations) and movies in columns (features)
# Creating a function that will work for the training set and test set
def convert(data): 
    new_data = [] # square brackets initialises a list 
    for id_users in range(1, nb_users + 1): # Last number is excluded, we add +1
        id_movies = data[:,1][data[:,0] == id_users] # The second square brackets serves as a condition to obtain all the movie IDs for that specific user
        id_ratings = data[:,2][data[:,0] == id_users] # Obtaining all the ratings for each specific user
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings # Replaces each movie index where there is a movie rating, with the movie rating
        new_data.append(list(ratings))
    return new_data 
    # Creating a list of a list for pytorch. One list for each movie, and another list for each user.
training_set = convert(training_set) #using the convert function to convert the training set into a list of lists
test_set = convert(test_set) #using the function to conver the test set into a list of lists

# Converting the data into Torch tensors
 #Tensor is a type of pytorch multidimensional array. It is an array containing elements of the same type
training_set = torch.FloatTensor(training_set) # Creating a float tensor using the FloatTensor class expects a list of lists
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module): #creating an inheritance class from the pytorch Module class
    def __init__(self, ): #Considers the variables of the parent class
        super(SAE, self).__init__() #This makes sure that we get all the inherited class and methods from the parent class
        self.fc1 = nn.Linear(nb_movies, 20) # first full conection related to our SAE
        # First input is number of features (number of movies)
        # Second is the number of nodes in the first encoded vector (first hidden layer). Each node is a new feature detected
        self.fc2 = nn.Linear(20, 10) # second fully conected hidden layer
        self.fc3 = nn.Linear(10, 20) # third fully conected hidden layer. Starts the decoding phase
        self.fc4 = nn.Linear(20, nb_movies) # Output layer with the same number of features as the input movies
        self.activation = nn.Sigmoid() #Getting the Sigmoid activation function
    def forward(self, x): # this function encodes and decodes the features
        x = self.activation(self.fc1(x)) #returns encoded input vector of features'x' using at the left full connection
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x)) #decoding starts
        x = self.fc4(x) # reconstructed input vecor without activation function
        return x # this is our vector of predictions
sae = SAE() #Creating the object from the class Stacked AutoEncoder
criterion = nn.MSELoss() #Created the criterion object to measure the error (nn module, MSELoss class)
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01,  weight_decay = 0.5) 
# creating the object for optimising (stochastic gradient descent) (optim module, RMSprop class)
# The first input uses an attribute to obtain the parameters from our SAE class
# Second input is the learning rate
# Third input is the decay used to reduce the learning rate after some epochs to regulate the convergence


# Training the SAE (using PyTorch related methods)
nb_epoch = 200
for epoch in range(1, nb_epoch +1):
    train_loss = 0
    s = 0. # VAriable that will serve to count the number of users that rated at least one movie
    # excluding the number of users that didn't rate any movie, to optimize memory
    for id_user in range(nb_users): # All the steps that happen per epoch. REMEMBER: last number is excluded
        input = Variable(training_set[id_user]).unsqueeze(0) 
        # We are creating a batch of one input vector per batch
        # 0 is the index of a new dimension we created because Pytorch can't accept inputs of only one dimension
        target = input.clone() #Creating a copy of the original inputs
        if torch.sum(target.data > 0) > 0: #target.data gets all the data from the input vector. Excluding the users that didn't have any ratings for any movie
            output = sae(input)
            target.require_grad = False # don't compute the gradient with respect to the target (optimizing the code)
            output[target == 0] = 0 #Setting the output equals to 0 when the target movies ratings equal 0 (optimizing the code)
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data>0) + 1e-10)#avoiding it to be equal to 0
            #Considering the movies that have positive ratings. This is a mean corrector for only the movies that were rated
            loss.backward() #this tells in which direction we need to update the weights (decrease for backward, increase the weights for forward)
            train_loss += np.sqrt(loss.data * mean_corrector) #Accesing the data of the loss variable that contains the values of the loss
            s += 1.
            optimizer.step() # optimizer decided the amoun to which the weights are updated
    print('epoch: ' +str(epoch)+' train loss: '+str(train_loss/s))
            
# Testing the SAE 
test_loss = 0
s = 0. # VAriable that will serve to count the number of users that rated at least one movie
for id_user in range(nb_users): 
    input = Variable(training_set[id_user]).unsqueeze(0) 
    target = Variable(test_set[id_user]).unsqueeze(0) #This is the test data for which we will compare our predictions to
    output = sae(input)
    target.require_grad = False # don't compute the gradient with respect to the target (optimizing the code)
    output[target == 0] = 0 #Setting the output equals to 0 when the target movies ratings equal 0 (optimizing the code)
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data>0) + 1e-10)#avoiding it to be equal to 0
    test_loss += np.sqrt(loss.data * mean_corrector) #Accesing the data of the loss variable that contains the values of the loss
    s += 1.
print(' test loss: '+str(test_loss/s))



