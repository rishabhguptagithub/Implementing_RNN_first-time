# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:20:37 2018

@author: Rishabh Gupta
"""

import numpy as np


#loading data and we have counted the numbers of words and and diffe
#rent characters in the file
data=open('JEE_FOR_ME.txt','r').read()
chars=list(set(data))
data_size,char_size=len(data),len(chars)
print(data_size,char_size)
print(chars)



#Now we have to create vectors of the chars that we have extracted
#First we crea
char_to_ix = { ch:i for i,ch in enumerate(chars)}
ix_to_char = { i:ch for i, ch in enumerate(chars)}
print (char_to_ix)
print (ix_to_char)

#hyperparameters

hidden_state=100
seq_len=25
learning_rate=1e-8


#tHIS IS THE WEIGHTS 
Wxx=np.random.rand(hidden_state,char_size)
Wxh=np.random.rand(hidden_state,hidden_state)
Wxy=np.random.rand(char_size,hidden_state)
bh=np.zeros((hidden_state,1))
by=np.zeros((char_size,1))



def Loss(inputs,targets,hprev):
    #input is series of integers.the program in further you will 
    #see that it picks up 25 chars as inputs and 25 chars just 
    #after one position from the input char for the target char.
    p,h,x,y={},{},{},{}
    h[-1]=hprev
    loss=0
    for t in range(len(inputs)):
        #vectorizing the input[t]
        x[t]=np.zeros((char_size,1))
        x[t][inputs[t]]=1
        
        
        #forward pass
        h[t]=np.tanh(np.dot(Wxx,x[t])+np.dot(Wxh,h[t-1])+bh)
        y[t]=np.dot(Wxy,h[t])+by
        p[t]=np.exp(y[t])/np.sum(np.exp(y[t]))
        loss+=-np.log(p[t][targets[t],0])
        
        #backward pass
        dWxy,dWxx,dWxh=np.zeros_like(Wxy),np.zeros_like(Wxx),np.zeros_like(Wxh)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by)
        dhnext = np.zeros_like(h[0])
        
        
        
    for t in reversed(range(len(inputs))):
        dy=np.copy(p[t])
        dy[targets[t]] -= 1
        dWxy+=np.dot(dy,h[t].T)
        dby+=by
        dh=np.dot(Wxy.T,dy)+dhnext
        dhraw=(1-h[t]*h[t])*dh
        dbh+=dhraw
        dWxx+=np.dot(dhraw,x[t].T)
        dWxh+=np.dot(dhraw,h[t-1].T)
    for dparam in [dWxh, dWxh, dWxy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxx, dWxh, dWxy, dbh, dby,h[len(inputs)-1]

def sample(h, seed_ix, n):
  """                                                                                                                                                                                         
  sample a sequence of integers from the model                                                                                                                                                
  h is memory state, seed_ix is seed letter for first time step   
  n is how many characters to predict
  """
  #create vector
  x = np.zeros((char_size, 1))
  #customize it for our seed char
  x[seed_ix] = 1
  #list to store generated chars
  ixes = []
  #for as many characters as we want to generate
  for t in range(n):
    #a hidden state at a given time step is a function 
    #of the input at the same time step modified by a weight matrix 
    #added to the hidden state of the previous time step 
    #multiplied by its own hidden state to hidden state matrix.
    h = np.tanh(np.dot(Wxx, x) + np.dot(Wxh, h) + bh)
    #compute output (unnormalised)
    y = np.dot(Wxy, h) + by
    ## probabilities for next chars
    p = np.exp(y) / np.sum(np.exp(y))
    #pick one with the highest probability 
    ix = np.random.choice(range(char_size), p=p.ravel())
    #create a vector
    x = np.zeros((char_size, 1))
    #customize it for the predicted char
    x[ix] = 1
    #add it to the list
    ixes.append(ix)

  txt = ''.join(ix_to_char[ix] for ix in ixes)
  print ('----\n %s \n----' % (txt, ))
hprev = np.zeros((hidden_state,1)) # reset RNN memory



n, p = 0, 0
mWxx, mWxh, mWhy = np.zeros_like(Wxx), np.zeros_like(Wxh), np.zeros_like(Wxy)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad                                                                                                                smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0                                                                                                                        
smooth_loss = -np.log(1.0/char_size)*seq_len # loss at iteration 0 
while n<=1000*100:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        # check "How to feed the loss function to see how this part works
        
        
    if p+seq_len+1 >= len(data) or n == 0:
     hprev = np.zeros((hidden_state,1)) # reset RNN memory                                                                                                                                      
    p = 0 # go from start of data                                                                                                                                                             
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_len]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_len+1]]
    
    # forward seq_length characters through the net and fetch gradient                                                                                                                          
    loss, dWxx, dWxh, dWhy, dbh, dby, hprev = Loss(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    
    # sample from the model now and then                                                                                                                                                        
    if n % 1000 == 0:
        print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
        sample(hprev, inputs[0], 200)
        
        # perform parameter update with Adagrad                                                                                                                                                     
        for param, dparam, mem in zip([Wxx, Wxh, Wxy, bh, by],
                                      [dWxx, dWxh, dWhy, dbh, dby],
                                      [mWxx, mWxh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update                                                                                                                   
            
            p += seq_len # move data pointer                                                                                                                                                         
            n += 1 # iteration counter
    
  
        
        
        
        
        
        
        
        
        
        
        
        
        














