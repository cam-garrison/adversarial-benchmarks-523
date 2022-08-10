import torch as nn
import torch.nn.functional as F
import numpy as np

from tqdm import trange

class SimBA_Attack(object):

    def __init__(self, model=None ):
        self.model = model
    
    def get_probs(self, x):
        if self.model is None:
            print("Model not defined!")
            return

        self.model.eval()

        with nn.no_grad():
            logits = self.model( x.view(-1, 3, 224, 224) )

        preds = F.softmax( logits, dim=1 )
        # copy result to the cpu 
        preds = preds.cpu()
        probs = preds.numpy()
        
        return probs[0]
    
    def simba( self, x, y, device, epsilon=0.05, steps=100000):
        if self.model is None:
            print("Model not defined!")
            return
        
        self.model = self.model.to(device)
        x = x.to(device)
    
        #last classification distribution (softmax)
        #of input image.
        last_probs = self.get_probs( x )
        
        #number of dimensions of the input
        ndims = x.numel()
        
        #random basis dimensions
        perm = nn.randperm( ndims )
        
        #collect history of successful perturbations
        pert_history = nn.zeros( ndims )

        finished = False
        num_changes = 0
        
        for i in trange( steps ):
            
            # if the true label is no longer the entry
            # with the highest probability in the prob dist,
            # # we're done w/ the untargeted attack. 
            if y != np.argmax(last_probs):
                finished = True
                break
            
            #make the perturbation for this step
            pert = nn.zeros( ndims )
            pert[ perm[ i % ndims ] ] = epsilon
            
            
            x_temp = (x.cpu() - pert.view(x.size()))#.clamp(0, 1)
            x_temp = x_temp.to(device)
            left_prob = self.get_probs( x_temp )
            
            if( left_prob[y] < last_probs[y] ):
                x = x_temp
                last_probs = left_prob
                pert_history -= pert
                num_changes += 1
                
            else:
                x_temp = (x.cpu() + pert.view(x.size()))#.clamp(0, 1)
                x_temp = x_temp.to(device)
                right_prob = self.get_probs( x_temp )
                
                if( right_prob[y] < last_probs[y] ):
                    x = x_temp
                    last_probs = right_prob
                    pert_history += pert
                    num_changes += 1

        return x.cpu(), pert_history, last_probs, finished,  num_changes

