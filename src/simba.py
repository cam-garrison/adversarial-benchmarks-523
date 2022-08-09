import torch as nn
import torch.nn.functional as F

from tqdm import trange

class SimBA_Attack(object):

    def __init__(self, model=None ):
        self.model = model
    
    def get_probs(self, x):
        if self.model is None:
            print("Model not defined!")
            return

        with nn.no_grad():
            output = self.model( x.view(-1, 3, 224, 224) )

        sf = F.softmax( output, dim=1 )
        probs = list( sf.numpy() )
        
        return probs[0]
    
    def simba( self, x, y, epsilon=0.05, steps=100000 ):
        if self.model is None:
            print("Model not defined!")
            return
    
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
        
        for i in trange( steps ):
            
            #if the model misclassifies the input
            if( last_probs[y] != last_probs.max() ):
                print(last_probs[y])
                print(last_probs.max())
                finished = True
                break
            
            #make the perturbation for this step
            pert = nn.zeros( ndims )
            pert[ perm[ i % ndims ] ] = epsilon
            
            
            x_temp = (x - pert.view(x.size()))#.clamp(0, 1)
            left_prob = self.get_probs( x_temp )
            
            if( left_prob[y] <= last_probs[y] ):
                print()
                print(f"Acc: {last_probs[y]}")
                print()
                x = x_temp
                last_probs = left_prob
                pert_history -= pert
                
            else:
                x_temp = (x + pert.view(x.size()))#.clamp(0, 1)
                right_prob = self.get_probs( x_temp )
                
                if( right_prob[y] <= last_probs[y] ):
                    print()
                    print(f"Acc: {last_probs[y]}")
                    print()
                    x = x_temp
                    last_probs = right_prob
                    pert_history += pert
        
        return x, pert_history, last_probs, finished 

