import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'} # class 1 will be in red and class -1 will be in blue
        if self.visualization:
            self.fig = plt.figure() # Figure to whole window
            self.ax = self.fig.add_subplot(1,1,1) # 1*1 grid plot no. 1
    
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}
        
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
                      
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature) # Simple list of individual features
                    
        # print("all_data : ",all_data)            
                    
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data) 
        # print("Max : ",self.max_feature_value,"\n")
        # print("Min : ",self.min_feature_value,"\n")
        all_data = None
    
        step_sizes = [self.max_feature_value*0.1,
                      self.max_feature_value*0.01,
                      # point of expenses:
                      self.max_feature_value*0.001]  
        # print("Step : ",step_sizes)
        # extremely exoensive              
        b_range_multiple = 5 
        # we don't need to take as small of steps
        # with b as we do with w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        # print("latest_optimum : ",latest_optimum)
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # print("w : ",w)
            # we can do this because its a convex problem
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple, 
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        # print("w_t",w_t)
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w + b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i # Class
                                if not yi*(np.dot(w_t,xi)+b) >= 1: # yi((w.x) + b) always should be >=1
                                    found_option = False
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b] # ||w|| : [w,b]        
                if w[0] < 0:
                    optimized = True
                    print('optimized a step...')
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict]) # Minimum norm (because we want to minimze ||w||)
            # print("norms[0] : ",norms[0])
            # print("opt_dict : ",opt_dict)
            opt_choice = opt_dict[norms[0]]
            # print("opt_choice : ",opt_choice)
           
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step*2
            # print("Opt_choice[0][0] : ",opt_choice[0][0])
                      
             
    def predict(self, features):
        # sign(x.w + b)
        classification = np.sign(np.dot(np.array(features),self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])  
        return classification
        
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in data_dict[i]] for i in data_dict]
          
        # hyperplane = x.w + b
        # v = x.w + b
        # psv = 1
        # nsv = -1
        # decision_boundary = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1] # Remember
        
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x + b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k') # Plot pairs : [hyp_x_min,psv1],[hyp_x_max,psv2]
        
        # (w.x + b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k') # Plot pairs : [hyp_x_min,nsv1],[hyp_x_max,nsv2]
        
        # (w.x + b) = 0
        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2],'y') # # Plot pairs : [hyp_x_min,db1],[hyp_x_max,db2]
        
        plt.show()

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8]]), 
              1:np.array([[5,1],
                          [6,-1],
                          [7,3]])} # -1 represents one class and 1 represents the other class

svm = Support_Vector_Machine()
svm.fit(data_dict)

predict_us = [  [0,10],
                [1,3],
                [3,4],
                [3,5],
                [5,5],
                [5,6],
                [6,-5],
                [5,8]]
                
for p in predict_us:
    svm.predict(p)

svm.visualize()                
                