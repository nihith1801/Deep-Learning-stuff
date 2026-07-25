import numpy as np
try:
    import mlx.core as mx
    engine=mx
    BACKEND="mlx"

except ImportError:
        engine=np
        BACKEND="numpy"

class BaseLayer: #Simple abstract class to implement other methods 
    def __init__(self):
        self.input=input
        self.output=self.output
    
    def forward_pass(self,input_data):
        raise NotImplementedError
    
    def backward_pass(self,output_gradient,learning_rate):
        raise NotImplementedError

    def regularization_loss(self):
        return 0.0

    

class Tensor: #Wrapping np array into mlx or cuda py
    def __init__(self,data,dtype=None):
        if isinstance(data,(list,tuple)):
            data=np.array(data)
        if isinstance(data,np.ndarray):
            if BACKEND=="mlx":
                self.data=mx.array(data,dtype=dtype) if dtype else mx.array(data)
            else:
                self.data=data.astype(dtype) if dtype else data
        else:
            self.data=data
    
    def __array__(self):
        #In order to convert back from cuda or mlx to numpy
        if BACKEND=="mlx":
            return np.array(self.data)
    
    def numpy(self):
        return np.array(self)

    @property
    def shape(self):
        return tuple(self.data.shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape},backned={BACKEND})"


#Helper functions define!!!! 
def _to_engine(arr):
    if isinstance(arr, Tensor):
        return arr.data
    if isinstance(arr,np.ndarray):
        return mx.array(arr) if BACKEND=="mlx" else arr

    return arr

def __to_numpy(arr):
    if isinstance(arr,Tensor): return arr.numpy()
    if BACKEND=="mlx" and "mlx" in str(type(arr)):
        return np.array(arr) 
    return np.array(arr)

def _zeroes(shape):
    return engine.zeros(shape,dtype=engine.float32) #Shape of the zeroes

def _ones(shape):
    return engine.ones(shape,dtype=engine.float32) #Handling of the non zero values

def _randn(shape,mean=0.0,std=1.0):
    return engine.random.normal(mean,std,shape)   #Rnadom using normla distrubution

def _exp(x):
    return engine.exp(_to_engine(x))   #Converting to exponent


def _log(x,eps=1e-8):  #Setting eps=1e-8
    return np.log(_to_engine(x)+eps)


def _sum(x,axis=None,keepdims=False):
    return engine.sum(_to_engine(x),axis=axis,keepdims=keepdims)

def _max(x,axis=None,keepdims=False):
    return engine.max(_to_engine(x),axis=axis)

def _mean(x,axis=None):
    return engine.mean(_to_engine(x),axis=axis)

def _clip(x,low,hi):
    return engine.clip(_to_engine(x),low,hi)

def _dot(a,b):
    return engine.matmul(_to_engine(a),_to_engine(b)) if engine=="mlx" else np.dot(a,b)

def _transpose(a):
    return engine.transpose(_to_engine(a))

def _reshape(x,shape):
    return engine.reshape(_to_engine(x),shape)

def _where(condition,a,b): #Values to choose and condition to be applied (condition, reutrn value a if condition true, return value b if condition false)
    return engine.where(_to_engine(condition),_to_engine(a),_to_engine(b))

def _sign(x):
    return engine.sign(_to_engine(x))

def _abs(x):
    return engine.abs(_to_engine(x))

class ReLU(BaseLayer):
    '''
    Relu is recitified linear unit activation function. 
    It only returns f(x)=max(0,x)
    S it only returns positive values otherwise 0. Since the images we handle
    are 0-255, relu is a nice activation function we can use.

    Using forward from base layer. This class will act as a dense layer, 
    contianing relu activation functions
    '''
    def forward_pass(self,x):
        self.input=_to_engine(x)
        self.mask=_where(self.input>0, _ones(self.input.shape),_zeroes(self.input.shape)) #To prevent negative values/ applying relu
        self.output=self.input*self.mask
        return Tensor(self.output)

    def backward_pass(self, output_gradient, learning_rate=None):
        output_gradient=_to_engine(output_gradient) #Converting from numpy to mlx engine
        return Tensor(output_gradient*self.mask)


class LeakyReLU(BaseLayer):
    '''
    Sometimes..simply relu is too much.
    Cutting off the values ( i mean clipping them to 0 if it becomes negative) then 
    we do a alpha*x . now alpha a smaller value which is multiplied with x
    example x=-1 -> f(x)= x if x>0 which is not so it gives alpha*x = 0.01 *x -> -0.01
    This small values trail of -ve values between 0 and -1 is leaky relu
    now this can be used during masking like if negative values appear.
    '''

    def __init__(self,alpha=0.01):
        super().__init__()
        self.alpha=alpha

    def forward_pass(self, x):
        self.input=_to_engine(x)
        self.output=_where(self.input>0, self.input,self.alpha*self.input)
        return Tensor(self.output)

    def backward_pass(self, output_gradient, learning_rate=None):
        output_gradient=_to_engine(output_gradient)
        mask=_where(self.input>0,_ones(self.input.shape),_ones(self.input.shape)*self.alpha)
        return Tensor(output_gradient*mask)


class Softmax(BaseLayer):
    '''
    Now softmax is an interesting function
    f(x)= e^xi/Sigma(e^-xj) where j=0->k ( all classes)
    We iterate across the classes.
    Basically sincee we're comparing with other classes
    this kind of gives like a probability. A score which which is wrt to other classes
    thus the formula is the exponential of that class value / others
    this is done for each class, and the the softmax function,
    since it's a probability, transfomes those logits (raw values),
    which we get from the dense layers into probability scores (distirbution).
    in the end,we take the one which has the maximum probability
    Very useful in multi class classificaiton.  
    '''

    def forward_pass(self, x):
        self.input(_to_engine(x))
        max_values=_max(self.input,axis=1,keepdims=True)
        shifted=self.input-max_values
        exponential_shift=_exp(shifted)
        self.output=exponential_shift/_sum(exponential_shift,axis=1,keepdims=True) #Softmax f(x)
        return Tensor(self.output)

    def backward_pass(self, output_gradient, learning_rate=None):
        output_gradient=_to_engine(output_gradient)
        sum_gs=_sum(output_gradient*self.output,axis=1,keepdims=True)
        return Tensor(self.output*(output_gradient-sum_gs)) #Weight updation


class Sigmoid(BaseLayer):
    '''
    Now sigmoid is an interesting one as well. It is generally of two types
    Binary and bipolar which we also call tanh

    Now sigmoid is generally used for a binary classification. We can use tanh as well
    but we don't because we're delaing with positive value classes. Since,
    Sigmoid gives values between 0 and 1, it is always positive

    f(x)=1/1+e^x where 0<f(x)<1 never is pure 0 or 1
    '''
    def forward_pass(self, x):
        self.input=_to_engine(x)
        self.output=1.0/(1.0+_exp(-self.input)) #Binary sigmoid
        return Tensor(self.output)

    def backward_pass(self, output_gradient, learning_rate=None):
        output_gradient=_to_engine(output_gradient)
        return Tensor(output_gradient*self.output*(1.0-self.output))


#Now coming to dense layers

class Dense(BaseLayer):
    '''
    During forward pass, we take wreighted sums
    which is weights*inputs+bias
    now bias does have a weight assigned to it but it generally is set to 1
    Now we will define which one to use in the dense layer as activation function.
    "relu" for Relu and "sigmoid" for Sigmoid function
    '''
    def __init__(self,input_size,output_size,activation_function="relu",l1=0.0,l2=0.0):   #Default is set to Relu
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        #Introducting l1 and l2. Will be discussed later
        self.l1=l1
        self.l2=l2

        '''
        Now I did some research on why we must use 2.0 for relu and 1.0 for simgmoid
        Since normal disturbution is perfectly symmetrical, relu cuts off half of it
        I mean we can use leaky relu to handle negative values but to keep things simple here
        we use relu.
        since relu is value or 0, if and a big if a negative value comes, the neurons give it 0
        and thus empty gradient and the neurons shut off.
        While the 2.0 value doesn't essentially solve the dying neurons,
        it stabilizes it's data variance. Thus cancelling out the relu's dividing effect.
        So no data get's lost.

        Now sigmoid can give a value negative value but it's bipolar sigmoid ( tanh )
        since data is linear, the numerator is 1.

        
        '''
        if activation_function=="relu":
            #Applying standard deviation
            std=np.sqrt(2.0/input_size)
        elif activation_function=="sigmoid":
            std=np.sqrt(1.0/input_size)

        #Assinging random weights distribution based on the standard deviation we get.
        self.Weights=_randn((input_size,output_size),std=std)
        self.bias=_zeroes((1,output_size))

        self.differential_of_weight=None
        self.differenital_of_bias=None

    def forward_pass(self, x):
        self.input=_to_engine(x)
        '''
        Taking weighted sum ((Input * input's weight) + bias)
        '''
        self.output=_dot(self.input,self.Weights)+self.bias 
        return Tensor(self.output)

    '''
    Now doing backpropogation here
    '''
    def backward_pass(self, output_gradient, learning_rate):
        output_gradient=_to_engine(output_gradient)
        self.differential_of_weight=_dot(_transpose(input),output_gradient) #Taking weight differential delta w
        self.differenital_of_bias=_sum(output_gradient,axis=0,keepdims=True) #delta bias
        '''
        Now there might be cases where the model learns too much. Focuses on 
        noise uncesseraily and fits too much. Thus it has low bias but high variance
        OVERFITTING!

        Now to solve this issue, we use a term called regularization.
        There are 3 types of regularization
        1) L1 which is called Lasso
        2) L2 is called Ridge
        3) Hybrid which is basically a combination of these two ( l1+ l2)
        '''
        if self.l1>0:
            #Applying l1
            self.differential_of_weight=self.differential_of_weight+self.l1*_sign(self.Weights)
        if self.l2>0:
            self.differential_of_weight=self.differential_of_weight+self.l2*self.Weights #Ridge just squares so sign doesn't matter
        dx=_dot(output_gradient,_transpose(self.Weights))
        '''
        Now updating weights w(new)=w(old)-( delta w * learning rate)
        '''
        self.Weights=self.Weights-(learning_rate*self.differential_of_weight)
        self.bias=self.bias-(learning_rate*self.differenital_of_bias)
        return Tensor(dx)

    def regularization_loss(self):
        '''Computing l1+l2 penalty for loss'''
        regularization=0.0
        if self.l1>0:
            regularization=regularization+self.l1*float(__to_numpy(_sum(_abs(self.Weights)))) #Lasso taking absolute
        if self.l2>0:
            regularization=regularization+self.l2*float(__to_numpy(_sum(self.Weights**2))) #Ridge taking squared
        return regularization

    def params(self):
        return {"Weight": Tensor(self.Weights), "Bias": Tensor(self.bias)} #Returning a key value pair

    def set_parameters(self,p):
        if "W" in p: self.Weights=_to_engine(p['W'])
        if "b" in p: self.bias=_to_engine(p['b'])


'''
Now to introduce this regularization here, we add a layer called dropout layer.
Basically wieght stabilzers in simple terms
'''

class Dropout(BaseLayer):
    def __init__(self,rate=0.5):
        super().__init__()
        self.rate=rate
        self.scale=1.0/(1.0-rate)
        self.mask=None
        self.training=True

    def forward_pass(self, x):
        input=_to_engine(x)
        if self.training:
            rand=engine.random.unfiorm(0,1,self.input.shape)
            self.mask=_where(rand>self.rate,_ones(self.input.shape),_zeroes(self.input.shape))
            self.output=self.input*self.mask*self.scale
        else:
            self.output=self.input
        return Tensor(self.output)

    def backward_pass(self, output_gradient, learning_rate=None):
        output_gradient=_to_engine(output_gradient)
        if self.training:
            return Tensor(output_gradient*self.mask*self.scale)
        return Tensor(output_gradient)


'''

Now flatten layer. Converts this n * n array into a 1*n array. A single linear array.
So if an image is 256x256x1 then the image is flattened into (1, (256x256x1)) which is (1,65536)
'''

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape=None

    def forward_pass(self, x):
        self.input=_to_engine(x)
        self.input_shape=self.input.shape
        N=self.input_shape[0] #No. of columns, when we flatten it so no of values in the resultant 1d array
        flat=np.prod(self.input_shape[1:])
        self.output=_reshape(self.input,(N,flat))
        return Tensor(self.output)

    def backward_pass(self, output_gradient, learning_rate):
        return Tensor(_reshape(_to_engine(output_gradient),self.input_shape))

    
'''
Defining LOSS. Now loss functions, also knowns as the cost functions,
help us analyze the cost or difference between the expected value
and the actual value. Now few types are cross entorpy, rmse, mse, mae, etc
'''

class CrossEntropyLoss:
    def __init__(self):
        self.probabilities=None
        self.lables=None
        self.batch_size=None

    def forward_pass(self,predicitons,labels):
        self.probabilities=_to_engine(self.probabilities)
        self.lables=labels
        self.batch_size=self.probabilities.shape[0]
        n_classes=self.probabilities.shape[1]

        if labels.ndim==1:
            '''
            One hot coding 
            '''
            if BACKEND=="mlx":
                y_onehot=engine.zeros((self.batch_size,n_classes))
                for i in range(self.batch_size):
                    y_onehot[i,int(labels[i])]=1.0
            else:
                y_onehot=np.zeros((self.batch_size,n_classes))
                y_onehot[np.arrange(self.batch_size),labels.astype(int)]
            self.y_onehot=_to_engine(labels)

    def backward(self):
        return Tensor((self.probabilities-self.y_onehot)/self.batch_size)


'''
Now MSELoss is Mean Squared Error Loss
 f(x)= 1/n (sigma(actual value - predicted value)**2)
'''
class MSELoss:
    def forward_pass(self,predicitons,labels):
        self.predictions=_to_engine(predicitons)
        self.labels=_to_engine(labels)
        difference=self.predictions-self.labels
        '''Now implmenting cost function'''
        return float(__to_numpy(_sum(difference**2)/self.predictions.shape[0]))

    def backward_pass(self):
        return Tensor(2*(self.predictions-self.labels)/self.predictions.shape[0])
    

'''
Finally model class.
'''

class Model:
    def __init__(self):
        '''Defining Hyper parameters'''
        self.layers=[]
        self.loss_function=None
        self.learning_rate=0.01
        self.history={'loss':[],'accuracy':[],'regularization_loss':[]}

    def add(self,layer):
        self.layers.append(layer)
        return self
    '''Setting up loss functions'''
    def compile(self,loss="crossentropy",learning_rate=0.01):
        self.learning_rate=learning_rate
        self.loss_function=CrossEntropyLoss() if loss=="crossentropy" else MSELoss()

    '''Forward pass'''
    def _forward_pass(self,x):
        output=x
        for layer in self.layers:
            output=layer.forward_pass(output)
        return output
    '''Backward pass'''
    def _backward_pass(self,gradient):
        g=gradient
        for layer in reversed(self.layers):
            g=layer.backward(g,self.learning_rate)
        return g
    
    def _compute_reg_loss(self):
        '''Sum of L1/L2 penalties across all layers'''
        total=0.0 #If this remains 0.0 life would have been so easier :)
        for layer in self.layers:
            total+=layer.regularization_loss()
        return total
    
    def predict(self,x):
        for layer in self.layers:
            if isinstance(layer,Dropout):
                layer.training=False
        output=self._forward_pass(x)
        for layer in self.layers:
            if isinstance(layer,Dropout):
                layer.training=True
        return output


    def model_fit(self,X,y,epochs=10,batch_size=32,val_data=None,verbose=1,class_weights=None):
        n=X.shape[0]
        batches = max(1,n//batch_size)
        for epoch in range(epochs):
            idx=np.random.permutation(n) #Taking random indices for training
            X_s,y_s = X[idx], y[idx]
            
            