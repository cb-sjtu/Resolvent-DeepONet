from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing
from keras import layers
import numpy as np
# import tensorflow as tf


class MIONet(NN):
    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class MIONetCartesianProd(MIONet):
    """MIONet with two input functions for Cartesian product format."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True




class   MIONet_CNN(NN):
    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,85*85,self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2),tf.expand_dims(y_func2, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            self.y = tf.multiply(y_func1, y_loc)
            self.y = tf.multiply(self.y, y_func2)
            self.y = tf.reduce_sum(self.y, 1, keepdims=True)



        
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None,None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class MIONet_CNN_no_average_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            self.y_net = y_func1



        
        b = tf.Variable(tf.zeros(1))
        self.y_net += b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class SVD_DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,10))
            y_func=y_func1[:,:24*10]
            y_c=y_func1[:,24*10:24*10+87*24]
            y_d=y_func1[:,24*10+87*24:]
            y_func=tf.reshape(y_func,(-1,1,24,10))
            self.y_net = tf.multiply(y_func, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24))
        self.y_net=tf.multiply(self.y_net,y_c)+y_d

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        # b = tf.Variable(tf.zeros(1))
        # self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1],activation=activation, kernel_regularizer=self.regularizer)

class flex_DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        # layer_sizes_trunk2,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        # self.layer_trunk2 = layer_sizes_trunk2
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
            # self.activation_trunk2 = activations.get(activation["trunk2"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        # self.X_loc2 = tf.placeholder(config.real(tf), [None,self.layer_trunk2[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1,self.X_func2, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        y_func2_1=tf.reshape(y_func2[:,:87],(-1,1))
        # y_func2_2=tf.reshape(y_func2[:,87:],(-1,1))
        self.X_loc=tf.multiply(self.X_loc,y_func2_1)

        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)
        
        

        # # Trunk net2
        # if callable(self.layer_trunk2[1]):
        #     # User-defined network
        #     y_loc2 = self.layer_trunk2[1](self.X_loc2)
        # else:   
        #     y_loc2 = self._net(self.X_loc2, self.layer_trunk2[1:], self.activation_trunk2)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,10))
            y_func=y_func1[:,:24*10]
            y_c=y_func1[:,24*10:24*10+87*24]
            y_d=y_func1[:,24*10+87*24:]
            y_func=tf.reshape(y_func,(-1,1,24,10))
            self.y_net = tf.multiply(y_func, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24))
        self.y_net=tf.multiply(self.y_net,y_c)+y_d
        

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        # b = tf.Variable(tf.zeros(1))
        # self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        # y_loc2=tf.reshape(y_loc2,(-1,1,100))
        # print(y_loc2.shape)
        # q=tf.multiply(q,y_loc2)
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)

class DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,10))
            y_func1=tf.reshape(y_func1,(-1,1,24,10))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)
    
class DeepONet_resolvent_2d(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,50))
            y_func1=tf.reshape(y_func1,(-1,1,24,50))
          
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')
    
        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1],activation=activation, kernel_regularizer=self.regularizer)
    


class DeepONet_resolvent_jiami(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,173*47,200])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,46])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,173,47,50))
            y_func1=tf.reshape(y_func1,(-1,1,47,50))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,173*47,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b


        #积分
       
            

        q=self.trunk_out*self.y_net

        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,173,47,200))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,46,1))
        x_sum3=tf.multiply(x_sum1,first)
        x_sum3=tf.multiply(x_sum3,0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        

        self.y=tf.reshape(x_1,(-1,173*200))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        # self.y=self.y_net

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1],kernel_regularizer=self.regularizer)



