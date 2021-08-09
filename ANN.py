import numpy
import scipy.special

class neuralNetwork:

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate
        self.wih=(numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who =(numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        self.activation_function=lambda x: scipy.special.expit(x)
        pass

    def train(self,input_list,target_list):
        inputs=numpy.array(input_list,ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_ouputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_ouputs)

        final_outputs = self.activation_function(final_inputs)

        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(inputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_ouputs * (1.0 - hidden_ouputs)),numpy.transpose(inputs))
        pass
    def query(self,input_list):
        inputs=numpy.array(input_list,ndmin=2).T

        hidden_inputs=numpy.dot(self.wih,inputs)

        hidden_ouputs=self.activation_function(hidden_inputs)

        final_inputs=numpy.dot(self.who, hidden_ouputs)

        final_outputs=self.activation_function(final_inputs)

        return final_outputs

        pass


input_nodes=3
hidden_nodes=3
output_nodes=3

learning_rate=0.3

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

print(n.query([1.0,0.5,-1.5]))

