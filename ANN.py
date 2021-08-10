import numpy
import scipy.special
import matplotlib.pyplot

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

    def train(self,input_list,targets_list):
        inputs=numpy.array(input_list,ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_ouputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_ouputs)

        final_outputs = self.activation_function(final_inputs)

        output_errors=targets-final_outputs
        hidden_errors=numpy.dot(self.who.T,output_errors)
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_ouputs))
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


input_nodes=784
hidden_nodes=100
output_nodes=10
learning_rate=0.3



n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
training_data_file=open("mnist_dataset/mnist_train_100.csv",'r')
training_data_list=training_data_file.readlines()
training_data_file.close()



for record in training_data_list:
    all_values=record.split(',')
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    targets=numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])]=0.99
    n.train(inputs,targets)
    pass


test_data_file=open("mnist_dataset/mnist_test_10.csv",'r')
test_data_list=test_data_file.readlines()
test_data_file.close()

all_values2= test_data_list[0].split(',')
print(all_values2[0])

image_array=numpy.asfarray(all_values2[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

print(n.query((numpy.asfarray(all_values2[1:])/255.0*0.99)+0.01))





