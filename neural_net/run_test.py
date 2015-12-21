import network
import mnist_loader


if __name__ == "__main__":
    image_size = 784
    hidden_layer_size = 20
    num_outputs = 10
    learning_rate = 3.0
    net = network.Network([image_size, hidden_layer_size, num_outputs])
    
    training_data, validation_wrapper, test_data = mnist_loader.load_data_wrapper()

    net.SGD(training_data, hidden_layer_size, num_outputs, learning_rate, test_data=test_data)

