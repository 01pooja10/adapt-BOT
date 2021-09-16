def hyperparams(xtrain,ytrain):
    input_size = len(xtrain[0])
    output_size = len(ytrain)
    #hidden_size = 80
    hidden_size = int(input('Enter the hidden size: '))
    #num_layers = 2
    num_layers = int(input('Enter the number of layers in the LSTM: '))
    #learning_rate = 0.001
    learning_rate = int(input('Enter the learning rate')
    return input_size, output_size, num_layers, hidden_size, learning_rate
