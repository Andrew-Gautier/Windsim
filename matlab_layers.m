import torch
import scipy.io as sio

# Load your trained PyTorch model
model = WindFaultCNN(input_channels=..., num_classes=..., window_size=...)
model.load_state_dict(torch.load('model.pth'))

# Export model weights
weights = {
    'conv1.weight': model.conv1.weight.detach().numpy().transpose(2, 1, 0),  # [kernel, in_ch, out_ch]
    'conv1.bias': model.conv1.bias.detach().numpy(),
    'conv2.weight': model.conv2.weight.detach().numpy().transpose(2, 1, 0),
    'conv2.bias': model.conv2.bias.detach().numpy(),
    'fc1.weight': model.fc1.weight.detach().numpy(),
    'fc1.bias': model.fc1.bias.detach().numpy(),
    'fc2.weight': model.fc2.weight.detach().numpy(),
    'fc2.bias': model.fc2.bias.detach().numpy(),
}
sio.savemat('model_weights.mat', weights)

# Export scaler parameters (from your dataset)
sio.savemat('scaler_params.mat', {
    'scaler_min': train_dataset.scaler.min_,
    'scaler_scale': train_dataset.scaler.scale_
})

function y = FaultPredictor(input_window) %#codegen
% Real-time fault prediction for Simulink

persistent net scaler_min scaler_scale

if isempty(net)
    % Load parameters once
    weights = coder.load('model_weights.mat');
    scaler = coder.load('scaler_params.mat');
    scaler_min = scaler.scaler_min;
    scaler_scale = scaler.scaler_scale;
    
    % Define network architecture
    inputChannels = size(weights.conv1_weight, 2);
    windowSize = size(input_window, 1); % Adjust based on your data
    numClasses = size(weights.fc2_weight, 1);
    
    layers = [
        sequenceInputLayer([inputChannels, windowSize], 'Name', 'input')
        convolution1dLayer(size(weights.conv1_weight, 1), 16, ...
            'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool1')
        convolution1dLayer(size(weights.conv2_weight, 1), 32, ...
            'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'pool2')
        flattenLayer('Name', 'flatten')
        fullyConnectedLayer(64, 'Name', 'fc1')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(numClasses, 'Name', 'fc2')
        softmaxLayer('Name', 'softmax')
    ];
    
    % Assign weights
    layers(2).Weights = weights.conv1_weight;
    layers(2).Bias = weights.conv1_bias;
    layers(5).Weights = weights.conv2_weight;
    layers(5).Bias = weights.conv2_bias;
    layers(9).Weights = weights.fc1_weight;
    layers(9).Bias = weights.fc1_bias;
    layers(11).Weights = weights.fc2_weight;
    layers(11).Bias = weights.fc2_bias;
    
    % Create network
    net = assembleNetwork(layers);
end

% Preprocess input
normalized = (input_window - scaler_min) .* scaler_scale;
input_data = normalized'; % Transpose to [channels, sequence]

% Predict
y = predict(net, input_data);
end