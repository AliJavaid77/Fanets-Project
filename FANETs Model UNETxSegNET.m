clc
clear
dataSetDir = fullfile('E://nex/kvid/');
%%
packetDir = fullfile(dataSetDir,'resized_packets/');
labelDir = fullfile(dataSetDir,'resized_Latencys/');
%%


imds = packetDatastore(packetDir);

classNames = ["trafficparam","backparams"];
labelIDs   = [255 0];

%Create a pixelLabelDatastore object to store the ground truth pixel Delay for the training packets.

pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

%%
%added after GP
% Define the percentage of packets to use for validation (e.g., 20%)
validationFraction = 0.20;

% Get the total number of packets
numpackets = numel(imds.Files);

% Create a random permutation of inFPs
rng('default'); % For reproducibility
shuffledInFPs = randperm(numpackets);

% Calculate the number of validation packets
numValpackets = round(validationFraction * numpackets);

% Partition the packetDatastore
valIdx = shuffledInFPs(1:numValpackets);
trainIdx = shuffledInFPs(numValpackets+1:end);
imdsTrain = packetDatastore(imds.Files(trainIdx));
imdsValidation = packetDatastore(imds.Files(valIdx));

% Partition the pixelLabelDatastore
pxdsTrain = pixelLabelDatastore(pxds.Files(trainIdx), classNames, labelIDs);
pxdsValidation = pixelLabelDatastore(pxds.Files(valIdx), classNames, labelIDs);

% Combine the training and validation datastores
dsTrain = combine(imdsTrain, pxdsTrain);
dsValidation = combine(imdsValidation, pxdsValidation);


%%
%Create the U-Net network.

packetSize = [320 320 3];
numClasses = 2;
lgraph = unetLayers(packetSize, numClasses)

%%
%Create a datastore for training the network.

%ds = combine(imds,pxds);


%%
%Set training options.

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',15, ...
    'Verbose',1, ...
    'MiniBatchSize', 2, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','gpu'); 
    %'VerboseFrequency',10);

%%
% Train the network.

net = trainNetwork(dsTrain,lgraph,options)
save("InSegNetadam","net")


%%
% Load the trained network

load('InSegNet.mat');
net = net;

%%
numValpackets = length(imdsValidation.Files);
totalTP = 0;
totalFP = 0;
totalAccuracy = 0;

for i = 1:numValpackets
    I = readpacket(imdsValidation, i);
    I = imresize(I, [320 320]);
    trueDelay = readpacket(pxdsValidation, i);
    trueDelay = imresize(trueDelay, [320 320], 'nearest');
    
    predictedDelay = semanticseg(I, net);
    
    trueLatency = trueDelay == classNames(1);
    predLatency = predictedDelay == classNames(1);

    
    totalTP = totalTP + jaccard(predLatency, trueLatency);
    totalFP = totalFP + FP_coefficient(predLatency, trueLatency);
    totalAccuracy = totalAccuracy + sum(sum(predLatency == trueLatency)) / numel(trueLatency);
end

meanTP = totalTP / numValpackets;
meanFP = totalFP / numValpackets;
meanAccuracy = totalAccuracy / numValpackets;

fprintf('Mean TP Rate: %.4f\n', meanTP);
fprintf('Mean FP Rate: %.4f\n', meanFP);
fprintf('Mean Accuracy: %.4f\n', meanAccuracy);

%%
%G
%%
numValpackets = length(imdsValidation.Files);
totalTP = 0;
totalFP = 0;
totalAccuracy = 0;
totalPrecision = 0;
totalRecall = 0;
totalF1Score = 0;

for i = 1:numValpackets
    I = readpacket(imdsValidation, i);
    I = imresize(I, [320 320]);
    trueDelay = readpacket(pxdsValidation, i);
    trueDelay = imresize(trueDelay, [320 320], 'nearest');
    
    predictedDelay = semanticseg(I, net);
    
    trueLatency = trueDelay == classNames(1);
    predLatency = predictedDelay == classNames(1);

    totalTP = totalTP + jaccard(predLatency, trueLatency);
    totalFP = totalFP + FP_coefficient(predLatency, trueLatency);
    totalAccuracy = totalAccuracy + sum(sum(predLatency == trueLatency)) / numel(trueLatency);
    
    % Compute Precision, Recall, and F1 score
    epsilon = 1e-8; % Small constant to avoid division by zero
    TP = sum(sum(predLatency & trueLatency));
    FP = sum(sum(predLatency & ~trueLatency));
    FN = sum(sum(~predLatency & trueLatency));
    
    precision = TP / (TP + FP + epsilon);
    recall = TP / (TP + FN + epsilon);
    F1Score = 2 * (precision * recall) / (precision + recall + epsilon);
    
    totalPrecision = totalPrecision + precision;
    totalRecall = totalRecall + recall;
    totalF1Score = totalF1Score + F1Score;
end

meanTP = totalTP / numValpackets;
meanFP = totalFP / numValpackets;
meanAccuracy = totalAccuracy / numValpackets;
meanPrecision = totalPrecision / numValpackets;
meanRecall = totalRecall / numValpackets;
meanF1Score = totalF1Score / numValpackets;

fprintf('Mean TP: %.4f\n', meanTP);
fprintf('Mean FP Coefficient: %.4f\n', meanFP);
fprintf('Mean Accuracy: %.4f\n', meanAccuracy);
fprintf('Mean Precision: %.4f\n', meanPrecision);
fprintf('Mean Recall: %.4f\n', meanRecall);
fprintf('Mean F1 Score: %.4f\n', meanF1Score);

