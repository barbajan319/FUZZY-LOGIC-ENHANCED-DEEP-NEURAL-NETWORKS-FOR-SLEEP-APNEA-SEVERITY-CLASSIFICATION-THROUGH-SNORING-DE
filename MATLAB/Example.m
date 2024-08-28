clear;
clc;

%Load Dataset into a datastore 
datafolder = "C:\Users\luisb\OneDrive\Documents\Audio_Classification\CNNAudio";

dataset = fullfile(datafolder);
ads = audioDatastore(fullfile(dataset), ...
    "IncludeSubfolders",true, ...
   "LabelSource","foldernames", ...
  "FileExtensions",".wav");

%split dataset into training and testing
[adsTrain,adsValidation] = splitEachLabel(ads,0.7,0.3);

%Parameters for each clip
fs = 16e3; % Known sample rate of the data set.

segmentDuration = 3;
frameDuration = 0.030;
hopDuration = 0.020;

FFTLength = 1024;
numBands = 50;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

%Extracts the feature from each file in the datastore 
afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    FFTLength=FFTLength, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);
setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);

x = read(adsTrain);
numSamples = size(x,1);

% numToPadFront = floor((segmentSamples - numSamples)/2);
% numToPadBack = ceil((segmentSamples - numSamples)/2);
% xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

features = extract(afe, x);
[numhops, numFeatures] = size(features)
%Pads audio to a certain length
transform1 = transform(adsTrain,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
%extracts features 
transform2 = transform(transform1,@(x)extract(afe,x));
%turns spectogram into logarithm spectrogram
transform3 = transform(transform2,@(x){log10(x+1e-6)});

%Reads each file of datastore and applies the transform into spectrogram
XTrain = readall(transform3);

XTrain = cat(4,XTrain{:});

[numHops,numBands,numChannels,numFiles] = size(XTrain);
transform1 = transform(adsValidation,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
transform2 = transform(transform1,@(x)extract(afe,x));
transform3 = transform(transform2,@(x){log10(x+1e-6)});
XValidation = readall(transform3);
XValidation = cat(4,XValidation{:});

TTrain = adsTrain.Labels;
TValidation = adsValidation.Labels;

%Visualize 
specMin = min(XTrain,[],"all");
specMax = max(XTrain,[],"all");
idx = randperm(numel(adsTrain.Files),3);
figure(Units="normalized",Position=[0.2,0.2,0.6,0.6]);

tiledlayout(2,3)
for ii = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(ii)});

    nexttile(ii)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(ii))))
    
    nexttile(ii+3)
    spect = XTrain(:,:,1,idx(ii))';
    pcolor(spect)
    clim([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

classes = categories(TTrain);
classWeights = 1./countcats(TTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(classes);

layers = [
    imageInputLayer([numHops,afe.FeatureVectorLength])

    convolution2dLayer(3, 32, "Padding","same","Stride",2)
    batchNormalizationLayer
    reluLayer

%     groupedConvolution2dLayer(3,1, 32, "Padding","same", Stride = 1)
%     batchNormalizationLayer
%     reluLayer

    convolution2dLayer(1, 64, "Padding","same","Stride",1)
    batchNormalizationLayer
    reluLayer
% 
%     groupedConvolution2dLayer(3,1,64,"Padding","same","Stride",2)
%     batchNormalizationLayer
%     reluLayer
    

    convolution2dLayer(3, 128, "Padding","same","Stride",1)
    batchNormalizationLayer
    reluLayer

%     groupedConvolution2dLayer(3,1,128,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer

    convolution2dLayer(1, 128, "Padding","same","Stride",1)
    batchNormalizationLayer
    reluLayer

%     groupedConvolution2dLayer(3,1,128,"Padding","same","Stride",2)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(3, 256, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer

%     groupedConvolution2dLayer(3,1,256,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% % 
%     convolution2dLayer(1, 256, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     groupedConvolution2dLayer(3,1,256,"Padding","same","Stride",2)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(3, 512, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     groupedConvolution2dLayer(3,1,512,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
    convolution2dLayer(1, 512, "Padding","same","Stride",1)
    batchNormalizationLayer
    reluLayer
% 
%     groupedConvolution2dLayer(3,1,512,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(1, 512, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     groupedConvolution2dLayer(3,1,512,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(1, 512, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     groupedConvolution2dLayer(3,1,512,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(1, 512, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     groupedConvolution2dLayer(3,1,512,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(1, 512, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer

%     groupedConvolution2dLayer(3,1,512,"Padding","same","Stride",2)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(1, 1024, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     groupedConvolution2dLayer(3,1,1024,"Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(1, 1024, "Padding","same","Stride",1)
%     batchNormalizationLayer
%     reluLayer

%     globalAveragePooling2dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

miniBatchSize = 128;
validationFrequency = floor(numel(TTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    MaxEpochs=15, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    LearnRateSchedule="piecewise",...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=validationFrequency);

trainedNet_2 = trainNetwork(XTrain,TTrain,layers,options);
save trainedNet_2;
YValidation = classify(trainedNet_2,XValidation);
validationError = mean(YValidation ~= TValidation);
YTrain = classify(trainedNet_2,XTrain);
trainError = mean(YTrain ~= TTrain);

disp(["Training error: " + trainError*100 + "%";"Validation error: " + validationError*100 + "%"])

figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
cm = confusionchart(TValidation,YValidation, ...
    Title="Confusion Matrix for Validation Data", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");


% outputDir = "C:\Users\luisb\OneDrive\Documents\Audio_Classification";
% outputFile = fullfile(outputDir, "SnoringDetection.mat");
% Snoring_CNN = trainedNet_2;
% save Snoring_CNN

% saveLearnerForCoder(trainedNet_2,'CNNModel.mat');
% save(outputFile, "Example");
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 






% % Initialize a variable to stor
% e the spectrograms
% spectrograms = cell(length(ads.Files),1);
% 
% % Loop over the audio files in the datastore
% for i = 1:length(ads.Files)
%     % Read in one audio file at a time
%     audio = read(ads);
%     % Compute the spectrogram of the audio file
%     spectrograms{i}= spectrogram(audio);
% end

