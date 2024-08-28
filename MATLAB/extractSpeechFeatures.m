clear;
clc;

%Load Dataset into a datastore 
datafolder = "C:\Users\luisb\OneDrive\Desktop\1sec";

dataset = fullfile(datafolder);
ads = audioDatastore(fullfile(dataset), ...
    "IncludeSubfolders",true, ...
   "LabelSource","foldernames", ...
  "FileExtensions",".wav");

%split dataset into training and testing
[adsTrain,adsValidation] = splitEachLabel(ads,0.7,0.3);

%Parameters for each clip
fs = 16e3; % Known sample rate of the data set.

segmentDuration = 1;
frameDuration = 0.030;
hopDuration = 0.0201;

FFTLength = 512;
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