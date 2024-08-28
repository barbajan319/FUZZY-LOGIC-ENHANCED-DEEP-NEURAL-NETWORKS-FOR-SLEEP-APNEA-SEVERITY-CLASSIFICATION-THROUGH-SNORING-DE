clear;
clc;

rng default


datafolder = "C:\Users\luisb\OneDrive\Documents\Audio_Classification\Train";
ads = audioDatastore(datafolder,IncludeSubfolders=true,LabelSource="foldernames");
[adsTrain,adsValidation,adsTest] = splitEachLabel(ads,0.7,0.2,0.1);
tdsTrain = transform(adsTrain,@audioPreprocess,IncludeInfo=true);
tdsValidation = transform(adsValidation,@audioPreprocess,IncludeInfo=true);
tdsTest = transform(adsTest,@audioPreprocess,IncludeInfo=true);

deepNetworkDesigner
% dataset = fullfile(datafolder);
% ads = audioDatastore(fullfile(dataset, "train"), ...
%     "IncludeSubfolders",true, ...
%    "LabelSource","foldernames", ...
%   "FileExtensions",".wav");
% outputs = categorical(["Snoring","Non Snoring"]);
% adsTrain = subset(ads,isOutput);
% adsTrain.Labels = removecats(adsTrain.Labels);





%Non_Snoring_spectrograms = Non_Snoring_spectrograms(:,:,1:2599);

%dataset = cat(4, Snoring_spectrograms, Non_Snoring_spectrograms);


function [data,info] = audioPreprocess(audioIn,info)
class = info.Label;
fs = info.SampleRate;
features = yamnetPreprocess(audioIn,fs);

numSpectrograms = size(features,4);

data = cell(numSpectrograms,2);
for index = 1:numSpectrograms
    data{index,1} = features(:,:,:,index);
    data{index,2} = class;
end
end