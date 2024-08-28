fs = 16e3;
classificationRate = 20;
samplesPerCapture = fs/classificationRate;

segmentDuration = 1;
segmentSamples = round(segmentDuration*fs);

frameDuration = 0.025;
frameSamples = round(frameDuration*fs);

hopDuration = 0.010;
hopSamples = round(hopDuration*fs);


overlapSamples = frameSamples - hopSamples;

numSpectrumPerSpectrogram = floor((segmentSamples-frameSamples)/hopSamples) + 1;


afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);
setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);

numBands = 50;
setExtractorParameters(afe,'barkSpectrum','NumBands',numBands,'WindowNormalization',false);

numElementsPerSpectrogram = numSpectrumPerSpectrogram*numBands;

load('snoringcnn.mat')
labels = trainedNet.Layers(end).Classes;
NumLabels = numel(labels);

probBuffer = single(zeros([NumLabels,classificationRate/2]));
YBuffer = single(NumLabels * ones(1, classificationRate/2)); 

countThreshold = ceil(classificationRate*0.2);
probThreshold = single(0.7);

adr = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',samplesPerCapture,'OutputDataType','single');
audioBuffer = dsp.AsyncBuffer(fs);

matrixViewer = dsp.MatrixViewer("ColorBarLabel","Power per band (dB/Band)",...
    "XLabel","Frames",...
    "YLabel","Bark Bands", ...
    "Position",[400 100 600 250], ...
    "ColorLimits",[-4 2.6445], ...
    "AxisOrigin","Lower left corner", ...
    "Name","Speech Command Recognition using Deep Learning");

timeScope = timescope("SampleRate",fs, ...
    "YLimits",[-1 1], ...
    "Position",[400 380 600 250], ...
    "Name","Speech Command Recognition Using Deep Learning", ...
    "TimeSpanSource","Property", ...
    "TimeSpan",1, ...
    "BufferLength",fs, ...
    "YLabel","Amplitude", ...
    "ShowGrid",true);


show(timeScope)
show(matrixViewer)

timeLimit = inf;

tic
while isVisible(timeScope) && isVisible(matrixViewer) && toc < timeLimit
    % Capture audio
    x = adr();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-samplesPerCapture);
    
    % Compute auditory features
    features = extract(afe,y);
    auditoryFeatures = log10(features + 1e-6);
    
    % Perform prediction
    probs = predict(trainedNet, auditoryFeatures);      
    [~, YPredicted] = max(probs);
    
    % Perform statistical post processing
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];

    [YModeIdx, count] = mode(YBuffer);
    maxProb = max(probBuffer(YModeIdx,:));
    speechCommandIdx = YModeIdx;

    % Update plots
    matrixViewer(auditoryFeatures');
    timeScope(x);

    timeScope.Title = char(labels(speechCommandIdx));

    drawnow limitrate 
end  

hide(matrixViewer)
hide(timeScope)
