D = '/Users/admin/Documents/Exchange/appliedML/Ass2/data/Cyrillic'
imds = imageDatastore(D,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@customreader);
dbpathSave={'/Users/admin/Documents/Exchange/appliedML/Ass2/data/Cyrillic/Ж',...
     '/Users/admin/Documents/Exchange/appliedML/Ass2/data/Cyrillic/I',...
      '/Users/admin/Documents/Exchange/appliedML/Ass2/data/Cyrillic/Ю',};
exts ={'.png'};
     imds = imageDatastore( dbpathSave,'FileExtensions',exts,'IncludeSubfolders',0,'LabelSource' ,'foldernames','ReadFcn',@customreader);
     
[imds_c, imds_extra] = splitEachLabel(imds,100);
% figure;
read(imds_c)
labelCount = countEachLabel(imds_c)
numTrainFiles = 100*0.8;
[imdsTrain,imdsValidation] = splitEachLabel(imds_c,numTrainFiles,'randomize');
labelCount = countEachLabel(imdsTrain)
inputSize=[28 28 1]
numClasses=numel(categories(imdsTrain.Labels))
layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
%     fullyConnectedLayer(12)
%     batchNormalizationLayer
%     tanhLayer
   
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% layers1 = [
%     imageInputLayer(inputSize)
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     leakyReluLayer
%     
%     convolution2dLayer(3,40,'Padding','same')
%     batchNormalizationLayer
%     
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% layers2 = [
%     imageInputLayer(inputSize)
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     tanhLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     tanhLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     tanhLayer
%     
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% layers3 = [
%     imageInputLayer(inputSize)
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(3,40,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% 
% layers4 = [
%     imageInputLayer(inputSize)
%     
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     crossChannelNormalizationLayer(3)
%     batchNormalizationLayer
%     reluLayer
%     
%     convolution2dLayer(3,40,'Padding','same')
%     crossChannelNormalizationLayer(3)
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
 net = trainNetwork(imdsTrain,layers,options);
 YPred = classify(net,imdsValidation);
 YValidation = imdsValidation.Labels;
  
   accuracy = sum(YPred == YValidation)/numel(YValidation)

function data = customreader(filename)
    [A,map,transparency] = imread(filename);
    gray_I=mat2gray(transparency);
    data=imresize(gray_I,[28 28]);
end