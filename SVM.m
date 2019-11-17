D = '/Users/admin/Documents/Exchange/appliedML/Ass2/data/Cyrillic'
imds = imageDatastore(D,'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames','ReadFcn',@customreader);
[imds_c, imds_extra] = splitEachLabel(imds,200);
[trainingSet, validationSet] = splitEachLabel(imds_c, 0.6, 'randomize');
opts_g = templateSVM('BoxConstraint',1.1,'KernelFunction','gaussian');

opts_l = templateSVM('BoxConstraint',1.1,'KernelFunction','linear');

opts_q = templateSVM('BoxConstraint',1.1,'KernelFunction','polynomial', 'PolynomialOrder',2);
bag = bagOfFeatures(trainingSet);
categoryClassifier_g = trainImageCategoryClassifier(trainingSet, bag,'LearnerOptions', opts_g);
categoryClassifier_q = trainImageCategoryClassifier(trainingSet, bag,'LearnerOptions',opts_q);
categoryClassifier_l = trainImageCategoryClassifier(trainingSet, bag,'LearnerOptions',opts_l);
% confMatrix = evaluate(categoryClassifier, trainingSet);
confMatrix_g = evaluate(categoryClassifier_g, validationSet);
confMatrix_q = evaluate(categoryClassifier_q, validationSet);
confMatrix_l = evaluate(categoryClassifier_l, validationSet);
result=[mean(diag(confMatrix_l)),mean(diag(confMatrix_q)),mean(diag(confMatrix_g))]
disp(result)
function data = customreader(filename)
    [A,map,transparency] = imread(filename);
    gray_I=mat2gray(transparency);
    data=imresize(gray_I,[28 28]);
end