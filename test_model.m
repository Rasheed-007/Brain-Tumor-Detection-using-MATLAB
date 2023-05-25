% Load the trained KNN model
load('brain_cancer_knn_model.mat', 'mdl');

% Set the input image folder path
testFolder = "E:\ENG Sem 6\Projects\BrainTUmor\archive\test"; % Folder containing test images

% Read test image files from the folder
testFiles = dir(fullfile(testFolder, '*.jpg'));

% Initialize the predictions
predictions = [];

% Process test images
for i = 1:numel(testFiles)
    imagePath = fullfile(testFolder, testFiles(i).name);
    testImage = imread(imagePath);
    
    % Preprocess the test image to match the training data format
    % Resize the test image to the same size as the training images
    testImage = imresize(testImage, [256, 256]);
    
    % Convert the test image to grayscale if needed
    if size(testImage, 3) > 1
        testImage = rgb2gray(testImage);
    end
    
    % Normalize the image pixel values to the range [0, 1]
    testImage = double(testImage) / 255.0;
    
    % Reshape the test image to match the number of columns in the training data
    testImage = reshape(testImage', 1, numel(testImage));
    
    % Predict using the trained KNN model
    label = predict(mdl, testImage);
    predictions = [predictions; label];
end

% Display the predictions
for i = 1:numel(testFiles)
    fprintf('Image: %s, Prediction: %d\n', testFiles(i).name, predictions(i));
end
