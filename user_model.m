% Load the trained KNN model
load('brain_cancer_knn_model.mat', 'mdl');

% Prompt the user to select the image file
[filename, pathname] = uigetfile('*.jpg', 'Select the image file');

% Check if a file was selected
if isequal(filename, 0) || isequal(pathname, 0)
    fprintf('No file was selected. Exiting...\n');
    return;
end

% Construct the full image path
imagePath = fullfile(pathname, filename);

% Read the input image
inputImage = imread(imagePath);

% Preprocess the image if required
% e.g., resizing, normalization, feature extraction
% Resize the input image to match the size of the training images
inputImage = imresize(inputImage, [256, 256]);

% Convert the input image to grayscale if needed
if size(inputImage, 3) > 1
    inputImage = rgb2gray(inputImage);
end

% Normalize the image pixel values to the range [0, 1]
inputImage = double(inputImage) / 255.0;

% Reshape the input image to match the number of columns in the training data
inputImage = reshape(inputImage', 1, numel(inputImage));

% Predict using the trained KNN model
label = predict(mdl, inputImage);

% Display the prediction
if label == 1
    fprintf('The image is predicted to be cancerous (brain cancer positive).\n');
else
    fprintf('The image is predicted to be non-cancerous (brain cancer negative).\n');
end
