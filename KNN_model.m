% Set the input image folder paths
positiveFolder = "E:\ENG Sem 6\Projects\BrainTUmor\archive\yes"; % Folder containing positive images
negativeFolder = "E:\ENG Sem 6\Projects\BrainTUmor\archive\no"; % Folder containing negative images

% Read positive image files from the folder
positiveFiles = dir(fullfile(positiveFolder, '*.jpg'));

% Read negative image files from the folder
negativeFiles = dir(fullfile(negativeFolder, '*.jpg'));

% Initialize the training data and labels
X = [];
Y = [];

% Process positive images
for i = 1:numel(positiveFiles)
    imagePath = fullfile(positiveFolder, positiveFiles(i).name);
    Image = imread(imagePath);
    
    % Preprocess the image
    % Example preprocessing steps:
    % - Resize the image to a fixed size
    % - Convert the image to grayscale if needed
    % - Normalize the image pixel values
    % - Perform feature extraction if desired
    
    % Resize the image to a fixed size (e.g., 256x256)
    Image = imresize(Image, [256, 256]);
    
    % Convert the image to grayscale
    if size(Image, 3) > 1
        Image = rgb2gray(Image);
    end
    
    % Normalize the image pixel values to the range [0, 1]
    Image = double(Image) / 255.0;
    
    % Add the preprocessed image to the training data
    X = [X; reshape(Image', 1, numel(Image))];
    
    % Assign the label 1 (positive) to the image
    Y = [Y; 1];
end

% Process negative images
for i = 1:numel(negativeFiles)
    imagePath = fullfile(negativeFolder, negativeFiles(i).name);
    Image = imread(imagePath);
    
    % Preprocess the image (similar to the positive images)
    
    % Resize the image to a fixed size (e.g., 256x256)
    Image = imresize(Image, [256, 256]);
    
    % Convert the image to grayscale
    if size(Image, 3) > 1
        Image = rgb2gray(Image);
    end
    
    % Normalize the image pixel values to the range [0, 1]
    Image = double(Image) / 255.0;
    
    % Add the preprocessed image to the training data
    X = [X; reshape(Image', 1, numel(Image))];
    
    % Assign the label 0 (negative) to the image
    Y = [Y; 0];
end

% Train the KNN model
k = 5; % Number of neighbors
mdl = fitcknn(X, Y, 'NumNeighbors', k);

% Save the trained model
save('brain_cancer_knn_model.mat', 'mdl');
