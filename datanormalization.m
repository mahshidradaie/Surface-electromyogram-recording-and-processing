% Load your dataset (replace 'your_dataset.mat' and 'your_variable_name')
load('S1_E3_A1', 'emg'); 

data = emg;
[num_samples, num_channels] = size(data);

% 1. Split into Training and Testing Sets
train_percentage = 0.8; % 80% for training
num_train_samples = round(train_percentage * num_samples);

% Random permutation of indices
perm_indices = randperm(num_samples);

train_indices = perm_indices(1:num_train_samples);
test_indices = perm_indices(num_train_samples+1:end);

train_data = data(train_indices, :);
test_data = data(test_indices, :);

% 2. Normalize the Data (Choose ONE method)

normalization_method = 'standardization'; % Or 'minmax'

if strcmp(normalization_method, 'minmax')
    % Min-Max Normalization

    % Calculate min and max for each channel in the TRAINING data
    min_values = min(train_data);
    max_values = max(train_data);

    % Normalize TRAINING data
    train_data_normalized = (train_data - min_values) ./ (max_values - min_values);

    % Normalize TESTING data using TRAINING data's min and max
    test_data_normalized = (test_data - min_values) ./ (max_values - min_values);

elseif strcmp(normalization_method, 'standardization')
    % Standardization (Z-score)

    % Calculate mean and std for each channel in the TRAINING data
    mean_values = mean(train_data);
    std_values = std(train_data);

    % Normalize TRAINING data
    train_data_normalized = (train_data - mean_values) ./ std_values;

    % Normalize TESTING data using TRAINING data's mean and std
    test_data_normalized = (test_data - mean_values) ./ std_values;

else
    error('Invalid normalization method selected.');
end

% Display or Save the Results (Optional)
disp('Normalized Training Data:');
disp(train_data_normalized(1:5,:)); % Display first 5 rows (or as needed)
disp('Normalized Testing Data:');
disp(test_data_normalized(1:5,:)); % Display first 5 rows (or as needed)

% If you want to save:
% save('normalized_data.mat', 'train_data_normalized', 'test_data_normalized');


% --- Explanation of Data Splitting and Why We Do It ---

% 1. Why Split Data?
% - To evaluate the performance of a machine learning model on unseen data.
% - We train the model on the 'train_data' and then assess how well it 
% generalizes to the 'test_data', which the model has never seen during training.

% 2. Why Not Normalize Before Splitting?
% - To prevent "data leakage" from the test set into the training process.
% - If we normalize before splitting, the test data's statistics (min, max, mean, std)
% influence the normalization of the training data. This gives the model 
% an unfair advantage and leads to overly optimistic (but unrealistic) 
% performance estimates.

% 3. Why Normalize After Splitting, Based on Training Data?
% - We normalize the test data using the training data's statistics to simulate a 
% real-world scenario. In practice, we train a model on available data and 
% then apply it to new, unseen data. We don't have access to the statistics 
% of the future data, so we shouldn't use them during training or preprocessing.

% Now you can use train_data_normalized and test_data_normalized in your
% machine learning model training and evaluation.

