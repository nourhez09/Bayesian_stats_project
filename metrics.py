import numpy as np

def calculate_mmad(posterior_samples, x, y_true):
    """
    Calculate the Median of Mean Absolute Deviations (MMAD) from a given posterior sample.
    
    Parameters:
    - posterior_samples: An array of shape (num_samples, num_parameters), where each row is a sample from the posterior.
    - x: The design matrix of predictors (shape: num_samples x num_features).
    - y_true: The true observed response values (shape: num_samples).
    
    Returns:
    - MMAD: The Median of Mean Absolute Deviations from the posterior sample.
    """
    # n_samples = posterior_samples.shape[0]
    
    # Initialize an array to store the MAD values for each sample
    mad_values = []
    
    # Loop over each posterior sample
    for sample in posterior_samples:
        # Calculate the predicted values for the current sample (x * sample)
        y_pred = np.dot(x, sample)
        
        # Compute the residuals (difference between true values and predicted values)
        residuals = y_true - y_pred
        
        # Calculate the Mean Absolute Deviation (MAD) for this sample
        mad = np.mean(np.abs(residuals))
        
        # Append the MAD value for this sample to the list
        mad_values.append(mad)
    
    # Convert the list of MAD values to a numpy array
    mad_values = np.array(mad_values)
    
    # Calculate and return the Median of Mean Absolute Deviations (MMAD)
    mmad = np.median(mad_values)
    
    return mmad