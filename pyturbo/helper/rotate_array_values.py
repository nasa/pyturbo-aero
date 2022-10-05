import numpy as np

def rotate_array_vals(array_values,rot_angle,min_max):
    '''
        Rotate array values by their min and max values 
    '''
    max_val = np.max(min_max)
    min_val = np.min(min_max)

    rot_angle = rot_angle % 360
    if (rot_angle<0):
        rot_angle = rot_angle+360
    
    dx = (max_val-min_val)/360 * rot_angle

    array_values = array_values + dx
    array_values[array_values>max_val] = array_values[array_values>max_val] - max_val + min_val

    return array_values

