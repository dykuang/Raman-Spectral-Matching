import numpy as np
import os


def text_data_to_numpy_array(filepath):
    """
    Transforms the data saved as a text or csv file to a numpy array.

    Parameters
    ----------
    filepath : string
        The path of the text or csv file
    filetype : string
        The type of the files (tex or csv)

    Returns
    -------
    data : numpy array
        Numpy array with two rows, given by the x and y values of the data
    """

    filetype = filepath[-3:]
    if filetype == "csv":
        delim = ','
    elif filetype == "txt":
        delim = '\t'

    f = open(filepath, 'r')
    X, Y = [], []
    for row in f:
        xi, yi = row.split(delim)
        X.append(float(xi))
        Y.append(float(yi))
    data = np.array([X, Y])
    return data


def extract_data(folder):
    """
    Extracts the data from all the files in the folder, transforms them into
    arrays, and puts them in a dictionary.

    Parameters
    ----------
    folder : string
        The path of the folder

    Returns
    -------
    data_in_folder : dictionary
        The keys of data_in_folder are the names of the files in the folder,
        the values are the corresponding data, saved as a numpy array with two
        rows, the first containing the x axis, athe second the y axis
    """

    data_in_folder = {}

    for file_name in os.listdir(folder):

        try:
#            print(folder + '\\' + file_name)
            data_in_folder[file_name] = \
                text_data_to_numpy_array(folder + '\\' + file_name)
                
        except:
            print("Problem with file", file_name)
    return data_in_folder
