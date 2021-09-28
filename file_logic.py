import os
from datetime import datetime
import pandas as pd
from glob import glob
from inspect import getfullargspec

DATA_DIRECTORY = ' \\Experiment\\'


def save_file(*data,
              file_name="test",
              directory=DATA_DIRECTORY):
    """
    Function to save data in file

    :param data: lists or arrays of data
    :param file_name: string
    :param directory: string
    """
    df = pd.DataFrame(zip(*data))
    if isinstance(file_name, dict):
        file_name = create_filename(**file_name)
    if file_name[-4:] != ".csv":
        file_name = file_name + ".csv"
    path_to_file = os.path.join(directory, file_name)
    dir_of_file = os.path.split(path_to_file)[0]

    os.makedirs(dir_of_file, exist_ok=True)

    df.to_csv(path_to_file, sep="\t", line_terminator="\n", index=False, header=False)


def load_file(file_name,
              axis=(0, 1),
              directory=DATA_DIRECTORY,
              return_dataframe=False):
    """Function to load a csv file


    :param file_name: string of file name with subdirectory, if file_name_dict is given creates file_name
    :param axis: tuple of columns (x,y), defaults to (0,1)
    :param directory: string of directory, defaults to DATA_DIRECTORY
    :param bool return_dataframe: if returned data is pandas dataframe
    :return: two lists (time or frequency, amplitude)
    """
    if isinstance(file_name, dict):
        file_name = create_filename(**file_name)

    if file_name[-4:] != ".csv":
        file_name = file_name + ".csv"

    if isinstance(axis, int):
        axis = [axis]

    path_to_file = os.path.join(directory, file_name)
    data = pd.read_csv(path_to_file, sep='\t', usecols=axis, header=None).T

    if axis is None:
        axis = range(len(data))

    if len(axis) == 2 and return_dataframe:
        data.index = pd.MultiIndex.from_product([[file_name[:-4]], ["time", "ampl"]], names=["file_name", "type"])

    if return_dataframe:
        return data
    else:
        return data.to_numpy() if len(axis) > 1 else data.to_numpy()[0]


def rename_file(file_dicts, directory=DATA_DIRECTORY, **kwargs_to_change):
    def _do_rename_file(old_file_dict):
        if not old_file_dict["raw_data"]:
            new_file_dict = dict(old_file_dict, **kwargs_to_change)
            old_file_name = os.path.join(directory, create_filename(**old_file_dict) + ".csv")
            new_file_name = os.path.join(directory, create_filename(**new_file_dict) + ".csv")

            os.rename(old_file_name, new_file_name)

        else:
            raise ValueError("raw_data should not be renamed")

    if isinstance(file_dicts, dict):
        _do_rename_file(file_dicts)
    else:
        for file_dict in file_dicts:
            _do_rename_file(file_dict)


def remove_file(file_dicts, directory=DATA_DIRECTORY):
    def _do_remove_file(fn_dict):
        if not fn_dict["raw_data"]:
            file_path = os.path.join(directory, create_filename(**fn_dict) + ".csv")

            os.remove(file_path)

        else:
            raise ValueError("raw_data should not be removed")

    if isinstance(file_dicts, dict):
        _do_remove_file(file_dicts)
    else:
        for file_dict in file_dicts:
            _do_remove_file(file_dict)


def create_filename(raw_data=False,
                    date=None,
                    title="test",
                    description=None,
                    demodulator_freq=0,
                    noise=False,
                    version_index=1,
                    measurement_index=0,
                    filtering=None, filter_freq=None,
                    is_fft_data=False,
                    is_average_data=False,
                    is_cut=False):
    """Creates a file name, with subfolder structure.

    Different options for naming the file are given.

    Returns the file name in format:\n
    [data_folder]/[date]/[measurement_title]/[date]_[measurement_description](_[measurement_index])\n
    [measurement_description] is in format [description](_n)(_v[version_index])

    :param raw_data: bool, decides in which data folder the measurement will be saved
    :param str date: string of the date, defaults to today
    :param title: string to title file
    :param description: string to describe file
    :param demodulator_freq:
    :param noise: if noise measurement
    :param version_index: int
    :param measurement_index: int
    :param filtering: if Data is filtered, None is no filtering, 'high' Highpass, 'low' Lowpass, 'band' Bandpass
    :param int or tuple filter_freq: Frequency of the filter-cutoff (will be rounded)
    :param is_fft_data: bool
    :param is_average_data: bool
    :param is_cut:

    :return: string of file name
    """
    if isinstance(raw_data, dict):
        return create_filename(**raw_data)

    date = datetime.today().strftime('%y-%m-%d') if date is None else date

    # create lists
    file_name_list = [date, title]
    measurement_name_list = [title]

    # check different naming options
    if description is not None:
        file_name_list.append(description)

    if demodulator_freq >= 1000:
        file_name_list.append("{:.0f}kHz".format(demodulator_freq / 1000))

    if noise:
        file_name_list.append("n")

    if version_index > 0:
        version_name = "v{:02d}".format(version_index)
        measurement_name_list.append(version_name)
        file_name_list.append(version_name)

    if measurement_index > -1 and not is_average_data:
        file_name_list.append("{:02d}".format(measurement_index))

    if filtering == 'high':
        file_name_list.append("HP{:.0f}Hz".format(filter_freq) if filter_freq is not None else "HP")
    elif filtering == 'low':
        file_name_list.append("LP{:.0f}Hz".format(filter_freq) if filter_freq is not None else "LP")
    elif filtering == 'band':
        file_name_list.append("BP{:.0f}-{:.0f}Hz".format(
            filter_freq[0], filter_freq[1])
                              if filter_freq is not None and isinstance(filter_freq, (tuple, list))
                              else "BP")

    if is_fft_data:
        file_name_list.append("fft")

    if is_average_data:
        file_name_list.append("av")

    if is_cut:
        file_name_list.append("cut")

    file_name = "_".join(file_name_list)
    measurement_name = "_".join(measurement_name_list)

    # checks if data should be stored in raw data folder
    if raw_data:
        path_to_file = os.path.join("1_data", date, measurement_name, file_name)
    else:
        path_to_file = os.path.join("2_data_analysis", date, measurement_name, file_name)

    return path_to_file


def split_filename(filename):
    """
    Function to split the given filename into a dictionary of keyword arguments for the create_filename function.

    :param filename: string of a filename
    :return: dictionary of keyword arguments for the create_filename function
    """
    # initialize dictionary
    filename_dict = dict(zip(getfullargspec(create_filename)[0], getfullargspec(create_filename)[3]))

    # set values if they are not found in filename
    filename_dict["version_index"] = 0
    filename_dict["measurement_index"] = -1

    # check for raw_data
    if "1_data" in filename:
        filename_dict["raw_data"] = True
    elif '2_data_analysis' in filename:
        filename_dict["raw_data"] = False

    # keep only actual filename
    filename = os.path.split(filename)[-1]
    if filename.endswith('.csv'):
        filename = filename[:-4]

    # make list of items in filename
    filename_list = filename.split("_")
    filename_list_copy = filename_list.copy()

    # check for date and title
    try:
        filename_dict['date'] = filename_list[0]
        filename_dict['title'] = filename_list[1]
        filename_list_copy = filename_list_copy[2:]
    except IndexError:
        raise IndexError("File can not be processed, name not created by create_filename-function")

    # check for different naming options
    for item in filename_list[2:]:
        # demodulator_freq
        if item[:-3].isdigit() and item[-3:] == "kHz":
            filename_list_copy.remove(item)
            filename_dict["demodulator_freq"] = int(item[:-3]) * 1000
        # noise
        elif item == "n":
            filename_list_copy.remove(item)
            filename_dict["noise"] = True
        # versioning
        elif item[0] == "v" and item[1:].isdigit():
            filename_list_copy.remove(item)
            filename_dict["version_index"] = int(item[1:])
        # measurement_indexing
        elif item.isdigit():
            filename_list_copy.remove(item)
            filename_dict["measurement_index"] = int(item)
        # HP LP BP
        elif item[:2] == "HP":
            filename_list_copy.remove(item)
            filename_dict["filtering"] = 'high'
            if len(item) > 2:
                filename_dict["filter_freq"] = int(item[2:-2])
        elif item[:2] == "LP":
            filename_list_copy.remove(item)
            filename_dict["filtering"] = 'low'
            if len(item) > 2:
                filename_dict["filter_freq"] = int(item[2:-2])
        elif item[:2] == "BP":
            filename_list_copy.remove(item)
            filename_dict["filtering"] = 'band'
            if len(item) > 2:
                filename_dict["filter_freq"] = [int(x) for x in item[2:-2].split("-")]
        # average data
        elif item == "av":
            filename_list_copy.remove(item)
            filename_dict["is_average_data"] = True
        # fft data
        elif item == "fft":
            filename_list_copy.remove(item)
            filename_dict["is_fft_data"] = True
        # cut data
        elif item == "cut":
            filename_list_copy.remove(item)
            filename_dict["is_cut"] = True

    if filename_list_copy:
        filename_dict["description"] = "_".join(filename_list_copy)

    return filename_dict


def file_search(search_parameter=None, *, squeeze=True, directory=DATA_DIRECTORY, **kwargs_for_subdirectory):
    """
    Searches in the subdirectory, specified by the kwargs_for_subdirectory,
    for *.csv files and returns a list of keyword argument dictionaries for the
    create_filename function.\n
    If no kwargs_for_subdirectory are given, it will list all files.

    :param squeeze: when True, returns only dict, when only one dict is found
    :param search_parameter: str, what to search for.
    :param directory: str, defaults to DATA_DIRECTORY
    :param  kwargs_for_subdirectory: keyword arguments of the create_filename function. Considers raw_data, date and
        title
    :return: list of dictionaries of all files found, containing keyword arguments for the
        create_filename function.
    """
    key_list = ["raw_data", "date"]
    if kwargs_for_subdirectory:
        pathname_list = [directory]

        for key in key_list:
            if key in kwargs_for_subdirectory.keys():
                if key == "raw_data":
                    if kwargs_for_subdirectory["raw_data"]:
                        pathname_list.append("1_data")
                    else:
                        pathname_list.append("2_data_analysis")
                elif key == "date":
                    pathname_list.append(kwargs_for_subdirectory["date"])
            else:
                pathname_list.append("*")

        pathname_list.append("*")
        pathname_list.append("*.csv")
        pathname = os.path.join(*pathname_list)
    else:
        pathname = os.path.join(directory, "**", "*.csv")

    file_list = glob(pathname=pathname, recursive=True)

    if search_parameter is not None:
        new_file_list = [filename for filename in file_list if search_parameter in filename]
        file_list = new_file_list

    filename_dict_list = []
    for filename in file_list:
        filename_dict = split_filename(filename)
        filename_dict_list.append(filename_dict)

    for key in kwargs_for_subdirectory:
        if key in key_list:
            continue
        if key in getfullargspec(create_filename)[0]:
            new_dict_list = []
            for dict in filename_dict_list:
                if key in dict.keys():
                    if dict[key] == kwargs_for_subdirectory[key]:
                        new_dict_list.append(dict)
            filename_dict_list = new_dict_list

    if squeeze is True and len(filename_dict_list) is 1:
        return filename_dict_list[0]
    else:
        return filename_dict_list


def load_files_in_one_dataframe(**search_kwargs):
    file_dicts = file_search(**dict(search_kwargs, raw_data=False, is_cut=True))
    if not file_dicts:
        raise FileExistsError("no cut data in 2_data_analysis with that search parameters")

    data = pd.concat(map(lambda x: load_file(x, return_dataframe=True), file_dicts))

    time_data_mean = data.xs('time', level='type').mean()
    data = data.xs('ampl', level='type')
    data.columns = time_data_mean
    return data
