def normalize_range(original_array, original_min, original_max, new_min, new_max):
    """ Normalizing data to a new range (e.g. to [-1, 1] or [1, 1])
    :param original_array:   input array
    :param original_min: current minimum (array, can be derived from a larger sample)
    :param original_max: current max (array, can be derived from a larger sample)
    :param new_min: new minimum (float)
    :param new_max: new maximum (float)
    :return: normalized array
    """
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (original_array - a) / (b - a) * (d - c) + c


def un_normalize_range(normalized_array, original_min, original_max, new_min, new_max):
    """ Un-normalizing data to its original range (e.g. to [0, 1400])
    :param normalized_array:  normalized array
    :param original_min: minimum value (array, can be derived from a larger sample)
    :param original_max: current max (array, can be derived from a larger sample)
    :param new_min: new minimum (float)
    :param new_max: new maximum (float)
    :return: original array
    """
    a = original_min
    b = original_max
    c = new_min
    d = new_max
    return (normalized_array - c) / (d - c) * (b - a) + a