def bytes_to_gigabytes(bytes_value):
    """
    Convert bytes to gigabytes.

    Args:
    bytes_value (int or float): Value in bytes.

    Returns:
    float: Value converted to gigabytes.
    """
    return bytes_value / (1024 ** 3)  # 1 GB = 1024^3 bytes

def bytes_to_megabytes(bytes_value):
    """
    Convert bytes to megabytes.

    Args:
        bytes_value (int or float): Value in bytes.

    Returns:
        float: Value converted to megabytes.
    """
    return bytes_value / (1024 ** 2)  # 1 MB = 1024^2 bytes


def kilobytes_per_second_to_megabytes_per_second(kbps_value):
    """
    Convert kilobytes per second to megabytes per second.

    Args:
    kbps_value (int or float): Value in kilobytes per second.

    Returns:
    float: Value converted to megabytes per second.
    """
    return kbps_value / 1024  # 1 MB = 1024 KB


def milliseconds_to_hours(milliseconds_value):
    """
    Convert milliseconds to hours.

    Args:
    milliseconds_value (int or float): Value in milliseconds.

    Returns:
    float: Value converted to hours.
    """
    return milliseconds_value / (1000 * 60 * 60)  # 1 hour = 3600000 milliseconds

def milliseconds_to_minutes(milliseconds_value):
    """
    Convert milliseconds to minutes.

    Args:
    milliseconds_value (int or float): Value in milliseconds.

    Returns:
    float: Value converted to minutes.
    """
    return milliseconds_value / (1000 * 60)  # 1 minute = 60000 milliseconds

def milliseconds_to_seconds(milliseconds_value):
    """
    Convert milliseconds to seconds.

    Args:
        milliseconds_value (int or float): Value in milliseconds.

    Returns:
        float: Value converted to seconds.
    """
    return milliseconds_value / 1000.0  # Using float division