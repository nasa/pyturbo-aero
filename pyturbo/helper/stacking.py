import enum

class StackType(enum.Enum):
    """class defining the type of stacking for Airfoil2D profiles

    Args:
        enum (enum.Emum): inherits enum
    """
    leading_edge = 1
    centroid = 2
    trailing_edge = 3