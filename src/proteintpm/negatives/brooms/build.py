import numpy as np


def calc_vector(start_coord, end_coord):

    return np.asarray(end_coord) - np.asarray(start_coord)


def get_broom(start_coord, protein, neighbour_indices):
    # Calculate broom
    positions = []
    classes = []

    for n in neighbour_indices:
        positions.append(calc_vector(start_coord, protein.iloc[n][["x", "y", "z"]]))
        classes.append(protein.iloc[n][["target atom_id"]])

    positions = np.asarray(positions)
    classes = np.asarray(classes)

    positions = positions[np.newaxis, :, :]
    classes = classes[np.newaxis, :, :]

    # return (tf.convert_to_tensor(positions, tf.float32),
    #        tf.convert_to_tensor(classes, tf.float32))
    return (positions, classes)
