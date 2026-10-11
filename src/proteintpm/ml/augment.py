import tensorflow as tf


def random_rotation_matrix_tf():
    """
    Generate a random 3x3 rotation matrix using a random quaternion.
    """
    q = tf.random.normal([4])
    q = q / tf.linalg.norm(q)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    R = tf.stack(
        [
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        ]
    )

    return R


def apply_rotation_tf(vectors, R):
    """
    Applies a random rotation to a set of 3D vectors.
    :param vectors: Tensor of shape (100, 3)
    :return: Rotated tensor of shape (100, 3)
    """
    # R = random_rotation_matrix_tf()  # (3, 3)
    # Multiply each vector by the transpose of R.
    return tf.matmul(vectors, tf.transpose(R))


def duplicate_and_augment(ch1, ch2, label, num_augmentations):
    # Create a dataset from the current sample (which is already a tensor)

    if num_augmentations == 0:
        return tf.data.Dataset.from_tensors(
            rotate_and_convert_spherical(ch1, ch2, label, use_identity=True)
        )

    else:
        ds_identity = tf.data.Dataset.from_tensors(
            rotate_and_convert_spherical(ch1, ch2, label, use_identity=True)
        )

        ds_random = (
            tf.data.Dataset.from_tensors((ch1, ch2, label))
            .repeat(num_augmentations - 1)
            .map(lambda a, b, d: rotate_and_convert_spherical(a, b, d, use_identity=False))
        )

        # Concatenate the two datasets so the original sample is included.
        return ds_identity.concatenate(ds_random)


def rotate_and_convert_spherical(ch1, ch2, label, use_identity=False):

    if use_identity:
        R = tf.eye(3)
    else:
        R = random_rotation_matrix_tf()

    ch1_rotated = apply_rotation_tf(ch1, R)

    # Reshape ori (shape (6,)) into two 3D vectors (shape (2, 3))
    # ori_rotated = apply_rotation_tf(tf.reshape(ori, (2, 3)), R)

    # ch3_rotated = calc_spheri(ori_rotated)

    return (ch1_rotated, ch2), label
