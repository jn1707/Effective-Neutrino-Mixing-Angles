import numpy as np
from scipy.spatial.transform import Rotation
from itertools import permutations


def is_unitary(matrix) :
    """ Check matrix is unitary, e.g. U U^dagger = 1 """

    return np.allclose( np.dot( matrix, matrix.conjugate().T), np.diagflat([1.]*matrix.shape[0]).astype(np.complex128), atol=1e-8)


def pmns_angles_from_PMNS(matrix, deg=True):
    """
    Returns the mixing angles.
    """
    #assert is_unitary(matrix)

    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Try canonical rotation order
    order = 'zyx'
    angles = rotation.as_euler(order, degrees=True)

    # Correct for wrong sign of theta23 and theta12
    angles[0] = -angles[0]
    angles[2] = -angles[2]
        
    # Extract the angles of rotation around x, y, and z axes
    theta12, theta13, theta23 = angles 

    # Check whether angles lie in first quadrant
    #if np.all(angles >= 0) and np.all(angles <= 90):
    #    pass
    #else:
    #    raise Exception("Angles are not in the first quadrant. The PMNS matrix might have been calculated using a wrong order. The calculated angels are ", angles)

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)


def is_diagonal(matrix, threshold= 1e-10):
    # Check if the matrix is diagonal within a specified threshold
    return np.all(np.abs(matrix - np.diag(np.diagonal(matrix))) < threshold)


def pmns_angles_from_get_mixing_matrix2(matrix, H2E, threshold=1e-10, deg=True):
    # All possible rotation orders
    rotation_orders = ['xyz', 'yzx', 'zxy', 'xzy', 'zyx', 'yxz']

    found_diagonal = None
    angles_order = None

    for permuted_indices in permutations(range(matrix.shape[1])):
        # Create a matrix with columns permuted according to the current indices
        permuted_matrix = matrix[:, permuted_indices]

        found_diagonal = [False] * len(rotation_orders)
        angles_order = [None] * len(rotation_orders)

        for i, order in enumerate(rotation_orders):
            # Convert the permuted matrix to a rotation object
            rotation = Rotation.from_matrix(permuted_matrix)
            
            # Extract the angles of rotation around x, y, and z axes
            angles = rotation.as_euler(order, degrees=True)
            
            # Save values
            angles_order[i] = angles

            # Calculate PMNS matrix
            U_PMNS = get_pmns_matrix(np.deg2rad(angles), dcp=0.)

            # Calculate U^dagger H2E U
            result = np.dot(np.dot(U_PMNS.conjugate().T, H2E), U_PMNS)

            # Set values close to zero to zero
            result_zeroed = np.where(np.abs(result) < threshold, 0, result)
            print("\n permuted_indices", permuted_indices, " \n Order: ",order, "\n Attempt at Diagonalization \n", angles)

            # Check if the result is diagonal
            if is_diagonal(result_zeroed, threshold):
                found_diagonal[i] = True
                print("Diagonal matrix:\n", result_zeroed)
                # Stop for loop if a diagonal arrangement is found
                break

        # Check if a diagonal arrangement was found for any order
        if any(found_diagonal):
            print("Found a diagonal arrangement:")
            print("Permutated Indices:", permuted_indices)
            print("Angles Order:", angles_order[found_diagonal])
            break

    # Check if a diagonal matrix was found
    if np.any(found_diagonal):
        # Extract the angles of the first diagonal matrix
        angles = angles_order[found_diagonal][0]
        theta12, theta13, theta23 = angles
        # If more than one diagonal matrix was found, check if the angles are the same
        if np.sum(found_diagonal) > 1:
            print("Multiple diagonal matrices found. The calculated angles are ", angles_order[found_diagonal])
    else:
        raise Exception("No diagonal matrix found.")

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)
    
    
def pmns_angles_from_get_mixing_matrix(matrix, H2E, deg=True):
    #assert is_unitary(matrix)

    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Try specific rotation order
    order = 'yzx'

    # Extract the angles of rotation around x, y, and z axes
    angles = rotation.as_euler(order, degrees=True)

    # Correct for wrong order of rotations
    angles[0] = - angles[0] % 90
    angles[1] = - angles[1]
    angles[2] = 90 - angles[2]
    print("Angles 1st try: ",angles)
    theta12, theta13, theta23 = angles
    
    # Correct for wrong order of eigenvectors
    pmns_try = get_pmns_matrix(np.deg2rad(angles), dcp=0.)
    diagonal_try = 2*np.dot(pmns_try.conjugate().T, H2E, pmns_try)
    print("Diagonalized matrix:\n", diagonal_try)

    if np.allclose(diagonal_try[0,1], 0, atol=1e-8) and np.allclose(diagonal_try[0,2], 0, atol=1e-8):
        pass
    else:
        matrix[:, 0] = -matrix[:, 0]
        # Extract the angles again
        rotation = Rotation.from_matrix(matrix)
        new_angles = rotation.as_euler(order, degrees=True)

        # Correct for wrong order of rotations
        new_angles[0] = - new_angles[0] % 90
        new_angles[1] = - new_angles[1]
        new_angles[2] = 90 - new_angles[2]
        print("Angles 2nd try: ",new_angles)

        if np.all(new_angles >= 0) and np.all(new_angles <= 90):
            theta12, theta13, theta23 = new_angles
            angles = new_angles
        else:
            pass

    # Check whether angles lie in first quadrant
    if np.all(angles >= 0) and np.all(angles <= 90):
        pass
    else:
        raise Exception("Angles are not in the first quadrant. The PMNS matrix might have been calculated using a wrong order. The calculated angels are ", angles)


    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)


def get_pmns_matrix(theta, dcp=0.) :
    """ Get the PMNS matrix (rotation from mass to flavor basis)"""

    if len(theta) == 1 :
        assert (dcp is None) or np.isclose(dcp, 0.)
        # This is just the standard unitary rotation matrix in 2D
        pmns = np.array( [ [np.cos(theta[0]),np.sin(theta[0])], [-np.sin(theta[0]),np.cos(theta[0])] ], dtype=np.complex128 )
    elif len(theta) == 3 :
        # Check if mixing angles are in the first quadrant
        #if not np.all((0 <= np.array(theta)) & (np.array(theta) <= np.pi/2)):
        #    raise ValueError("Mixing angles should be in the first quadrant.")
        
        # Using definition from https://en.wikipedia.org/wiki/Neutrino_oscillation
        pmns = np.array( [
            [   np.cos(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[1])*np.exp(-1.j*dcp)  ], 
            [  -np.sin(theta[0])*np.cos(theta[2]) -np.cos(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[0])*np.cos(theta[2]) -np.sin(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.sin(theta[2])*np.cos(theta[1])  ], 
            [  np.sin(theta[0])*np.sin(theta[2]) -np.cos(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    -np.cos(theta[0])*np.sin(theta[2]) -np.sin(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[2])*np.cos(theta[1])  ],
        ], dtype=np.complex128 )
    else :
        raise Exception("Only 2x2 or 3x3 PMNS matrix supported")

    return pmns


if __name__ == "__main__" :
    # Example usage:

    orthogonal_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    print(is_unitary(orthogonal_matrix))

    for i in np.random.uniform(0, np.pi/2, 1):
        j = np.random.uniform(0, np.pi/2)
        k = np.random.uniform(0, np.pi/2)
        mixing_angles = [i, j, k]

        print("Initial mixing angles:", np.rad2deg(mixing_angles))
        theta12, theta13, theta23 = pmns_angles_from_PMNS(
            get_pmns_matrix(mixing_angles, dcp=0.), deg=False)  
         
        print("Difference between calculated and input mixing angles:")
        print(f"\Delta Theta12: {theta12-mixing_angles[0]} rad")
        print(f"\Delta Theta13: {theta13-mixing_angles[1]} rad")
        print(f"\Delta Theta23: {theta23-mixing_angles[2]} rad")
        if np.allclose([theta12, theta13, theta23], mixing_angles):
            print("Angles are correct.")
