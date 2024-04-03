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
    # Convert the matrix to a rotation object
    rotation = Rotation.from_matrix(matrix)
    
    # Use canonical rotation order
    order = 'zyx'
    angles = rotation.as_euler(order, degrees=True)

    # Correct for opposite sign of theta23 and theta12 compared to physics convention
    '''
    Code to check signs of angles
    r_z = R.from_matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r_y = R.from_matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    r_x = R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r.as_euler('xyz', degrees=True)
    '''
    angles[0] = -angles[0]
    angles[2] = -angles[2]
        
    # Extract the angles of rotation around z, y, and x axes
    theta12, theta13, theta23 = angles 

    # Return mixing angles
    if deg:
        return theta12, theta13, theta23
    else:
        return np.deg2rad(theta12), np.deg2rad(theta13), np.deg2rad(theta23)
    

def is_diagonal(matrix, threshold= 1e-10):
    """ Check if the matrix is diagonal within a specified threshold """
    return np.all(np.abs(matrix - np.diag(np.diagonal(matrix))) < threshold)


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
            [   np.cos(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[0])*np.cos(theta[1]),                                                                        np.sin(theta[1])*np.exp(1.j*dcp)  ], #Janni
            [  -np.sin(theta[0])*np.cos(theta[2]) -np.cos(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[0])*np.cos(theta[2]) -np.sin(theta[0])*np.sin(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.sin(theta[2])*np.cos(theta[1])  ], 
            [  np.sin(theta[0])*np.sin(theta[2]) -np.cos(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    -np.cos(theta[0])*np.sin(theta[2]) -np.sin(theta[0])*np.cos(theta[2])*np.sin(theta[1])*np.exp(1.j*dcp),    np.cos(theta[2])*np.cos(theta[1])  ],
        ], dtype=np.complex128 )
    else :
        raise Exception("Only 2x2 or 3x3 PMNS matrix supported")

    return pmns


if __name__ == "__main__" :
    
    #
    # Example usage
    #
    orthogonal_matrix = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    print("Is matrix unitary?", is_unitary(orthogonal_matrix))
    print("Is matrix diagonal?", is_diagonal(orthogonal_matrix))

    #
    # Check whether get_pmns_matrix method works
    #

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
