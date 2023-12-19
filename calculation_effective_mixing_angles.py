from PMNS import pmns_angles_from_get_mixing_matrix, get_pmns_matrix, is_unitary
import numpy as np

def generate_neutrino_hamiltonian(mass_squared_diff, mixing_angles, E_eV):
    """
    Generate a 3x3 neutrino flavor Hamiltonian matrix based on mass squared differences and mixing angles.

    Parameters:
    - mass_squared_diff (tuple): Mass squared differences in eV^2.
    - mixing_angles (triple): Vacuum mixing angles in radians.

    Returns:
    - numpy.ndarray: 3x3 neutrino flavor Hamiltonian matrix.
    """
    # Check if input lists/tuples have correct length
    if len(mass_squared_diff) != 2 or len(mixing_angles) != 3:
        raise ValueError("Incorrect input dimensions. Mass squared differences and mixing angles should have length 2.")

    # Check if mixing angles are in the first quadrant
    if not np.all((0 <= np.array(mixing_angles)) & (np.array(mixing_angles) <= np.pi/2)):
        raise ValueError("Mixing angles should be in the first quadrant.")
    
    # Unpack mass squared differences and mixing angles
    delta_m21_sq, delta_m31_sq = mass_squared_diff
    #theta12, theta13, theta23 = mixing_angles

    # Construct the neutrino flavor Hamiltonian matrix
    H = np.zeros((3, 3))

    # Diagonal elements (normal mass hierarchy)
    H[1, 1] = delta_m21_sq
    H[2, 2] = delta_m31_sq
    
    U_PMNS = get_pmns_matrix(mixing_angles, dcp=0.)

    #flavor state matrix
    H_f = 1/(2*E_eV) * np.dot(U_PMNS, np.dot(H, U_PMNS.conjugate().T))

    return H_f


def get_mixing_matrix(matrix, E_eV=1):
    # Rescale matrix to account for energy dependence
    matrix = matrix*2*E_eV

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Order eigenvectors based on eigenvalues
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    #eigenvectors[:, 0] = -eigenvectors[:, 0]

    # Check if the matrix is diagonalizable
    if not np.all(np.iscomplex(eigenvalues) | np.isreal(eigenvalues)):
        raise ValueError("Matrix is not diagonalizable.")

    return eigenvectors, eigenvalues


if __name__ == "__main__" :

    nu_mass_squared_diff = [1, 0.2]

    for i in np.random.uniform(0, np.pi/2, 1):
        j = np.random.uniform(0, np.pi/2)
        k = np.random.uniform(0, np.pi/2)
        mixing_angles = [i, j, k]
        #mixing_angles = np.deg2rad([58.92861939,  7.6486031,  41.83700821])
        #mixing_angles = np.deg2rad([43.48994929, 16.44984926, 25.04209752])
        #mixing_angles = np.deg2rad([12.47424932, 40.54584511, 41.75163916])
        #mixing_angles = np.deg2rad([60.8246798,  62.16921519, 79.54490906])
        #mixing_angles = np.deg2rad([0,  0, 45])
        
        H = generate_neutrino_hamiltonian(nu_mass_squared_diff, mixing_angles, 1/2)

        mixing_matrix, eigenvalues = get_mixing_matrix(H, E_eV=1/2)
        print("Mixing matrix:\n", mixing_matrix)
        # Calculate the product of conjugate transpose of mixing_matrix, H, and mixing_matrix
        result_array = np.dot(mixing_matrix.conjugate().T, np.dot(H, mixing_matrix))

        # Set values close to zero to zero based on the tolerance
        tolerance = 1e-9
        result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

        print("Diagonalized matrix:\n", result_array)

        print("Initial mixing angles:", np.rad2deg(mixing_angles))


        theta12, theta13, theta23 = pmns_angles_from_get_mixing_matrix(matrix=mixing_matrix, H2E=H, deg=False)  

        print("Difference between calculated and input mixing angles:")
        print(f"\Delta Theta12: {theta12-mixing_angles[0]} rad")
        print(f"\Delta Theta13: {theta13-mixing_angles[1]} rad")
        print(f"\Delta Theta23: {theta23-mixing_angles[2]} rad")
        if np.allclose([theta12, theta13, theta23], mixing_angles):
            print("Angles are correct.")

