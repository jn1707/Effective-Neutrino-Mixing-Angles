from PMNS import pmns_angles_from_get_mixing_matrix, get_pmns_matrix, pmns_angles_from_get_mixing_matrix2, pmns_angles_from_PMNS
import numpy as np
import itertools

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
    print("U_PMNS: ", U_PMNS)

    #flavor state matrix
    H_f = 1/(2*E_eV) * np.dot(U_PMNS, np.dot(H, U_PMNS.conjugate().T))

    return H_f


def get_mixing_matrix(matrix, E_eV=1):
    # Rescale matrix to account for energy dependence
    matrix = matrix*2*E_eV

    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Normalize eigenvectors
    normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    print("eigenvectors: \n", np.real(normalized_eigenvectors))  # Show only the real part of normalized eigenvectors

    # Check if the matrix is diagonalizable
    if not np.all(np.iscomplex(eigenvalues) | np.isreal(eigenvalues)):
        raise ValueError("Matrix is not diagonalizable.")

    return normalized_eigenvectors, eigenvalues


def try_all_mixing_orders_and_signs(matrix, H2E, deg=True):
    """
    Returns the mixing angles that will diagonalize the matrix in ascending order of the diagonal elements.
    Has problems if two diagonal elements are equal.
    """
    n = matrix.shape[0]
    signs = [-1, 1]
    column_orders = list(itertools.permutations(range(n)))

    for order in column_orders:
        for sign_combination in itertools.product(signs, repeat=n):
            modified_matrix = matrix[:, order] * np.array(sign_combination)
            theta12, theta13, theta23 = pmns_angles_from_PMNS(matrix=modified_matrix, deg=False)
            
            if 0 <= theta12 <= np.pi/2 and 0 <= theta13 <= np.pi/2 and 0 <= theta23 <= np.pi/2:
                
                result_array = np.dot(modified_matrix.conjugate().T, np.dot(H2E, modified_matrix))
                # Set values close to zero to zero based on the tolerance
                tolerance = 1e-9
                result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

                if np.diag(result_array)[0] <= np.diag(result_array)[1] and np.diag(result_array)[1] <= np.diag(result_array)[2]:
                    print("Angles in the first quadrant for sign combination", sign_combination, "and the following order of eigenvectors", order)
                    if deg:
                        return np.rad2deg(theta12), np.rad2deg(theta13), np.rad2deg(theta23)
                    else:
                        return theta12, theta13, theta23
            

if __name__ == "__main__" :

    # Set random values for mass squared differences and mixing angles
    for i in np.random.uniform(0, np.pi/2, 1):
        j = np.random.uniform(0, np.pi/2)
        k = np.random.uniform(0, np.pi/2)
        m2 = np.random.uniform(0, 1)
        m1 = np.random.uniform(0, m2)
        nu_mass_squared_diff = [m1, m2]
        mixing_angles = [i, j, k]

        # Calculate the neutrino Hamiltonian
        Ham = generate_neutrino_hamiltonian(nu_mass_squared_diff, mixing_angles, 1/2)

        # Calculate the mixing matrix
        mixing_matrix, eigenvalues = get_mixing_matrix(Ham, E_eV=1/2)

        # Calculate the product of conjugate transpose of mixing_matrix, H, and mixing_matrix
        result_array = np.dot(mixing_matrix.conjugate().T, np.dot(Ham, mixing_matrix))

        # Set values close to zero to zero based on the tolerance
        tolerance = 1e-9
        result_array[np.isclose(result_array, 0, atol=tolerance)] = 0

        print("Diagonalized matrix:\n", result_array)

        print("Initial mixing angles:", np.rad2deg(mixing_angles))
        print("Initial mass squared differences:", nu_mass_squared_diff)

        # Calculate mixing angles
        theta12, theta13, theta23 =try_all_mixing_orders_and_signs(mixing_matrix, H2E=Ham, deg=False)
        
        print("Difference between calculated and input mixing angles:")
        print(f"\Delta Theta12: {theta12-mixing_angles[0]} rad")
        print(f"\Delta Theta13: {theta13-mixing_angles[1]} rad")
        print(f"\Delta Theta23: {theta23-mixing_angles[2]} rad")
        if np.allclose([theta12, theta13, theta23], mixing_angles):
            print("Angles are correct.")