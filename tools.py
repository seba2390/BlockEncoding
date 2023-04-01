from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.visualization import plot_histogram


def plot_count_histogram(counts: Dict) -> None:
    """
    Function for plotting a histogram of the states and
    their corresponding probability.

    :param counts: dictionary, e.g.: {state1:count1, state2:counts2,...}
    """

    # ---------- Getting values ----------#
    initial_states = np.array(list(counts.keys()))
    initial_counts = np.array(list(counts.values()))

    # ---------- Sorting ----------#
    initial_counts = np.array([count / np.sum(initial_counts) for count in initial_counts])

    # ---------- Plotting -----------#
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    xs = np.arange(0, len(initial_states))
    x_labels = [r"$|$" + state + r"$\rangle$" for state in initial_states]
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=45, size=15)
    bar = ax.bar(initial_states, initial_counts, width=0.3, align="center", color="tab:blue")

    for idx, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{initial_counts[idx]:.3f}', ha='center', va='bottom')

    ax.set_ylim(0, np.max(initial_counts) * 1.2)
    ax.set_ylabel("Probability", size=18)
    fig.subplots_adjust(bottom=0.2)  # Increasing space below fig (in case of large states)
    plt.show()


def qubit_vector(state: str) -> np.ndarray:
    """
    Takes tensor product state like |0>|1>|0>|0> and
    returns vector representation of state.

    :param state: str. repr of state, e.g. |0>|0>|1> as '001'.
    :return _result: vector representation of tensor product qubit state.
    """
    _basis_dict = {'0': np.array([[1], [0]]), '1': np.array([[0], [1]])}
    _result = np.kron(_basis_dict[state[0]], _basis_dict[state[1]])
    if len(state) > 2:
        for remaining_state in range(2, len(state)):
            _result = np.kron(_result, _basis_dict[state[remaining_state]])
    return _result


def get_probabilities(counts: Dict) -> Dict:
    """
    Returns dictionary of states probabilities as relative frequency.

    :param: counts: dictionary, e.g.: {state1:count1, state2:counts2,...}
    :return: probabilities dictionary, e.g.: {state1:prob1, state2:prob2,...}
    """
    probabilities = {}
    for state in counts.keys():
        probabilities[state] = counts[state] / sum(list(counts.values()))
    return probabilities


def get_state_vector(counts: Dict) -> np.ndarray:
    """
    Approximates the state vector up til a sign.

    :param: counts: dictionary, e.g.: {state1:count1, state2:counts2,...}
    :return: state_vector: np.array vector.
    """
    probabilities = get_probabilities(counts)
    state_vector = np.zeros(shape=(2 ** len(list(probabilities.keys())[0])))
    for state in probabilities:
        state_vector += np.sqrt(probabilities[state]) * qubit_vector(state=state).flatten()
    return state_vector


def check_symmetric(a: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if a given matrix is symmetric within a certain tolerance.

    Parameters
    ----------
    a : array_like
        The matrix to check for symmetry.
    rtol : float, optional
        The relative tolerance for the symmetry check. Default is 1e-5.
    atol : float, optional
        The absolute tolerance for the symmetry check. Default is 1e-8.

    Returns
    -------
    bool
        True if the matrix is symmetric within the specified tolerances, False otherwise.

    Notes
    -----
    A matrix is symmetric if it is equal to its transpose. This function checks if the difference between the input matrix and its transpose is within the specified tolerances. If both the relative and absolute differences are less than or equal to their respective tolerances, the matrix is considered symmetric.

    This function uses the `allclose` method from NumPy to compare the matrix and its transpose.
    """

    return np.allclose(a, a.T, rtol=rtol, atol=atol)


