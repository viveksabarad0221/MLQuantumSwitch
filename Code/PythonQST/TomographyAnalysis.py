import numpy as np
import matplotlib.pyplot as plt
from QuantumTomography import Tomography
import QuantumTomography.TomoFunctions as qf
import os
import pathlib
from foundationsbib.plotclasses.plot_3d import barplot_3d


class QSTAnalysis:
    def __init__(self):
        """
        This class performs the State Tomography on the recorded data. The actual Tomography class is taken from
        http://research.physics.illinois.edu/QI/Photonics/tomography/
        This code is written by the Kwiat Quantum Information group from the University of Illinois.
        """
        self.qt = Tomography()

        self.path_singles = None        # File name of the recorded singles.
        self.path_coincidences = None   # File name of the recorded coincidences.
        self.path_efficiencies = None   # File name of the measured efficiencies.
        self.path_rho = None            # Path of density matrix file

        self.window = np.array([])  # In nanoseconds.
        self.int_time = np.array([])  # In seconds.
        self.efficiencies = np.array([])  # Array for different detector efficiencies.
        self.singles = np.array([])  # Array for recorded singles.
        self.coincidences = np.array([])  # Array for recorded coincidences.
        self.measurements = np.array([])  # Array for measurement specification.

        self.rho = np.array([])  # Array for density matrix.
        self.intensity = None  # The predicted overall intensity used to normalize the state.
        self.fvalp = None   # Final value of the internal optimization function. Values greater than the number
                            # of measurements indicate poor agreement with a quantum state.
        self.trace_rho = None           # Variable for trace of rho
        self.purity_rho = None          # Variable for purity of rho
        self.concurrence_rho = None     # Variable for concurrence of rho

    @staticmethod
    def find_between_strings(s: str, start: str, end: str):
        """
        This function returns a string which is between two sub-strings of a string
        :param s: string to search in
        :param start: start substring
        :param end: end substring
        :return: string between start and end
        """
        return (s.split(start))[1].split(end)[0]

    def read_data(self, num_qubits, window_used=None, int_time=None, has_herald_tomo_stage=True):
        """
        This function reads in the coincidence and single data and prepares the required array for accidental correction.
        :param num_qubits: Number of Qubits.
        :param window_used: Coincidence window in ns.
        :param int_time: Integration time of the measurement in s.
        :param has_herald_tomo_stage: If true the herald photon has also a tomography stage.
                                        If False, its directly send to detector. Data file looks different
        :return:
        """
        if num_qubits == 2:
            # For accidental correction:
            self.window = np.empty(4)  # For 4 detectors. # Make coincidence window array for accidental correction.
            self.window.fill(window_used * 1e-9)
            # Make integration time array for accidental correction.
            self.int_time = np.empty(9)  # For 4 detectors.
            self.int_time.fill(int_time)
            # Read in singles data.
            dummy_s = np.genfromtxt(fname=self.path_singles, delimiter='\t', comments='#')
            self.singles = dummy_s[:, [2, 4, 3, 5]]  # Hs, Hi, Vs, Vi
        # For Tomography:
        # Read in coincidences data.
        dummy_c = np.genfromtxt(fname=self.path_coincidences, delimiter='\t', comments='#')
        if num_qubits == 1:
            if has_herald_tomo_stage:
                # need to post select on one herald detector, to not be affected from state mixing by leakage in the Sagnac
                self.coincidences = dummy_c[:, [3, 4]]
            else:
                self.coincidences = dummy_c[:, [1, 2]]  # H, V
        elif num_qubits == 2:
            self.coincidences = dummy_c[:, [2, 3, 4, 5]]  # Hs-Hi, Hs-Vi, Vs-Hi, Vs-Vi
        # Create measurement array.
        dummy_m = self.qt.getTomoInputTemplate(numBits=num_qubits, numDet=2)
        self.measurements = dummy_m[:, 2 ** num_qubits + 2 * num_qubits + 1: 2 ** num_qubits + 4 * num_qubits + 1]

    def load_efficiencies(self, verbose=False):
        """
        This function loads the measured detection efficiencies for detection efficiency correction.
        :param verbose: If true, it prints the efficiencies.
        :return:
        """
        self.efficiencies = np.genfromtxt(fname=self.path_efficiencies, comments='#')
        if verbose:
            print(self.efficiencies)

    def run_tomography(self, num_qubits, verbose=False):
        """
        This function performs the tomography using the Maximum Likelihood Method.
        :return:
        """
        # StateTomography runs by default the Maximum Likelihood Estimation method.
        if num_qubits == 1:
            dummy = self.qt.StateTomography(measurements=self.measurements, counts=self.coincidences,
                                            efficiency=self.efficiencies)
        elif num_qubits == 2:
            dummy = self.qt.StateTomography(measurements=self.measurements, counts=self.coincidences,
                                            efficiency=self.efficiencies, time=self.int_time,
                                            singles=self.singles, window=self.window, method='MLE')
        else:
            print(f'This method does not support {num_qubits}-qubit tomography.')
        self.rho = dummy[0]
        self.intensity = dummy[1]
        self.fvalp = dummy[2]
        self.extract_complex_numbers()
        self.analyse_rho(num_qubit=num_qubits)
        self.save_density_matrix(num_qubit=num_qubits)
        if verbose:
            print('The estimated density matrix is: \n')
            print(self.rho)
            print('\n The estimated intensity for normalizing the state is : ' + str(self.intensity))
            print('The final value of the optimization function is: ' + str(self.fvalp))

    def save_density_matrix(self, num_qubit=2):
        """
        This function saves the reconstructed density matrix.
        :return:
        """
        head_tail = os.path.split(self.path_coincidences)
        self.path_rho = os.path.join(head_tail[0], 'rho_' + head_tail[1])
        if num_qubit == 1:
            comment = 'H, V'
        if num_qubit == 2:
            comment = 'HH,  HV, VH, VV'
        np.savetxt(self.path_rho, self.rho, delimiter='\t', header=comment, comments='#', fmt='%s')

    def read_density_matrix(self, num_qubit):
        """
        This function is required for plotting an already constructed density matrix.
        :param num_qubit: Number of qubits.
        :return:
        """
        self.rho = np.genfromtxt(fname=self.path_rho, delimiter='\t', comments='#', dtype=str)
        self.convert_rho()
        self.extract_complex_numbers()
        self.analyse_rho(num_qubit=num_qubit)

    def extract_complex_numbers(self, verbose=False):
        """
        This function extracts the real and complex part out of the (string) array.
        :param array: (string) array
        :return:
        """
        self.rho_real = np.zeros_like(self.rho)
        self.rho_imag = self.rho_real.copy()
        for i in range(len(self.rho[:, 0])):
            for j in range(len(self.rho[0, :])):
                self.rho_real[i, j] = complex(self.rho[i, j]).real
                self.rho_imag[i, j] = complex(self.rho[i, j]).imag
        self.rho_real = self.rho_real.astype(float)
        if verbose:
            print('Real part of rho:')
            print(self.rho_real)
            print('\n')
        self.rho_imag = self.rho_imag.astype(float)
        self.rho = self.rho_real + 1j * self.rho_imag

    def convert_rho(self):
        """When the density matrix is saved to file as in this class, it needs to undergo a string-edit
        when it is read back in."""
        for i in range(len(self.rho[:, 0])):
            for j in range(len(self.rho[0, :])):
                self.rho[i, j] = self.rho[i, j].replace(' ', '')
                self.rho[i, j] = self.rho[i, j].replace('(', '')
                self.rho[i, j] = self.rho[i, j].replace(')', '')
                self.rho[i, j] = complex(self.rho[i, j])

    def get_trace(self, verbose=False):
        self.trace_rho = np.real(np.trace(self.rho))
        if verbose:
            print(r'$\Tr{\rho} = $' + str(self.trace_rho))

    def get_purity(self, verbose=False):
        self.purity_rho = qf.purity(self.rho)
        if verbose:
            print(r'$\Tr{{\rho} ** 2} = $#' + str(self.purity_rho))

    def get_concurrence(self, verbose=False):
        self.concurrence_rho = qf.concurrence(self.rho)
        if verbose:
            print(r'$C({\rho}) = $' + str(self.concurrence_rho))

    def analyse_rho(self, num_qubit):
        self.get_trace()
        self.get_purity()
        if num_qubit == 2:
            self.get_concurrence()

    def build_textbox(self, two_qubit=True):
        digit = 3
        text = r'$Tr(\rho)$' + ' = {}\n'.format(round(float(self.trace_rho), digit))
        text += r'$\gamma = Tr({\rho}^2)$' + ' = {}\n'.format(round(float(self.purity_rho), digit))
        if two_qubit:
            text += r'$C(\rho)$' + ' = {}'.format(round(float(self.concurrence_rho), digit))
        text += '\n\n\n\n\n'
        if type(self.path_coincidences) == str:
            c_name = os.path.split(self.path_coincidences)
            #e_name = os.path.split(self.path_efficiencies)
            text += 'CC_File: {}\n'.format(c_name[1])
            #text += 'Eff_File: {}'.format(e_name[1])
        return text

    def plot_text(self, num_qubit):
        plt.subplot(131)
        if num_qubit == 1:
            text = self.build_textbox(two_qubit=False)
        elif num_qubit == 2:
            text = self.build_textbox(two_qubit=True)
        else:
            text = f'Text-box not defined for {num_qubit} qubit'
        xpos = -0.17  # self.wl[0] - 25
        ypos = 0
        plt.text(xpos, ypos, text, fontsize=28)
        plt.axis('off')

    def plot_3d_bars_subplots(self, num_qubit, show=True, save=True):
        fig = plt.figure(figsize=(16, 8))
        plt.subplots_adjust(wspace=0.6)     # 0.4
        barplot_3d(figure=fig, matrix=self.rho_real, z_label='r', num_qubit=num_qubit, sub_row=1, sub_col=3, sub_idx=2,
                   barwidth=0.8)
        barplot_3d(figure=fig, matrix=self.rho_imag, z_label='i', num_qubit=num_qubit, sub_row=1, sub_col=3, sub_idx=3,
                   barwidth=0.8)
        plt.suptitle(f'{num_qubit}-Qubit Quantum State Tomography', fontsize=35, color='grey', style='italic')
        self.plot_text(num_qubit=num_qubit)
        if show:
            plt.show()
        if save:
            fig.savefig(self.path_rho + '.png')


def reconstruct_and_plot_rho(num_qubit):
    a = QSTAnalysis()
    folder = '/Users/michi/Library/CloudStorage/OneDrive-Persönlich/Physik/Master/WaltherGroup/Projects/HCF_TimBin-Entanglement/Data/01_03_23/HCF_Time-Bin/Tomography_measurement/1qubit_final/HCF'

    a.path_coincidences = os.path.join(folder, '1qubit_Tomo_w5ns_t10s_coincidences_20221007_1546.txt')
    a.path_singles = os.path.join(folder, '2qubit_Tomo_w2ns_t60s_singles_20220718_1141.txt')
    a.path_efficiencies = os.path.join(folder, '2_qubit_efficiency_measurement_230105-1551.txt')
    if num_qubit == 1:
        window = None
        int_time = None
    elif num_qubit == 2:
        window = 0.126      # In nanoseconds.
        int_time = 10   # in seconds.
    a.efficiencies = np.ones(2 * num_qubit)
    a.read_data(num_qubits=num_qubit, window_used=window, int_time=int_time)
    #a.load_efficiencies()
    a.run_tomography(num_qubits=num_qubit)
    a.plot_3d_bars_subplots(num_qubit=num_qubit)


def plot_rho_from_file(num_qubit):
    a = QSTAnalysis()
    folder = '/Users/michi/Library/CloudStorage/OneDrive-Persönlich/Physik/Master/WaltherGroup/Projects/HCF_TimBin-Entanglement/Data/01_03_23/HCF_Time-Bin/Tomography_measurement'
    a.path_rho = os.path.join(folder, 'rho_2qubit_Tomo_w161ns_t20s_coincidences_20230113_1753.txt')
    a.read_density_matrix(num_qubit=num_qubit)
    a.plot_3d_bars_subplots(num_qubit=num_qubit)


if __name__ == '__main__':
    reconstruct_and_plot_rho(num_qubit=1)
    #plot_rho_from_file(num_qubit=2)
