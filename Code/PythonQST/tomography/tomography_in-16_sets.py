
import numpy as np
import time
from datetime import datetime
import os


class StateTomography:
    def __init__(self, detector_readout=None, motor_addresses=None):
        """
        This class takes the required data for reconstructing the 2qubit density matrix.
        :param detector_readout: Class to do basic preparations for measurements and reading out the tagger.
        :param motor_addresses: A dictionary mapping waveplate names to motor addresses.
        """
        self.detection_data = detector_readout
        self.motor_addresses = motor_addresses  # Example: {'1QWP': '2', '1HWP': '3', '2QWP': '0', '2HWP': '1'}
        self._datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Lists with waveplate angles and projection bases.
        self._base_1 = np.array([])
        self._base_2 = np.array([])
        self._angle_1QWP = np.array([])
        self._angle_1HWP = np.array([])
        self._angle_2QWP = np.array([])
        self._angle_2HWP = np.angle([])

    def _assign_waveplate_angles_1qubit(self):
        """Assign angles and bases for 1-qubit tomography"""
        self._base_1 = np.array(['H', 'D', 'R'])
        self._angle_1QWP = np.array([0, 45, 45])
        self._angle_1HWP = np.array([0, -22.5, 0])

    def _assign_waveplate_angles_3detectors(self):
        """Assign angles and bases for 2-qubit tomography with 16 measurements"""
        self._base_1 = np.array(['H', 'H', 'H', 'D', 'D', 'D', 'R', 'R', 'R'])
        self._base_2 = np.array([['H', 'V'], ['D','A'], ['R','L'], ['H','V'], ['D','A'], ['R','L'], ['H','V'], ['D','A'], ['R','L']])
        # 22.5 in negative direction:
        self._angle_1QWP = np.array([0, 0, 0, 0, 0, 0, 0, 0, 45, 45, 45, 45, 45, 45, 45, 45])
        self._angle_1HWP = np.array([0, 0, 0, 0, -45, -45, -45, -45, -22.5, -22.5, -22.5, -22.5, 0, 0, 0, 0])

        self._angle_2QWP = np.array([0, 0, 45, 45, 0, 0, 45, 45, 0, 0, 45, 45, 0, 0, 45, 45])
        self._angle_2HWP = np.array([0, -45, -22.5, 0, 0, -45, -22.5, 0, 0, -45, -22.5, 0, 0, -45, -22.5, 0])

    def _assign_waveplate_angles_2qubit(self):
        """Assign angles and bases for 2-qubit tomography"""
        self._base_1 = np.array(['H', 'H', 'H', 'D', 'D', 'D', 'R', 'R', 'R'])
        self._base_2 = np.array(['H', 'D', 'R', 'H', 'D', 'R', 'H', 'D', 'R'])
        self._angle_1QWP = np.array([0, 0, 0, 45, 45, 45, 45, 45, 45])
        self._angle_1HWP = np.array([0, 0, 0, -22.5, -22.5, -22.5, 0, 0, 0])
        self._angle_2QWP = np.array([0, 45, 45, 0, 45, 45, 0, 45, 45])
        self._angle_2HWP = np.array([0, -22.5, 0, 0, -22.5, 0, 0, -22.5, 0])

    def _move_waveplates(self, motor, waveplate_keys, angles):
        """
        Moves the specified waveplates by looking up their addresses in `motor_addresses`.

        :param motor: The motor control object
        :param waveplate_keys: List of waveplate names (e.g., ['1QWP', '1HWP'])
        :param angles: List of angles corresponding to the waveplates
        """
        addresses = [self.motor_addresses[key] for key in waveplate_keys]
        motor.calmove(addresses, angles)

    def _run_measurement(self, iterations, motor_1, waveplate_keys_1, motor_2=None, waveplate_keys_2=None):
        """
        Generalized measurement function for 1-qubit and 2-qubit tomography.
        :param iterations: Number of angle settings to iterate over.
        :param motor_1: Motor object for waveplate control of photon 1.
        :param waveplate_keys_1: List of waveplate keys (e.g., ['1QWP', '1HWP']).
        :param motor_2: (Optional) Motor object for photon 2.
        :param waveplate_keys_2: (Optional) List of waveplate keys for photon 2 (e.g., ['2QWP', '2HWP']).
        """
        for i in range(iterations):
            moves = []  # Store motor movements

            # Move waveplates for photon 1
            if i == 0:
                moves.append((motor_1, waveplate_keys_1, [self._angle_1QWP[i], self._angle_1HWP[i]]))
            else:
                if int(self._angle_1QWP[i]) != int(self._angle_1QWP[i - 1]):
                    moves.append((motor_1, [waveplate_keys_1[0]], [self._angle_1QWP[i]]))
                if int(self._angle_1HWP[i]) != int(self._angle_1HWP[i - 1]):
                    moves.append((motor_1, [waveplate_keys_1[1]], [self._angle_1HWP[i]]))

            # Move waveplates for photon 2 (if doing 2-qubit tomography)
            if motor_2 and waveplate_keys_2:
                if i == 0:
                    moves.append((motor_2, waveplate_keys_2, [self._angle_2QWP[i], self._angle_2HWP[i]]))
                else:
                    if int(self._angle_2QWP[i]) != int(self._angle_2QWP[i - 1]):
                        moves.append((motor_2, [waveplate_keys_2[0]], [self._angle_2QWP[i]]))
                    if int(self._angle_2HWP[i]) != int(self._angle_2HWP[i - 1]):
                        moves.append((motor_2, [waveplate_keys_2[1]], [self._angle_2HWP[i]]))

            # Execute motor movements
            for motor, plate_keys, angles in moves:
                self._move_waveplates(motor, plate_keys, angles)

            time.sleep(0.2 if motor_2 is None else 0.5)  # Shorter delay for 1-qubit tomography
            print(f'BASE ROUND {i + 1} SET!')
            print('START detection!')
            self.detection_data.read_out_tagger()
            print('Detection DONE!')

    def measurement_1qubit(self, motor, photon_id='1'):
        """
        Perform a 1-qubit quantum state tomography measurement on either photon 1 or photon 2.

        :param motor: The motor object controlling the waveplates.
        :param photon_id: '1' for photon 1, '2' for photon 2.
        """
        self._assign_waveplate_angles_1qubit()  # Assign angles for 1-qubit tomography

        # Select the correct waveplate addresses based on photon_id
        if photon_id == '1':
            waveplate_keys_1 = ['1QWP', '1HWP']
        elif photon_id == '2':
            waveplate_keys_1 = ['2QWP', '2HWP']
        else:
            raise ValueError("Invalid photon_id. Choose '1' or '2'.")

        # Pass waveplate keys and motor to _run_measurement
        self._run_measurement(len(self._angle_1QWP), motor_1=motor, waveplate_keys_1=waveplate_keys_1)

    def measurement_parallel_1qubit(self, motor_1, motor_2):
        self._assign_waveplate_angles_1qubit()
        self._base_2 = self._base_1
        self._angle_2QWP = self._angle_1QWP
        self._angle_2HWP = self._angle_1HWP
        # Define waveplate keys for both photons
        waveplate_keys_1 = ['1QWP', '1HWP']
        waveplate_keys_2 = ['2QWP', '2HWP']
        self._run_measurement(len(self._angle_1QWP),
                              motor_1=motor_1, waveplate_keys_1=waveplate_keys_1,
                              motor_2=motor_2, waveplate_keys_2=waveplate_keys_2)

    def measurement_2qubit(self, motor_1, motor_2):
        """
        Perform a 2-qubit quantum state tomography measurement.

        :param motor_1: Motor object for photon 1.
        :param motor_2: Motor object for photon 2.
        """
        self._assign_waveplate_angles_2qubit()  # Assign angles for 2-qubit tomography

        # Define waveplate keys for both photons
        waveplate_keys_1 = ['1QWP', '1HWP']
        waveplate_keys_2 = ['2QWP', '2HWP']

        # Pass both motors and waveplate keys to _run_measurement
        self._run_measurement(len(self._angle_1QWP), motor_1, waveplate_keys_1, motor_2, waveplate_keys_2)

    def save_data_2qubit(self, directory, fname,
                         cc_hs_hi, cc_hs_vi, cc_vs_hi, cc_vs_vi,
                         s_hs, s_vs, s_hi, s_vi):
        """
        This function saves the recorded data to the files.
        To keep the script general, the corresponding count container from the detection_setup script have to be
        passed here manually (can be improved).
        The cc_... container correspond the coincidences (e.g. cc_hs_hi, where s and i correspond to
        signal and idler respectively.)
        The s_... container correspond the singles (e.g. s_hs, s_vi, where s and i correspond to
        signal and idler respectively.)
        """
        data_coincidences = np.transpose([self._base_1,
                                          self._base_2,
                                          cc_hs_hi,
                                          cc_hs_vi,
                                          cc_vs_hi,
                                          cc_vs_vi])
        data_singles = np.transpose([self._base_1,
                                     self._base_2,
                                     s_hs,
                                     s_vs,
                                     s_hi,
                                     s_vi])
        comment_singles = '1_base, 2_base, 1H, 1V, 2H, 2V; ----- date: ' + str(self._datetime)
        comment_coincidences = '1_base, 2_base, h1_h2, h1_v2, v1_h2, v1_v2; ----- date: ' + str(self._datetime)
        path_singles = os.path.join(directory, str(self._datetime) + 'singles_' + fname + '.txt')
        np.savetxt(path_singles, data_singles,
                   delimiter='\t', header=comment_singles, comments='#', fmt='%s')
        path_coincidences = os.path.join(directory, str(self._datetime) + 'coincidences_' + fname + '.txt')
        np.savetxt(path_coincidences, data_coincidences,
                   delimiter='\t', header=comment_coincidences, comments='#', fmt='%s')
        return path_singles, path_coincidences

    def save_data_1qubit(self, directory, fname, cc_herald_h, cc_herald_v):
        data_coincidences = np.transpose([self._base_1,
                                          cc_herald_h,
                                          cc_herald_v])
        comment_coincidences = '1_base, herald_h, herald_v; ----- date: ' + str(self._datetime)
        path_coincidences = os.path.join(directory, str(self._datetime) + 'coincidences_' + fname + '.txt')
        np.savetxt(path_coincidences, data_coincidences,
                   delimiter='\t', header=comment_coincidences, comments='#', fmt='%s')
        return path_coincidences


def run_tomography_measurement(num_qubit: int):
    from qubib.devices.thorlabs.ElliptecMotor import ElliptecMotor
    from qubib.devices.swabian_instruments.TimeTaggerSwabian import TimeTaggerSwabian

    tt = TimeTaggerSwabian()
    tt.connect_network_tagger(ip_addrs='10.42.20.150')
    detection = None #Insert class here
    'Motor assignment'
    sQWP = 0
    sHWP = 0
    ms = ElliptecMotor(dev='COM0',
                  addrs=['0', '1'],
                  cal={'0': sQWP,
                       '1': sHWP})
    iQWP = 0
    iHWP = 0
    mi = ElliptecMotor(dev='COM0',
                  addrs=['0', '1'],
                  cal={'0': iQWP,
                       '1': iHWP})
    motor_addresses = {'1QWP': '0', '1HWP': '1',
                       '2QWP': '0', '2HWP': '1'}
    qst = StateTomography(detector_readout=detection, motor_addresses=motor_addresses)
    'Prepare detectors:'
    detection.window = 5e-9         # in seconds
    detection.exposure_time = 10    # in seconds

    if num_qubit == 1:
        qst.measurement_1qubit(motor_1=ms)
        #qst.save_data_1qubit(directory='', fname='')
    elif num_qubit == 2:
        qst.measurement_2qubit(motor_1=ms, motor_2=mi)
        """make the directory that it checks does it exist, if not create. See some swabian example"""
        #qst.save_data_2qubit(directory=, fname=)
    else:
        print(f'Measurement for {num_qubit} qubit is not available.')

    ms.close()
    mi.close()
    tt.free_time_tagger()


if __name__ == '__main__':
    run_tomography_measurement(num_qubit=2)

