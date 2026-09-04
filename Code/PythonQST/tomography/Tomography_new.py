import threading

import pathlib

import numpy as np
import time
from datetime import datetime
import os


class StateTomography:
    def __init__(self, detector_readout):
        """
        This class takes the required data for reconstructing the 2qubit density matrix.
        :param detector_readout: Class to do basic preparations for measurements and reading out the tagger.
        """
        self.detection_data = detector_readout
        self.datetime = datetime.now().strftime("%Y%m%d_%H%M")
        # Lists with waveplate angles and projection bases.
        self.base_signal = np.array([])
        self.base_idler = np.array([])
        self.angle_sQWP = np.array([])
        self.angle_sHWP = np.array([])
        self.angle_iQWP = np.array([])
        self.angle_iHWP = np.angle([])

    def assign_waveplate_angles_2qubit(self):
        """Assign angles and bases for 2-qubit tomography"""
        self.base_signal = np.array(['H', 'H', 'H', 'D', 'D', 'D', 'R', 'R', 'R'])
        self.base_idler = np.array(['H', 'D', 'R', 'H', 'D', 'R', 'H', 'D', 'R'])

        """
        # positive directions:
        self.angle_sQWP = np.array([0, 0, 0, 45, 45, 45, 45, 45, 45])
        self.angle_sHWP = np.array([0, 0, 0, 22.5, 22.5, 22.5, 0, 0, 0])
        self.angle_iQWP = np.array([0, 45, 45, 0, 45, 45, 0, 45, 45])
        self.angle_iHWP = np.array([0, 22.5, 0, 0, 22.5, 0, 0, 22.5, 0])
        """

        # 22.5 in negative direction:
        self.angle_sQWP = np.array([0, 0, 0, 45, 45, 45, 45, 45, 45])
        self.angle_sHWP = np.array([0, 0, 0, -22.5, -22.5, -22.5, 0, 0, 0])
        self.angle_iQWP = np.array([0, 45, 45, 0, 45, 45, 0, 45, 45])
        self.angle_iHWP = np.array([0, -22.5, 0, 0, -22.5, 0, 0, -22.5, 0])

    def measurement_2qubit(self, signal_motor, idler_motor, ms_addrs, mi_addrs):
        self.assign_waveplate_angles_2qubit()
        # Charlie:
        key_hwp_s = ms_addrs[1]
        key_qwp_s = ms_addrs[0]
        # Bob:
        key_hwp_i = mi_addrs[1]
        key_qwp_i = mi_addrs[0]
        for i in range(len(self.angle_sQWP)):
            if i == 0:
                # Signal waveplates:
                signal_motor.calmove([key_qwp_s, key_hwp_s], [self.angle_sQWP[i], self.angle_sHWP[i]])
                # Idler waveplates:
                idler_motor.calmove([key_qwp_i, key_hwp_i], [self.angle_iQWP[i], self.angle_iHWP[i]])
                time.sleep(0.5)
                print('BASE ROUND ' + str(i + 1) + ' SET!')
                print('START detection!')
                self.detection_data.read_out_tagger()
                print('Detection DONE!')
            else:
                # Signal waveplates:
                if int(self.angle_sQWP[i]) != int(self.angle_sQWP[i - 1]):
                    signal_motor.calmove([key_qwp_s], [self.angle_sQWP[i]])
                if int(self.angle_sHWP[i]) != int(self.angle_sHWP[i - 1]):
                    signal_motor.calmove([key_hwp_s], [self.angle_sHWP[i]])
                # Idler waveplates:
                if int(self.angle_iQWP[i]) != int(self.angle_iQWP[i - 1]):
                    idler_motor.calmove([key_qwp_i], [self.angle_iQWP[i]])
                if int(self.angle_iHWP[i]) != int(self.angle_iHWP[i - 1]):
                    idler_motor.calmove([key_hwp_i], [self.angle_iHWP[i]])
                time.sleep(0.5)
                print('BASE ROUND ' + str(i + 1) + ' SET!')
                print('START detection!')
                self.detection_data.read_out_tagger()
                print('Detection DONE!')

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
        data_coincidences = np.transpose([self.base_signal,
                                          self.base_idler,
                                          cc_hs_hi,
                                          cc_hs_vi,
                                          cc_vs_hi,
                                          cc_vs_vi])
        data_singles = np.transpose([self.base_signal,
                                     self.base_idler,
                                     s_hs,
                                     s_vs,
                                     s_hi,
                                     s_vi])
        comment_singles = 's_base, i_base, sH, sV, iH, iV; ----- date: ' + str(self.datetime)
        comment_coincidences = 's_base, i_base, hs_hi, hs_vi, vs_hi, vs_vi; ----- date: ' + str(self.datetime)
        path_singles = os.path.join(directory, 'singles_' + str(self.datetime) + fname + '.txt')
        np.savetxt(path_singles, data_singles,
                   delimiter='\t', header=comment_singles, comments='#', fmt='%s')
        path_coincidences = os.path.join(directory, 'coincidences_' + str(self.datetime) + fname + '.txt')
        np.savetxt(path_coincidences, data_coincidences,
                   delimiter='\t', header=comment_coincidences, comments='#', fmt='%s')
        return path_singles, path_coincidences

    def assign_waveplate_angles_1qubit(self):
        """Assign angles and bases for 1-qubit tomography"""
        self.base_signal = np.array(['H', 'D', 'R'])
        self.angle_sQWP = np.array([0, 45, 45])
        self.angle_sHWP = np.array([0, 22.5, 0])

    def measurement_1qubit(self, signal_motor, herald_motor):
        self.assign_waveplate_angles_1qubit()
        for i in range(len(self.angle_sQWP)):
            if i == 0:
                # Herald waveplates:
                herald_motor.calmove(['0', '1'], [0, 0])
                # Signal waveplates:
                signal_motor.calmove(['0', '1'], [self.angle_sQWP[i], self.angle_sHWP[i]])
                time.sleep(0.2)
                print('BASE ROUND ' + str(i + 1) + ' SET!')
                print('START detection!')
                self.detection_data.read_out_tagger()
                print('Detection DONE!')
            else:
                # Signal waveplates:
                if int(self.angle_sQWP[i]) != int(self.angle_sQWP[i - 1]):
                    signal_motor.calmove(['0'], [self.angle_sQWP[i]])
                if int(self.angle_sHWP[i]) != int(self.angle_sHWP[i - 1]):
                    signal_motor.calmove(['1'], [self.angle_sHWP[i]])
                time.sleep(0.2)
                print('BASE ROUND ' + str(i + 1) + ' SET!')
                print('START detection!')
                self.detection_data.read_out_tagger()
                print('Detection DONE!')

    def save_data_1qubit(self, directory, fname,
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
        TODO: Detector configuration should be written in file (window, exposure time)"""
        data_coincidences = np.transpose([self.base_signal,
                                          cc_hs_hi,
                                          cc_hs_vi,
                                          cc_vs_hi,
                                          cc_vs_vi])
        data_singles = np.transpose([self.base_signal,
                                     s_hs,
                                     s_vs,
                                     s_hi,
                                     s_vi])
        comment_singles = 's_base, sH, sV, iH, iV; ----- date: ' + str(self.datetime)
        comment_coincidences = 's_base, hs_hi, hs_vi, vs_hi, vs_vi; ----- date: ' + str(self.datetime)
        path_singles = os.path.join(directory, str(self.datetime) + 'singles_' + fname + '.txt')
        np.savetxt(path_singles, data_singles,
                   delimiter='\t', header=comment_singles, comments='#', fmt='%s')
        path_coincidences = os.path.join(directory, str(self.datetime) + 'coincidences_' + fname + '.txt')
        np.savetxt(path_coincidences, data_coincidences,
                   delimiter='\t', header=comment_coincidences, comments='#', fmt='%s')
        return path_singles, path_coincidences


def run_tomography_measurement(num_qubit: int, cc_window: float, exposure_time: float, name: str = None,
                               verbose: bool = True, signal: str = 'Charlie0'):
    """
    :param signal: switch between waveplate pairs - 'Charlie0' or 'Charlie1'
    :param name: add comment to file name
    :param num_qubit: 2 or 1
    :param cc_window: in seconds
    :param exposure_time: also in seconds
    :param verbose: True or False
    :return: file paths of single and coincidence counts
    """
    from elliptec.controller import Elliptec
    from qubib.devices.swabian_instruments.TimeTaggerSwabian import TimeTaggerSwabian
    from switch.tomography.detectionSetup import DetectionSetup
    from foundationsbib.helpers.DirectoryManagment import directory_generator
    from foundationsbib.helpers.motor_helper import InitElliptic

    tt = TimeTaggerSwabian()
    tt.connect_network_tagger(ip_addrs='131.130.102.185')

    detection = DetectionSetup(tagger=tt, detector_init=True, delay_set=signal)  # Insert class here
    qst = StateTomography(detector_readout=detection)
    # server path:
    directory = os.path.join('Z:', os.sep, 'quantumSWITCH_DI', 'Data', 'Tomography',
                             datetime.now().strftime("%y_%m_%d"))
    # local path:
    # directory = os.path.join(pathlib.Path.home(), 'OneDrive', 'Desktop', 'SWITCH_LCI', 'Tomography_measurement')
    directory_generator(dir_name=directory)
    if name:
        file = name
    else:
        file = 'oT'

    if verbose:
        print('connecting to motors...')

    'Motor assignment'
    init = InitElliptic()

    if signal == 'Charlie1':
        ts = threading.Thread(target=init.initialize_motor,
                              args=(1, detection.Charlie1_com, detection.Charlie1_addrs,
                                    [detection.Charlie1_QWP, detection.Charlie1_HWP]))
    elif signal == 'Charlie0':
        ts = threading.Thread(target=init.initialize_motor,
                              args=(1, detection.Charlie0_com, detection.Charlie0_addrs,
                                    [detection.Charlie0_QWP, detection.Charlie0_HWP]))
    # Bob as idler:
    ti = threading.Thread(target=init.initialize_motor, args=(2, detection.Bob_com, detection.Bob_addrs,
                                                              [detection.Bob_QWP, detection.Bob_HWP]))
    ts.start()
    ti.start()
    ts.join()
    ti.join()

    ms = init.motors1
    mi = init.motors2

    'Prepare detectors:'
    detection.cc_window = cc_window  # in seconds
    detection.exposure_time = exposure_time  # in seconds

    if num_qubit == 1:
        qst.measurement_1qubit(signal_motor=ms, herald_motor=mi)
        # qst.save_data_1qubit(directory='', fname='')

    elif num_qubit == 2:
        qst.measurement_2qubit(signal_motor=ms, idler_motor=mi, ms_addrs=detection.Charlie1_addrs,
                               mi_addrs=detection.Bob_addrs)

        """make the directory that it checks does it exist, if not create. See some swabian example"""
        cc_hs_hi = detection.cc_charlie1_h__bob_h
        cc_hs_vi = detection.cc_charlie1_h__bob_v
        cc_vs_hi = detection.cc_charlie1_v__bob_h
        cc_vs_vi = detection.cc_charlie1_v__bob_v
        s_hs = detection.s_charlie1_h
        s_vs = detection.s_charlie1_v
        s_hi = detection.s_bob_h
        s_vi = detection.s_bob_v

        s_path, cc_path = qst.save_data_2qubit(directory=directory, fname=file, cc_hs_hi=cc_hs_hi, cc_hs_vi=cc_hs_vi,
                                               cc_vs_hi=cc_vs_hi, cc_vs_vi=cc_vs_vi, s_hs=s_hs, s_vs=s_vs,
                                               s_hi=s_hi, s_vi=s_vi)
    else:
        print(f'Measurement for {num_qubit} qubit is not available.')

    if verbose:
        print('\n')
        print('singles-data saved to: ', s_path)
        print('coincidence counts saved to: ', cc_path)
    ms.close()
    mi.close()
    tt.free_time_tagger()
    return s_path, cc_path


def measure_and_plot():
    from foundationsbib.proccessingclasses.TomographyAnalysis import reconstruct_and_plot_rho_files

    qubits = 2
    cc_window = 647e-12  # 5e-9
    exposure_time = 3
    name = '_Bob_Charlie0'
    signal = 'Charlie0'

    s_path, cc_path = run_tomography_measurement(num_qubit=qubits, cc_window=cc_window, exposure_time=exposure_time,
                                                 name=name, signal=signal)
    reconstruct_and_plot_rho_files(qubits, cc_path, s_path, window=cc_window, int_time=exposure_time, eff_zero=True)


if __name__ == '__main__':
    measure_and_plot()
