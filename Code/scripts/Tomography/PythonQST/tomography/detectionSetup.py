import time
from foundationsbib.helpers.motor_helper import InitElliptic
from switch.waveplates_and_motors.WaveplateSettings import WaveplateSettings
import numpy as np
import os
import threading
import TimeTagger


class DetectionSetup:
    def __init__(self, tagger=None, detector_init=True, is_virtual=False, delay_set=None, replay_file=''):
        """

        :param tagger: tagger object
        :param detector_init: True or False
        :param is_virtual: True or False
        :param delay_set:'Charlie0', 'Charlie1'
        :param replay_file:
        """
        self.tt = tagger  # Time tagger object
        self.parent_dir = os.path.join('Z:', os.sep, 'quantumSWITCH_DI', 'Data')  # Parent directory of the experiment
        self.threshold = 0.05  # 50 mV
        self.cc_window = None  # in seconds
        self.exposure_time = None  # in seconds
        self._is_virtual = is_virtual  # True when, the tagger object is virtual
        self._replay_file = replay_file
        self.delay_set = delay_set
        self.wp = WaveplateSettings()

        ############ swabian settings: ###########

        self.h_channel_bob = 9
        self.v_channel_bob = 14
        self.h_channel_charlie1 = 4
        self.v_channel_charlie1 = 8
        self.h_channel_charlie0 = 4
        self.v_channel_charlie0 = 8

        if self.delay_set == 'Charlie1':  # delays for Bob and the path directly to Charlie1 (no circulator)
            self.delay_h_channel_bob = 0.141e-9  # 17
            self.delay_v_channel_bob = 0  # 1
            self.delay_h_channel_charlie1 = 7.884e-9  # 4
            self.delay_v_channel_charlie1 = 6.609e-9  # 8
            # ch-key, ch-number, ch-delay

            self.channel_settings = np.array(
                [['ch_charlie1_h', self.h_channel_charlie1, self.delay_h_channel_charlie1],
                 ['ch_charlie1_v', self.v_channel_charlie1, self.delay_v_channel_charlie1],
                 ['ch_bob_h', self.h_channel_bob, self.delay_h_channel_bob],
                 ['ch_bob_v', self.v_channel_bob, self.delay_v_channel_bob]])

        elif self.delay_set == 'Charlie0':  # delays for Bob and Charlie_0 with input delay (for compensation)
            self.delay_h_channel_bob = 389.142e-9  # 9
            self.delay_v_channel_bob = 403.783e-9  # 1
            self.delay_h_channel_charlie0 = 1.318e-9  # 4
            self.delay_v_channel_charlie0 = 0  # 8
            # ch-key, ch-number, ch-delay
            self.channel_settings = np.array(
                [['ch_charlie0_h', self.h_channel_charlie0, self.delay_h_channel_charlie0],
                 ['ch_charlie0_v', self.v_channel_charlie0, self.delay_v_channel_charlie0],
                 ['ch_bob_h', self.h_channel_bob, self.delay_h_channel_bob],
                 ['ch_bob_v', self.v_channel_bob, self.delay_v_channel_bob]])

        # Single count containers
        self.s_charlie0_h = np.array([])
        self.s_charlie0_v = np.array([])
        self.s_charlie1_h = np.array([])
        self.s_charlie1_v = np.array([])
        self.s_bob_h = np.array([])
        self.s_bob_v = np.array([])

        # Coincidence count containers
        self.cc_charlie0_h__bob_h = np.array([])
        self.cc_charlie0_h__bob_v = np.array([])
        self.cc_charlie0_v__bob_h = np.array([])
        self.cc_charlie0_v__bob_v = np.array([])

        self.cc_charlie1_h__bob_h = np.array([])
        self.cc_charlie1_h__bob_v = np.array([])
        self.cc_charlie1_v__bob_h = np.array([])
        self.cc_charlie1_v__bob_v = np.array([])

        if detector_init:
            self.initialize_detector_settings()

    def initialize_detector_settings_qubib(self):
        """This function sets the trigger level and delay per channel."""
        for i in range(len(self.channel_settings[:, 0])):
            ch = int(self.channel_settings[i, 1])
            if not self._is_virtual:
                self.tt.set_trigger_level(channel=ch, voltage=self.threshold)
                # Delays are a bit tricky now, before 26.4.24 the TimeTaggerSwabian driver did set hardware delays,
                # which are also in the timetag file. Now I changed to software delays.
                self.tt.set_delay_zero(ch)
                time.sleep(0.5)
                #print('channel: ', ch)
                self.tt.set_channel_delay(channel=ch, delay=float(self.channel_settings[i, 2]))

    def initialize_detector_settings(self):
        """This function sets the trigger level and delay per channel."""
        for i in range(len(self.channel_settings[:, 0])):
            ch = int(self.channel_settings[i, 1])
            if not self._is_virtual:
                self.tt.set_trigger_level(channel=ch, voltage=self.threshold)
                # Delays are a bit tricky now, before 26.4.24 the TimeTaggerSwabian driver did set hardware delays,
                # which are also in the timetag file. Now I changed to software delays.
                self.tt.set_delay_zero(ch)
                time.sleep(0.5)
                #print('channel: ', ch)
                self.tt.set_channel_delay(channel=ch, delay=float(self.channel_settings[i, 2]))
        #self.tt.set_trigger_level(channel=9, voltage=self.threshold)
        #self.tt.set_delay_zero(9)
        #self.tt.set_channel_delay(channel=9, delay=float(self.channel_settings[i, 2]))


    def get_single_channels(self):

        if self.delay_set == 'Charlie1':
            user_ch = np.array([self.ch('ch_charlie1_h'), self.ch('ch_charlie1_v'),
                                self.ch('ch_bob_h'), self.ch('ch_bob_v')])
            return user_ch

        elif self.delay_set == 'Charlie0':
            user_ch = np.array([self.ch('ch_charlie0_h'), self.ch('ch_charlie0_v'),
                                self.ch('ch_bob_h'), self.ch('ch_bob_v')])
            return user_ch

        else:
            print('no valid delay_set')

    def get_channels(self):

        if self.delay_set == 'Charlie0':
            cc_channels = np.array([[self.ch('ch_charlie0_h'), self.ch('ch_bob_h')],
                                    [self.ch('ch_charlie0_h'), self.ch('ch_bob_v')],
                                    [self.ch('ch_charlie0_v'), self.ch('ch_bob_h')],
                                    [self.ch('ch_charlie0_v'), self.ch('ch_bob_v')]])
            return cc_channels

        elif self.delay_set == 'Charlie1':
            cc_channels = np.array([[self.ch('ch_charlie1_h'), self.ch('ch_bob_h')],
                                    [self.ch('ch_charlie1_h'), self.ch('ch_bob_v')],
                                    [self.ch('ch_charlie1_v'), self.ch('ch_bob_h')],
                                    [self.ch('ch_charlie1_v'), self.ch('ch_bob_v')]])
            return cc_channels

        else:
            print('no valid delay_set')

    def read_out_tagger1(self):
        count_channels = self.get_single_channels()
        cc_channels = self.get_channels()
        print('single channels:', count_channels)

        # measure singles:
        total_counts = self.tt.get_total_counts(channels=count_channels,
                                                t_measure=self.exposure_time,
                                                is_virtual=self._is_virtual, replay_file=self._replay_file)
        # measure coincidences
        coincidences = self.tt.get_coincidence_counts(cc_groups=cc_channels,
                                                      cc_window=self.cc_window,
                                                      t_measure=self.exposure_time,
                                                      is_virtual=self._is_virtual, replay_file=self._replay_file)

        # read out tagger channels:
        if self.delay_set == 'Charlie1':

            self.s_charlie1_h = np.append(self.s_charlie1_h, total_counts[0])
            self.s_charlie1_v = np.append(self.s_charlie1_v, total_counts[1])
            self.s_bob_h = np.append(self.s_bob_h, total_counts[2])
            self.s_bob_v = np.append(self.s_bob_v, total_counts[3])

            self.cc_charlie1_h__bob_h = np.append(self.cc_charlie1_h__bob_h, coincidences[0])
            self.cc_charlie1_h__bob_v = np.append(self.cc_charlie1_h__bob_v, coincidences[1])
            self.cc_charlie1_v__bob_h = np.append(self.cc_charlie1_v__bob_h, coincidences[2])
            self.cc_charlie1_v__bob_v = np.append(self.cc_charlie1_v__bob_v, coincidences[3])

        elif self.delay_set == 'Charlie0':

            self.s_charlie0_h = np.append(self.s_charlie0_h, total_counts[0])
            self.s_charlie0_v = np.append(self.s_charlie0_v, total_counts[1])
            self.s_bob_h = np.append(self.s_bob_h, total_counts[2])
            self.s_bob_v = np.append(self.s_bob_v, total_counts[3])

            self.cc_charlie0_h__bob_h = np.append(self.cc_charlie0_h__bob_h, coincidences[0])
            self.cc_charlie0_h__bob_v = np.append(self.cc_charlie0_h__bob_v, coincidences[1])
            self.cc_charlie0_v__bob_h = np.append(self.cc_charlie0_v__bob_h, coincidences[2])
            self.cc_charlie0_v__bob_v = np.append(self.cc_charlie0_v__bob_v, coincidences[3])

    def read_out_tagger(self):
        count_channels = self.get_single_channels()
        cc_channels = self.get_channels()
        #for ch in count_channels:
            #print('single channel:', ch)
            #print('type: ', type(ch))

        # measure singles:
        self.tt.tagger.sync()
        with TimeTagger.Coincidences(self.tt.tagger, cc_channels,
                                     coincidenceWindow=int(self.tt.seconds_to_picoseconds(time_s=self.cc_window)),
                                     timestamp=TimeTagger.CoincidenceTimestamp.Last) as cc_measurement, \
                TimeTagger.Countrate(self.tt.tagger, channels=count_channels) as s_measurement:
            coincidence_data = TimeTagger.Countrate(tagger=self.tt.tagger, channels=cc_measurement.getChannels())

            time.sleep(self.exposure_time)

            total_counts = s_measurement.getCountsTotal()
            coincidences = coincidence_data.getData()
            coincidence_data.clear()
            s_measurement.clear()

        if self.delay_set == 'Charlie1':

            self.s_charlie1_h = np.append(self.s_charlie1_h, total_counts[0])
            self.s_charlie1_v = np.append(self.s_charlie1_v, total_counts[1])
            self.s_bob_h = np.append(self.s_bob_h, total_counts[2])
            self.s_bob_v = np.append(self.s_bob_v, total_counts[3])

            self.cc_charlie1_h__bob_h = np.append(self.cc_charlie1_h__bob_h, coincidences[0])
            self.cc_charlie1_h__bob_v = np.append(self.cc_charlie1_h__bob_v, coincidences[1])
            self.cc_charlie1_v__bob_h = np.append(self.cc_charlie1_v__bob_h, coincidences[2])
            self.cc_charlie1_v__bob_v = np.append(self.cc_charlie1_v__bob_v, coincidences[3])

        elif self.delay_set == 'Charlie0':

            self.s_charlie0_h = np.append(self.s_charlie0_h, total_counts[0])
            self.s_charlie0_v = np.append(self.s_charlie0_v, total_counts[1])
            self.s_bob_h = np.append(self.s_bob_h, total_counts[2])
            self.s_bob_v = np.append(self.s_bob_v, total_counts[3])

            self.cc_charlie0_h__bob_h = np.append(self.cc_charlie0_h__bob_h, coincidences[0])
            self.cc_charlie0_h__bob_v = np.append(self.cc_charlie0_h__bob_v, coincidences[1])
            self.cc_charlie0_v__bob_h = np.append(self.cc_charlie0_v__bob_h, coincidences[2])
            self.cc_charlie0_v__bob_v = np.append(self.cc_charlie0_v__bob_v, coincidences[3])

    def get_channel_info_from_key(self, key: str):
        """This function returns the channel number and delay corresponding to a given key."""
        idx = np.where(self.channel_settings[:, 0] == key)[0][0]
        print(self.channel_settings[idx, :])
        ch = int(self.channel_settings[idx, 1])
        delay = float(self.channel_settings[idx, 2])
        print(f'Channel with key "{key}" has channel number {ch} and is {delay} seconds delayed.')
        return ch, delay

    def ch(self, key):
        """Returns the physical channel number of time tagger corresponding to the 'key' as an integer."""
        idx = np.where(self.channel_settings[:, 0] == key)[0][0]
        channel = int(self.channel_settings[idx, 1])
        return channel

    @staticmethod
    def initialize_motors(settings):
        """
        :param settings: motor settings
        :return: initialized motor objects
        """
        i = InitElliptic()
        se = settings
        delay_set = se.delay_set
        tau = None
        tbu = None

        Bob_QWP = se.Bob_QWP  # addrs= '0'
        Bob_HWP = se.Bob_HWP  # addrs = '1'
        Bob_port = se.Bob_com

        Charlie1_QWP = se.Charlie1_QWP  # addrs = '0'
        Charlie1_HWP = se.Charlie1_HWP  # addrs = '1'
        Charlie1_port = se.Charlie1_com

        Charlie0_QWP = se.Charlie0_QWP  # addrs = '0'
        Charlie0_HWP = se.Charlie0_HWP  # addrs = '1'
        Charlie0_port = se.Charlie0_com

        if delay_set == 'Charlie1':
            tau = threading.Thread(target=i.initialize_motor, args=(2, Bob_port, ['0', '1'], [Bob_QWP, Bob_HWP]))
            tbu = threading.Thread(target=i.initialize_motor, args=(1, Charlie1_port, ['0', '1'], [Charlie1_QWP,
                                                                                                   Charlie1_HWP]))

        elif delay_set == 'Charlie0':
            tau = threading.Thread(target=i.initialize_motor, args=(2, Bob_port, ['0', '1'], [Bob_QWP, Bob_HWP]))
            tbu = threading.Thread(target=i.initialize_motor, args=(1, Charlie0_port, ['0', '1'], [Charlie0_QWP,
                                                                                                   Charlie0_HWP]))
        if tau:
            tau.start()
            tbu.start()

            tau.join()
            tbu.join()
            return i.motors1, i.motors2
        else:
            print('failed to initialize motors')
            return





if __name__ == '__main__':
    DetectionSetup(tagger=None, detector_init=False)
    #d.get_channel_info_from_key(key='ch_alice_user_h')
    #cc = d.get_verifier_cc_channels()

