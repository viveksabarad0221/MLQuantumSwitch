import time


class Settings:
    def __init__(self):
        # motor settings:
        # Charlie1:
        self.Charlie1_com = 'COM5'
        self.Charlie1_HWP = 48.85  # 44.3  # 95.6
        self.Charlie1_QWP = 40.05  # 38.1

        # Bob:
        self.Bob_com = 'COM4'
        self.Bob_HWP = 88.1  # 86.1
        self.Bob_QWP = 115.3  # 122.7

        # swabian settings:
        self.h_channel_bob = 17
        self.delay_h_channel_bob = 0.172e-9

        self.v_channel_bob = 1
        self.delay_v_channel_bob = 0

        self.h_channel_charlie1 = 4
        self.delay_h_channel_charlie1 = 7.95e-9

        self.v_channel_charlie1 = 8
        self.delay_v_channel_charlie1 = 6.63e-9

        self.channel_list = [self.h_channel_bob, self.v_channel_bob, self.h_channel_charlie1, self.v_channel_charlie1]
        self.delays = [self.delay_h_channel_bob, self.delay_v_channel_bob, self.delay_h_channel_charlie1,
                       self.delay_v_channel_charlie1]

        self.swabian_thresholds = [0.05, 0.05, 0.05, 0.05]
        self.dead_time = 70

    def give_settings_to_tagger(self, tagger):
        tt = tagger
        for i in range(len(self.channel_list)):
            ch = self.channel_list[i]
            thr = self.swabian_thresholds[i]
            delay = self.delays[i]
            tt.set_trigger_level(channel=ch, voltage=thr)
            tt.set_delay_zero(i)  # making sure software and hardware delay are both zero
            time.sleep(0.5)
            tt.set_channel_delay(channel=ch, delay=delay)



    def start_motors(self):
        from elliptec.controller import Elliptec

        sQWP = self.signalQWP  # 40.28  # Charlie 1
        sHWP = self.signalHWP  # 119.04
        ms = Elliptec(dev=self.signal_com,
                      addrs=['0', '1'],
                      cal={'0': sQWP,
                           '1': sHWP})

        iQWP = self.idlerQWP  # 120.5   Bob
        iHWP = self.idlerHWP  # 82.06
        mi = Elliptec(dev=self.idler_com,
                      addrs=['0', '1'],
                      cal={'0': iQWP,
                           '1': iHWP})
        return ms, mi



