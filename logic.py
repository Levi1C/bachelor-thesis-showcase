import MFLI
import arduino
import Keysight
import file_logic as file
import inspect
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import analyse_logic as al


class Logic:
    def __init__(self, use_servo=True, use_mfli=True, use_signal_gen=False):
        if use_servo:
            self.servo = arduino.Arduino(servo_min=900, servo_max=1972, position_printing=False)
        else:
            self.servo = None

        if use_mfli:
            self.mfli = MFLI.Mfli(time_constant=0.0001)
        else:
            self.mfli = None

        if use_signal_gen:
            self.signal_gen = Keysight.SignalGenerator()
        else:
            self.signal_gen = None

    def toggle_servo(self):
        if self.servo is None:
            self.servo = arduino.Arduino(servo_min=900, servo_max=1972, position_printing=False)
        else:
            self.servo.close_connection()
            self.servo = None

    def toggle_mfli(self):
        if self.mfli is None:
            self.mfli = MFLI.Mfli()
        else:
            self.mfli = None

    def toggle_signal_gen(self):
        if self.signal_gen is None:
            self.signal_gen = Keysight.SignalGenerator()
        else:
            self.signal_gen = None

    def measurement(self, poll_time=15., noise=False, pre_subscribed=False, stop_time=0.8):
        """Measurement logic to be used in every measurement.

        First it subscribes to the nodes of the mfli. Then plucks the string
        with or without stopping it (is omitted if noise
        is set to True).

        Then it polls for the given time and unsubscribes from the mfli.

        :param poll_time: Time in [s] to measure (poll) for
        :param noise: bool if just noise should be recorded, omits plucking
        :param pre_subscribed: if subscribe is called before the measurement function is called
        :param stop_time: time to stop the string for
        :return: two lists with time and demodulated amplitude R
        """
        if not pre_subscribed:
            # subscribe to data node
            self.mfli.subscribe()
        # sync the nodes
        self.mfli.sync()

        # plucking
        if not noise:
            self.servo.move_servo(1)

        # Poll data
        time_data, amplitude = self.mfli.poll_r(poll_time)

        # stop string
        if not noise:
            self.servo.stop_string(stop_time)

        if not pre_subscribed:
            # unsubscribe from data node
            self.mfli.unsubscribe()

        return time_data, amplitude

    def measurement_cycle(self, cycle_total=1,
                          manual_continue=False,
                          **kwargs):
        """
        Function to repeat measurement for cycle_total times. Data is stored for every iteration.

        :param cycle_total: int total number of cycles
        :param manual_continue: bool, if cycle waits for input before continuing
        :param kwargs: keyword arguments for measurement function and create_filename function
        """

        # get keyword-arguments (kwargs) of measurement function out of kwargs
        kwargs_for_measurement = {}
        for keys in inspect.getfullargspec(self.measurement)[0]:
            if keys in kwargs.keys():
                kwargs_for_measurement[keys] = kwargs[keys]

        # get keyword-arguments (kwargs) of create_filename function out of kwargs
        kwargs_for_filename = {}
        for keys in inspect.getfullargspec(file.create_filename)[0]:
            if keys in kwargs.keys():
                kwargs_for_filename[keys] = kwargs[keys]
        kwargs_for_filename['raw_data'] = True

        # repeat the measurement cycle for cycle_total times
        for cycle_index in range(cycle_total):

            # set measurement_index
            kwargs_for_filename['measurement_index'] = cycle_index

            if manual_continue:
                # wait for input before continuing
                inp = input("To start cycle {0} press ENTER:".format(cycle_index))
                if inp is 'exit':
                    break

            else:
                print('Starting cycle {} of {}.'.format(cycle_index, cycle_total))

            # measure
            time_data, ampl_data = self.measurement(**kwargs_for_measurement)

            # create filename
            file_name = file.create_filename(**kwargs_for_filename)

            # save data
            file.save_file(time_data, ampl_data, file_name=file_name)

        self.mfli.unsubscribe('*')

    def measurement_cycle_live(self,
                               cutoff_freq=0,
                               save_data=False,
                               **kwargs):
        """
        Function to display measurement directly. Data can be filter with a highpass filter and optionally stored on harddrive.


        :param cutoff_freq: cutoff frequency for the highpass-filter, negative values result in no filtering
        :param save_data: Bool, if data should be saved
        :param kwargs: keyword arguments for measurement function and create_filename function
        """

        # get keyword-arguments (kwargs) of measurement function out of kwargs
        kwargs_for_measurement = {}
        for keys in inspect.getfullargspec(self.measurement)[0]:
            if keys in kwargs.keys():
                kwargs_for_measurement[keys] = kwargs[keys]
        kwargs_for_measurement['pre_subscribed'] = True

        # get keyword-arguments (kwargs) of create_filename function out of kwargs
        kwargs_for_filename = {}
        for keys in inspect.getfullargspec(file.create_filename)[0]:
            if keys in kwargs.keys():
                kwargs_for_filename[keys] = kwargs[keys]
        kwargs_for_filename['raw_data'] = True

        self.mfli.subscribe()

        plt.figure()

        cycle_index = 0
        # repeat the measurement until exit is input
        while True:
            # set measurement_index
            kwargs_for_filename['measurement_index'] = cycle_index

            # wait for input before continuing
            inp = input("To start Cycle {0} press ENTER:".format(cycle_index))
            if inp == "exit":
                break
            elif inp[:4] == "freq":
                cutoff_freq = inp[4:]

            cycle_index += 1

            # measure
            time_data, ampl_data = self.measurement(**kwargs_for_measurement)

            if save_data:
                # create filename
                file_name = file.create_filename(**kwargs_for_filename)

                # save data
                file.save_file(time_data, ampl_data, file_name=file_name)

            if cutoff_freq > 0:
                sampling_freq = al.get_data_sampling_rate(time_data)

                # apply highpass-filter
                ampl_data = al.data_filter(ampl_data, cutoff_freq, sampling_freq)

            plt.clf()
            plt.plot(time_data, ampl_data)
            plt.pause(0.01)

        self.mfli.unsubscribe('*')

    def demodulator_freq_sweep(self, freq_start, freq_stop, freq_step,
                               reversed_direction=False, time_to_settle=30, measurements_per_step=1,
                               save_data=False, do_fft=False, do_plot=False, **kwargs):

        global fft_freq, fft_ampl

        # get keyword-arguments (kwargs) of measurement function out of kwargs
        kwargs_for_measurement = {}
        for keys in inspect.getfullargspec(self.measurement)[0]:
            if keys in kwargs.keys():
                kwargs_for_measurement[keys] = kwargs[keys]
        kwargs_for_measurement["noise"] = True
        kwargs_for_measurement["pre_subscribed"] = True


        # get keyword-arguments (kwargs) of create_filename function out of kwargs
        kwargs_for_filename = {}
        for keys in inspect.getfullargspec(file.create_filename)[0]:
            if keys in kwargs.keys():
                kwargs_for_filename[keys] = kwargs[keys]
        kwargs_for_filename['measurement_index'] = -1
        kwargs_for_filename['raw_data'] = True
        kwargs_for_filename["noise"] = True

        self.signal_gen.activate_output()
        self.mfli.subscribe()

        frequency = []
        noise_std = []
        noise_mean = []
        signal_to_noise = []

        freq_stop += 1
        demod_freq_range = np.arange(freq_start, freq_stop, freq_step)
        if reversed_direction:
            demod_freq_range = demod_freq_range[::-1]

        for demod_freq in demod_freq_range:

            # set demodulator frequency on the signal generator
            self.signal_gen.set_frequency(demod_freq)
            sleep(time_to_settle)

            temp_std = []
            temp_mean = []
            for measurement_index in range(measurements_per_step):
                # measure
                time_data, ampl_data = self.measurement(**kwargs_for_measurement)
                if do_fft:
                    fft_freq, fft_ampl = al.fft(time_data, ampl_data)
                if save_data:
                    # update kwargs_for_filename
                    kwargs_for_filename["demodulator_freq"] = demod_freq
                    if measurements_per_step > 1:
                        kwargs_for_filename['measurement_index'] = measurement_index
                    # create filename
                    kwargs_for_filename["is_fft_data"] = False
                    file_name = file.create_filename(**kwargs_for_filename)
                    # save data
                    file.save_file(time_data, ampl_data, file_name=file_name)
                    if do_fft:
                        kwargs_for_filename["is_fft_data"] = True
                        file_name = file.create_filename(**kwargs_for_filename)
                        file.save_file(fft_freq, fft_ampl, file_name=file_name)

                # calculate std and mean of noise
                ampl_data_std = np.std(ampl_data)
                ampl_data_mean = np.mean(ampl_data)
                temp_std.append(ampl_data_std)
                temp_mean.append(ampl_data_mean)

            # mean and std means
            ampl_data_std = np.mean(temp_std)
            ampl_data_mean = np.mean(temp_mean)
            # append lists
            frequency.append(demod_freq)
            noise_std.append(ampl_data_std)
            noise_mean.append(ampl_data_mean)
            signal_to_noise.append(ampl_data_mean / ampl_data_std)

        kwargs_for_filename['description'] = "SNRData"
        kwargs_for_filename['demodulator_freq'] = 0
        stn_file_name = file.create_filename(**kwargs_for_filename)
        file.save_file(frequency, noise_std, noise_mean, signal_to_noise, file_name=stn_file_name)

        self.mfli.unsubscribe('*')

    def magnet_position_test(self, pos1=0.5, pos2=0.3, window_length=30):

        amplitudes = []
        self.mfli.subscribe()

        _, ax = plt.subplots()

        while True:
            self.servo.move_servo(pos1)
            sleep(0.5)
            _, pl_data1 = self.measurement(1, noise=True, pre_subscribed=True)

            self.servo.move_servo(pos2)
            sleep(0.5)
            _, pl_data2 = self.measurement(1, noise=True, pre_subscribed=True)

            amplitude = abs(np.mean(pl_data1) - np.mean(pl_data2))

            amplitudes.append(amplitude)

            if len(amplitudes) > window_length:
                amplitudes = amplitudes[-window_length:]

            ax.clear()
            ax.plot(amplitudes)
            plt.pause(0.2)

    def magnet_pos_test_fft(self, window_length=30, **kwargs_for_measurement):
        fig, [ax1, ax2] = plt.subplots(2)
        ax1.set_xlim(0, 250)

        peaks = []
        while True:
            # self.servo.stop_and_pluck()
            # time_data, ampl_data = self.measurement(5,True,False)
            # self.servo.stop_and_pluck()

            time_data, ampl_data = self.measurement(**kwargs_for_measurement)
            sampling_freq = al.get_data_sampling_rate(time_data)
            ampl_data = al.data_filter(ampl_data, 20, sampling_freq)

            freq_data, fft_data = al.fft(time_data, ampl_data, sampling_freq)

            freq_data_peaks, fft_data_peaks = al.fft_peaks(freq_data, fft_data, 1000)

            peaks.append(fft_data_peaks[0])
            if len(peaks) > window_length:
                peaks = peaks[-window_length:]

            ax1.clear()
            ax1.plot(freq_data[:600], fft_data[:600])
            ax1.plot(freq_data_peaks, fft_data_peaks, 'rx')

            ax2.clear()
            ax2.plot(peaks)

            plt.pause(2)
