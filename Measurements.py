import logic


"Abklingverhalten"
meas = logic.Logic(True, True, False)
meas.measurement_cycle(cycle_total=50, poll_time=120, noise=False,
                       title='AbklingBSaite', version_index=1)

"zDip"
meas = logic.Logic(True, True, False)
meas.measurement_cycle(cycle_total=200, poll_time=120, noise=False,
                       title='AbklingBSaite', version_index=1)


"magnet-test"
# meas = logic.Logic(True, True, False)
# # meas.magnet_position_test(0.35, 0.5)
# meas.magnet_pos_test_fft(poll_time=4, stop_time=1)


"freq_sweep"
# meas = logic.Logic(False, True, True)
# meas.signal_gen.activate_output()
#
# start_freq = 50e3
#
# meas.signal_gen.set_frequency(start_freq)
# input("start?")
# sleep(60)
#
# meas.demodulator_freq_sweep(start_freq,600e3,1e3,time_to_settle=0.5, save_data=False, reversed_direction=False,
#                             measurements_per_step=10,
#                             title="demodSlowSweep", version_index=1,
#                             poll_time=0.5)
