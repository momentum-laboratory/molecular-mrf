import math
from types import SimpleNamespace
from typing import Tuple, Union

import numpy as np

from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.opts import Opts

# simplified version of pypulseq.make_gauss_pulse 
def make_gauss_pulse(b1: float, apodization: float = 0, bandwidth: float = 0, center_pos: float = 0.5,
                     delay: float = 0, duration: float = 0, freq_offset: float = 0, max_grad: float = 0,
                     max_slew: float = 0, phase_offset: float = 0, return_gz: bool = False, slice_thickness: float = 0,
                     system: Opts = Opts(), time_bw_product: float = 4,
                     use: str = str()) -> Union[SimpleNamespace,
                                                Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]]:
    """
    Creates a radio-frequency gauss pulse event and optionally accompanying slice select and slice select rephasing
    trapezoidal gradient events.

    Parameters
    ----------
    b1 : float
        b1.
    apodization : float, optional, default=0
        Apodization.
    bandwidth : float, optional, default=0
        Bandwidth in Hertz (Hz).
    center_pos : float, optional, default=0.5
        Position of peak.
    delay : float, optional
        Delay in milliseconds (ms).
    duration : float, optional, default=0
        Duration in milliseconds (ms).
    freq_offset : float, optional, default=0
        Frequency offset in Hertz (Hz).
    max_grad : float, optional, default=0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, optional, default=0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, optional, default=0
        Phase offset in Hertz (Hz).
    return_gz : bool, default=False
        Boolean flag indicating if slice-selective gradient has to be returned.
    slice_thickness : float, optional, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, optional, default=Opts()
        System limits.
    time_bw_product : int, optional, default=4
        Time-bandwidth product.
    use : str, optional, default=str()
        Use of radio-frequency gauss pulse event. Must be one of 'excitation', 'refocusing' or 'inversion'.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency gauss pulse event.
    gz : SimpleNamespace
        Accompanying slice select trapezoidal gradient event.
    gzr : SimpleNamespace
        Accompanying slice select rephasing trapezoidal gradient event.

    Raises
    ------
    ValueError
        If invalid `use` is passed. Must be one of 'excitation', 'refocusing' or 'inversion'.
        If `return_gz=True` and `slice_thickness` was not passed.
    """
    valid_use_pulses = ['excitation', 'refocusing', 'inversion']
    if use != '' and use not in valid_use_pulses:
        raise ValueError(
            f"Invalid use parameter. Must be one of 'excitation', 'refocusing' or 'inversion'. Passed: {use}")

    if bandwidth == 0:
        BW = time_bw_product / duration
    else:
        BW = bandwidth
    alpha = apodization
    N = int(round(duration / 1e-6))
    t = np.arange(1, N + 1) * system.rf_raster_time
    tt = t - (duration * center_pos)
    window = 1 - alpha + alpha * np.cos(2 * np.pi * tt / duration)
    signal = np.multiply(window, __gauss(BW * tt))
    # flip = np.sum(signal) * system.rf_raster_time * 2 * np.pi
    # signal = signal * flip_angle / flip
    signal = signal * b1

    rf = SimpleNamespace()
    rf.type = 'rf'
    rf.signal = signal
    rf.t = t
    rf.freq_offset = freq_offset
    rf.phase_offset = phase_offset
    rf.dead_time = system.rf_dead_time
    rf.ringdown_time = system.rf_ringdown_time
    rf.delay = delay
    if use != '':
        rf.use = use

    if rf.dead_time > rf.delay:
        rf.delay = rf.dead_time

    if return_gz:
        if slice_thickness == 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system.max_grad = max_grad

        if max_slew > 0:
            system.max_slew = max_slew

        amplitude = BW / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
        gzr = make_trapezoid(channel='z', system=system, area=-area * (1 - center_pos) - 0.5 * (gz.area - area))

        if rf.delay > gz.rise_time:
            gz.delay = math.ceil((rf.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rf.delay < (gz.rise_time + gz.delay):
            rf.delay = gz.rise_time + gz.delay

    if rf.ringdown_time > 0:
        t_fill = np.arange(1, round(rf.ringdown_time / 1e-6) + 1) * 1e-6
        rf.t = np.concatenate((rf.t, rf.t[-1] + t_fill))
        rf.signal = np.concatenate((rf.signal, np.zeros(len(t_fill))))

    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rf.signal == -0.0)
    rf.signal[negative_zero_indices] = 0

    if return_gz:
        return rf, gz, gzr
    else:
        return rf


def __gauss(x):
    return np.exp(-np.square(x))
