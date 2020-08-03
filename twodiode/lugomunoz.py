import numpy as np
from scipy.special import lambertw
from scipy import constants


def dark_current(voltage: float, i01: float, i02: float, rs1: float, rs2: float, rsh: float, n1: float = 1.,
                 n2: float = 2.,
                 temperature: float = 289.15):
    """
    Models the dark current using a two diode model based on

    D. Lugo-Munoz, J. Muci, A. Ortiz-Conde, F. J. Garcia-Sanchez, M. de Souza, and M. A. Pavanello,
    Microelectron. Reliab. 51 (12), 2044 (2011).

    Eq. (5)

    (1/Rs) approx sum_k (1/Rsk)

    Parameters
    ----------
    voltage: float
        The voltage at which to estimate the dark current measurement (V)
    i01: float
        The recombination current for the first diode (A)
    i02: float
        The recombination current for the first diode (A)
    rs1: float
        The series resistance for the first diode (Ohm).
    rs2: float
        The series resistance for the second diode (Ohm).
    rsh: float
        The shunt resistance (Ohm)
    n1: float
        The ideality factor for the first diode (default 1)
    n2: float
        The ideality factor for the second diode
    temperature

    Returns
    -------
    float
        The current at the specified voltage (A)
    """
    kb = constants.value('Boltzmann constant in eV/K')

    # The thermal voltage
    vth = kb * temperature  # Volts

    # The parallel conductance
    gp = 1 / rsh

    # The products n1 x vth, n2 x vth
    n1_vth = n1 * vth
    n2_vth = n2 * vth

    # The products i01 x rs1, i02 x rs2
    rs1_i01 = rs1 * i01
    rs2_i02 = rs2 * i02

    # Define the arguments for Lambert W0
    w0_argument_1 = (rs1_i01 / n1_vth) * np.exp((voltage + rs1_i01) / n1_vth)
    w0_argument_2 = (rs2_i02 / n2_vth) * np.exp((voltage + rs2_i02) / n2_vth)

    current = (n1_vth / rs1) * lambertw(z=w0_argument_1, k=0)
    current += (n2_vth / rs2) * lambertw(z=w0_argument_2, k=0)
    current -= (i01 + i02)
    current += (gp * voltage)

    return current


def illuminated_current(voltage: float, i_l: float, i01: float, i02: float, rs1: float, rs2: float, rsh: float,
                        n1: float = 1., n2: float = 2., temperature: float = 289.15):
    """
    Models the illuminatio  current using a two diode model based on

    D. Lugo-Munoz, J. Muci, A. Ortiz-Conde, F. J. Garcia-Sanchez, M. de Souza, and M. A. Pavanello,
    Microelectron. Reliab. 51 (12), 2044 (2011).

    Eq. (5)

    (1/Rs) approx sum_k (1/Rsk)

    Parameters
    ----------
    voltage: float
        The voltage at which to estimate the dark current measurement (V)
    i_l: float
        The illuminated current at V=0 (A)
    i01: float
        The recombination current for the first diode (A)
    i02: float
        The recombination current for the first diode (A)
    rs1: float
        The series resistance for the first diode (Ohm).
    rs2: float
        The series resistance for the second diode (Ohm).
    rsh: float
        The shunt resistance (Ohm)
    n1: float
        The ideality factor for the first diode (default 1)
    n2: float
        The ideality factor for the second diode
    temperature

    Returns
    -------
    float
        The current at the specified voltage (A)
    """
    return i_l - dark_current(voltage=voltage, i01=i01, i02=i02, rs1=rs1, rs2=rs2, rsh=rsh,
                              n1=n1, n2=n2, temperature=temperature)
