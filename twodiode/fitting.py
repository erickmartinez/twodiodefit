import twodiode.lugomunoz as lgm
import numpy as np
import pymc3 as pm
import theano


class FitTwoDiode:
    __burn: int = 3000
    __tune: int = 1000
    __target_accept: float = 0.95
    __result: dict = None
    __current: np.ndarray = None
    __voltage: np.ndarray = None
    __vshared: theano.compile.sharedvalue = None
    __ishared: theano.compile.sharedvalue = None
    __fixed_parameters: dict = {}
    __available_parameters: list = ['i01', 'i02', 'rs1', 'rs2', 'rsh', 'n1', 'n2']
    __temperature_k: float = 298.15
    __area: float = 1  # cm^2

    def __init__(self, voltage: np.ndarray, current: np.ndarray, temperature_c: float = 25, area: float = 1):
        """
        Parameters
        ----------
        voltage: np.ndarray
            The experimental voltage (V)
        current: np.ndarray
            The experimental current (A)
        temperature_c: float
            The experimental temperature (°C). Default 25 °C
        area: float
            The area of the device (cm^2)
        """
        if len(voltage) != len(current):
            error_message = 'The number voltage points ({0}) must equal the number of current points ({0})'.format(
                len(voltage), len(current)
            )
            raise ValueError(error_message)
        # If there's data at negative votlages, remove it
        idx = voltage > 0
        self.__voltage = voltage[idx]
        self.__current = current[idx]
        self.__temperature_k = temperature_c + 273.15

        self.__vshared = theano.shared(voltage)
        self.__ihsared = theano.shared(current)
        self.area = area
        self.__model = pm.Model()

    @property
    def current(self) -> np.ndarray:
        return self.__current

    @property
    def voltage(self) -> np.ndarray:
        return self.__voltage

    @property
    def temperature_c(self) -> float:
        return self.__temperature_k - 273.15

    @property
    def temperature_k(self) -> float:
        return self.__temperature_k

    @temperature_c.setter
    def temperature_c(self, value: float):
        self.__temperature_k = value + 273.15

    @temperature_k.setter
    def temperature_k(self, value):
        self.__temperature_k = abs(value)

    @property
    def fixed_parameters(self) -> dict:
        return self.__fixed_parameters

    @property
    def area(self) -> float:
        return self.__area

    @area.setter
    def area(self, value: float):
        if value == 0:
            raise ValueError('Trying to set the area to 0!')
        self.__area = abs(value)

    @property
    def burn(self) -> int:
        return self.__burn

    @burn.setter
    def burn(self, value: int):
        if value < 100:
            raise ValueError('The number of burns {0} is too low for decent accuracy.'.format(value))
        self.__burn = abs(int(value))

    @property
    def tune(self) -> int:
        return self.__tune

    @tune.setter
    def tune(self, value: int):
        if value < 100:
            raise ValueError('The number of tunes {0} is too low for decent accuracy.'.format(value))
        self.__tune = abs(int(value))

    @property
    def target_accept(self) -> float:
        return self.__target_accept

    @target_accept.setter
    def target_accept(self, value: float):
        if value < 0.5:
            raise ValueError('Trying to set target accept to a very low value ({0}).'.format(value))
        self.__target_accept = value

    def fix_i01(self, value: float):
        value = abs(value)
        self.__fixed_parameters['i01'] = value

    def fix_i02(self, value: float):
        value = abs(value)
        self.__fixed_parameters['i02'] = value

    def fix_rs1(self, value: float):
        value = abs(value)
        self.__fixed_parameters['rs1'] = value

    def fix_rs2(self, value: float):
        value = abs(value)
        self.__fixed_parameters['rs2'] = value

    def fix_rsh(self, value: float):
        value = abs(value)
        self.__fixed_parameters['rsh'] = value

    def fix_n1(self, value: float):
        value = abs(value)
        self.__fixed_parameters['n1'] = value

    def fix_n2(self, value: float):
        value = abs(value)
        self.__fixed_parameters['n2'] = value

    def fit_dark(self) -> dict:
        # Define the priors
        priors = {}
        with self.__model:
            for p in self.__available_parameters:
                if p not in self.__fixed_parameters:
                    if p == 'i01' or p == 'i02':
                        priors[p] = pm.Lognormal(p, mu=1E-10, sigma=30)
                    if p == 'rs1' or p == 'rs2':
                        priors[p] = pm.Lognormal(p, mu=1E-3, sigma=30)
                    if p == 'rsh':
                        priors[p] = pm.Lognormal(p, mu=1E8, sigma=30)
                    if p == 'n1':
                        priors[p] = pm.Uniform(p, lower=0.5, upper=1.9)
                    if p == 'n2':
                        priors[p] = pm.Uniform(p, lower=2.0, upper=100)
                else:
                    priors[p] = self.__fixed_parameters[p]
            # Expected value of outcome
            gauss = pm.Deterministic('gauss', lgm.dark_current(
                self.__voltage,
                i01=priors["i01"], i02=priors['i02'], rs1=priors['rs1'], rs2=priors['rs2'],
                rsh=priors['rsh'], n1=priors['n1'], n2=priors['n2'], temperature=self.__temperature_k
            ))

            sigma = pm.HalfNormal('sigma', sigma=10)
            # Likelihood (sampling distribution) of observations
            y_obs = pm.Normal('y_obs', mu=gauss, sigma=sigma, observed=self.__ishared)

            trace = pm.sample(self.__burn, chains=2, cores=1, tune=self.__tune, target_accept=self.__target_accept)

        y_min = np.percentile(trace.gauss, 2.5, axis=0)
        y_max = np.percentile(trace.gauss, 97.5, axis=0)
        y_fit = np.percentile(trace.gauss, 50, axis=0)
