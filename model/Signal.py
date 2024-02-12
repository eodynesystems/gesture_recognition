import numpy as np 
import bottleneck as bn
import matplotlib.pyplot as plt
import ruptures as rpt

class Signal():
    def __init__(self, path_to_file=None, signal=None, step=10, window_size = 50):
        """
        Args:
        path_to_file (str): path to file containing numpy array for the signal
        signal (np.array): numpy array containing the signal
        Both arguments are optional. 
        This class can be instantiated to just use the functions.
        """
        if path_to_file:
            self.signal = np.load(path_to_file)
            self.n_samples = self.signal.shape[0]
            self.step = step
            self.window_size = window_size
            if self.n_samples < 50:
                raise Exception ("signal too short! minimum size = 50 samples")
    
        self.features = {"mav":self.mav, "rms":self.rms, "ssc":self.ssc, "wl":self.wl, "var":self.var, 
                         "iasd":self.iasd, "iatd":self.iatd}
        self.n_features = 8*len(self.features)
        
    def get_features(self, list_features = "all", remove_transition=False):
        if remove_transition:
            self.remove_transition()
        features = np.empty((0, self.n_features))
        for idx in range(0, self.n_samples-self.window_size, self.step):
            x = self.signal[idx:idx+self.window_size, :]
            features = np.concatenate((features, self.get_features_window(x, list_features)))
        return features
    
    def get_features_window(self, x, list_features="all"):
        features = np.empty((1, 0))
        for f in self.features.values():
            features = np.concatenate((features, f(x).reshape(1, 8)), axis=1)
        return features

    def remove_transition(self, w=1):
        # Extract features
        self.signal = self.sliding_avg(self.signal, w)
        s = self.get_features(list_features=["mav"]).sum(axis=1)
    
        # Change point detection with Pelt
        detect = rpt.Pelt(model="l2", min_size=40).fit(s)
        result = detect.predict(pen=10)
    
        # Find first big change in direction
        cutting_index = result[0] if result else 0
    
        # Remove phase of transition
        self.signal = self.signal[cutting_index:, :]
        self.n_samples = self.signal.shape[0]
    
        # Return altered signal
        return self
    
    # ops 
    def mav(self, x):                                      # mean absolute value
        return sum(abs(x)) / x.shape[0]

    def rms(self, x):                                      # root mean square
        return ((sum(x ** 2)) / x.shape[0]) ** (1 / 2)

    def wl (self, x):                                      # waveform length
        return sum(abs(x[:-1] - x[1:]))

    def ssc(self, x, delta = 1):                           # slope of sign change
        f = lambda x: (x >= delta).astype(float)
        return sum(f(-(x[1:-1, :] - x[:-2, :])*(x[1:-1] - x[2:])))
    
    def var(self, x):                                      # variance
        return sum((x ** 2)) / (x.shape[0] - 1)
    
    def derivative(self, x):
        return x[1:] - x[:-1]
    
    def iasd(self, x):                                     # integrated absolute of second derivative
        return (sum(abs(self.derivative(self.derivative(x)))))
    
    def iatd(self, x):                                     # integrated absolute of third derivative
        return (sum(abs(self.derivative(self.derivative(self.derivative(x))))))
    
    def sliding_avg(self, x, w=1):
        if x.ndim > 1:
            return np.apply_along_axis(lambda y: np.convolve(y, np.ones(w), 'valid') / w, axis=0, arr=x)
        else:
            return np.convolve(x, np.ones(w), 'valid') / w
    
    def sliding_var(self, x, w=1):
        return bn.move_var(x, window=w)
    
    #vis 
    def display(self, attr = "energy", w = 5):
        plt.figure()
        """
        w (window size) only applicable for functions of energy
        """
        if attr == "energy":
            energy = self.get_features(list_features=["mav"])
            energy = energy.sum(axis = 1)
            plt.plot(energy)
            plt.xlabel("time (samples)")
            plt.ylabel("Energy")
        if attr == "avg_slope_energy": 
            energy = self.get_features(list_features=["mav"])
            energy = energy.sum(axis = 1)
            slope = energy[1:] - energy[:-1]
            avg_slope = self.sliding_avg(slope, w)
            plt.plot(avg_slope)
            plt.xlabel("time (samples)")
            plt.ylabel(f"Avg. slope (window = {w})")
        if attr == "sliding_var_energy":
            energy = self.get_features(list_features=["mav"])
            energy = energy.sum(axis = 1)
            mov_var = bn.move_var(energy, window=w)
            mov_avg = bn.move_mean(energy, window=w)
            plt.plot(mov_var/mov_avg)
            plt.xlabel("time (samples)")
            plt.ylabel(f"Moving Variance of Energy (window = {w})")
        ylim_bottom = min(0, plt.gca().get_ylim()[0])
        plt.gca().set_ylim(bottom=ylim_bottom)


