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
            self.remove_transition(w=40) 
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

    def remove_transition(self, algorithm="Binseg", w=50, min_s=30):
        
        if algorithm == "Binseg":

            # detection using Binseg
            s = self.get_features(list_features=["mav"]).sum(axis=1)
            algo = rpt.Binseg(model="l2").fit(s)
            result = algo.predict(n_bkps=1)[0]
            if result > len(s) // 2:
                result = 20
            trans_idx = result * self.step
            if self.signal.shape[0] > 2 * trans_idx:
                self.signal = self.signal[trans_idx:, :]
                self.n_samples = self.signal.shape[0]

        if algorithm == "Pelt":

            # store original signal
            original_signal = self.signal.copy()

            # extract features using sliding average
            self.signal = self.sliding_avg(self.signal, w)
            s = self.get_features(list_features=["mav"]).sum(axis=1)

            # initialize change point detection with Pelt
            result = None

            # attempt valid removal, otherwise make window shorter to make the algorithm less aggressive
            while min_s > 5 and not result:
                try:
                    detect = rpt.Pelt(model="l2", min_size=min_s).fit(s)
                    result = detect.predict(pen=20)
                    if result:  # valid removal found
                        break
                except Exception:  # if window size too large 
                    min_s -= 1  # decrease min_size 

            # find first big change in direction
            if result:
                cutting_index = result[0]
            else:
                # cut the signal at 10% of its length if no cutting point is found with minimal window
                cutting_index = int(len(original_signal) * 0.1)

            # transform this index into the original space
            norm_cutting_index = cutting_index + (w - 1) if result else cutting_index

            # remove phase of transition
            self.signal = original_signal[norm_cutting_index:, :]
            self.n_samples = self.signal.shape[0]
    
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


