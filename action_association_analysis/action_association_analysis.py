import numpy as np
class MotionCorrelationEnhanced:
    
    @staticmethod
    def pearson_correlation(a, b):
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        numerator = np.sum((a - mean_a) * (b - mean_b))
        denominator = np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2))
        return numerator / denominator if denominator != 0 else 0

    class PursuitsAlgorithm:
        
        def __init__(self, tau, N):
            self.tau = tau
            self.N = N

        def run(self, P_in, P_out_list):
            T = P_in.shape[0]
            I = len(P_out_list)
            Selected = [None] * T

            for t in range(T):
                if t < self.N:
                    Selected[t] = None
                else:
                    max_correlation = -np.inf
                    selected_target = None
                    for i in range(I):
                        P_out = P_out_list[i]
                        c_x = MotionCorrelationEnhanced.pearson_correlation(P_in[t-self.N:t, 0], P_out[t-self.N:t, 0])
                        c_y = MotionCorrelationEnhanced.pearson_correlation(P_in[t-self.N:t, 1], P_out[t-self.N:t, 1])
                        correlation = min(c_x, c_y)

                        if correlation > self.tau and correlation > max_correlation:
                            max_correlation = correlation
                            selected_target = i

                    Selected[t] = selected_target

            return Selected

    class PathSyncAlgorithm:
        
        def __init__(self, tau_lo, tau_hi, N, h):
            self.tau_lo = tau_lo
            self.tau_hi = tau_hi
            self.N = N
            self.h = h

        def run(self, P_in, P_out_list):
            T = P_in.shape[0]
            I = len(P_out_list)
            States = np.full((T, I), 'Deactivated', dtype=object)

            for t in range(T):
                if t < self.N:
                    continue

                for i in range(I):
                    P_out = P_out_list[i]
                    v1, v2 = np.linalg.eig(np.cov(P_out[t-self.N:t].T))
                    rotation_matrix = np.vstack((v1, v2)).T

                    P_in_rotated = P_in[t-self.N:t] @ rotation_matrix
                    P_out_rotated = P_out[t-self.N:t] @ rotation_matrix

                    c_x = MotionCorrelationEnhanced.pearson_correlation(P_in_rotated[:, 0], P_out_rotated[:, 0])
                    c_y = MotionCorrelationEnhanced.pearson_correlation(P_in_rotated[:, 1], P_out_rotated[:, 1])
                    correlation = min(c_x, c_y)

                    if correlation < self.tau_lo:
                        States[t, i] = 'Deactivated'
                    elif correlation > self.tau_hi:
                        States[t, i] = 'Activated'
                    elif States[t-1, i] == 'Activated':
                        States[t, i] = 'Activated'
                    else:
                        States[t, i] = 'Deactivated'

            return States

if __name__ == "__main__":
    # show data
    P_in = np.random.rand(100, 2)  # T x 2 matrix
    P_out_list = [np.random.rand(100, 2) for _ in range(5)]  #  many T x 2 matrix

    # Pursuits Algorithm
    pursuits = MotionCorrelationEnhanced.PursuitsAlgorithm(tau=0.5, N=10)
    selected_targets = pursuits.run(P_in, P_out_list)
    print("Selected Targets (Pursuits Algorithm):", selected_targets)

    # PathSync Algorithm
    pathsync = MotionCorrelationEnhanced.PathSyncAlgorithm(tau_lo=0.4, tau_hi=0.7, N=10, h=5)
    states = pathsync.run(P_in, P_out_list)
    print("States (PathSync Algorithm):", states)
