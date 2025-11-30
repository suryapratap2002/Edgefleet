import numpy as np

class KalmanBallTracker:
    """
    2D Kalman filter tracker for small object centroid tracking.
    State vector: [x, y, vx, vy]^T
    """
    def __init__(self, dt=1.0, process_var=1e-2, meas_var=20.0):
        self.dt = dt
        self.state = np.zeros((4,1))
        self.P = np.eye(4) * 500.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.eye(2) * meas_var
        self.Q = np.eye(4) * process_var
        self.initialized = False

    def initialize(self, x, y):
        self.state[0,0] = x
        self.state[1,0] = y
        self.initialized = True

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.state[0,0]), float(self.state[1,0])

    def update(self, x, y):
        if not self.initialized:
            self.initialize(x, y)
            return
        Z = np.array([[x],[y]])
        y_res = Z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def get_state(self):
        return float(self.state[0,0]), float(self.state[1,0]), float(self.state[2,0]), float(self.state[3,0])
