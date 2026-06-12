import numpy as np

class Calibrator:
    @staticmethod
    def sigmoid_calibrate(score: float, center: float = 0.5, temperature: float = 0.1) -> float:
        """
        Calibrates a raw score to a [0, 1] probability using a sigmoid function.
        Lower temperature values create a sharper transition.
        """
        # Sigmoid calibration: 1 / (1 + exp(-(score - center) / temp))
        try:
            val = 1.0 / (1.0 + np.exp(-(score - center) / temperature))
            return float(val)
        except Exception:
            return float(score)

    @staticmethod
    def PlattScaling(raw_predictions: np.ndarray, labels: np.ndarray):
        """
        Computes Platt scaling parameters A and B for a set of raw predictions:
        P(y=1|x) = 1 / (1 + exp(A * x + B))
        """
        from sklearn.linear_model import LogisticRegression
        # Fit a simple logistic regression on the raw predictions
        lr = LogisticRegression(penalty=None if hasattr(LogisticRegression, "penalty") else "none")
        X = raw_predictions.reshape(-1, 1)
        lr.fit(X, labels)
        
        # A corresponds to the coefficient, B to the intercept
        A = float(lr.coef_[0][0])
        B = float(lr.intercept_[0])
        return A, B
