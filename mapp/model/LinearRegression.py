from sklearn.linear_model import LinearRegression

from ..utilities.gregression import Regressor

class linearregression:
    """
    The class that performs the linear regression algorithm with Scikit-learn.
    
    Parameters
    ----------
    adescriptors: list of arrays
        The fingerprints.
    features: list of floats
        The predicted values.
    weights: list of floats
        The weights for evaluating the cost functions.
    """
    def __init__(self, train_adescriptors, test_adescriptors, train_features, test_features, train_weights=None, test_weights=None, stress=False):
        self.X_train = train_adescriptors
        self.X_test = test_adescriptors
        self.y_train = train_features[0]
        self.y_test = test_features[0]
        self.type_train = train_features[1]
        self.type_test = train_features[1]
        self.w_train = train_weights
        self.w_test = test_weights
        self.stress = stress

        if w_train == None or w_test == None:
            raise ValueError("You must input the value for the weights")


    def fit(self, train_adescriptors, test_adescriptors, train_features, test_features, train_weights, test_weights):
        """
        Run the linear regression.
        """
        #self.reg = LinearRegression().fit(self.X_train, self.y_train, self.w_train)
            
        #train_result = self.get_mae_rsquare(self.X_train, self.y_train, self.w_train, self.type_train)
        #test_result = self.get_mae_rsquare(self.X_test, self.y_test, self.w_test, self.type_test)

        #return train_result, test_result

        self.regressor = gregressor()


    def get_coefficients(self):
        """
        Get the linearly fitted coefficients.
        """
        coeff = {}

        coeff['intercept'] = [self.reg.intercept_]
        coeff['slope'] = reg.coef_

        return coeff


    def get_mae_rsquare(self, X, y, w, styles):
        """
        Calculate the mae and rsquare of energies, forces, and stress (if applicable).
        """
        X1, X2, X3 = [], [], []
        y1, y2, y3 = [], [], []
        w1, w2, w3 = [], [], []

        for i, style in styles:
            if style == 'energy':
                X1.append(X[i])
                y1.append(y[i])
                w1.append(w[i])
            elif style == 'force':
                X2.append(X[i])
                y2.append(y[i])
                w2.append(w[i])
            else:
                X3.append(X[i])
                y3.append(y[i])
                w3.append(w[i])

        # Evaluate the mae and r square of energy
        y1_ = self.reg.predict(X1)
        mae1 = mean_absolute_error(y1, y1_)
        rsquare1 = reg.score(X1, y1, w1)

        # Evaluate the mae and r square of force
        y2_ = self.reg.predict(X2)
        mae2 = mean_absolute_error(y2, y2_)
        rsquare2 = reg.score(X2, y2, w2)

        # Evaluate the mae and r square of stress
        if self.stress == True:
            y3_ = self.reg.predict(X3)
            mae3 = mean_absolute_error(y3, y3_)
            rsquare3 = reg.score(X3, y3, w3)

            result = {'energy_r2': [rsquare1],
                      'energy_mae': [mae1],
                      'force_r2': [rsquare2],
                      'force_mae': [mae2],
                      'stress_r2': [rsquare3],
                      'stress_mae': [mae3]}
        
            return result

        else:
            result = {'energy_r2': [rsquare1],
                      'energy_mae': [mae1],
                      'force_r2': [rsquare2],
                      'force_mae': [mae2]}

            return result
