from __future__ import absolute_import, division

from copy import deepcopy
from math import log, exp, sqrt
import sys
import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z
import tensorflow as tf



class KalmanFilter(object):
    r""" Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults  will
    not give you a functional filter.
    For now the best documentation is my free book Kalman and Bayesian
    Filters in Python [2]_. The test files in this directory also give you a
    basic idea of use, albeit without much description.
    In brief, you will first construct this object, specifying the size of
    the state vector with dim_x and the size of the measurement vector that
    you will be using with dim_z. These are mostly used to perform size checks
    when you assign values to the various matrices. For example, if you
    specified dim_z=2 and then try to assign a 3x3 matrix to R (the
    measurement noise matrix you will get an assert exception because R
    should be 2x2. (If for whatever reason you need to alter the size of
    things midstream just use the underscore version of the matrices to
    assign directly: your_filter._R = a_3x3_matrix.)
    After construction the filter will have default matrices created for you,
    but you must specify the values for each. It’s usually easiest to just
    overwrite them rather than assign to each element yourself. This will be
    clearer in the example below. All are of type numpy.array.
    Examples
    --------
    Here is a filter that tracks position and velocity using a sensor that only
    reads position.
    First construct the object with the required dimensionality. Here the state
    (`dim_x`) has 2 coefficients (position and velocity), and the measurement
    (`dim_z`) has one. In FilterPy `x` is the state, `z` is the measurement.
    .. code::
        from filterpy.kalman import KalmanFilter
        f = KalmanFilter (dim_x=2, dim_z=1)
    Assign the initial value for the state (position and velocity). You can do this
    with a two dimensional array like so:
        .. code::
            f.x = np.array([[2.],    # position
                            [0.]])   # velocity
    or just use a one dimensional array, which I prefer doing.
    .. code::
        f.x = np.array([2., 0.])
    Define the state transition matrix:
        .. code::
            f.F = np.array([[1.,1.],
                            [0.,1.]])
    Define the measurement function. Here we need to convert a position-velocity
    vector into just a position vector, so we use:
        .. code::
        f.H = np.array([[1., 0.]])
    Define the state's covariance matrix P. 
    .. code::
        f.P = np.array([[1000.,    0.],
                        [   0., 1000.] ])
    Now assign the measurement noise. Here the dimension is 1x1, so I can
    use a scalar
    .. code::
        f.R = 5
    I could have done this instead:
    .. code::
        f.R = np.array([[5.]])
    Note that this must be a 2 dimensional array.
    Finally, I will assign the process noise. Here I will take advantage of
    another FilterPy library function:
    .. code::
        from filterpy.common import Q_discrete_white_noise
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
    Now just perform the standard predict/update loop:
    .. code::
        while some_condition_is_true:
            z = get_sensor_reading()
            f.predict()
            f.update(z)
            do_something_with_estimate (f.x)
    **Procedural Form**
    This module also contains stand alone functions to perform Kalman filtering.
    Use these if you are not a fan of objects.
    **Example**
    .. code::
        while True:
            z, R = read_sensor()
            x, P = predict(x, P, F, Q)
            x, P = update(x, P, z, R, H)
    See my book Kalman and Bayesian Filters in Python [2]_.
    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.
    Parameters
    ----------
    dim_x : int
        Number of state variables for the Kalman filter. For example, if
        you are tracking the position and velocity of an object in two
        dimensions, dim_x would be 4.
        This is used to set the default size of P, Q, and u
    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.
    dim_u : int (optional)
        size of the control input, if it is being used.
        Default value of 0 indicates it is not used.
    compute_log_likelihood : bool (default = True)
        Computes log likelihood by default, but this can be a slow
        computation, so if you never use it you can turn this computation
        off.
    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.
    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.
    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convenience; they store the  prior and posterior of the
        current epoch. Read Only.
    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.
    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.
    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.
    z : numpy.array
        Last measurement used in update(). Read only.
    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.
    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. Also known as the transition
        covariance.
    F : numpy.array()
        State Transition matrix. Also known as `A` in some formulation.
    H : numpy.array(dim_z, dim_x)
        Measurement function. Also known as the observation matrix, or as `C`.
    y : numpy.array
        Residual of the update step. Read only.
    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.
    S :  numpy.array
        System uncertainty (P projected to measurement space). Read only.
    SI :  numpy.array
        Inverse system uncertainty. Read only.
    log_likelihood : float
        log-likelihood of the last measurement. Read only.
    likelihood : float
        likelihood of last measurement. Read only.
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
    mahalanobis : float
        mahalanobis distance of the innovation. Read only.
    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv
        This is only used to invert self.S. If you know it is diagonal, you
        might choose to set it to filterpy.common.inv_diagonal, which is
        several times faster than numpy.linalg.inv for diagonal matrices.
    alpha : float
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
    References
    ----------
    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)
    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        #self.x = zeros((dim_x, 1))        # state
        self.x = tf.Variable(tf.zeros((dim_x, 1)), dtype=tf.float32)
        #self.P = eye(dim_x)               # uncertainty covariance
        self.P = tf.Variable(tf.eye(dim_x), dtype=tf.float32)
        #self.Q = eye(dim_x)               # process uncertainty
        self.Q = tf.Variable(tf.eye(dim_x), dtype=tf.float32)
        self.B = None                     # control transition matrix
        #self.F = eye(dim_x)               # state transition matrix
        self.F = tf.eye(dim_x)
        #self.H = zeros((dim_z, dim_x))    # measurement function
        self.H = tf.zeros((dim_z, dim_x), dtype=tf.float32)
        #self.R = eye(dim_z)               # measurement uncertainty
        self.R = tf.Variable(tf.eye(dim_z), dtype=tf.float32)
        self._alpha_sq = 1.               # fading memory control
        #self.M = np.zeros((dim_x, dim_z)) # process-measurement cross correlation
        self.M = tf.zeros((dim_x, dim_z), dtype=tf.float32)
        #self.z = np.array([[None]*self.dim_z]).T
        self.z = tf.zeros(dim_z, dtype=tf.float32)

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = tf.zeros((dim_x, dim_z), dtype=tf.float32) # kalman gain
        self.y = tf.zeros((dim_z, 1), dtype=tf.float32)
        self.S = tf.zeros((dim_z, dim_z), dtype=tf.float32) # system uncertainty
        self.SI = tf.zeros((dim_z, dim_z), dtype=tf.float32) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = tf.eye(dim_x, dtype=tf.float32)

        # these will always be a copy of x,P after predict() is called
        #self.x_prior = self.x.copy()
        self.x_prior = tf.identity(self.x)
        #self.P_prior = self.P.copy()
        self.P_prior = tf.identity(self.P)

        # these will always be a copy of x,P after update() is called
        #self.x_post = self.x.copy()
        self.x_post = tf.identity(self.x)
        #self.P_post = self.P.copy()
        self.P_post = tf.identity(self.P)

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = tf.linalg.inv


    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.
        Parameters
        ----------
        u : np.array, default 0
            Optional control vector.
        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.
        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = tf.cast(self.F, tf.float32)
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = tf.eye(self.dim_x, dtype=tf.float32) * Q


        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = tf.matmul(F, self.x) + tf.matmul(B, u)
        else:
            #print(F)
            #print(self.x)
            self.x = tf.matmul(F, self.x, output_type=tf.float32)
            

        # P = FPF' + Q
        self.P = self._alpha_sq * tf.matmul(tf.matmul(F, self.P), tf.transpose(F)) + Q

        # save prior
        self.x_prior = tf.identity(self.x)
        self.P_prior = tf.identity(self.P)


    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter.
        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.
        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
            If you pass in a value of H, z must be a column vector the
            of the correct size.
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.
        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            #self.z = np.array([[None]*self.dim_z]).T
            self.z = tf.zeros(self.dim_z, dtype=tf.float32)
            self.x_post = tf.identity(self.x)
            self.P_post = tf.identity(self.P)
            self.y = tf.zeros((self.dim_z, 1), dtype=tf.float32)
            return

        if R is None:
            R = self.R
        elif isscalar(R):
            R = tf.eye(self.dim_z, dtype=tf.float32) * R

        if H is None:
            # z = reshape_z(z, self.dim_z, self.dim_x)
            z = reshape_z_tf(z, self.dim_z, self.dim_x)
            H = tf.cast(self.H, tf.float32)

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - tf.matmul(H, self.x)

        # common subexpression for speed
        PHT = tf.matmul(self.P, tf.transpose(H))

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = tf.matmul(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = tf.matmul(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + tf.matmul(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - tf.matmul(self.K, H)
        self.P = tf.matmul(tf.matmul(I_KH, self.P), tf.transpose(I_KH)) + tf.matmul(tf.matmul(self.K, R), tf.transpose(self.K))

        # save measurement and posterior state
        self.z = z
        self.x_post = tf.identity(self.x)
        self.P_post = tf.identity(self.P)

def reshape_z_tf(z, dim_z, ndim):
    """Ensure z is a (dim_z, 1) shaped tensor, then reshape according to ndim
    
    Args:
        z: Input tensor to reshape
        dim_z: Expected first dimension size
        ndim: Target number of dimensions (0, 1, or 2)
    
    Returns:
        Reshaped tensor according to ndim:
        - ndim=2: shape (dim_z, 1)
        - ndim=1: shape (dim_z,)
        - ndim=0: scalar
    
    Raises:
        tf.errors.InvalidArgumentError: If z cannot be converted to shape (dim_z, 1)
    """
    # Convert to tensor if needed
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    
    # Ensure at least 2D (equivalent to np.atleast_2d)
    if len(z.shape) == 0:  # scalar
        z = tf.reshape(z, (1, 1))
    elif len(z.shape) == 1:  # 1D vector
        z = tf.expand_dims(z, axis=1)  # Make it (n, 1)
    
    # Get current shape
    current_shape = tf.shape(z)
    
    # Check if we need to transpose (if shape is (1, dim_z) instead of (dim_z, 1))
    def transpose_if_needed():
        return tf.cond(
            tf.equal(current_shape[1], dim_z),
            lambda: tf.transpose(z),
            lambda: z
        )
    
    # Apply transpose logic only if we have exactly 2 dimensions
    z = tf.cond(
        tf.equal(tf.rank(z), 2),
        transpose_if_needed,
        lambda: z
    )
    
    # Verify the shape is correct
    final_shape = tf.shape(z)
    
    # Check if shape is (dim_z, 1)
    shape_check = tf.logical_and(
        tf.equal(final_shape[0], dim_z),
        tf.equal(final_shape[1], 1)
    )
    
    # Assert shape is correct (this will raise an error if not)
    tf.debugging.assert_equal(
        shape_check, True,
        message=f'z must be convertible to shape ({dim_z}, 1), got shape: ' + tf.as_string(final_shape)
    )
    
    if ndim == 1:
        z = z[:, 0]  # Return as (dim_z,)
    elif ndim == 0:
        z = z[0, 0]  # Return as scalar

    return z
