import numpy as np
from sympy.utilities.misc import filldedent

"""
Gaussian optics, by Lukas.

focal distance
    positive for convergent lenses
object distance
    positive for real objects
image distance
    positive for real images
"""

###
# A, B, C, D matrices
###


class RayTransferMatrix(np.ndarray):

    def __new__(cls, *args):

        if len(args) == 4:
            temp = ((args[0], args[1]), (args[2], args[3]))
        elif len(args) == 1 \
            and isinstance(args[0], np.ndarray) \
                and args[0].shape == (2, 2):
            temp = args[0]
        else:
            raise ValueError(filldedent('''
                Expecting 2x2 Matrix or the 4 elements of
                the Matrix but got %s''' % str(args)))
        return np.asarray(temp).view(cls)

    def __mul__(self, other):
        if isinstance(other, RayTransferMatrix):
            return RayTransferMatrix(np.ndarray.dot(self, other))
        elif isinstance(other, GeometricRay):
            return GeometricRay(np.ndarray.dot(self, other))
        elif isinstance(other, BeamParameter):
            temp = np.ndarray.dot(self, np.array((other.q, 1)))
            q = (temp[0]/temp[1])
            return BeamParameter(other.wavelen, q.real, other.n,
                                  z_r=q.imag)
        else:
            return np.ndarray.dot(self, other)

    @property
    def A(self):
        return self[0, 0]
    @property
    def B(self):
        return self[0, 1]
    @property
    def C(self):
        return self[1, 0]
    @property
    def D(self):
        return self[1, 1]

class FreeSpace(RayTransferMatrix):
    def __new__(cls, d):
        return RayTransferMatrix.__new__(cls, 1, d, 0, 1)

class FlatRefraction(RayTransferMatrix):
    def __new__(cls, n1, n2):
        inst = RayTransferMatrix.__new__(cls, 1, 0, 0, n1/n2)
        inst.n1, inst.n2 = n1, n2
        return inst

class CurvedRefraction(RayTransferMatrix):
    def __new__(cls, R, n1, n2):
        inst = RayTransferMatrix.__new__(cls, 1, 0, (n1 - n2)/R/n2, n1/n2)
        inst.n1, inst.n2 = n1, n2
        return inst

# class FlatMirror(RayTransferMatrix):
#     def __new__(cls):
#         return RayTransferMatrix.__new__(cls, 1, 0, 0, 1)

# class CurvedMirror(RayTransferMatrix):
#     def __new__(cls, R):
#         return RayTransferMatrix.__new__(cls, 1, 0, -2/R, 1)

class ThinLens(RayTransferMatrix):
    def __new__(cls, f, n1 = 1, n2 = 1):
        inst = RayTransferMatrix.__new__(cls, 1, 0, -1/f, 1)
        inst.n1, inst.n2 = n1, n2
        return inst

###
# Representation for geometric ray
###

class GeometricRay(np.ndarray):
    """
    Representation for a geometric ray in the Ray Transfer Matrix formalism.

    Parameters
    ==========

    h : height, and
    angle : angle, or
    matrix : a 2x1 matrix (Matrix(2, 1, [height, angle]))

    Examples
    ========

    >>> from sympy.physics.optics import GeometricRay, FreeSpace
    >>> from sympy import symbols, Matrix
    >>> d, h, angle = symbols('d, h, angle')

    >>> GeometricRay(h, angle)
    Matrix([
    [    h],
    [angle]])

    >>> FreeSpace(d)*GeometricRay(h, angle)
    Matrix([
    [angle*d + h],
    [      angle]])

    >>> GeometricRay( Matrix( ((h,), (angle,)) ) )
    Matrix([
    [    h],
    [angle]])

    See Also
    ========

    RayTransferMatrix

    """

    def __new__(cls, *args):
        if len(args) == 1 and isinstance(args[0], np.ndarray) \
                and args[0].shape == (2, 1):
            temp = args[0]
        elif len(args) == 2:
            temp = ((args[0],), (args[1],))
        else:
            raise ValueError(filldedent('''
                Expecting 2x1 Matrix or the 2 elements of
                the Matrix but got %s''' % str(args)))
        return np.asarray(temp).view(cls)

    @property
    def height(self):
        return self[0]
    @property
    def angle(self):
        return self[1]


###
# Representation for gauss beam
###

class BeamParameter():
    
    """
    Parameters
    ==========

    wavelen : the wavelength,
    z : the distance to waist, and
    w : the waist, or
    z_r : the rayleigh range

    """

    __slots__ = ['z', 'z_r', 'n', 'wavelen']

    def __init__(self, wavelen, z, n, **kwargs):

        self.z = z
        self.wavelen = wavelen
        self.n = n

        if len(kwargs) != 1:
            raise ValueError('Constructor expects exactly one named argument.')
        elif 'z_r' in kwargs:
            self.z_r = kwargs['z_r']
        elif 'w' in kwargs:
            w = kwargs['w']
            t = (z*wavelen/np.pi/n)**2
            w_0 = np.sqrt(w**2/2+np.sqrt(w**4/4-t))
            self.z_r = waist2rayleigh(w_0, wavelen, n)
        else:
            raise ValueError('The constructor needs named argument w or z_r')

    @property
    def q(self):
        """
        The complex parameter representing the beam.
        """
        return self.z + 1j*self.z_r

    @property
    def radius(self):
        """
        The radius of curvature of the phase front.
        """
        return self.z*(1 + (self.z_r/self.z)**2)

    @property
    def w(self):
        """
        The beam radius at `1/e^2` intensity.
        """
        return self.w_0*np.sqrt(1 + (self.z/self.z_r)**2)

    @property
    def w_0(self):
        """
        The beam waist (minimal radius).
        """
        return np.sqrt(self.z_r/np.pi/self.n*self.wavelen)

    # @property
    # def divergence(self):
    #     """
    #     Half of the total angular spread.

    #     Examples
    #     ========

    #     >>> from sympy.physics.optics import BeamParameter
    #     >>> p = BeamParameter(530e-9, 1, w=1e-3)
    #     >>> p.divergence
    #     0.00053/pi
    #     """
    #     return self.wavelen/pi/self.w_0

    # @property
    # def gouy(self):
    #     """
    #     The Gouy phase.

    #     Examples
    #     ========

    #     >>> from sympy.physics.optics import BeamParameter
    #     >>> p = BeamParameter(530e-9, 1, w=1e-3)
    #     >>> p.gouy
    #     atan(0.53/pi)
    #     """
    #     return atan2(self.z, self.z_r)

    # @property
    # def waist_approximation_limit(self):
    #     """
    #     The minimal waist for which the gauss beam approximation is valid.

    #     The gauss beam is a solution to the paraxial equation. For curvatures
    #     that are too great it is not a valid approximation.

    #     Examples
    #     ========

    #     >>> from sympy.physics.optics import BeamParameter
    #     >>> p = BeamParameter(530e-9, 1, w=1e-3)
    #     >>> p.waist_approximation_limit
    #     1.06e-6/pi
    #     """
    #     return 2*self.wavelen/pi


###
# Utilities
###

def waist2rayleigh(w_0, wavelen, n):
    """
    Calculate the rayleigh range from the waist of a gaussian beam.
    """
    return w_0**2*np.pi/wavelen*n

def rayleigh2waist(z_r, wavelen, n):
    """Calculate the waist from the rayleigh range of a gaussian beam.
    """
    return np.sqrt(z_r/np.pi/n*wavelen)


# def geometric_conj_ab(a, b):
#     """
#     Conjugation relation for geometrical beams under paraxial conditions.

#     Takes the distances to the optical element and returns the needed
#     focal distance.

#     See Also
#     ========

#     geometric_conj_af, geometric_conj_bf

#     Examples
#     ========

#     >>> from sympy.physics.optics import geometric_conj_ab
#     >>> from sympy import symbols
#     >>> a, b = symbols('a b')
#     >>> geometric_conj_ab(a, b)
#     a*b/(a + b)
#     """
#     a, b = map(sympify, (a, b))
#     if a.is_infinite or b.is_infinite:
#         return a if b.is_infinite else b
#     else:
#         return a*b/(a + b)


# def geometric_conj_af(a, f):
#     """
#     Conjugation relation for geometrical beams under paraxial conditions.

#     Takes the object distance (for geometric_conj_af) or the image distance
#     (for geometric_conj_bf) to the optical element and the focal distance.
#     Then it returns the other distance needed for conjugation.

#     See Also
#     ========

#     geometric_conj_ab

#     Examples
#     ========

#     >>> from sympy.physics.optics.gaussopt import geometric_conj_af, geometric_conj_bf
#     >>> from sympy import symbols
#     >>> a, b, f = symbols('a b f')
#     >>> geometric_conj_af(a, f)
#     a*f/(a - f)
#     >>> geometric_conj_bf(b, f)
#     b*f/(b - f)
#     """
#     a, f = map(sympify, (a, f))
#     return -geometric_conj_ab(a, -f)

# geometric_conj_bf = geometric_conj_af


# def gaussian_conj(s_in, z_r_in, f):
#     """
#     Conjugation relation for gaussian beams.

#     Parameters
#     ==========

#     s_in : the distance to optical element from the waist
#     z_r_in : the rayleigh range of the incident beam
#     f : the focal length of the optical element

#     Returns
#     =======

#     a tuple containing (s_out, z_r_out, m)
#     s_out : the distance between the new waist and the optical element
#     z_r_out : the rayleigh range of the emergent beam
#     m : the ration between the new and the old waists

#     Examples
#     ========

#     >>> from sympy.physics.optics import gaussian_conj
#     >>> from sympy import symbols
#     >>> s_in, z_r_in, f = symbols('s_in z_r_in f')

#     >>> gaussian_conj(s_in, z_r_in, f)[0]
#     1/(-1/(s_in + z_r_in**2/(-f + s_in)) + 1/f)

#     >>> gaussian_conj(s_in, z_r_in, f)[1]
#     z_r_in/(1 - s_in**2/f**2 + z_r_in**2/f**2)

#     >>> gaussian_conj(s_in, z_r_in, f)[2]
#     1/sqrt(1 - s_in**2/f**2 + z_r_in**2/f**2)
#     """
#     s_in, z_r_in, f = map(sympify, (s_in, z_r_in, f))
#     s_out = 1 / ( -1/(s_in + z_r_in**2/(s_in - f)) + 1/f )
#     m = 1/sqrt((1 - (s_in/f)**2) + (z_r_in/f)**2)
#     z_r_out = z_r_in / ((1 - (s_in/f)**2) + (z_r_in/f)**2)
#     return (s_out, z_r_out, m)


# def conjugate_gauss_beams(wavelen, waist_in, waist_out, **kwargs):
#     """
#     Find the optical setup conjugating the object/image waists.

#     Parameters
#     ==========

#     wavelen : the wavelength of the beam
#     waist_in and waist_out : the waists to be conjugated
#     f : the focal distance of the element used in the conjugation

#     Returns
#     =======

#     a tuple containing (s_in, s_out, f)
#     s_in : the distance before the optical element
#     s_out : the distance after the optical element
#     f : the focal distance of the optical element

#     Examples
#     ========

#     >>> from sympy.physics.optics import conjugate_gauss_beams
#     >>> from sympy import symbols, factor
#     >>> l, w_i, w_o, f = symbols('l w_i w_o f')

#     >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[0]
#     f*(1 - sqrt(w_i**2/w_o**2 - pi**2*w_i**4/(f**2*l**2)))

#     >>> factor(conjugate_gauss_beams(l, w_i, w_o, f=f)[1])
#     f*w_o**2*(w_i**2/w_o**2 - sqrt(w_i**2/w_o**2 -
#               pi**2*w_i**4/(f**2*l**2)))/w_i**2

#     >>> conjugate_gauss_beams(l, w_i, w_o, f=f)[2]
#     f
#     """
#     #TODO add the other possible arguments
#     wavelen, waist_in, waist_out = map(sympify, (wavelen, waist_in, waist_out))
#     m = waist_out / waist_in
#     z = waist2rayleigh(waist_in, wavelen)
#     if len(kwargs) != 1:
#         raise ValueError("The function expects only one named argument")
#     elif 'dist' in kwargs:
#         raise NotImplementedError(filldedent('''
#             Currently only focal length is supported as a parameter'''))
#     elif 'f' in kwargs:
#         f = sympify(kwargs['f'])
#         s_in = f * (1 - sqrt(1/m**2 - z**2/f**2))
#         s_out = gaussian_conj(s_in, z, f)[0]
#     elif 's_in' in kwargs:
#         raise NotImplementedError(filldedent('''
#             Currently only focal length is supported as a parameter'''))
#     else:
#         raise ValueError(filldedent('''
#             The functions expects the focal length as a named argument'''))
#     return (s_in, s_out, f)

#TODO
#def plot_beam():
#    """Plot the beam radius as it propagates in space."""
#    pass

#TODO
#def plot_beam_conjugation():
#    """
#    Plot the intersection of two beams.
#
#    Represents the conjugation relation.
#
#    See Also
#    ========
#
#    conjugate_gauss_beams
#    """
#    pass
