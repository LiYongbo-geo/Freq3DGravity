import numpy
import numpy as np
import scipy.interpolate
import scipy.special
import copy as cp
import json
import struct
import numpy as np
import scipy.interpolate as interp
from matplotlib import pyplot as plt
from matplotlib import pyplot
import warnings
#import properties
from struct import unpack
try:
    from . import _prism
except ImportError:
    import _prism

'''try:
    from . import _prism
except ImportError:
    _prism = None'''

#: Conversion factor from SI units to mGal: :math:`1\ m/s^2 = 10^5\ mGal`
SI2MGAL = 100000.0


#: The gravitational constant in :math:`m^3 kg^{-1} s^{-1}`
G = 0.00000000006673

###
def gridderRegular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    .. warning::

        As of version 0.4, the ``shape`` argument was corrected to be
        ``shape = (nx, ny)`` instead of ``shape = (ny, nx)``.


    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples:

    >>> x, y = regular((0, 10, 0, 5), (5, 3))
    >>> print(x)
    [  0.    0.    0.    2.5   2.5   2.5   5.    5.    5.    7.5   7.5   7.5 10.   10.   10. ]
    >>> print(x.reshape((5, 3)))
    [[  0.    0.    0. ]
     [  2.5   2.5   2.5]
     [  5.    5.    5. ]
     [  7.5   7.5   7.5]
     [ 10.   10.   10. ]]
    >>> print(y.reshape((5, 3)))
    [[ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]
     [ 0.   2.5  5. ]]
    >>> x, y = regular((0, 0, 0, 5), (1, 3))
    >>> print(x.reshape((1, 3)))
    [[ 0.  0.  0.]]
    >>> print(y.reshape((1, 3)))
    [[ 0.   2.5  5. ]]
    >>> x, y, z = regular((0, 10, 0, 5), (5, 3), z=-10)
    >>> print(z.reshape((5, 3)))
    [[-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]
     [-10. -10. -10.]]


    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    _check_area(area)
    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = np.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*np.ones(nx*ny, dtype=np.float))
    return [i.ravel() for i in arrays]

class GeometricElement(object):
    """
    Base class for all geometric elements.
    """

    def __init__(self, props):
        self.props = {}
        if props is not None:
            for p in props:
                self.props[p] = props[p]

    def addprop(self, prop, value):
        """
        Add a physical property to this geometric element.

        If it already has the property, the given value will overwrite the
        existing one.

        Parameters:

        * prop : str
            Name of the physical property.
        * value : float
            The value of this physical property.

        """
        self.props[prop] = value

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)

class Prism(GeometricElement):
    """
    A 3D right rectangular prism.

    .. note:: The coordinate system used is x -> North, y -> East and z -> Down

    Parameters:

    * x1, x2 : float
        South and north borders of the prism
    * y1, y2 : float
        West and east borders of the prism
    * z1, z2 : float
        Top and bottom of the prism
    * props : dict
        Physical properties assigned to the prism.
        Ex: ``props={'density':10, 'magnetization':10000}``

    Examples:

        >>> from geoist.inversion import Prism
        >>> p = Prism(1, 2, 3, 4, 5, 6, {'density':200})
        >>> p.props['density']
        200
        >>> print p.get_bounds()
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | density:200
        >>> p = Prism(1, 2, 3, 4, 5, 6)
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6
        >>> p.addprop('density', 2670)
        >>> print p
        x1:1 | x2:2 | y1:3 | y2:4 | z1:5 | z2:6 | density:2670

    """

    def __init__(self, x1, x2, y1, y2, z1, z2, props=None):
        super().__init__(props)
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)
        self.z1 = float(z1)
        self.z2 = float(z2)

    def __str__(self):
        """Return a string representation of the prism."""
        names = [('x1', self.x1), ('x2', self.x2), ('y1', self.y1),
                 ('y2', self.y2), ('z1', self.z1), ('z2', self.z2)]
        names.extend((p, self.props[p]) for p in sorted(self.props))
        return ' | '.join('%s:%g' % (n, v) for n, v in names)

    def get_bounds(self):
        """
        Get the bounding box of the prism (i.e., the borders of the prism).

        Returns:

        * bounds : list
            ``[x1, x2, y1, y2, z1, z2]``, the bounds of the prism

        Examples:

            >>> p = Prism(1, 2, 3, 4, 5, 6)
            >>> print p.get_bounds()
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        """
        return [self.x1, self.x2, self.y1, self.y2, self.z1, self.z2]

    def center(self):
        """
        Return the coordinates of the center of the prism.

        Returns:

        * coords : list = [xc, yc, zc]
            Coordinates of the center

        Example:

            >>> prism = Prism(1, 2, 1, 3, 0, 2)
            >>> print prism.center()
            [ 1.5  2.   1. ]

        """
        xc = 0.5 * (self.x1 + self.x2)
        yc = 0.5 * (self.y1 + self.y2)
        zc = 0.5 * (self.z1 + self.z2)
        return np.array([xc, yc, zc])

###
def interp_at(x, y, v, xp, yp, algorithm='cubic', extrapolate=False):
    """
    Interpolate spacial data onto specified points.

    Wraps ``scipy.interpolate.griddata``.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * xp, yp : 1D arrays
        Points where the data values will be interpolated
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata)
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * v : 1D array
        1D array with the interpolated v values.

    """
    vp = scipy.interpolate.griddata((x, y), v, (xp, yp),
                                    method=algorithm).ravel()
    if extrapolate and algorithm != 'nearest' and np.any(np.isnan(vp)):
        fill_nans(x, y, v, xp, yp, vp)
    return vp

def gInterp(x, y, v, shape, area=None, algorithm='cubic', extrapolate=False):
    """
    Interpolate spacial data onto a regular grid.

    Utility function that generates a regular grid with
    :func:`~geoist.gridder.regular` and calls
    :func:`~geoist.gridder.interp_at` on the generated points.

    Parameters:

    * x, y : 1D arrays
        Arrays with the x and y coordinates of the data points.
    * v : 1D array
        Array with the scalar value assigned to the data points.
    * shape : tuple = (nx, ny)
        Shape of the interpolated regular grid, ie (nx, ny).
    * area : tuple = (x1, x2, y1, y2)
        The are where the data will be interpolated. If None, then will get the
        area from *x* and *y*.
    * algorithm : string
        Interpolation algorithm. Either ``'cubic'``, ``'nearest'``,
        ``'linear'`` (see scipy.interpolate.griddata).
    * extrapolate : True or False
        If True, will extrapolate values outside of the convex hull of the data
        points.

    Returns:

    * ``[x, y, v]``
        Three 1D arrays with the interpolated x, y, and v

    """
    if area is None:
        area = (x.min(), x.max(), y.min(), y.max())
    x1, x2, y1, y2 = area
    xp, yp = gridderRegular(area, shape)
    vp = interp_at(x, y, v, xp, yp, algorithm=algorithm,
                   extrapolate=extrapolate)
    return xp, yp, vp

def contour(x, y, v, shape, levels, interp=False, extrapolate=False, color='k',
            label=None, clabel=True, style='solid', linewidth=1.0,
            basemap=None):
    """
    Make a contour plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * color : str
        Color of the contour lines.
    * label : str
        String with the label of the contour that would show in a legend.
    * clabel : True or False
        Wether or not to print the numerical value of the contour lines
    * style : str
        The style of the contour lines. Can be ``'dashed'``, ``'solid'`` or
        ``'mixed'`` (solid lines for positive contours and dashed for negative)
    * linewidth : float
        Width of the contour lines
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~geoist.vis.giplt.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if style not in ['solid', 'dashed', 'mixed']:
        raise ValueError("Invalid contour style %s" % (style))
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = gInterp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(colors=color)
    #kwargs = dict(colors=color, picker=True)
    if basemap is None:
        ct_data = pyplot.contour(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contour(lon, lat, V, levels, **kwargs)
    if clabel:
        ct_data.clabel(fmt='%g')
    if label is not None:
        ct_data.collections[0].set_label(label)
    if style != 'mixed':
        for c in ct_data.collections:
            c.set_linestyle(style)
    for c in ct_data.collections:
        c.set_linewidth(linewidth)
    return ct_data.levels

def contourf(x, y, v, shape, levels, interp=False, extrapolate=False,
             vmin=None, vmax=None, cmap=pyplot.cm.jet, basemap=None):
    """
    Make a filled contour plot of the data.

    Parameters:

    * x, y : array
        Arrays with the x and y coordinates of the grid points. If the data is
        on a regular grid, then assume x varies first (ie, inner loop), then y.
    * v : array
        The scalar value assigned to the grid points.
    * shape : tuple = (ny, nx)
        Shape of the regular grid.
        If interpolation is not False, then will use *shape* to grid the data.
    * levels : int or list
        Number of contours to use or a list with the contour values.
    * interp : True or False
        Wether or not to interpolate before trying to plot. If data is not on
        regular grid, set to True!
    * extrapolate : True or False
        Wether or not to extrapolate the data when interp=True
    * vmin, vmax
        Saturation values of the colorbar. If provided, will overwrite what is
        set by *levels*.
    * cmap : colormap
        Color map to be used. (see pyplot.cm module)
    * basemap : mpl_toolkits.basemap.Basemap
        If not None, will use this basemap for plotting with a map projection
        (see :func:`~geoist.vis.giplt.basemap` for creating basemaps)

    Returns:

    * levels : list
        List with the values of the contour levels

    """
    if x.shape != y.shape != v.shape:
        raise ValueError("Input arrays x, y, and v must have same shape!")
    if interp:
        x, y, v = gInterp(x, y, v, shape, extrapolate=extrapolate)
    X = numpy.reshape(x, shape)
    Y = numpy.reshape(y, shape)
    V = numpy.reshape(v, shape)
    kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap)
    #kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, picker=True)
    if basemap is None:
        ct_data = pyplot.contourf(X, Y, V, levels, **kwargs)
        pyplot.xlim(X.min(), X.max())
        pyplot.ylim(Y.min(), Y.max())
    else:
        lon, lat = basemap(X, Y)
        ct_data = basemap.contourf(lon, lat, V, levels, **kwargs)
    return ct_data.levels

###
def pf_nextpow2(i):
    buf = numpy.ceil(numpy.log(i)/numpy.log(2))
    return int(2**buf)

class SmoothOperator:
    def __init__(self):
        self.axis = {'zyx':{'x':-1,'y':-2,'z':-3},
                     'xyz':{'x':-3,'y':-2,'z':-1}}

    def derivation(self,v,component='dx',array_order='zyx'):
        for axis_i in component[1:]:
            slices = [slice(None)]*v.ndim
            slices[self.axis[array_order][axis_i]] = slice(-1,None,-1)
            v = np.diff(v[tuple(slices)],axis=self.axis[array_order][axis_i])[tuple(slices)]
        return v

    def rderivation(self,v,component='dx',array_order='zyx'):
        for axis_i in component[-1:0:-1]:
            slices = [slice(None)]*v.ndim
            slices[self.axis[array_order][axis_i]] = 0
            shape = list(v.shape)
            shape[self.axis[array_order][axis_i]] = 1
            prepend=np.zeros_like(v[tuple(slices)].reshape(tuple(shape)))
            append=np.zeros_like(v[tuple(slices)].reshape(tuple(shape)))
            v = np.diff(v,
                           axis=self.axis[array_order][axis_i],
                           prepend=prepend,
                           append=append)
        return v

    def shapes(self,component='dx',model_size=None):
        '''shape information
        Returns:
            shapes(list): shapes[0] is the correct shape of a vector for rderivation operating on.
                          shapes[1] is the shape of derivation matrix.
        '''
        testv = np.zeros(model_size)
        resv = self.derivation(testv,component=component)
        return [resv.shape,(len(resv.ravel()),len(testv.ravel()))]

class PrismMesh(object):
    """
    A 3D regular mesh of right rectangular prisms.

    Prisms are ordered as follows: first layers (z coordinate),
    then EW rows (y) and finaly x coordinate (NS).

    .. note:: Remember that the coordinate system is x->North, y->East and
        z->Down

    Ex: in a mesh with shape ``(3,3,3)`` the 15th element (index 14) has z
    index 1 (second layer), y index 1 (second row), and x index 2 (third
    element in the column).

    :class:`~geoist.inversion.PrismMesh` can used as list of prisms. It acts
    as an iteratior (so you can loop over prisms). It also has a
    ``__getitem__`` method to access individual elements in the mesh.
    In practice, :class:`~geoist.inversion.PrismMesh` should be able to be
    passed to any function that asks for a list of prisms, like
    :func:`geoist.pfm.prism.gz`.

    To make the mesh incorporate a topography, use
    :meth:`~geoist.inversion.PrismMesh.carvetopo`

    Parameters:

    * bounds : list = [xmin, xmax, ymin, ymax, zmin, zmax]
        Boundaries of the mesh.
    * shape : tuple = (nz, ny, nx)
        Number of prisms in the x, y, and z directions.
    * props :  dict
        Physical properties of each prism in the mesh.
        Each key should be the name of a physical property. The corresponding
        value should be a list with the values of that particular property on
        each prism of the mesh.

    Examples:

        >>> from geoist.inversion import PrismMesh
        >>> mesh = PrismMesh((0, 1, 0, 2, 0, 3), (1, 2, 2))
        >>> for p in mesh:
        ...     print p
        x1:0 | x2:0.5 | y1:0 | y2:1 | z1:0 | z2:3
        x1:0.5 | x2:1 | y1:0 | y2:1 | z1:0 | z2:3
        x1:0 | x2:0.5 | y1:1 | y2:2 | z1:0 | z2:3
        x1:0.5 | x2:1 | y1:1 | y2:2 | z1:0 | z2:3
        >>> print mesh[0]
        x1:0 | x2:0.5 | y1:0 | y2:1 | z1:0 | z2:3
        >>> print mesh[-1]
        x1:0.5 | x2:1 | y1:1 | y2:2 | z1:0 | z2:3

    One with physical properties::

        >>> props = {'density':[2670.0, 1000.0]}
        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2), props=props)
        >>> for p in mesh:
        ...     print p
        x1:0 | x2:1 | y1:0 | y2:4 | z1:0 | z2:3 | density:2670
        x1:1 | x2:2 | y1:0 | y2:4 | z1:0 | z2:3 | density:1000

    or equivalently::

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2))
        >>> mesh.addprop('density', [200, -1000.0])
        >>> for p in mesh:
        ...     print p
        x1:0 | x2:1 | y1:0 | y2:4 | z1:0 | z2:3 | density:200
        x1:1 | x2:2 | y1:0 | y2:4 | z1:0 | z2:3 | density:-1000

    You can use :meth:`~geoist.inversion.PrismMesh.get_xs` (and similar
    methods for y and z) to get the x coordinates of the prisms in the mesh::

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2))
        >>> print mesh.get_xs()
        [ 0.  1.  2.]
        >>> print mesh.get_ys()
        [ 0.  4.]
        >>> print mesh.get_zs()
        [ 0.  3.]

    The ``shape`` of the mesh must be integer!

        >>> mesh = PrismMesh((0, 2, 0, 4, 0, 3), (1, 1, 2.5))
        Traceback (most recent call last):
            ...
        AttributeError: Invalid mesh shape (1, 1, 2.5). shape must be integers

    """

    celltype = Prism

    def __init__(self, bounds, shape, props=None):
        nz, ny, nx = shape
        if not isinstance(nx, int) or not isinstance(ny, int) or \
                not isinstance(nz, int):
            raise AttributeError(
                'Invalid mesh shape {}. shape must be integers'.format(
                    str(shape)))
        size = int(nx * ny * nz)
        x1, x2, y1, y2, z1, z2 = bounds
        dx = (x2 - x1)/nx
        dy = (y2 - y1)/ny
        dz = (z2 - z1)/nz
        self.shape = tuple(int(i) for i in shape)
        self.size = size
        self.dims = (dx, dy, dz)
        self.bounds = bounds
        if props is None:
            self.props = {}
        else:
            self.props = props
        # The index of the current prism in an iteration. Needed when mesh is
        # used as an iterator
        self.i = 0
        # List of masked prisms. Will return None if trying to access them
        self.mask = []
        # Wether or not to change heights to z coordinate
        self.zdown = True

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index >= self.size or index < -self.size:
            raise IndexError('mesh index out of range')
        # To walk backwards in the list
        if index < 0:
            index = self.size + index
        if index in self.mask:
            return None
        nz, ny, nx = self.shape
        k = index//(nx*ny)
        j = (index - k*(nx*ny))//nx
        i = (index - k*(nx*ny) - j*nx)
        x1 = self.bounds[0] + self.dims[0] * i
        x2 = x1 + self.dims[0]
        y1 = self.bounds[2] + self.dims[1] * j
        y2 = y1 + self.dims[1]
        z1 = self.bounds[4] + self.dims[2] * k
        z2 = z1 + self.dims[2]
        props = dict([p, self.props[p][index]] for p in self.props)
        return self.celltype(x1, x2, y1, y2, z1, z2, props=props)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        prism = self.__getitem__(self.i)
        self.i += 1
        return prism

    def addprop(self, prop, values):
        """
        Add physical property values to the cells in the mesh.

        Different physical properties of the mesh are stored in a dictionary.

        Parameters:

        * prop : str
            Name of the physical property.
        * values :  list or array
            Value of this physical property in each prism of the mesh. For the
            ordering of prisms in the mesh see
            :class:`~geoist.inversion.PrismMesh`

        """
        self.props[prop] = values

    def carvetopo(self, x, y, height, below=False):
        """
        Mask (remove) prisms from the mesh that are above the topography.

        Accessing the ith prism will return None if it was masked (above the
        topography).
        Also mask prisms outside of the topography grid provided.
        The topography height information does not need to be on a regular
        grid, it will be interpolated.

        Parameters:

        * x, y : lists
            x and y coordinates of the grid points
        * height : list or array
            Array with the height of the topography
        * below : boolean
            Will mask prisms below the input surface if set to *True*.

        """
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        # The coordinates of the centers of the cells
        xc = np.arange(x1, x2, dx) + 0.5 * dx
        # Sometimes arange returns more due to rounding
        if len(xc) > nx:
            xc = xc[:-1]
        yc = np.arange(y1, y2, dy) + 0.5 * dy
        if len(yc) > ny:
            yc = yc[:-1]
        zc = np.arange(z1, z2, dz) + 0.5 * dz
        if len(zc) > nz:
            zc = zc[:-1]
        XC, YC = np.meshgrid(xc, yc)
        topo = scipy.interpolate.griddata((x, y), height, (XC, YC),
                                          method='cubic').ravel()
        if self.zdown:
            # -1 if to transform height into z coordinate
            topo = -1 * topo
        # griddata returns a masked array. If the interpolated point is out of
        # of the data range, mask will be True. Use this to remove all cells
        # below a masked topo point (ie, one with no height information)
        if np.ma.isMA(topo):
            topo_mask = topo.mask
        else:
            topo_mask = [False for i in range(len(topo))]
        c = 0
        for cellz in zc:
            for h, masked in zip(topo, topo_mask):
                if below:
                    if (masked or
                            (cellz > h and self.zdown) or
                            (cellz < h and not self.zdown)):
                        self.mask.append(c)
                else:
                    if (masked or
                            (cellz < h and self.zdown) or
                            (cellz > h and not self.zdown)):
                        self.mask.append(c)
                c += 1

    def get_xs(self):
        """
        Return an array with the x coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        xs = np.arange(x1, x2 + dx, dx)
        if xs.size > nx + 1:
            return xs[:-1]
        return xs

    def get_ys(self):
        """
        Return an array with the y coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        ys = np.arange(y1, y2 + dy, dy)
        if ys.size > ny + 1:
            return ys[:-1]
        return ys

    def get_zs(self):
        """
        Return an array with the z coordinates of the prisms in mesh.
        """
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        nz, ny, nx = self.shape
        zs = np.arange(z1, z2 + dz, dz)
        if zs.size > nz + 1:
            return zs[:-1]
        return zs

    def get_layer(self, i):
        """
        Return the set of prisms corresponding to the ith layer of the mesh.

        Parameters:

        * i : int
            The index of the layer

        Returns:

        * prisms : list of :class:`~geoist.inversion.Prism`
            The prisms in the ith layer

        Examples::

            >>> mesh = PrismMesh((0, 2, 0, 2, 0, 2), (2, 2, 2))
            >>> layer = mesh.get_layer(0)
            >>> for p in layer:
            ...     print p
            x1:0 | x2:1 | y1:0 | y2:1 | z1:0 | z2:1
            x1:1 | x2:2 | y1:0 | y2:1 | z1:0 | z2:1
            x1:0 | x2:1 | y1:1 | y2:2 | z1:0 | z2:1
            x1:1 | x2:2 | y1:1 | y2:2 | z1:0 | z2:1
            >>> layer = mesh.get_layer(1)
            >>> for p in layer:
            ...     print p
            x1:0 | x2:1 | y1:0 | y2:1 | z1:1 | z2:2
            x1:1 | x2:2 | y1:0 | y2:1 | z1:1 | z2:2
            x1:0 | x2:1 | y1:1 | y2:2 | z1:1 | z2:2
            x1:1 | x2:2 | y1:1 | y2:2 | z1:1 | z2:2


        """
        nz, ny, nx = self.shape
        if i >= nz or i < 0:
            raise IndexError('Layer index %d is out of range.' % (i))
        start = i * nx * ny
        end = (i + 1) * nx * ny
        layer = [self.__getitem__(p) for p in range(start, end)]
        return layer

    def layers(self):
        """
        Returns an iterator over the layers of the mesh.

        Examples::

            >>> mesh = PrismMesh((0, 2, 0, 2, 0, 2), (2, 2, 2))
            >>> for layer in mesh.layers():
            ...     for p in layer:
            ...         print p
            x1:0 | x2:1 | y1:0 | y2:1 | z1:0 | z2:1
            x1:1 | x2:2 | y1:0 | y2:1 | z1:0 | z2:1
            x1:0 | x2:1 | y1:1 | y2:2 | z1:0 | z2:1
            x1:1 | x2:2 | y1:1 | y2:2 | z1:0 | z2:1
            x1:0 | x2:1 | y1:0 | y2:1 | z1:1 | z2:2
            x1:1 | x2:2 | y1:0 | y2:1 | z1:1 | z2:2
            x1:0 | x2:1 | y1:1 | y2:2 | z1:1 | z2:2
            x1:1 | x2:2 | y1:1 | y2:2 | z1:1 | z2:2

        """
        nz, ny, nx = self.shape
        for i in range(nz):
            yield self.get_layer(i)

    def dump(self, meshfile, propfile, prop):
        r"""
        Dump the mesh to a file in the format required by UBC-GIF program
        MeshTools3D.

        Parameters:

        * meshfile : str or file
            Output file to save the mesh. Can be a file name or an open file.
        * propfile : str or file
            Output file to save the physical properties *prop*. Can be a file
            name or an open file.
        * prop : str
            The name of the physical property in the mesh that will be saved to
            *propfile*.

        .. note:: Uses -10000000 as the dummy value for plotting topography

        Examples:

            >>> from StringIO import StringIO
            >>> meshfile = StringIO()
            >>> densfile = StringIO()
            >>> mesh = PrismMesh((0, 10, 0, 20, 0, 5), (1, 2, 2))
            >>> mesh.addprop('density', [1, 2, 3, 4])
            >>> mesh.dump(meshfile, densfile, 'density')
            >>> print meshfile.getvalue().strip()
            2 2 1
            0 0 0
            2*10
            2*5
            1*5
            >>> print densfile.getvalue().strip()
            1.0000
            3.0000
            2.0000
            4.0000

        """
        if prop not in self.props:
            raise ValueError("mesh doesn't have a '%s' property." % (prop))
        isstr = False
        if isinstance(meshfile, str):
            isstr = True
            meshfile = open(meshfile, 'w')
        nz, ny, nx = self.shape
        x1, x2, y1, y2, z1, z2 = self.bounds
        dx, dy, dz = self.dims
        meshfile.writelines([
            "%d %d %d\n" % (ny, nx, nz),
            "%g %g %g\n" % (y1, x1, -z1),
            "%d*%g\n" % (ny, dy),
            "%d*%g\n" % (nx, dx),
            "%d*%g" % (nz, dz)])
        if isstr:
            meshfile.close()
        values = np.fromiter(self.props[prop], dtype=np.float)
        # Replace the masked cells with a dummy value
        values[self.mask] = -10000000
        reordered = np.ravel(np.reshape(values, self.shape), order='F')
        np.savetxt(propfile, reordered, fmt='%.4f')

    def copy(self):
        """ Return a deep copy of the current instance."""
        return cp.deepcopy(self)

def kron_matvec(t, V):
    '''matrix M multiply vectors V where M is kronecker product of A and B,
    i.e. M = A\otimes B.
    Args:
        t (list of ndarray): M = tds[0] \otimes t[1] \otimes t[2] ...
        V (ndarray): vectors to be multiplied.
    Returns:
        res (ndarray): results.
    '''
    shapes = [m.shape[1] for m in t]
    if V.ndim == 1:
        tmp = V.reshape(1,*shapes)
    else:
        tmp = V.reshape(V.shape[0],*shapes)
    n = len(t)
    params = []
    for i,m in enumerate(t):
        params.append(m)
        params.append([i,n+i])
    params.append(tmp)
    params.append([2*n]+list(range(n,2*n)))
    params.append([2*n]+list(range(n)))
    path = np.einsum_path(*params,optimize='optimal')
    res = np.einsum(*params,optimize=path[0])
    res = res.reshape(res.shape[0],-1)
    return res.squeeze()

class FreqInvModel:
    def __init__(self,conf_file=None,**kwargs):
        self.confs = {'nzyx':[4,4,4],
                      'nobsyx':None,
                      'smooth_components':None,
                      'depth_scaling':None,
                      'model_density':None,
                      'refer_densities':None,
                      'weights':None,
                      'source_volume':None,
                      'obs_area':None,
                      'data_dir':'./'}
        confs = dict()
        if not conf_file is None:
            with open(conf_file) as f:
                confs = json.load(f)
        self.confs = {**self.confs,**confs,**kwargs}
        self.nz,self.ny,self.nx = self.confs['nzyx']
        if self.confs['nobsyx'] is None:
            self.nobsx = self.nx
            self.nobsy = self.ny
        else:
            self.nobsy,self.nobsx = self.confs['nobsyx']
        self.source_volume = self.confs['source_volume']
        self.obs_area= self.confs['obs_area']
        if self.confs['model_density'] is None:
            self._model_density = None
        else:
            self._model_density = self.confs['model_density'].ravel()
        self._smooth_components = self.confs['smooth_components']
        if self.confs['depth_scaling'] is None:
            self.depth_scaling = np.ones(self.nz)
        else:
            self.depth_scaling = self.confs['depth_scaling']
        self.refer_densities = self.confs['refer_densities']
        self._weights = self.confs['weights']
        self.smop = SmoothOperator()
        self.kernel_op = None

    def gen_mesh(self,height = 0):
        ''' Generate mesh of the model.
        Args:
            height (float): height of the observations.
        '''
        shape = (self.nz, self.ny, self.nx)
        self.mesh = PrismMesh(self.source_volume, shape)
        self.mesh.addprop('density', self._model_density)
        # generate obs grid
        # coordinate: x North-South,y East-West
        # gridder is in the order: (nx,ny)
        self.gen_obs_grid(height=height)

    def gen_obs_grid(self,height=0):
        if self.obs_area is None:
            self.obs_area = (self.source_volume[0]+0.5*self.mesh.dims[0],
                              self.source_volume[1]-0.5*self.mesh.dims[0],
                              self.source_volume[2]+0.5*self.mesh.dims[1],
                              self.source_volume[3]-0.5*self.mesh.dims[1])
        obs_shape = (self.nobsx, self.nobsy)
        self.xp, self.yp, self.zp = gridderRegular(self.obs_area, obs_shape, z=height)

    def _pad_data(self, data, shape):
        n0 = pf_nextpow2(2*shape[0])
        n1 = pf_nextpow2(2*shape[1])
        nx, ny = shape
        padx = (n0 - nx)//2
        pady = (n1 - ny)//2
        padded = np.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
        return padded, padx, pady

    def gen_kernel(self,gtype='z'):
        xs = np.array(self.mesh.get_xs())
        ys = np.array(self.mesh.get_ys())
        zs = np.array(self.mesh.get_zs())
        x0 = (xs[:-1] + xs[1:])/2.0
        y0 = (ys[:-1] + ys[1:])/2.0
        a = np.abs(xs[:-1]-xs[1:])/2.0
        b = np.abs(ys[:-1]-ys[1:])/2.0
        nx, ny = self.nobsx, self.nobsy
        xmin, xmax, ymin, ymax = self.obs_area
        dx = (xmax - xmin)/(nx - 1)
        dy = (ymax - ymin)/(ny - 1)

        # Pad the array with the edge values to avoid instability
        shape = (nx,ny)
        self.padded, self.padx, self.pady = self._pad_data(self.zp, shape)
        nxe, nye = self.padded.shape
        M_left=(nxe-nx)/2+1
        M_right=M_left+nx-1
        N_down=(nye-ny)/2+1
        N_up=N_down+ny-1

        XXmin=xmin-dx*(M_left-1)
        XXmax=xmax+dx*(nxe-M_right)
        YYmin=ymin-dy*(N_down-1)
        YYmax=ymax+dy*(nye-N_up)
        # we store kx and ky as 1d array
        self.kx = 2*np.pi*np.array(np.fft.fftfreq(self.padded.shape[0], dx))
        self.ky = 2*np.pi*np.array(np.fft.fftfreq(self.padded.shape[1], dy))
        # kz is 2d array
        self.kz = np.sqrt(np.add.outer(self.ky**2,self.kx**2)).ravel()
        self.kxm = np.ma.array(self.kx, mask= self.kx==0)
        self.kym = np.ma.array(self.ky, mask= self.ky==0)
        self.kzm = np.ma.array(self.kz, mask= self.kz==0)

        complex1 = 0+1j
        ## zs should be depth or coordinate?
        self.C = -8*np.pi*G*SI2MGAL
        self.W = np.exp(self.kz*self.zp[0])
        if gtype == 'zz':
            self.W = self.kz * self.W
        self.dW =((np.exp(-np.outer(self.kz,zs[1:]))
                  -np.exp(-np.outer(self.kz,zs[:-1])))/self.kzm.reshape(-1,1))
        self.dW[self.kzm.mask,:] = 0.
        self.dW[self.kzm.mask,:] += zs[:-1] - zs[1:]
        self.dW = self.dW.data
        self.WX = np.exp(complex1*(XXmin-XXmax+xmax+xmin)*self.kx/2.)/dx
        #self.WX = np.ones_like(self.kx)/dx
        if gtype == 'zx':
            self.WX = complex1 * self.kx
        self.FX = np.sin(np.outer(self.kx,a))/self.kxm.reshape(-1,1)
        self.FX[self.kxm.mask,:] = 0.
        self.FX[self.kxm.mask,:] += a
        self.FX = self.FX * np.exp(-complex1*np.outer(self.kx,x0)) * self.WX.reshape(-1,1)
        self.FX = self.FX.data
        self.WY = np.exp(complex1*(YYmin-YYmax+ymax+ymin)*self.ky/2.)/dy
#        self.WY = np.ones_like(self.ky)/dy
        if gtype == 'zy':
            self.WY = complex1 * self.ky
        self.FY = np.sin(np.outer(self.ky,b))/self.kym.reshape(-1,1)
        self.FY[self.kym.mask,:] = 0.
        self.FY[self.kym.mask,:] += b
        self.FY = self.FY * np.exp(-complex1*np.outer(self.ky,y0)) * self.WY.reshape(-1,1)
        self.FY = self.FY.data
        self.solve_prepared = False

    def _gen_rhs(self):
        # default pad 0s
        self.rhs = self.C*self.W*self.obs_freq.T.ravel()
        self.rhs = self.dW.T*self.rhs
#        self.rhs = kron_matvec([np.eye(self.nz),
#                                np.conj(self.FY.T),
#                                np.conj(self.FX.T)],
#                               self.rhs.ravel())
        self.rhs = kron_matvec([np.conj(self.FY.T),
                                np.conj(self.FX.T)],
                               self.rhs.reshape(self.nz,-1)).ravel()

    def forward(self,v=None,update=False):
        ''' Forward modelling gravity and its frequency representation
        Args:
            v (ndarray): model density, if not given, use self.model_density
            update (bool): If False, self.freq and self.obs_field won't be touched.
                           If True, self.freq and self.obs_field will be updated.
        Returns:
            obs_field (ndarray): gravity.
            freq (ndarray): gravity in frequency domain.
        '''
        if v is None:
            v = self._model_density
        freq = kron_matvec([self.FY,self.FX],v.reshape(self.nz,-1))
        freq = freq * self.dW.T
        freq = self.W * np.sum(freq,axis=0)
        freq = self.C * freq.reshape(self.padded.shape[1],self.padded.shape[0]).T
        obs_field = np.real(np.fft.ifft2(freq))
        obs_field = obs_field[self.padx: self.padx + self.nobsx, self.pady: self.pady + self.nobsy].ravel()
        if update:
            self.freq = freq
            self.obs_field = obs_field
        return freq,obs_field

    def _prepare_solve(self):
        if self.solve_prepared:
            return
        tmpX = self.FX @ np.conj(self.FX.T)
        tmpY = self.FY@np.conj(self.FY.T)
        tmp = np.kron(tmpY,tmpX)
        self.woodSUB = np.zeros_like(tmp)
        for iz in range(self.nz):
            self.woodSUB += self.dW[:,iz].reshape(-1,1)*tmp*np.conj(self.dW[:,iz].reshape(1,-1))
        self.solve_prepared = True

    def do_linear_solve_quiet(self):
        self._gen_rhs()
        if not self.solve_prepared:
            self._prepare_solve()
        weight = np.sum(self._weights['refers'])
        invSUB = self.C**2*weight*self.woodSUB
        invSUB_diag = np.einsum('ii->i',invSUB)
        invSUB_diag += weight**2 * 1./self.W**2
        invSUB = np.linalg.inv(invSUB)
        self.solution = kron_matvec([self.FY,self.FX],self.rhs.reshape(self.nz,-1))
        self.solution = np.sum(self.solution * self.dW.T, axis=0)
        self.solution = invSUB @ self.solution
        self.solution = self.dW.T*self.solution.reshape(1,-1)
        self.solution = kron_matvec([np.conj(self.FY.T),np.conj(self.FX.T)],
                                    self.solution.reshape(self.nz,-1)).ravel()
        self.solution = self.rhs/weight - self.C**2*self.solution

    def calc_res(self):
        pass

###
def gz(xp, yp, zp, prisms, dens=None):
    """
    Calculates the :math:`g_z` gravity acceleration component.

    .. note:: The coordinate system of the input parameters is to be
        x -> North, y -> East and z -> **DOWN**.

    .. note:: All input values in **SI** units(!) and output in **mGal**!

    Parameters:

    * xp, yp, zp : arrays
        Arrays with the x, y, and z coordinates of the computation points.
    * prisms : list of :class:`~geoist.mesher.Prism`
        The density model used to calculate the gravitational effect.
        Prisms must have the property ``'density'``. Prisms that don't have
        this property will be ignored in the computations. Elements of *prisms*
        that are None will also be ignored. *prisms* can also be a
        :class:`~geoist.mesher.PrismMesh`.
    * dens : float or None
        If not None, will use this value instead of the ``'density'`` property
        of the prisms. Use this, e.g., for sensitivity matrix building.

    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """
    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        _prism.gz(xp, yp, zp, x1, x2, y1, y2, z1, z2, density, res)
    res *= G * SI2MGAL
    return res

def gzfreq(xp, yp, zp, shape, prisms, dens=None, gtype = 'z'):
    """
    Calculates the :math:`g_z` gravity acceleration component in frequency domain.
    Returns:

    * res : array
        The field calculated on xp, yp, zp

    """

    if xp.shape != yp.shape or xp.shape != zp.shape:
        raise ValueError("Input arrays xp, yp, and zp must have same length!")
    size = len(xp)
    res = numpy.zeros(size, dtype=numpy.float)
    for prism in prisms:
        if prism is None or ('density' not in prism.props and dens is None):
            continue
        if dens is None:
            density = prism.props['density']
        else:
            density = dens
        x1, x2 = prism.x1, prism.x2
        y1, y2 = prism.y1, prism.y2
        z1, z2 = prism.z1, prism.z2
        r1 = _gzfreq(xp, yp, zp, x1, x2, y1, y2, z1, z2, shape, gtype)
        res = res + r1 *2*numpy.pi*G *density* SI2MGAL
    #res *=
    return res

def _gzfreq(x, y, data, x1, x2, y1, y2, z1, z2, shape, gtype):

    x0 = (x1+x2)/2.0
    y0 = (y1+y2)/2.0
    a = numpy.abs(x1-x2)/2.0
    b = numpy.abs(y1-y2)/2.0
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    dy = (y.max() - y.min())/(ny - 1)

    nx, ny = shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, shape)
    kx, ky = _fftfreqs(x, y, shape, padded.shape)
    kz = numpy.sqrt(kx**2 + ky**2)
    kxm = numpy.ma.array(kx, mask= kx==0)
    kym = numpy.ma.array(ky, mask= ky==0)
    kzm = numpy.ma.array(kz, mask= kz==0)

    complex1 = 0+1j
    ker1 = (numpy.exp(-kz*z2)-numpy.exp(-kz*z1))/kzm
    ker2 = numpy.sin(ky*b)/kym
    ker3 = numpy.sin(kx*a)/kxm
    keruv = -4*ker1*ker2*ker3

    keruv[kxm.mask] = -4*a*numpy.sin(ky[kxm.mask]*b)*ker1[kxm.mask]/ky[kxm.mask]
    keruv[kym.mask] = -4*b*numpy.sin(kx[kym.mask]*a)*ker1[kym.mask]/kx[kym.mask]
    keruv[kzm.mask] = 4*a*b*(z2-z1)
    nxe, nye = padded.shape

    M_left=(nxe-nx)/2+1
    M_right=M_left+nx-1
    N_down=(nye-ny)/2+1
    N_up=N_down+ny-1

    XXmin=x.min()-dx*(M_left-1)
    XXmax=x.max()+dx*(nxe-M_right)
    YYmin=y.min()-dy*(N_down-1)
    YYmax=y.max()+dy*(nye-N_up)

    keruv = keruv*numpy.exp(-ky*y0*complex1)*numpy.exp(-kx*x0*complex1)*numpy.exp(kz*data[0])
    ## scale transform, can be transformed before inversion
    keruv = keruv*numpy.exp(complex1*((x.max()+x.min())*kx/2+(y.max()+y.min())*ky/2))*numpy.exp(complex1*((XXmin-XXmax)*kx/2+(YYmin-YYmax)*ky/2))/dx/dy
    if gtype == 'zz':
        keruv = kz * keruv
    elif gtype == 'zx':
        keruv = complex1 * kx * keruv
    elif gtype == 'zy':
        keruv = complex1 * ky * keruv

    res = numpy.real(numpy.fft.ifft2(keruv))
    res0 = res[padx: padx + nx, pady: pady + ny].ravel()
    return res0

def _fftfreqs(x, y, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    dx = (x.max() - x.min())/(nx - 1)
    fx = 2*numpy.pi*numpy.fft.fftfreq(padshape[0], dx)
    dy = (y.max() - y.min())/(ny - 1)
    fy = 2*numpy.pi*numpy.fft.fftfreq(padshape[1], dy)
    return numpy.meshgrid(fy, fx)[::-1]

def _pad_data(data, shape):
    n = _nextpow2(numpy.max(shape))
    nx, ny = shape
    padx = (n - nx)//2
    pady = (n - ny)//2
    padded = numpy.pad(data.reshape(shape), ((padx, padx), (pady, pady)),
                       mode='edge')
    return padded, padx, pady

def _nextpow2(i):
    buf = numpy.ceil(numpy.log(i)/numpy.log(2))
    return int(2**buf)

def _check_area(area):
    """
    Check that the area argument is valid.
    For example, the west limit should not be greater than the east limit.
    """
    x1, x2, y1, y2 = area
    assert x1 <= x2, \
        "Invalid area dimensions {}, {}. x1 must be < x2.".format(x1, x2)
    assert y1 <= y2, \
        "Invalid area dimensions {}, {}. y1 must be < y2.".format(y1, y2)

def regular(area, shape, z=None):
    """
    Create a regular grid.

    The x directions is North-South and y East-West. Imagine the grid as a
    matrix with x varying in the lines and y in columns.

    Returned arrays will be flattened to 1D with ``numpy.ravel``.

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.
    * z
        Optional. z coordinate of the grid points. If given, will return an
        array with the value *z*.

    Returns:

    * ``[x, y]``
        Numpy arrays with the x and y coordinates of the grid points
    * ``[x, y, z]``
        If *z* given. Numpy arrays with the x, y, and z coordinates of the grid
        points

    Examples:

    >>> x, y = regular((0, 10, 0, 5), (5, 3))
    >>> print(x)
    [  0.    0.    0.    2.5   2.5   2.5   5.    5.    5.    7.5   7.5   7.5
      10.   10.   10. ]
    >>> print(x.reshape((5, 3)))
    [[  0.    0.    0. ]
     [  2.5   2.5   2.5]
     [  5.    5.    5. ]
     [  7.5   7.5   7.5]
     [ 10.   10.   10. ]]

    """
    nx, ny = shape
    x1, x2, y1, y2 = area
    _check_area(area)
    xs = np.linspace(x1, x2, nx)
    ys = np.linspace(y1, y2, ny)
    # Must pass ys, xs in this order because meshgrid uses the first argument
    # for the columns
    arrays = np.meshgrid(ys, xs)[::-1]
    if z is not None:
        arrays.append(z*np.ones(nx*ny, dtype=np.float))
    return [i.ravel() for i in arrays]

def spacing(area, shape):
    """
    Returns the spacing between grid nodes

    Parameters:

    * area
        ``(x1, x2, y1, y2)``: Borders of the grid
    * shape
        Shape of the regular grid, ie ``(nx, ny)``.

    Returns:

    * ``[dx, dy]``
        Spacing the y and x directions

    Examples:

    >>> print(spacing((0, 10, 0, 20), (11, 11)))
    [1.0, 2.0]
    >>> print(spacing((0, 10, 0, 20), (11, 21)))
    [1.0, 1.0]
    >>> print(spacing((0, 10, 0, 20), (5, 21)))
    [2.5, 1.0]
    >>> print(spacing((0, 10, 0, 20), (21, 21)))
    [0.5, 1.0]

    """
    x1, x2, y1, y2 = area
    nx, ny = shape
    dx = (x2 - x1)/(nx - 1)
    dy = (y2 - y1)/(ny - 1)
    return [dx, dy]

# class GridInfo(properties.HasProperties):
    # """Internal helper class to store Surfer grid properties and create
    # ``vtkImageData`` objects from them.
    # """
    # ny = properties.Integer('number of columns', min=2)
    # nx = properties.Integer('number of rows', min=2)
    # xll = properties.Float('x-value of lower-left corner')
    # yll = properties.Float('y-value of lower-left corner')
    # dx = properties.Float('x-axis spacing')
    # dy = properties.Float('y-axis spacing')
    # dmin = properties.Float('minimum data value', required=False)
    # dmax = properties.Float('maximum data value', required=False)
    # data = properties.Array('grid of data values', shape=('*',))

    # def mask(self):
        # """Mask the no data value"""
        # data = self.data
        # nans = data >= 1.701410009187828e+38
        # if np.any(nans):
            # data = np.ma.masked_where(nans, data)
        # err_msg = "{} of data ({}) doesn't match that set by file ({})."
        # if not np.allclose(self.dmin, np.nanmin(data)):
            # raise ValueError(err_msg.format('Min', np.nanmin(data), self.dmin))
        # if not np.allclose(self.dmax, np.nanmax(data)):
            # raise ValueError(err_msg.format('Max', np.nanmax(data), self.dmax))
        # self.data = data
        # return

    # def to_vtk(self, output=None, z=0.0, dz=1.0, data_name='Data'):
    #     """Convert to a ``vtkImageData`` object"""
    #     self.mask()
    #     self.validate()
    #     if output is None:
    #         output = vtk.vtkImageData()
    #     # Build the data object
    #     output.SetOrigin(self.xll, self.yll, z)
    #     output.SetSpacing(self.dx, self.dy, dz)
    #     output.SetDimensions(self.nx, self.ny, 1)
    #     vtkarr = interface.convert_array(self.data, name=data_name)
    #     output.GetPointData().AddArray(vtkarr)
    #     return output
    
class grddata(object):
    """
    Grid Data Object
    Attributes
    ----------
    data : numpy masked array
        array to contain raster data
    xmin : float
        min value X coordinate of raster grid
    ymin : float
        min value Y coordinate of raster grid
    xdim : float
        x-dimension of grid cell
    ydim : float
        y-dimension of grid cell
    typeofdata : int
        number of datatype
    dataname : str
        data name or id
    rows : int
        number of rows for each raster grid/band
    cols : int
        number of columns for each raster grid/band
    nullvalue : float
        grid null or nodata value
    norm : dictionary
        normalized data
    gtr : tuple
        projection information
    wkt : str
        projection information
    units : str
        description of units to be used with color bars
    """
    def __init__(self):
        self.data = np.ma.array([])
        self.data0 = np.array([])
        self.xmin = 0.0  # min value of X coordinate
        self.ymin = 0.0  # min value of Y coordinate
        self.xdim = 1.0
        self.ydim = 1.0
        self.dmin = 0.0
        self.dmax = 0.0
        self.typeofdata = 1 # 1- grav or 2- mag
        self.dataname = '' #name of data
        self.rows = -1
        self.cols = -1
        self.nullvalue = 1e+20
        self.norm = {}
        self.gtr = (0.0, 1.0, 0.0, 0.0, -1.0)
        self.wkt = ''
        self.units = ''

    def fill_nulls(self, method='nearest'):
        """
            Fill in the NaNs or masked values on interpolated points using nearest
            neighbors.
            method='nearest' or 'linear' or 'cubic'
        """
        if np.ma.is_masked(self.data):
            nans = self.data.mask
        else:
            nans = np.isnan(self.data)

        nx,ny = nans.shape
        ns = nans.reshape(nx*ny)
        shape = (nx, ny)
        xmax = self.xmin + (self.cols-1)*self.xdim
        ymax = self.ymin + (self.rows-1)*self.ydim
        area = (self.xmin, xmax, self.ymin, ymax)
        x, y = regular(area, shape)
        dtmp = self.data.copy() #数组copy，不改变源数组
        dtmp1 = dtmp.reshape(nx*ny)
        ns1 = (ns == False)
        dtmp1[ns] = interp.griddata((x[ns1], y[ns1]), dtmp1[ns1], (x[ns], y[ns]),
                                    method).ravel()
        self.data0 = dtmp1.reshape(nx,ny)

    def grd2xyz(self, flag = True):
        """
        Return x,y,z 1-D array data from 2-D grid array.

        Parameters:
          flag  : True  -  Output Grid Grid
                False -  Output Bak Grid Grid
        Returns:
          x,y,z 1-D array data
        """
        nx,ny = self.data.shape
        xmax = self.xmin + (self.cols-1)*self.xdim
        ymax = self.ymin + (self.rows-1)*self.ydim

        shape = (nx, ny)
        area = (self.xmin, xmax, self.ymin, ymax)
        x, y = regular(area, shape)
        if flag:
          z = self.data.reshape(nx*ny)
        else:
          z = self.data0.reshape(nx*ny)
        return (x, y, z)


    def load_grd(self,fname,*args,**kwargs):
        with open(fname,'rb') as f:
            tmp = f.read(4)
        if tmp == b'DSAA':
            self._load_surfer_ascii(fname,*args,**kwargs)
        elif tmp == b'DSBB':
            self._load_surfer_dsbb(fname,*args,**kwargs)
        elif tmp == b'ncol':
            self.load_ascii(fname,*args,**kwargs)
        else:
            raise ValueError("Unrecognized grd format.")

    def load_surfer(self, fname, *args, **kwargs):
        """
        Read data from a Surfer grid file.

        Parameters:

        * fname : str
            Name of the Surfer grid file
        * dtype : numpy dtype object or string
            The type of variable used for the data. Default is numpy.float64 for
            ascii data and is '=f' for binary data. Use numpy.float32 if the
            data are large and precision is not an issue.
        * header_format : header format (excluding the leading 'DSBB') following
            the convention of the struct module. Only used for binary data.

        Returns:

        """
        with open(fname,'rb') as f:
            tmp = f.read(4)
        if tmp == b'DSAA':
            self._load_surfer_ascii(fname,*args,**kwargs)
        elif tmp == b'DSBB':
            self._load_surfer_dsbb(fname,*args,**kwargs)
        else:
            raise ValueError("Unknown header info {}.".format(tmp)
                            +"Only DSAA or DSBB could be recognized.")

    def _load_surfer_dsbb(self,fname,dtype='=f',header_format='cccchhdddddd'):
        """
        Read data from a Surfer DSBB grid file.

        Parameters:

        * fname : str
            Name of the Surfer grid file
        * dtype : numpy dtype object or string
            The type of variable used for the data. Default is numpy.float64. Use
            numpy.float32 if the data are large and precision is not an issue.
        * header_format : header format following the convention of the
            struct module.

        Returns:

        """
        with open(fname,'rb') as f:
            # read header
            header_len = struct.calcsize(header_format)
            header = f.read(header_len)
            # read data
            data = b''
            for x in f:
                data += x

        # unpack header
        s = struct.Struct(header_format)
        (tmp,tmp,tmp,tmp,self.cols,self.rows,self.xmin,self.xmax,
         self.ymin,self.ymax,self.dmin,self.dmax) = s.unpack(header)
        if self.cols<=0 and self.rows<=0:
            raise ValueError("Array shape can't be infered.")

        # convert data to numpy array
        self.data = np.frombuffer(data,dtype=dtype).reshape(self.cols,self.rows)
        self.data = np.ma.MaskedArray(self.data)
        self.cols,self.rows = self.data.shape
        if self.data.min()+1<self.dmin or self.data.max()-1>self.dmax:
            warnings.warn("(min(z),max(z)) in the data is incompatible "
                          +"with (zmin,zmax) in the header. "
                          +"Please check whether the 'dtype' argument is "
                          +"correct.(default is '=f')")
        self.xdim = (self.xmax-self.xmin)/(self.rows-1)
        self.ydim = (self.ymax-self.ymin)/(self.cols-1)


    def _load_surfer_ascii(self, fname, dtype='float64'):
        """
        Read data from a Surfer ASCII grid file.

        Parameters:

        * fname : str
            Name of the Surfer grid file
        * dtype : numpy dtype object or string
            The type of variable used for the data. Default is numpy.float64. Use
            numpy.float32 if the data are large and precision is not an issue.

        Returns:

        """
        # Surfer ASCII grid structure
        # DSAA            Surfer ASCII GRD ID
        # nCols nRows     number of columns and rows
        # xMin xMax       X min max
        # yMin yMax       Y min max
        # zMin zMax       Z min max
        # z11 z21 z31 ... List of Z values
        with open(fname) as input_file:
            # DSAA is a Surfer ASCII GRD ID (discard it for now)
            input_file.readline()
            # Read the number of columns (ny) and rows (nx)
            ny, nx = [int(s) for s in input_file.readline().split()]
            #shape = (nx, ny)
            # Our x points North, so the first thing we read is y, not x.
            ymin, ymax = [float(s) for s in input_file.readline().split()]
            xmin, xmax = [float(s) for s in input_file.readline().split()]
            #area = (xmin, xmax, ymin, ymax)
            dmin, dmax = [float(s) for s in input_file.readline().split()]
            field = np.fromiter((float(s)
                                 for line in input_file
                                 for s in line.split()),
                                dtype=dtype)
            nans = field >= 1.70141e+38
            if np.any(nans):
                field = np.ma.masked_where(nans, field)
            #err_msg = "{} of data ({}) doesn't match one from file ({})."
            if dmin != field.min():
                dmin = field.min()
            if dmax != field.max():
                dmax = field.max()
#            assert np.allclose(dmin, field.min()), err_msg.format('Min', dmin,
#                                                                  field.min())
#            assert np.allclose(dmax, field.max()), err_msg.format('Max', dmax,
#                                                                  field.max())
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            self.xdim = (xmax-xmin)/(nx-1)
            self.ydim = (ymax-ymin)/(ny-1)
            self.dmin = dmin
            self.dmax = dmax
            self.cols = ny
            self.rows = nx
            self.nullvalue = 1.701410009187828e+38
            self.data = np.ma.masked_equal(field.reshape(nx,ny), self.nullvalue)
        #x, y = gridder.regular(area, shape)
        #data = dict(file=fname, shape=shape, area=area, data=field, x=x, y=y)
        #return data

    @staticmethod
    def _surfer7bin(filename):
        """See class notes.
        """
        with open(filename, 'rb') as f:
            if unpack('4s', f.read(4))[0] != b'DSRB':
                raise ValueError(
                    '''Invalid file identifier for Surfer 7 binary .grd
                    file. First 4 characters must be DSRB.'''
                )
            f.read(8)  #Size & Version

            section = unpack('4s', f.read(4))[0]
            if section != b'GRID':
                raise ValueError(
                    '''Unsupported Surfer 7 file structure. GRID keyword
                    must follow immediately after header but {}
                    encountered.'''.format(section)
                )
            size = unpack('<i', f.read(4))[0]
            if size != 72:
                raise ValueError(
                    '''Surfer 7 GRID section is unrecognized size. Expected
                    72 but encountered {}'''.format(size)
                )
            nrow = unpack('<i', f.read(4))[0]
            ncol = unpack('<i', f.read(4))[0]
            x0 = unpack('<d', f.read(8))[0]
            y0 = unpack('<d', f.read(8))[0]
            deltax = unpack('<d', f.read(8))[0]
            deltay = unpack('<d', f.read(8))[0]
            zmin = unpack('<d', f.read(8))[0]
            zmax = unpack('<d', f.read(8))[0]
            rot = unpack('<d', f.read(8))[0]
            if rot != 0:
                warnings.warn('Unsupported feature: Rotation != 0')
            blankval = unpack('<d', f.read(8))[0]

            section = unpack('4s', f.read(4))[0]
            if section != b'DATA':
                raise ValueError(
                    '''Unsupported Surfer 7 file structure. DATA keyword
                    must follow immediately after GRID section but {}
                    encountered.'''.format(section)
                )
            datalen = unpack('<i', f.read(4))[0]
            if datalen != ncol*nrow*8:
                raise ValueError(
                    '''Surfer 7 DATA size does not match expected size from
                    columns and rows. Expected {} but encountered
                    {}'''.format(ncol*nrow*8, datalen)
                )
            data = np.zeros(ncol*nrow)
            for i in range(ncol*nrow):
                data[i] = unpack('<d', f.read(8))[0]
            data = np.where(data >= blankval, np.nan, data)

            try:
                section = unpack('4s', f.read(4))[0]
                if section == b'FLTI':
                    warnings.warn('Unsupported feature: Fault Info')
                else:
                    warnings.warn('Unrecognized keyword: {}'.format(section))
                warnings.warn('Remainder of file ignored')
            except:
                pass

        grd = GridInfo(
            nx=ncol,
            ny=nrow,
            xll=x0,
            yll=y0,
            dx=deltax,
            dy=deltay,
            dmin=zmin,
            dmax=zmax,
            data=data
        )
        return grd

    @staticmethod
    def _surfer6bin(filename):
        """See class notes.
        """
        with open(filename, 'rb') as f:
            if unpack('4s', f.read(4))[0] != b'DSBB':
                raise ValueError(
                    '''Invalid file identifier for Surfer 6 binary .grd
                    file. First 4 characters must be DSBB.'''
                )
            nx = unpack('<h', f.read(2))[0]
            ny = unpack('<h', f.read(2))[0]
            xlo = unpack('<d', f.read(8))[0]
            xhi = unpack('<d', f.read(8))[0]
            ylo = unpack('<d', f.read(8))[0]
            yhi = unpack('<d', f.read(8))[0]
            dmin = unpack('<d', f.read(8))[0]
            dmax = unpack('<d', f.read(8))[0]
            data = np.ones(nx * ny)
            for i in range(nx * ny):
                zdata = unpack('<f', f.read(4))[0]
                if zdata >= 1.701410009187828e+38:
                    data[i] = np.nan
                else:
                    data[i] = zdata

        grd = GridInfo(
            nx=nx,
            ny=ny,
            xll=xlo,
            yll=ylo,
            dx=(xhi-xlo)/(nx-1),
            dy=(yhi-ylo)/(ny-1),
            dmin=dmin,
            dmax=dmax,
            data=data
        )
        return grd

    @staticmethod
    def _surfer6ascii(filename):
        """See class notes.
        """
        with open(filename, 'r') as f:
            if f.readline().strip() != 'DSAA':
                raise ValueError('''Invalid file identifier for Surfer 6 ASCII .grd file. First line must be DSAA''')
            [ncol, nrow] = [int(n) for n in f.readline().split()]
            [xmin, xmax] = [float(n) for n in f.readline().split()]
            [ymin, ymax] = [float(n) for n in f.readline().split()]
            [dmin, dmax] = [float(n) for n in f.readline().split()]
            # Read in the rest of the file as a 1D array
            data = np.fromiter((np.float(s) for line in f for s in line.split()), dtype=float)

        grd = GridInfo(
            nx=ncol,
            ny=nrow,
            xll=xmin,
            yll=ymin,
            dx=(xmax-xmin)/(ncol-1),
            dy=(ymax-ymin)/(nrow-1),
            dmin=dmin,
            dmax=dmax,
            data=data
        )
        return grd


    def _read_grids(self, idx=None):
        """This parses the first file to determine grid file type then reads
        all files set."""
        if idx is not None:
            filenames = [self.get_file_names(idx=idx)]
        else:
            filenames = self.get_file_names()
        contents = []
        f = open(filenames[0], 'rb')
        key = unpack('4s', f.read(4))[0]
        f.close()
        if key == b'DSRB':
            reader = self._surfer7bin
        elif key == b'DSBB':
            reader = self._surfer6bin
        elif key == b'DSAA':
            reader = self._surfer6ascii
        else:
            raise ValueError('''Invalid file identifier for Surfer .grd file.
            First 4 characters must be DSRB, DSBB, or DSAA. This file contains: %s''' % key)

        for f in filenames:
            try:
                contents.append(reader(f))
            except (IOError, OSError) as fe:
                raise IOError(str(fe))
        if idx is not None:
            return contents[0]
        return contents



    def export_surfer(self, fname, flag = True ,file_format='binary'):
        """
        Export a surfer grid

        Parameters
        ----------
        fname : filename of grid dataset to export
        flag  : True  -  Output Grid Grid
                False -  Output Bak Grid Grid
        file_format : binary/b - output binary format
                      ascii/a - output ascii format
        """
        if file_format == 'binary' or file_format == 'b':
            self._export_surfer_binary(fname,flag)
        elif file_format == 'ascii' or file_format == 'a':
            self._export_surfer_ascii(fname,flag)

    def _export_surfer_ascii(self, fname, flag = True):
        """
        Export a surfer binary grid

        Parameters
        ----------
        fname : filename of grid dataset to export
        flag  : True  -  Output Grid Grid
                False -  Output Bak Grid Grid
        """
        xmax = self.xmin + (self.cols-1)*self.xdim
        ymax = self.ymin + (self.rows-1)*self.ydim
        with open(fname,'w') as fno:
            fno.write('DSAA\n')
            fno.write('{} {}\n'.format(self.cols,self.rows))
            fno.write('{} {}\n'.format(self.xmin,xmax))
            fno.write('{} {}\n'.format(self.ymin,ymax))
            if flag:
                fno.write('{} {}\n'.format(np.min(self.data),
                                           np.max(self.data))
                          )
                ntmp = 1.701410009187828e+38
                tmp = self.data.astype('f')
                tmp = tmp.filled(ntmp)
            else:
                fno.write('{} {}\n'.format(np.min(self.data0),
                                           np.max(self.data0))
                          )
                tmp = self.data0.astype('f')
            np.savetxt(fno,tmp)

    def _export_surfer_binary(self, fname, flag = True):
        """
        Export a surfer binary grid

        Parameters
        ----------
        fname : filename of grid dataset to export
        flag  : True  -  Output Grid Grid
                False -  Output Bak Grid Grid
        """
        fno = open(fname, 'wb')
        xmax = self.xmin + (self.cols-1)*self.xdim
        ymax = self.ymin + (self.rows-1)*self.ydim
        if flag:
            bintmp = struct.pack('cccchhdddddd', b'D', b'S', b'B', b'B',
                             self.cols, self.rows,
                             self.xmin, xmax,
                             self.ymin, ymax,
                             np.min(self.data),
                             np.max(self.data))
            fno.write(bintmp)
            ntmp = 1.701410009187828e+38
            tmp = self.data.astype('f')
            tmp = tmp.filled(ntmp)
        else:
            bintmp = struct.pack('cccchhdddddd', b'D', b'S', b'B', b'B',
                             self.cols, self.rows,
                             self.xmin, xmax,
                             self.ymin, ymax,
                             np.min(self.data0),
                             np.max(self.data0))
            fno.write(bintmp)
            tmp = self.data0.astype('f')
        #tmp = tmp[::-1]
        fno.write(tmp.tostring())
        fno.close()


    def export_ascii(self, fname):
        """
        Export Ascii file

        Parameters
        ----------
        data : grid Data
            dataset to export
        """
        fno = open(fname, 'w')

        fno.write("ncols \t\t\t" + str(self.cols))
        fno.write("\nnrows \t\t\t" + str(self.rows))
        fno.write("\nxllcorner \t\t\t" + str(self.xmin))
        fno.write("\nyllcorner \t\t\t" + str(self.ymin))
        fno.write("\ncellsize \t\t\t" + str(self.xdim))
        fno.write("\nnodata_value \t\t" + str(self.nullvalue))

        tmp = self.data.filled(self.nullvalue)

        for j in range(self.rows):
            fno.write("\n")
            for i in range(self.cols):
                fno.write(str(tmp[j, i]) + " ")

        fno.close()

    def load_ascii(self,fname,dtype='float64'):
        """
        Load Ascii file

        Parameters
        ----------
        data : grid Data
            dataset to export
        """
        with open(fname) as fno:
            tmp = fno.readline().strip().split()
            self.cols = int(tmp[1])
            tmp = fno.readline().strip().split()
            self.rows = int(tmp[1])
            tmp = fno.readline().strip().split()
            self.xmin = float(tmp[1])
            tmp = fno.readline().strip().split()
            self.ymin = float(tmp[1])
            tmp = fno.readline().strip().split()
            self.xdim = float(tmp[1])
            tmp = fno.readline().strip().split()
            self.nullvalue = float(tmp[1])
            field = np.fromiter((float(s)
                                 for line in fno
                                 for s in line.strip().split()),
                                dtype=dtype)

        self.ydim = self.xdim
        self.dmin = field.min()
        self.dmax = field.max()
        self.xmax = self.xmin + self.xdim*(self.rows-1)
        self.ymax = self.ymin + self.ydim*(self.cols-1)
        self.data = np.ma.masked_equal(field.reshape(self.cols,self.rows),
                                       self.nullvalue)