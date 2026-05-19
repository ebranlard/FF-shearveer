import pickle
import numpy as np
import glob
import os
import matplotlib

# welib
from welib.weio.vtk_file import VTKFile
from welib.tools.clean_exceptions import *

# --------------------------------------------------------------------------------}
# --- Plotting 
# --------------------------------------------------------------------------------{
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['axes.labelsize']  = 15 
matplotlib.rcParams['axes.titlesize']  = 16
matplotlib.rcParams['legend.fontsize'] = 14
MathematicaBlue       = np.array([63 ,63 ,153 ])/255.;
MathematicaRed        = np.array([153,61 ,113 ])/255.;
MathematicaGreen      = np.array([61 ,153,86  ])/255.;
MW_Orange   =     np.array([218,124,48])/255.


cLES=MathematicaRed
cCurl=MathematicaBlue
# cCart=MathematicaGreen
cCart=MW_Orange
# # cPolr='k'
cPolr=MathematicaGreen
# cLES='k'

# --------------------------------------------------------------------------------}
# --- Saving Dictionaries
# --------------------------------------------------------------------------------{
def save_dict(filename, d):
    with open(filename, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(filename):
    with open(filename, 'rb') as handle:
        d = pickle.load(handle)
    return d


# def colorbar(mappable):
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     ax = mappable.axes
#     fig = ax.figure
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     return fig.colorbar(mappable, cax=cax)





# --------------------------------------------------------------------------------}
# --- FASTFarm 
# --------------------------------------------------------------------------------{
def vtkIndices(folder, base):
    """ """
    pattern = os.path.join(folder,'vtk_ff', base+'*.vtk')
    files = glob.glob(pattern)
    if len(files)==0:
        raise Exception('No file found with pattern',pattern)
    sIndices = np.array([ os.path.splitext(os.path.splitext(f)[0])[1][1:] for f in files ])
    n = len(sIndices[0]) # Number of characters used for time stamp
    Indices = sIndices.astype(int)
    iMax= np.max(Indices)
    Indices = np.sort(Indices)
    return Indices, iMax, n


def extractPlane(kind, folder, base, iPlane=1, iTime=None, removeBoundaries=None, U0=1, D=1, verbose=False, n=None):
    """ 

    kind: 'XY', 'XZ', or 'YZ'

    OPTIONAL INPUTS:
     - U0: if provided, used to scale the velocity field
     - D:  if provided, used to scale the coordinates
     - removeBoundaries: number of index to remove on the boundaries

    """
    vtkfiles = glob.glob(os.path.join(folder, 'vtk_ff', '*.vtk'))
    # Extract the last time index
    sIndices = np.array([os.path.splitext(os.path.splitext(f)[0])[1][1:] for f in vtkfiles ])
    Indices = sIndices.astype(int)
    if iTime is None:
        iTime= np.max(Indices)
    iTime= int(iTime)
    # Return plane data
    if n==None:
        # Extract number of digits
        n=len(sIndices[0])
    sFmt = '{:0'+str(n)+'d}'
    vtkFile = os.path.join(folder, 'vtk_ff', base + '.Low.Dis{}{:03d}.'.format(kind, iPlane)+sFmt.format(iTime)+'.vtk')

    if verbose:
        print('Reading: ',vtkFile)
    vtkc = VTKFile(vtkFile)
    x  = vtkc.xp_grid 
    if kind=='YZ':
        y  = vtkc.yp_grid/D
        z  = vtkc.zp_grid/D
        Uc = vtkc.point_data_grid['Velocity']/U0
    else:
        raise NotImplementedError()
    U = Uc[0, :, :, 0] # ny x nz
    V = Uc[0, :, :, 1] # ny x nz
    W = Uc[0, :, :, 2] # ny x nz

    if removeBoundaries is not None:
        i=removeBoundaries
        y  = y[i:-i]
        z  = z[i:-i]
        U = U[i:-i, i:-i]
        V = V[i:-i, i:-i]
        W = W[i:-i, i:-i]

    if kind=='YZ':
        Y,Z = np.meshgrid(y,z)
        return Y,Z,U,V,W


def extractLines(X1, X2, x10, x20, U):
    """ Extract "horizontal" and "vertical" lines from a 2d field"""
    x1 = X1[0,:]
    x2 = X2[:,0]
    id1 = np.argmin(np.abs(x1-x10))
    id2 = np.argmin(np.abs(x2-x20))
    return x1, U[:, id2], x2, U[id1, :] # E.g.: y, U[:,idz], z, U[idy,:] 
