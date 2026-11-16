"""

This library includes the functions needed for the cartesian general implementation of finite differences to solve the wake dynamics of a wind turbine for one plane

"""
import numpy as np
import os
from scipy.sparse.linalg import cg
from scipy.sparse import diags, kron, identity
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import welib.weio as weio
from welib.weio.vtk_file import VTKFile
from welib.tools.figure import *


def setup(nr, nx, dr):
    p = dict()
    p['nr'] = nr
    p['nx'] = nx
    p['ny'] = 2*p['nr']-1                               # Total number of nodes in y-direction, the origin is in the middle (0,0)
    p['nz'] = 2*p['nr']-1                               # Total number of nodes in z-direction, the origin is in the middle (0,0)
    p['dr'] = dr
    p['dx'] = 4.6                                       # Spacing between planes in x-direction: 4.6 default value, should be calculated
    p['dy'] = p['dr']                                   # Spacing between nodes in y-direction
    p['dz'] = p['dr']                                   # Spacing between nodes in z-direction
    p['x'] = np.linspace(0, (p['nx']-1)*p['dx'], p['nx'])                           # x-value at the plane
    p['y'] = np.linspace(-(p['nr']-1)*p['dy'], (p['nr']-1)*p['dy'], p['ny'])        # y-value at the node
    p['z'] = np.linspace(-(p['nr']-1)*p['dz'], (p['nr']-1)*p['dz'], p['nz'])        # z-value at the node

    # Variables for Poisson solver: TODO These all can be discarded for the solve_poisson_fixedExt function
    p['domainExtension'] = int(300 // p['dy'])                                      # Domain node amount is extended in the +- y/z direction by this
    p['pad_size'] = int((p['ny'] + 2*p['domainExtension'])//2)                      # Zero padding to imitate unbounded domain when doing FFT
    p['variableDimFull'] = int(p['ny'] + 2*p['domainExtension'] + p['pad_size'])    # Actual matrix dimension when domain is extended and padded fully
    p['y_ext'] = np.linspace(-(p['nr']+p['domainExtension']-1)*p['dy'], (p['nr']+p['domainExtension']-1)*p['dy'], p['ny']+2*p['domainExtension'])
    p['z_ext'] = np.linspace(-(p['nr']+p['domainExtension']-1)*p['dz'], (p['nr']+p['domainExtension']-1)*p['dz'], p['nz']+2*p['domainExtension'])
    p['G'] = np.zeros((p['variableDimFull'], p['variableDimFull']))
    p['G'][:p['y_ext'].shape[0], :p['z_ext'].shape[0]] = Green_function(p['y_ext'].shape[0], p['z_ext'].shape[0], p['y_ext'], p['z_ext'])
    p['G_fft'] = np.zeros((p['variableDimFull'], p['variableDimFull']))
    p['G_fft'] = np.fft.fft2(p['G'])
    
    # --- Intermediate storage variables
    m = dict()                                          # Coeff. a,b,c,d,e are entries in the A matrix of the linear system, f will be the right hand side
    m['dvtdy'] = np.zeros((p['ny'], p['nz']))           # Gradient of eddy viscosity in y-direction
    m['dvtdz'] = np.zeros((p['ny'], p['nz']))           # Gradient of eddy viscosity in z-direction 
    m['vt_tot'] = np.zeros((p['ny'], p['nz']))          # Eddy viscosity
    m['a'] = np.zeros((p['ny'], p['nz']))               # m['a'] to m['f'] are matrices where the indices are j,k
    m['b'] = np.zeros((p['ny'], p['nz']))
    m['c'] = np.zeros((p['ny'], p['nz']))
    m['d'] = np.zeros((p['ny'], p['nz']))
    m['e'] = np.zeros((p['ny'], p['nz']))
    m['f'] = np.zeros((p['ny'], p['nz']))
    m['U'] = np.zeros((p['ny'], p['nz']))              # Background flow
    m['V'] = np.zeros((p['ny'], p['nz']))
    m['W'] = np.zeros((p['ny'], p['nz']))
    m['phi'] = np.zeros((p['variableDimFull'], p['variableDimFull']))#np.zeros((2 * p['ny'], 2 * p['nz']))     # Velocity potential
    m['phi_ext'] = np.zeros((p['variableDimFull'], p['variableDimFull']))#np.zeros((2 * p['ny'], 2 * p['nz']))       # Velocity potential in extended domain
    m['phi_fft'] = np.zeros((p['variableDimFull'], p['variableDimFull']))#np.zeros((2 * p['ny'], 2 * p['nz']))
    m['dVxdx'] = np.zeros((p['variableDimFull'], p['variableDimFull']))#np.zeros((2 * p['ny'], 2 * p['nz']))     # Derivative of Vx in x direction
    m['dVxdx_fft'] = np.zeros((p['variableDimFull'], p['variableDimFull']))#np.zeros((2 * p['ny'], 2 * p['nz']))

    # --- Main states
    xd = dict()
    xd['Vx'] = np.zeros((p['ny'], p['nz']))
    xd['Vy'] = np.zeros((p['ny'], p['nz']))
    xd['Vz'] = np.zeros((p['ny'], p['nz']))
    xd['dUdy'] = np.zeros((p['ny'], p['nz']))           # Gradient of background flow U(y,z)
    xd['dUdz'] = np.zeros((p['ny'], p['nz']))
    xd['dVdy'] = np.zeros((p['ny'], p['nz']))           # Gradient of background flow V(y,z)
    xd['dVdz'] = np.zeros((p['ny'], p['nz']))
    xd['dWdy'] = np.zeros((p['ny'], p['nz']))           # Gradient of background flow W(y,z)
    xd['dWdz'] = np.zeros((p['ny'], p['nz']))
    xd['d2Udy2'] = np.zeros((p['ny'], p['nz']))         # Second derivative of U(y,z)
    xd['d2Udz2'] = np.zeros((p['ny'], p['nz']))
    return p, m, xd

def simplify_notations(xd, m, p):       # This function makes life a litter easier when writing the FD equations
    Vx = xd['Vx']
    Vy = xd['Vy']
    Vz = xd['Vz']
    U = m['U']
    V = m['V']
    W = m['W']
    dx = p['dx']
    dy = p['dy']
    dz = p['dz']
    x = p['x']
    y = p['y']
    z = p['z']
    vt_tot = m['vt_tot']
    return Vx, Vy, Vz, U, V, W, dx, dy, dz, x, y, z, vt_tot

i = 0                                           # TODO will have to be changed once we loop through the planes
def build_A_matrix_plane_i(i, xd, m, p):        # This function builds the A-matrix (calculation of Vx) at one given plane i
    
    # Derivatives
    m['dvtdy'] = gradient_y(m['vt_tot'],p['dy'])
    m['dvtdz'] = gradient_z(m['vt_tot'],p['dz'])
    xd['dUdy'] = gradient_y(m['U'],p['dy'])
    xd['dUdz'] = gradient_z(m['U'],p['dz'])
    xd['d2Udy2'] = gradient_y2(m['U'],p['dy'])
    xd['d2Udz2'] = gradient_z2(m['U'],p['dz'])

    # Simplify notations
    Vx, Vy, Vz, U, V, W, dx, dy, dz, x, y, z, vt_tot = simplify_notations(xd, m, p)
    
    # Compute A matrix coefficients from finite difference at one time step i
    for k in range(0+1, p['nz']-1, 1):  # TODO BC if not using Dirichlet BC
        for j in range(0+1, p['ny']-1, 1):
            m['a'][j, k] = -1/(4*dy)*(V[j,k]+Vy[j,k]) + 1/(4*dy)*m['dvtdy'][j,k] - 1/(2*dy**2)*vt_tot[j,k]
            m['b'][j, k] = (U[j,k]+Vx[j,k])/dx + vt_tot[j,k]/(dy**2) + vt_tot[j,k]/(dz**2)
            m['c'][j, k] = 1/(4*dy)*(V[j,k]+Vy[j,k]) - 1/(4*dy)*m['dvtdy'][j,k] - vt_tot[j,k]/(2*dy**2)
            m['d'][j, k] = -1/(4*dz)*(W[j,k]+Vz[j,k]) + 1/(4*dz)*m['dvtdz'][j,k] - vt_tot[j,k]/(2*dz**2)
            m['e'][j, k] = 1/(4*dz)*(W[j,k]+Vz[j,k]) - 1/(4*dz)*m['dvtdz'][j,k] - vt_tot[j,k]/(2*dz**2)
            m['f'][j, k] = ((1/(4*dy)*(V[j,k]+Vy[j,k]) - 1/(4*dy)*m['dvtdy'][j, k] + vt_tot[j,k]/(2*dy**2)) * Vx[j-1,k]
                            +((U[j,k]+Vx[j,k])/dx - vt_tot[j,k]/(dy**2) - vt_tot[j,k]/(dz**2)) * Vx[j,k]
                            +(-1/(4*dy)*(V[j,k]+Vy[j,k]) + 1/(4*dy)*m['dvtdy'][j, k] + vt_tot[j,k]/(2*dy**2)) * Vx[j+1,k]
                            +(1/(4*dz)*(W[j,k]+Vz[j,k]) - 1/(4*dz)*m['dvtdz'][j, k] + vt_tot[j,k]/(2*dz**2)) * Vx[j,k-1]
                            +(-1/(4*dz)*(W[j,k]+Vz[j,k]) + 1/(4*dz)*m['dvtdz'][j, k] + vt_tot[j,k]/(2*dz**2)) * Vx[j,k+1]
                            +(-(V[j,k]+Vy[j,k]) + m['dvtdy'][j, k])*xd['dUdy'][j,k] + vt_tot[j,k]*xd['d2Udy2'][j,k]
                            +(-(W[j,k]+Vz[j,k]) + m['dvtdz'][j, k])*xd['dUdz'][j,k] + vt_tot[j,k]*xd['d2Udz2'][j,k]
                            )       

    ### Populate A-matrix with coefficients
    b = np.zeros((p['ny']*p['nz']))                         # Both for Dirichlet and Neumann

    # Boundary conditions: Dirichlet
    #A = np.eye(p['ny']*p['nz'])

    # Boundary conditions: Neumann
    A = np.zeros((p['ny']*p['nz'], p['ny']*p['nz']))

    # y-direction
    # j = 0 (left boundary)
    for k in range(p['nz']):
        I = get_I(0, k, p)
        I_right = get_I(1, k, p)

        A[I, I] = 1
        A[I, I_right] = -1
        b[I] = 0

    # j = ny-1 (right boundary)
    for k in range(p['nz']):
        I = get_I(p['ny']-1, k, p)
        I_left = get_I(p['ny']-2, k, p)

        A[I, I] = 1
        A[I, I_left] = -1
        b[I] = 0
    
    # z-direction
    # k = 0 (bottom boundary)
    for j in range(p['ny']):
        I = get_I(j, 0, p)
        I_up = get_I(j, 1, p)

        A[I, I] = 1
        A[I, I_up] = -1
        b[I] = 0

    # k = nz-1 (top boundary)
    for j in range(p['ny']):
        I = get_I(j, p['nz']-1, p)
        I_down = get_I(j, p['nz']-2, p)

        A[I, I] = 1
        A[I, I_down] = -1
        b[I] = 0

    # Populate A-matrix with computed coefficients
    for k in range(0+1, p['nz']-1, 1):
        for j in range(0+1, p['ny']-1, 1):
            I = get_I(j, k, p)
            A[I,I-p['ny']] = m['d'][j,k]    # Far left diagonal
            A[I,I-1] = m['a'][j,k]          # Left diagonal
            A[I,I] = m['b'][j,k]            # Main diagonal
            A[I,I+1] = m['c'][j,k]          # Right diagonal
            A[I,I+p['ny']] = m['e'][j,k]    # Far right diagonal
    
    # Populate b-vector with coefficients
    for k in range(0+1, p['nz']-1, 1):
        for j in range(0+1, p['ny']-1, 1):
            I = get_I(j, k, p)
            b[I] = m['f'][j,k]

    return m, A, b    # For debugging return m, later only return A and b

def solve_Vx(i, xd, m, p):
    """
    This subroutine calculates Vx
    """
    # Build the A-matrix and right hand side vector b
    m, A, b = build_A_matrix_plane_i(i, xd, m, p)

    # Use conjugate gradient method to solve the system, info==0 means successful convergence
    Vx_new, info = cg(A, b)
    if info != 0:
        print('CG did not converge for Vx!')
        raise Exception()
    
    # Reshape Vx from a vector to a matrix
    #Vx_new_2 = Vx_new.reshape((p['ny'],p['nz']), order = 'F')
    Vx_new_2 = np.zeros((p['ny'], p['nz']))
    for I in range(p['ny']*p['nz']):
        j, k = get_I_inverse(I, p)
        I2 = get_I(j, k, p)
        if I2 != I:
            raise Exception()
        Vx_new_2[j, k] = Vx_new[I]
    return Vx_new_2

def get_I(j, k, p):         # This function returns the row index I for matrix A
    I = k*p['ny']+j
    return I

def get_I_inverse(I, p):         # This function returns j and k for a given I for matrix A
    j = mod(I, p['ny'])
    k = int(I/p['ny'])
    return j, k

def gradient_x(field_old, field_new, dx):
    gradient_x = (field_new-field_old)/(dx)
    return gradient_x

def gradient_y(field, dy):
    """
    This subroutine calculates the 2D gradient d/dy of a given vector field.
    """
    gradient_y = np.zeros((field.shape[0], field.shape[1]))
    for k in range(0, field.shape[1], 1):
        j = 0           # BC
        gradient_y[j,k] = (-3*field[j,k]+4*field[j+1,k]-field[j+2,k])/(2*dy)

        for j in range(0+1, field.shape[0]-1, 1):
            gradient_y[j,k] = (field[j+1,k]-field[j-1,k])/(2*dy)
        
        j = field.shape[0] - 1   # BC
        gradient_y[j,k] = (3*field[j,k]-4*field[j-1,k]+field[j-2,k])/(2*dy)
    return gradient_y

def gradient_z(field, dz):
    """
    This subroutine calculates the 2D gradient d/dz of a given vector field.
    """
    gradient_z = np.zeros((field.shape[0], field.shape[1]))
    for j in range(0, field.shape[0], 1):
        k = 0           # BC
        gradient_z[j,k] = (-3*field[j,k]+4*field[j,k+1]-field[j,k+2])/(2*dz)

        for k in range(0+1, field.shape[1]-1, 1):
            gradient_z[j,k] = (field[j,k+1]-field[j,k-1])/(2*dz)
        
        k = field.shape[1] - 1   # BC
        gradient_z[j,k] = (3*field[j,k]-4*field[j,k-1]+field[j,k-2])/(2*dz)
    return gradient_z

def gradient_y2(field, dy):
    """
    This subroutine calculates the second derivative d^2/dy^2 of a given vector field.
    """
    gradient_y2 = np.zeros((field.shape[0], field.shape[1]))
    for k in range(0, field.shape[1], 1):
        j = 0           # BC
        gradient_y2[j,k] = (field[j,k]-2*field[j+1,k]+field[j+2,k])/(dy**2)

        for j in range(0+1, field.shape[0]-1, 1):
            gradient_y2[j,k] = (field[j-1,k]-2*field[j,k]+field[j+1,k])/(dy**2)
        
        j = field.shape[0] - 1   # BC
        gradient_y2[j,k] = (field[j,k]-2*field[j-1,k]+field[j-2,k])/(dy**2)
    return gradient_y2

def gradient_z2(field, dz):
    """
    This subroutine calculates the second derivative d^2/dz^2 of a given vector field.
    """
    gradient_z2 = np.zeros((field.shape[0], field.shape[1]))
    for j in range(0, field.shape[0], 1):
        k = 0           # BC
        gradient_z2[j,k] = (field[j,k]-2*field[j,k+1]+field[j,k+2])/(dz**2)

        for k in range(0+1, field.shape[1]-1, 1):
            gradient_z2[j,k] = (field[j,k-1]-2*field[j,k]+field[j,k+1])/(dz**2)
        
        k = field.shape[1] - 1   # BC
        gradient_z2[j,k] = (field[j,k]-2*field[j,k-1]+field[j,k-2])/(dz**2)
    return gradient_z2

def getU(case, time_step, plane_number):
    """
    Reads a VTK file corresponding to a specific FAST.Farm case, time step, and plane number.
    Parameters:
        case: Case can be changed from polar ('') to ('cartesian')
        time_step (int): Time step (e.g., 500 for 000000500).
        plane_number (int): Plane number (e.g., 4 for 0004).
    Returns:
        vtkFile
    Data can be retrieved by:
        Vx = vtk.point_data_grid['Vx'][0,:,:,0] 
    """
    # Get current directory so this script can be called from any location
    MyDir=os.path.dirname(__file__)
    vtkFileName = os.path.join(MyDir,'TSinflow_curl/vtk_ff_planes',r'FAST.Farm{}.WT1.PlaneAtTime_{:09d}_Plane_{:04d}.vtk'.format(case, time_step, plane_number))
    vtk = VTKFile(vtkFileName)
    #print(vtk)
    return vtk

def backgroundFlow_lin_shearVeer(mean, slope_y, slope_z, y, z):
    """
    This subroutine computes the background flow U(y,z), V(y,z) or W(y,z) assuming linear shear/veer.
    Parameters:
        mean: Mean wind speed, for V and W this is 0
        slope_y, slope_z: Slope in the linear equation
        y, z: Coordinates in the grid
    """

    # Prevent code from crashing if U_min <= 0
    if mean != 0:
        U_min = mean + slope_y * y[0] + slope_z * z[0]
        if U_min <= 0:
            print('U_min =',U_min)
            raise Exception()

    backgroundFlow = np.zeros((y.shape[0], z.shape[0]))
    for k in range(0, z.shape[0], 1):
        for j in range(0, y.shape[0], 1):
            backgroundFlow[j, k] = mean + slope_y * y[j] + slope_z * z[k]
    return backgroundFlow

############################################################################################
# Finding Vy and Vz: Solve phi using Green's function and FFT with unbounded BC

def Green_function(ny, nz, y, z):
    """
    This subroutine calculates the values for Green's function for the 2D Laplacian
    """
    G = np.zeros((ny, nz))
    for k in range(0, nz, 1):
        for j in range(0, ny, 1):
            if y[j] == 0 and z[k] == 0:   # Remove the singularity for y=z=0
                #G[j,k] = G[j-1,k]
                G[j,k] = 0
            else:
                G[j,k] = 1/(2 * np.pi) * np.log(np.sqrt(y[j]**2 + z[k]**2))
    return G

# def fft2_convolve2d(f, G, dy, dz, pad_size):
#     """
#     This subroutine performs the 2D convolution of a source term f and the Green function G.
    
#     Parameters:
#         f: Source term of the Poisson Equation
#         G: Discretized Green function
#         dy, dz: Grid size
#         pad_size: What factor the inputs shall be padded by. Must be either 2 or 3
        
#     Returns:
#         phi: Velocity potential
#     """
#     # Zero pad f and G
#     if pad_size == 2:
#         f_pad = np.pad(f, pad_width=((0, f.shape[0]//2), (0, f.shape[1]//2)), mode='constant', constant_values=0)
#         G_pad = np.pad(G, pad_width=((0, G.shape[0]//2), (0, G.shape[1]//2)), mode='constant', constant_values=0)
#     elif pad_size == 3:
#         f_pad = np.pad(f, pad_width=((0, f.shape[0]//1), (0, f.shape[1]//1)), mode='constant', constant_values=0)
#         G_pad = np.pad(G, pad_width=((0, G.shape[0]//1), (0, G.shape[1]//1)), mode='constant', constant_values=0)
    
#     # Perform 2D FFT
#     f_fft = np.fft.fft2(f_pad)
#     G_fft = np.fft.fft2(G_pad)

#     # Element-wise multiplication in frequency domain
#     phi_fft = f_fft * G_fft

#     # Inverse FFT to get the result back in spatial domain
#     phi = np.fft.ifft2(phi_fft)

#     # Trim to original size and scale with grid distance
#     phi = phi[f.shape[0]//2:f.shape[0]//2 + f.shape[0], f.shape[0]//2:f.shape[0]//2 + f.shape[0]]*dy*dz

#     # Return the real part
#     return np.real(phi)

# def solve_poisson_domain_increase(f, dy, dz):
#     """
#     This subroutine solves the Poisson Equation and eliminates the noise from the FFT by increasing the size of the domain before the calculation.
#     """

#     # Increase domain size
#     pad_size = 300//dy      # Noise region is less than 100m for any dy when dy=dr

#     nr = (f.shape[0] // 2) + 1 # Number of radial nodes
#     y = np.linspace(-(nr+pad_size-1)*dy, (nr+pad_size-1)*dy, f.shape[0]+2*pad_size)        # y-value at the node
#     z = np.linspace(-(nr+pad_size-1)*dz, (nr+pad_size-1)*dz, f.shape[1]+2*pad_size)        # z-value at the node
#     f_padded = np.pad(f, pad_width=((pad_size,pad_size), (pad_size,pad_size)), mode='constant', constant_values=0)
    
#     # Discretize the Green function
#     G = Green_function(y.shape[0], z.shape[0], y, z)

#     # Perform 2D convolution
#     phi = fft2_convolve2d(f_padded, G, dy, dz, pad_size=2)

#     # Crop domain size back to the original size
#     start = (y.shape[0] - f.shape[0]) // 2
#     end = start + f.shape[0]
#     phi = phi[start:end, start:end]
#     return phi

def solve_poisson(source, p, m): # TODO This was the old function, can be deleted once everything works

    # Calculate the dimension of the variables
    # domainExtension = int(300 // p['dy'])                           # Domain node amount is extended in the +- y/z direction by this
    # pad_size = int((p['ny'] + 2*domainExtension)//2)                # Zero padding to imitate unbounded domain when doing FFT
    # variableDimFull = int(p['ny'] + 2*domainExtension + pad_size)   # Actual matrix dimension when domain is extended and padded fully

    # Initialize all variables with the domain extension and padding
    # phi = np.zeros((variableDimFull, variableDimFull))
    # dVxdx = np.zeros((variableDimFull, variableDimFull))
    # G = np.zeros((variableDimFull, variableDimFull))

    # Calculate the y,z grid that is used to discretize the Green function on the larger domain
    # y = np.linspace(-(p['nr']+domainExtension-1)*p['dy'], (p['nr']+domainExtension-1)*p['dy'], p['ny']+2*domainExtension)
    # z = np.linspace(-(p['nr']+domainExtension-1)*p['dz'], (p['nr']+domainExtension-1)*p['dz'], p['nz']+2*domainExtension)
    
    # Discretize the Green function
    # G[:y.shape[0], :z.shape[0]] = Green_function(y.shape[0], z.shape[0], y, z)

    # Calculate the source term for the poisson equation
    m['dVxdx'][p['domainExtension']:(source.shape[0]+p['domainExtension']), p['domainExtension']:(source.shape[1]+p['domainExtension'])] = source

    # Perform 2D FFT
    m['dVxdx_fft'] = np.fft.fft2(m['dVxdx'])

    # Element-wise multiplication in frequency domain
    m['phi_fft'] = m['dVxdx_fft'] * p['G_fft']

    # Inverse FFT to get the result back in spatial domain
    m['phi_ext'] = np.fft.ifft2(m['phi_fft'])

    # Scale with grid size
    m['phi_ext'] = m['phi_ext'] * p['dy'] * p['dz']

    # # Visualize Phi_num
    # plt.figure()
    # plt.imshow(np.real(m['phi_ext']).T, cmap='viridis')
    # plt.title("Phi_num")
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()

    # Trim to original domain size
    start = p['y_ext'].shape[0]//2 + (p['y_ext'].shape[0] - source.shape[0]) // 2
    end = p['y_ext'].shape[0]//2 + (p['y_ext'].shape[0] - source.shape[0]) // 2 + source.shape[0]
    m['phi'] = m['phi_ext'][start:end, start:end]
    
    return np.real(m['phi'])

def solve_poisson_fixedExt(f, y, z):
    
    # Compute the Green function
    y_ext = extend_vector(y)
    z_ext = extend_vector(z)
    G = Green_function(2*f.shape[0], 2*f.shape[1], y_ext, z_ext)

    # Initialize a zero-padded source term of size 2N x 2N
    f_padded = np.zeros((2 * f.shape[0], 2 * f.shape[1]))

    # Copy the original source term into the top-left corner
    f_padded[:f.shape[0], :f.shape[1]] = f

    # # Visualize f_padded
    # plt.figure()
    # plt.imshow(f_padded.T, cmap='viridis')
    # plt.title("f padded")
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()  

    # Perform 2D FFT
    f_fft = np.fft.fft2(f_padded)
    G_fft = np.fft.fft2(G)

    # Element-wise multiplication in frequency domain
    phi_fft = f_fft * G_fft

    # Inverse FFT to get the result back in spatial domain
    phi_padded = np.fft.ifft2(phi_fft)

    # Scale with grid size
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    phi_padded = phi_padded * dy *dz

    # # Visualize Phi
    # plt.figure()
    # plt.imshow(np.real(phi_padded).T, cmap='viridis')
    # plt.title("Phi padded")
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()  

    # Trim to original domain size
    phi = phi_padded[-f.shape[0]:, -f.shape[1]:]

    # Debugging
    d = dict()
    d['y_ext'] = y_ext
    d['z_ext'] = z_ext
    d['dy'] = dy
    d['dz'] = dz
    
    return np.real(phi), d

def extend_vector(y):
    """
    This subroutine extends a vector of length N to a vector of length 2N. It keeps the original spacing dy.
    """
    y = np.asarray(y)
    dy = np.diff(y).mean()   # spacing (robust for uniform grids)

    N = len(y)

    # number of extra points on each side
    lower_extra = N//2 + 1
    upper_extra = N//2

    # generate the extra points
    y_lower = y[0] - np.arange(lower_extra, 0, -1) * dy
    y_upper = y[-1] + np.arange(1, upper_extra + 1) * dy

    # concatenate new vector
    y_ext = np.concatenate([y_lower, y, y_upper])

    return y_ext

def extend_vector_right(y):
    """
    Extends a vector y of length N to a vector of length 2N,
    preserving the original spacing, by adding N points on the right.
    """
    y = np.asarray(y)
    dy = np.diff(y).mean()   # uniform spacing

    N = len(y)

    # add N new points on the right
    y_right = y[-1] + np.arange(1, N+1) * dy

    # concatenate
    y_ext = np.concatenate([y, y_right])

    return y_ext

def solve_Vy_Vz(Vx_old, Vx_new, p, m):
    """
    This subroutine calculates Vy and Vz
    """

    # Solve the poisson equation
    #m['phi'] = solve_poisson(-gradient_x(Vx_old, Vx_new, p['dx']), p, m) # TODO This was the old function, can be deleted once everything works
    m['phi'], debug = solve_poisson_fixedExt(-gradient_x(Vx_old, Vx_new, p['dx']), p['y'], p['z'])

    # Retrieve Vy and Vz from phi
    Vy_new = gradient_y(m['phi'], p['dy'])
    Vz_new = gradient_z(m['phi'], p['dz'])
    return Vy_new, Vz_new

############################################################################################
# Eddy viscosity
eddyParameter_amb = dict()          # The parameters are set to their default values here
eddyParameter_amb['k'] = 0.05
eddyParameter_amb['C_FMin'] = 1.0
eddyParameter_amb['C_DMin'] = 0.0
eddyParameter_amb['C_DMax'] = 1.0
eddyParameter_amb['C_Exp'] = 0.01
eddyParameter_shr = dict()          # The parameters are set to their default values here
eddyParameter_shr['k'] = 0.016
eddyParameter_shr['k_vCurl'] = 2
eddyParameter_shr['C_FMin'] = 0.2
eddyParameter_shr['C_DMin'] = 3.0
eddyParameter_shr['C_DMax'] = 25.0
eddyParameter_shr['C_Exp'] = 0.1

x_plane = 390   #TODO change this
def EddyFilter(x_plane,D_rotor,eddyParameter):
    """
    This subroutine calculates the eddy viscosity filter functions
    """
    if (x_plane <= eddyParameter['C_DMin']*D_rotor):
        EddyFilter = eddyParameter['C_FMin']
    elif (x_plane >= eddyParameter['C_DMax']*D_rotor):
        EddyFilter = 1
    else:
        EddyFilter = eddyParameter['C_FMin'] + (1-eddyParameter['C_FMin'])*( ( (x_plane/D_rotor) - eddyParameter['C_DMin'] ) / (eddyParameter['C_DMax']-eddyParameter['C_DMin']) )**eddyParameter['C_Exp']
    return EddyFilter

def WakeDiam(D_rotor):
    """
    This subroutine calculates the wake diameter. The most simple case is to use the rotor diameter
    """
    WakeDiam = D_rotor
    return WakeDiam

def EddyViscosity(x_plane,D_rotor,eddyParameter_amb,eddyParameter_shr,xd,p,m):
    """
    This subroutine calculates the eddy viscosity
    """
    D_wake = WakeDiam(D_rotor)
    TI_amb = 0.01    #TODO calculate TI
    # Compute eddy viscosity terms
    EddyTermA = EddyFilter(x_plane,D_rotor,eddyParameter_amb)*eddyParameter_amb['k']*TI_amb*np.mean(m['U'])*D_rotor/2 #TODO double check what U is
    EddyTermB = EddyFilter(x_plane,D_rotor,eddyParameter_shr)*eddyParameter_shr['k']
    vt_min = np.abs(1.e-4*D_rotor*np.mean(m['U']))     # Minimum eddy viscosity

    # Compute gradients of dVx/dy and dVx/dz
    dVxdy = gradient_y(xd['Vx'], p['dy'])
    dVxdz = gradient_z(xd['Vx'], p['dz'])

    # Eddy viscosity
    vt_amb = np.zeros((p['ny'], p['nz']))
    vt_shr = np.zeros((p['ny'], p['nz']))
    vt_tot = np.zeros((p['ny'], p['nz']))
    for k in range(0, p['nz'], 1):
        for j in range(0, p['ny'], 1):
            r_tmp = np.sqrt(p['y'][j]**2 + p['z'][k]**2)
            if r_tmp == 0:
                S = 0
                C = 0
                dvdtheta_r = 0
            else:
                S = p['z'][k]/r_tmp
                C = p['y'][j]/r_tmp
                dvdtheta_r = (dVxdy[j,k]*p['z'][k] + dVxdz[j,k]*p['y'][j])/r_tmp
            
            dvdr = dVxdy[j,k]*C + dVxdz[j,k]*S
            vt_amb[j,k] = max(EddyTermA, vt_min)
            vt_shr[j,k] = EddyTermB*max((D_wake/2)**2 * (np.abs(dvdr) + np.abs(dvdtheta_r)), D_wake/2 * np.min(xd['Vx']))
    vt_tot = vt_amb + eddyParameter_shr['k_vCurl'] * vt_shr
    return vt_tot
