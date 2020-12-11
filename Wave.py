# -*- coding: utf-8 -*-
"""
 ENSAM Campus Paris

 TIBA Azzeddine
 MONTEIRO Eric (Fourier Series)

"""
"""
    The problem is a rope displaced up at t = 0 in x = x0 
    and subjected to a Tension T

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.sparse as sp
import scipy.sparse.linalg
import os
from matplotlib import font_manager as fm, rcParams

#%% get_default_param
def get_default_param():
 ' get_default_param '
 RO=dict([ ('length',0.64), ('radius',5.e-4), ('rho',7.8e3), ('T',120),
        ('x0', 0.2), ('h',1), ('npts',51), ('dt',1e-4), ('Tend',50.e-3)])
 return RO

#%% plot results
def plot_results(mo1, res, i, dt_frames=50.):
 # force
 if i == 1:
     coor = mo1['x']
     ycoor = mo1['y0']
 else:
     coor = np.concatenate(([0],mo1['cells'],[mo1['x'][-1]]))
     ycoor = np.concatenate(([0],mo1['ycells'],[0]))
 def init_anim():

         ax = fig.get_axes();
         ax[0].cla();
         line1[0].set_data(coor, ycoor)
         plt.title('Evolution de la corde');
         plt.legend([res['method']]);
         plt.grid(True);
         plt.xlim(0, RO['length']);
         plt.ylim([-RO['h'], RO['h']]);
         plt.show()
         return line1


 def run_anim(i2):

     line1[0].set_ydata(res['u'][:, i2])
     return line1

# ------------------- Plot the force for the 3 methods -------------
 if type(res)==list:
     i = 0
     plt.figure(3)
     stl = ['--', '-', 'c-.']
     while i < len(res):

         plt.plot(mo1['t'] * 1.e3, res[i]['f'], stl[i], linewidth=1.7,label=res[i]['method'])

         i = i + 1
     plt.xlabel('Time (ms)')
     plt.grid(True)
     plt.ylabel('Force (N)')
     plt.legend()
     plt.show()



 else:
     if not ('f' in res): res['f'] = -(res['u'][-1, :] - res['u'][-2, :]) / mo1['dx']
     plt.figure(1);
     plt.plot(mo1['t'] * 1.e3, res['f']);
     plt.xlabel('Time (ms)');
     plt.grid(True);
     plt.ylabel('Force (N)');
     plt.legend([res['method']]);
     plt.show()
     # init figure
     fig = plt.figure(2)
     line1 = plt.plot([], [], 'r-')

     ani1 = anim.FuncAnimation(fig, run_anim, init_func=init_anim, frames=res['u'].shape[1], blit=True, interval=dt_frames, repeat=True, repeat_delay=2.e3)
     plt.show()

 if type(res) == list:
     print()
 else:

     return ani1

#%% create_mesh
def create_mesh(RO=[], fplt=False):  
 ' create mesh '
 #check input
 if type(RO)==list: RO=get_default_param()
 #points
 mo1=dict([ ('c',np.sqrt(RO['T']/(RO['rho']*np.pi*RO['radius']**2))) ])
 mo1['dx']=RO['length']/(RO['npts']-1)
 mo1['x']=np.arange(RO['npts'])*mo1['dx']
 #initial positions
 x0=RO['x0']*RO['length'];f1=mo1['x']/x0;
 f2=(RO['length']-mo1['x'])/(RO['length']-x0)
 mo1['y0']=((mo1['x']<=x0)*f1 + (mo1['x']>x0)*f2) * RO['h']
 #check 
 if fplt==True:
  plt.figure(0);plt.plot(mo1['x'],mo1['y0']);plt.xlabel('Position (m)');
  plt.grid(True);plt.ylabel('Initial position (m)');plt.show()
 #init time
 mo1['t']=np.arange(RO['Tend']//RO['dt']+2)*RO['dt']

 # cells
 cell_length = mo1['x'][1] - mo1['x'][0]
 cells = mo1['x'] - cell_length / 2
 mo1['cells'] = cells[1::]
 del cells

 # Number of cells
 N = mo1['cells'].size

 # Array of faces (indexes of neighbour cells)
 mo1['faces'] = np.concatenate((np.array([np.linspace(-1, N-1, N+1)]),\
                                np.array([np.linspace(0, N, N+1)])), axis = 0).astype(int)

 f1 = mo1['cells'] / x0;
 f2 = (RO['length'] - mo1['cells']) / (RO['length'] - x0)
 mo1['ycells'] = ((mo1['cells'] <= x0) * f1 + (mo1['cells'] > x0) * f2) * RO['h']

 mo1['connectivity'] = np.concatenate((np.array([np.arange(mo1['x'].size-1)]).T,\
                                       np.array([np.arange(1,mo1['x'].size)]).T),axis=1)
 return mo1

#%% sol_Fourier
def sol_Fourier(mo1=[],RO=[],N=10):
  ''' compute Fourier series '''
  #check input
  if type(RO)==list: RO=get_default_param()
  if type(mo1)==list: mo1=create_mesh(RO, False) 
  #init
  res=dict([('u',[]), ('f',[]), ('method','Fourier')])
  res['u']=np.zeros([mo1['x'].size, mo1['t'].size])
  res['f']=np.zeros(mo1['t'].size)
  #compute
  L=RO['length'];x0=RO['x0']*L
  c1t=np.pi*mo1['c']/L;c1x=np.pi/L
  for j1 in range(mo1['t'].size):
   for j2 in range(1,N):
    ak=(2*RO['h']*L**2)/((j2*np.pi)**2*x0*(L-x0)) * np.sin(j2*c1x*x0)
    res['u'][:,j1]=res['u'][:,j1]+ ak * np.sin(j2*c1x*mo1['x']) * np.cos(j2*c1t*mo1['t'][j1]);
    res['f'][j1] = res['f'][j1]  + ak * j2*c1x*np.cos(j2*np.pi) * np.cos(j2*c1t*mo1['t'][j1]);
  res['f']=-RO['T']*res['f']
  #output
  return res

def cell_contrib(index, mo1, RO):
    """
    Contribution d'une cellule à la matrice globale
        Inputs:
            index : indice global de la cellule dans la matrice
            mo1   : maillage
            RO    : Paramètre du problème
        Outputs:
            k: array (1,3) de raideur
            m: float; la contribution de la cellule dans la diagonale
             de la matrice M

    """

    # initialiser
    c = mo1['cells']
    f = mo1 ['faces']
    k = np.zeros((1,3))[0]

    T = RO['T']
    p = RO['rho']
    S = np.pi*RO['radius']**2
    if index == len(c)-1:
        L = c[f[0, index+1]] - c[f[0, index]]
    else:
        L = c[f[1, index+1]] - c[f[0, index+1]]

    #c'est en fait la distance entre 2 centres de cellules, mais dans
    # ce problème elle est égale à la longueur d'une cellule


    # remplissage de la matrice
    if index != len(c)-1:
        xfront = c[f[1, index+1]]
    else:
        xfront = c[index] + L/2

    if index != 0:
        xback = c[f[0,index]]
    else:
        xback = c[index] - L/2


    k[1] = T * ((1 / (xfront - c[index])) + (1 / (c[index] - xback)))

    if len(c) != index:
        k[2] = -T * (1 / (xfront - c[index]))

    if index != 0:
        k[0] = -T * (1 / (c[index] - xback))


    m = p * S * L

    return k, m

def assembly(mo1, RO):
    """
    Assemblage des matrices de chaque cellule dans la matrice globale
        Inputs:
            mo1   : maillage
            RO    : Paramètre du problème
        Outputs:
            K: array (N,N) avec N le nombre de cellules, la matrice de raideur
            M: array (N,N) avec N le nombre de cellules, la matrice de masse

    """
    c = mo1['cells']
    f = mo1['faces']

    # Taille du problème
    n = len(c)

    # -------------Initialisation des matrices--------------
    # K = np.zeros((n,n))
    # M = np.zeros((n,n))

    K = sp.lil_matrix((n, n))
    M = sp.lil_matrix((n, n))

    # --------------Assemblage des matrices-----------------
    i = 0
    while i < n:

        # ---------Appeler la matrice de la cellule locale--------
        cell_k, cell_m = cell_contrib(i, mo1, RO)

        # ---------Repérer les indices globales pour assembler-----
        indexes = np.array([f[0, i], i, f[1, i+1]])
        cond = np.logical_and(indexes >= 0, indexes < n)
        indexes = indexes[cond]

        # ---------Et finalement, assembler---------
        K[i, indexes] += sp.lil_matrix(cell_k[cond])
        M[i, i] = cell_m

        i += 1

    return K, M

def solve(mo1, theta, RO):
    """
    Résolution du système linéaire dans le temps
        Inputs:
            mo1   : maillage
            RO    : Paramètre du problème
            theta : paramètre theta pour le schéma theta
        Outputs:
            V     : liste avec des array (N, ) chaque élément de la liste contenant
                    la solution dans le pas de temps correspondant
            V_res : array (N,NT) avec N le nombre de cellules, NT le nombre de pas de temps
                    contenant la solution temporelle en accord avec la fonction plot_results

    """
    """
        Attention au fait que le nombre de points spatiaux pour les volumes finis,
        correspond au nombre de cellules qui n'est pas égal au nombre de noueds 
        pour les éléments finis ou pour la méthode de séries de Fourier

    """
    # Assembler les matrices -----------
    K, M = assembly(mo1,RO)

    # Liste pour stocker les vecteurs des résultats--------
    V = []

    theta_scheme(theta, V, mo1, K, M)

    # Transformer la liste des résultats en un array numpy
    V_res = np.concatenate((np.array([0]), V[0], np.array([0])))

    i = 1
    while i < mo1['t'].size:

        if i == 1:
            V_res = np.concatenate((np.array([V_res]), \
                                    np.array([np.concatenate((np.array([0]), V[i], np.array([0])))])), axis=0)
        else:
            V_res = np.concatenate((V_res, \
                                    np.array([np.concatenate((np.array([0]), V[i], np.array([0])))])), axis=0)
        i = i + 1
    V_res = V_res.T

    return V, V_res

def FV_Force(V, mo1, RO, order = 2):
    """
    Post calcul de la force en volumes finis
        Inputs:
            mo1   : maillage
            RO    : Paramètre du problème
            V     : Liste des solutions dans le temps
            order : Ordre de dérivation numérique de différences finis: 1 ou 2
        Outputs:
            F     : array(NT, ) contenant pour chaque élément la valeur de la force
                    dans le pas de temps correspondant

    """
    n_T = len(V)

    F = np.zeros((n_T,))

    i = 0
    while i < n_T:

        if order == 1:
            dV = V[i][-1] - V[i][-2]
            dx = mo1['x'][-1] - mo1['x'][-2]
        else:
            dV = 3 * V[i][-1] - 4 * V[i][-2] + V[i][-2]
            dx = 2 * (mo1['x'][-1] - mo1['x'][-2])

        F[i] = -RO['T'] * dV/dx

        i = i+1

    return F

def theta_scheme(theta, V, mo1, K, M, method = 'FV'):
    """
    Le schéma theta agit sur la liste V
        Inputs:
            mo1   : maillage
            RO    : Paramètre du problème
            theta : paramètre theta pour le schéma theta
            K     : Matrice de raideur globale
            M     : Matrice de masse globale
            method: Volumes finis ou Elements finis
            V     : Liste de solution dans le temps
        Outputs:


    """
    if method == 'FV':
        init = mo1['ycells']
    else:
        init = mo1['y0']

    # Initialisation--------------------
    deltaT = mo1['t'][1] - mo1['t'][0]
    n_T = mo1['t'].size
    V.append(init)

    # Accéleration nulle à t = 0
    tt = (1/(deltaT**2))

    # Solution dans l'instant t = delta T
    V.append(scipy.sparse.linalg.spsolve(2 * (M * tt + theta * K),
                                         (2 * tt * M - (1 - 2 * theta) * K).dot(V[0])))

    # Solution dans les autres pas
    i = 2
    while i < n_T:
        V.append(scipy.sparse.linalg.spsolve(tt * M + theta * K,
                                   tt * M.dot(-V[i-2] + 2 * V[i-1]) \
                                   - K.dot((theta*V[i-2] + (1-2*theta) * V[i-1]))))

        i += 1

def FEM(theta, mo1, RO):
    """
       La solution par éléments finis P1
           Inputs:
               theta : paramètre theta pour le schéma theta
               mo1   : maillage
               RO    : Paramètre du problème
           Outputs:
               V_FEM        : Liste contenant les solutions dans le temps, chaque élément est un array
                            (N, ) de solution dans un pas de temps
               V_FEM_res    :  Dictionnaire contenant la solution élémnents finis
               FEM_Force    : Array (NT, ) avec NT le nombre de pas de temps contenant les valeurs
                            de force dans le temps

       """
    T = RO['T']
    p = RO['rho']
    S = np.pi*RO['radius']**2

    def K_elem(mo1, index):
        """
           La matrice de raideur élémentaire

        """
        x1 = mo1['x'][mo1['connectivity'][index,0]]
        x2 = mo1['x'][mo1['connectivity'][index,1]]

        K =  T / (x2 - x1) * np.array([[1, -1],[-1, 1]])

        return K

    def M_elem(mo1, index):
        """
           La matrice de masse élémentaire

        """
        x1 = mo1['x'][mo1['connectivity'][index,0]]
        x2 = mo1['x'][mo1['connectivity'][index,1]]

        M = p * S * (x2 - x1) * (1/6) * np.array([[2, 1],[1, 2]])

        return M

    def FEM_Assembly(mo1):
        """
           Assemblage des matrices élémentaires dans la matrice globale
        """
        n = mo1['x'].size
        t = mo1['connectivity'].shape[0]

        K = sp.lil_matrix((n, n))
        M = sp.lil_matrix((n, n))

        i = 0
        while i < t:
            Ke = K_elem(mo1, i)
            Me = M_elem(mo1, i)

            jj = 0
            while jj < 2:
                kk = 0
                while kk < 2:
                    I = i + jj
                    J = i + kk

                    K[I, J] += Ke[jj, kk]
                    M[I, J] += Me[jj, kk]

                    kk += 1
                jj +=1
            i += 1

        return sp.lil_matrix(K), sp.lil_matrix(M)

    def FEM_BC(mo1):
        """
           Application de conditions aux bors de Dirichlet sur les 2 bords
        """
        Nn = mo1['x'].size
        Kb, Mb = FEM_Assembly(mo1)

        i = 0
        while i < Nn:
            Kb[0, i] = 0.
            Kb[Nn - 1, i] = 0.
            Kb[i, 0] = 0.
            Kb[i, Nn - 1] = 0.
            Mb[0, i] = 0.
            Mb[Nn - 1, i] = 0.
            Mb[i, 0] = 0.
            Mb[i, Nn - 1] = 0.

            i += 1

        Kb[0, 0] = 1; Mb[0, 0] = 1
        Kb[Nn - 1, Nn - 1] = 1; Mb[Nn - 1, Nn - 1] = 1

        return Kb, Mb

    def FEM_Solve(theta, mo1):
        """
           Solution de système linéaire dans le temps par le schéma theta
        """
        V_FEM = []

        K, M = FEM_BC(mo1)

        theta_scheme(theta, V_FEM, mo1, K, M, 'FEM')

        return V_FEM

    def FEM_Force(V, mo1):
        """
           Post calcul de la force de tension dans le temps
        """
        n_T = len(V)

        F = np.zeros((n_T,))

        i = 0
        while i < n_T:

            dphi_i = 1 / (mo1['x'][-1] - mo1['x'][-2])
            dphi_i_1 = - dphi_i

            F[i] = -RO['T'] * (V[i][-1] * dphi_i + V[i][-2] * dphi_i_1)

            i = i + 1

        return F

    V_FEM = FEM_Solve(theta, mo1)

    # Transformer la liste des résultats en un array numpy
    V_FEM_res = V_FEM[0]

    i = 1
    while i < mo1['t'].size:

        if i == 1:
            V_FEM_res = np.concatenate((np.array([V_FEM_res]), \
                                    np.array([V_FEM[i]])), axis=0)
        else:
            V_FEM_res = np.concatenate((V_FEM_res, \
                                    np.array([V_FEM[i]])), axis=0)
        i = i + 1
    V_FEM_res = V_FEM_res.T

    FEM_Force = FEM_Force(V_FEM, mo1)

    return V_FEM, V_FEM_res, FEM_Force

def animate_3_meth():
    """
        Animation des mouvement des cordes selon les 2 méthodes

    """
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    fpath = os.path.join(rcParams["datapath"],
                         "fonts/ttf/cmr10.ttf")

    prop = fm.FontProperties(fname=fpath)
    fname = os.path.split(fpath)[1]

    x1 = mo1['x']
    x3 = x1
    x2 = np.concatenate(([0],mo1['cells'],[mo1['x'][-1]]))
    y1 = res['u']
    y2 = res_FV['u']
    y3 = res_FEM['u']

    ax1.set_title(res['method'] , fontproperties=prop, fontsize=14)
    ax1.set_ylabel(u'v[m]')
    ax1.set_xlim(0, RO['length'])
    ax1.set_ylim([-RO['h'], RO['h']])
    ax1.get_xaxis().set_visible(False)

    ax2.set_ylabel(u'v[m]')
    ax2.set_xlim(0, RO['length'])
    ax2.set_ylim([-RO['h'], RO['h']])
    ax2.set_title(res_FV['method'],fontproperties=prop,fontsize=14)
    ax2.get_xaxis().set_visible(False)

    ax3.set_xlabel('x[m]')
    ax3.set_ylabel(u'v[m]')
    ax3.set_xlim(0, RO['length'])
    ax3.set_ylim([-RO['h'], RO['h']])
    ax3.set_title(res_FEM['method'],fontproperties=prop,fontsize=14)


    lines = []
    for i in range(len(mo1['t'])):

        line1, = ax1.plot(x1, y1[:,i], color='blue')
        line2, = ax2.plot(x2, y2[:,i], color='orange')
        line3, = ax3.plot(x3, y3[:,i], color='cyan')
        lines.append([line1, line2, line3])

    # Build the animation using ArtistAnimation function

    ani = anim.ArtistAnimation(fig, lines, interval=50, blit=True)
    plt.show()
    # Enregistrement de l'animation en gif (Partie commentée;
    # elle consomme beaucoup de temps CPU)
    """
    fn = 'comparaison'
    ani.save('%s.gif' % (fn), writer='imagemagick')
    import subprocess
    cmd = 'magick convert %s.gif -fuzz 10%% -layers Optimize %s_r.gif' % (fn, fn)
    subprocess.check_output(cmd)
    """

def animate_3_Forces():
    """
        Animation de comportement de la force de tensiond dans le temps pour
        les 3 méthodes

    """
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    fpath = os.path.join(rcParams["datapath"],
                         "fonts/ttf/cmr10.ttf")

    prop = fm.FontProperties(fname=fpath)
    fname = os.path.split(fpath)[1]

    t = mo1['t']
    y1 = res['f']
    y2 = res_FV['f']
    y3 = res_FEM['f']


    ax1.set_title(res['method'] , fontproperties=prop, fontsize=14)
    ax1.set_ylabel(u'F[N]')
    ax1.set_xlim(0, mo1['t'][-1])
    ax1.set_ylim(min(res['f']), max(res['f']))
    ax1.get_xaxis().set_visible(False)

    ax2.set_ylabel(u'F[N]')
    ax2.set_xlim(0, mo1['t'][-1])
    ax2.set_ylim(min(res_FV['f']), max(res_FV['f']))
    ax2.set_title(res_FV['method'],fontproperties=prop,fontsize=14)
    ax2.get_xaxis().set_visible(False)

    ax3.set_xlabel('t[s]')
    ax3.set_ylabel(u'F[N]')
    ax3.set_xlim(0, mo1['t'][-1])
    ax3.set_ylim(min(res_FEM['f']), max(res_FEM['f']))
    ax3.set_title(res_FEM['method'],fontproperties=prop,fontsize=14)

    lines = []
    for i in range(len(t)):

        head = i - 1
        head_slice = (t > t[i] - 1.0) & (t < t[i])
        line1,  = ax1.plot(t[:i], y1[:i], color='black')
        line1a, = ax1.plot(t[head_slice], y1[head_slice], color='blue', linewidth=2)
        line1e, = ax1.plot(t[head], y1[head], color='red', marker='o', markeredgecolor='r')
        line2,  = ax2.plot(t[:i], y2[:i], color='black')
        line2a, = ax2.plot(t[head_slice], y2[head_slice], color='orange', linewidth=2)
        line2e, = ax2.plot(t[head], y2[head], color='red', marker='o', markeredgecolor='r')
        line3,  = ax3.plot(t[:i], y3[:i], color='black')
        line3a, = ax3.plot(t[head_slice], y3[head_slice], color='cyan', linewidth=2)
        line3e, = ax3.plot(t[head], y3[head], color='red', marker='o', markeredgecolor='r')
        lines.append([line1,line1a,line1e,line2,line2a,line2e,line3,line3a,line3e])


    # Build the animation using ArtistAnimation function

    ani = anim.ArtistAnimation(fig,lines,interval=50,blit=True)
    plt.show()
    # Enregistrement de l'animation en gif (Partie commentée;
    # elle consomme beaucoup de temps CPU)
    """
    fn = 'comparaison_Forces'
    ani.save('%s.gif' % (fn), writer='imagemagick')
    import subprocess
    cmd = 'magick convert %s.gif -fuzz 10%% -layers Optimize %s_r.gif' % (fn, fn)
    subprocess.check_output(cmd)
    """


#%% main
# FOURIER
RO=get_default_param() #Change here the problem parameters
mo1=create_mesh(RO,fplt=True)
res=sol_Fourier(mo1,RO,N=10)
plot_results(mo1, res, 1, dt_frames=50)

# Theta for theta scheme
theta = 0.3

# ---------------- FINITE VOLUMES ----------------
V, V_res = solve(mo1, theta, RO)
FV_f = FV_Force(V, mo1, RO, 2)
res_FV = dict([('u', V_res), ('f', FV_f), ('method', 'Finite Volumes')])
plot_results(mo1, res_FV, 2, dt_frames=50)

# --------------- FINITE ELEMENTS ----------------
V_FEM, V_FEM_res, FEM_Force = FEM(theta, mo1, RO)
res_FEM = dict([('u', V_FEM_res), ('f', FEM_Force), ('method', 'Finite Elements')])
plot_results(mo1, res_FEM, 1, dt_frames=50)

# --------------- Comparison ---------------------
plot_results(mo1, [res, res_FV, res_FEM], 1, dt_frames=50)
animate_3_meth()

animate_3_Forces()

# ------------------- Plot the displacement at a time step 'i' for x = x0 -------------
# The index nearest to x = x0
ii = round(.2 * mo1['x'].size)

i = 0
plt.figure(13)
stl = ['--', '-', 'c:']
res_3 = [res, res_FV]
while i < len(res_3):

 plt.plot(mo1['t'] * 1.e3, res_3[i]['u'][ii,:], stl[i], linewidth=1.7,label=res_3[i]['method'])
 i = i + 1

plt.xlabel('Time [ms]')
plt.grid(True)
plt.ylim(-1,1)
plt.ylabel('Displacement [m]')
plt.legend()
plt.show()
