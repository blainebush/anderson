################################################################################
#
# Diagram of the simple four-site Hubbard model. The hopping terms are given by
# t12, ..., t41 and the on-site repulsion terms are given by u1, ..., u4. For
# the uniform lattice, t12 = t23 = t34 = t41 and u1 = u2 = u3 = u4. The four-
# site model reduces to two two-site models for t12 = t34 = 0 or t23 = t41 = 0.
#
#
#        1             2
#     u1 o-------------o u2
#        |     t12     |
#        |             |
#        | t41     t23 |
#        |             |
#        |     t34     |
#        o-------------o
#     u4 4             3 u3
#
#

import itertools as it
import numpy as np
import numpy.linalg as la
import scipy.interpolate as ir
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
np.seterr(divide='ignore', over='ignore', invalid='ignore')

class HubbardModel:
    def __init__(self, KB):
        """
        Initializes the physical constants used in the rest of the program.
        """
        self.KB = KB
    def operators(self, fill):
        # Construct all possible states. First four elements are the occupations (either
        # 0 or 1) for the up-electrons on each site, and the last four elements are the
        # same for the down-electrons on each site.
        states = list(it.product('01', repeat=8))
        states = [[int(y) for y in x] for x in states]
        if fill == 1:
            states = [pnt for pnt in states if sum(pnt) == 4]
        self.dim = len(states)

        self.states = states
        
        # Construct creation and annihilation operators for each spin species at each
        # site. C denotes a creation operator and A denotes an annihilation operator.
        # These operators are only created for the unrestricted filling case--they are
        # not used in the case of restricted filling.
        if fill == 0:
            C_1_up = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][1:] == states[j][1:]) and
                        (states[i][0] == 0) and
                        (states[j][0] == 1)):
                        C_1_up[j][i] = 1
            A_1_up = np.transpose(C_1_up)
            C_2_up = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][2:] == states[j][2:]) and
                        (states[i][:1] == states[j][:1]) and
                        (states[i][1] == 0) and
                        (states[j][1] == 1)):
                        C_2_up[j][i] = 1
            A_2_up = np.transpose(C_2_up)
            C_3_up = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][3:] == states[j][3:]) and
                        (states[i][:2] == states[j][:2]) and
                        (states[i][2] == 0) and
                        (states[j][2] == 1)):
                        C_3_up[j][i] = 1
            A_3_up = np.transpose(C_3_up)
            C_4_up = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][4:] == states[j][4:]) and
                        (states[i][:3] == states[j][:3]) and
                        (states[i][3] == 0) and
                        (states[j][3] == 1)):
                        C_4_up[j][i] = 1
            A_4_up = np.transpose(C_4_up)
            C_1_dn = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][5:] == states[j][5:]) and
                        (states[i][:4] == states[j][:4]) and
                        (states[i][4] == 0) and
                        (states[j][4] == 1)):
                        C_1_dn[j][i] = 1
            A_1_dn = np.transpose(C_1_dn)
            C_2_dn = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][6:] == states[j][6:]) and
                        (states[i][:5] == states[j][:5]) and
                        (states[i][5] == 0) and
                        (states[j][5] == 1)):
                        C_2_dn[j][i] = 1
            A_2_dn = np.transpose(C_2_dn)
            C_3_dn = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][7:] == states[j][7:]) and
                        (states[i][:6] == states[j][:6]) and
                        (states[i][6] == 0) and
                        (states[j][6] == 1)):
                        C_3_dn[j][i] = 1
            A_3_dn = np.transpose(C_3_dn)
            C_4_dn = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][:7] == states[j][:7]) and
                        (states[i][7] == 0) and
                        (states[j][7] == 1)):
                        C_4_dn[j][i] = 1
            A_4_dn = np.transpose(C_4_dn)

        # Construct number operators from respective creation and annihilation
        # operators for unrestricted filling, and manually for restricted filling.
        if fill == 0:
            N_1_up = np.dot(A_1_up, C_1_up)
            N_2_up = np.dot(A_2_up, C_2_up)
            N_3_up = np.dot(A_3_up, C_3_up)
            N_4_up = np.dot(A_4_up, C_4_up)
            N_1_dn = np.dot(A_1_dn, C_1_dn)
            N_2_dn = np.dot(A_2_dn, C_2_dn)
            N_3_dn = np.dot(A_3_dn, C_3_dn)
            N_4_dn = np.dot(A_4_dn, C_4_dn)
        elif fill == 1:
            N_1_up = np.zeros((self.dim, self.dim))
            N_2_up = np.zeros((self.dim, self.dim))
            N_3_up = np.zeros((self.dim, self.dim))
            N_4_up = np.zeros((self.dim, self.dim))
            N_1_dn = np.zeros((self.dim, self.dim))
            N_2_dn = np.zeros((self.dim, self.dim))
            N_3_dn = np.zeros((self.dim, self.dim))
            N_4_dn = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                N_1_up[i][i] = states[i][0]
                N_2_up[i][i] = states[i][1]
                N_3_up[i][i] = states[i][2]
                N_4_up[i][i] = states[i][3]
                N_1_dn[i][i] = states[i][4]
                N_2_dn[i][i] = states[i][5]
                N_3_dn[i][i] = states[i][6]
                N_4_dn[i][i] = states[i][7]
        N_up = N_1_up + N_2_up + N_3_up + N_4_up
        N_dn = N_1_dn + N_2_dn + N_3_dn + N_4_dn

        # Construct spin operators from respective number operators. Since these are
        # simply directly proportional to the number operators, there are not separate
        # cases for unrestricted and restricted filling.
        S_1_up = 0.5*N_1_up 
        S_2_up = 0.5*N_2_up 
        S_3_up = 0.5*N_3_up 
        S_4_up = 0.5*N_4_up 
        S_1_dn = -0.5*N_1_dn
        S_2_dn = -0.5*N_2_dn
        S_3_dn = -0.5*N_3_dn
        S_4_dn = -0.5*N_4_dn
        self.S_c = S_1_up + S_1_dn + S_2_up + S_2_dn
        self.S_f = S_3_up + S_3_dn + S_4_up + S_4_dn
                            
        # Construct hopping operators from creation and annihilation operators in the
        # unrestricted filling case, and manually in the restricted filling case. Since
        # fermionic operators anti-commute, transitions that hop over an odd number of
        # electrons pick up an extra negative sign.
        if fill == 0:
            self.T12_base = (np.dot(A_1_up, C_2_up) + np.dot(A_1_dn, C_2_dn))
            self.T23_base = (np.dot(A_2_up, C_3_up) + np.dot(A_2_dn, C_3_dn))
            self.T34_base = (np.dot(A_3_up, C_4_up) + np.dot(A_3_dn, C_4_dn))
            self.T41_base = (np.dot(A_4_up, C_1_up) + np.dot(A_4_dn, C_1_dn))
        elif fill == 1:
            self.T12_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][0] == 1 and states[j][0] == 0 and
                         states[i][1] == 0 and states[j][1] == 1 and
                         states[i][2:] == states[j][2:]) or
                        (states[i][4] == 1 and states[j][4] == 0 and
                         states[i][5] == 0 and states[j][5] == 1 and
                         states[i][:4] == states[j][:4] and
                         states[i][6:] == states[j][6:])):
                        self.T12_base[j][i] = 1
            self.T23_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][1] == 1 and states[j][1] == 0 and
                         states[i][2] == 0 and states[j][2] == 1 and
                         states[i][:1] == states[j][:1] and
                         states[i][3:] == states[j][3:]) or
                        (states[i][5] == 1 and states[j][5] == 0 and
                         states[i][6] == 0 and states[j][6] == 1 and
                         states[i][:5] == states[j][:5] and
                         states[i][7:] == states[j][7:])):
                        self.T23_base[j][i] = 1
            self.T34_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][2] == 1 and states[j][2] == 0 and
                         states[i][3] == 0 and states[j][3] == 1 and
                         states[i][:2] == states[j][:2] and
                         states[i][4:] == states[j][4:]) or
                        (states[i][6] == 1 and states[j][6] == 0 and
                         states[i][7] == 0 and states[j][7] == 1 and
                         states[i][:6] == states[j][:6])):
                        self.T34_base[j][i] = 1
            self.T41_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    if ((states[i][3] == 1 and states[j][3] == 0 and
                         states[i][0] == 0 and states[j][0] == 1 and
                         states[i][1:3] == states[j][1:3] and
                         states[i][4:] == states[j][4:]) or
                        (states[i][7] == 1 and states[j][7] == 0 and
                         states[i][4] == 0 and states[j][4] == 1 and
                         states[i][:4] == states[j][:4] and
                         states[i][5:7] == states[j][5:7])):
                        self.T41_base[j][i] = 1
                        
        # This adjusts the hopping operator between sites 1 and 4. The term picks up a
        # negative sign if there are an odd number of electrons between the sites, due
        # to the anti-commutation of fermionic operators.
        for i in range(self.dim):
            for j in range(self.dim):
                if self.T41_base[j][i] == 1:
                    diff1_up = states[j][0] - states[i][0]
                    diff4_up = states[j][3] - states[i][3]
                    diff1_dn = states[j][4] - states[i][4]
                    diff4_dn = states[j][7] - states[i][7]
                    if diff1_up == 1 and diff4_up == -1:
                        if (states[j][1] + states[j][2]) == 1:
                            self.T41_base[j][i] = -1
                    elif diff1_up == -1 and diff4_up == 1:
                        if (states[j][1] + states[j][2]) == 1:
                            self.T41_base[j][i] = -1
                    if diff1_dn == 1 and diff4_dn == -1:
                        if (states[j][5] + states[j][6]) == 1:
                            self.T41_base[j][i] = -1
                    elif diff1_dn == -1 and diff4_dn == 1:
                        if (states[j][5] + states[j][6]) == 1:
                            self.T41_base[j][i] = -1
                            
        # Hopping can occur in both directions, so the hopping operators must be added
        # to their transpose as well. 
        self.T12_base = self.T12_base + np.transpose(self.T12_base)
        self.T23_base = self.T23_base + np.transpose(self.T23_base)
        self.T34_base = self.T34_base + np.transpose(self.T34_base)
        self.T41_base = self.T41_base + np.transpose(self.T41_base)

        # Construct on-site repulsion operators from number operators. The -0.5 term is
        # to set the zero-energy point at half-filling, for symmetry purposes.
        if fill == 0:
            self.U1_base = np.dot(N_1_up - 0.5*np.eye(self.dim), N_1_dn - 0.5*np.eye(self.dim))
            self.U2_base = np.dot(N_2_up - 0.5*np.eye(self.dim), N_2_dn - 0.5*np.eye(self.dim))
            self.U3_base = np.dot(N_3_up - 0.5*np.eye(self.dim), N_3_dn - 0.5*np.eye(self.dim))
            self.U4_base = np.dot(N_4_up - 0.5*np.eye(self.dim), N_4_dn - 0.5*np.eye(self.dim))
        elif fill == 1:
            self.U1_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                if states[i][0] + states[i][4] == 0:
                    self.U1_base[i][i] = 0.0
                elif states[i][0] + states[i][4] == 1:
                    self.U1_base[i][i] = 0.0
                elif states[i][0] + states[i][4] == 2:
                    self.U1_base[i][i] = 1.0
            self.U2_base = np.zeros((self.dim, self.dim))                    
            for i in range(self.dim):
                if states[i][1] + states[i][5] == 0:
                    self.U2_base[i][i] = 0.0
                elif states[i][1] + states[i][5] == 1:
                    self.U2_base[i][i] = 0.0
                elif states[i][1] + states[i][5] == 2:
                    self.U2_base[i][i] = 1.0
            self.U3_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                if states[i][2] + states[i][6] == 0:
                    self.U3_base[i][i] = 0.0
                elif states[i][2] + states[i][6] == 1:
                    self.U3_base[i][i] = 0.0
                elif states[i][2] + states[i][6] == 2:
                    self.U3_base[i][i] = 1.0
            self.U4_base = np.zeros((self.dim, self.dim))
            for i in range(self.dim):
                if states[i][3] + states[i][7] == 0:
                    self.U4_base[i][i] = 0.0
                elif states[i][3] + states[i][7] == 1:
                    self.U4_base[i][i] = 0.0
                elif states[i][3] + states[i][7] == 2:
                    self.U4_base[i][i] = 1.0

        # Construct chemical potential operators from number operators. Generally, these
        # will be set to zero.
        self.M1_base = N_1_up + N_1_dn
        self.M2_base = N_2_up + N_2_dn
        self.M3_base = N_3_up + N_3_dn
        self.M4_base = N_4_up + N_4_dn
                          
    def eigen(self, t12, t23, t34, t41, u1, u2, u3, u4, m1, m2, m3, m4):
        """              
        The values t12, .., t41, u1, .., u4, and m1, ..., m4 are scalars. The object
        temp_vec is a numpy array. This function returns the susceptibility and
        Knight shift as a function of temperature for given energy parameters.
        """
        self.v = t23
        self.u = u3
        
        # Defines hopping operators from parameters and base operators.
        T12 = -t12*self.T12_base
        T23 = -t23*self.T23_base
        T34 = -t34*self.T34_base
        T41 = -t41*self.T41_base
        T = T12 + T23 + T34 + T41
                         
        # Defines on-site repulsion operators from parameters and base operators.
        U1 = u1*self.U1_base  
        U2 = u2*self.U2_base  
        U3 = u3*self.U3_base  
        U4 = u4*self.U4_base  
        U = U1 + U2 + U3 + U4
                         
        # Defines chemical potential operators from parameters and base operators.
        M1 = -m1*self.M1_base 
        M2 = -m2*self.M2_base 
        M3 = -m3*self.M3_base
        M4 = -m4*self.M4_base
        M = M1 + M2 + M3 + M4

        # Defines Hamiltonian from the sum of hopping, on-site repulsion, and
        # chemical potential operators, then calculates its eigenvalues and
        # eigenvectors.
        H = T + U + M
        evals, evecs = la.eig(H)
        self.evals = evals.real
        self.evecs = evecs.real

    def observables(self, temp_vec):
        """
        Defines terms used in the calculation of the components of magnetic
        susceptibility and spin-lattic relaxation time, then calculates those values
        for a given range of temperatures.
        """
        term1_c = np.dot(np.transpose(self.evecs), np.dot(self.S_c, self.evecs))
        term2_c = np.transpose(term1_c)
        term1_f = np.dot(np.transpose(self.evecs), np.dot(self.S_f, self.evecs))
        term2_f = np.transpose(term1_f)
        term3 = (np.transpose(np.tile(self.evals, (self.dim, 1)))
                 - np.tile(self.evals, (self.dim, 1)))

        chi_cc_vec = np.zeros(len(temp_vec))
        chi_cf_vec = np.zeros(len(temp_vec))
        chi_ff_vec = np.zeros(len(temp_vec))

        chi_cc_vec_hsb = np.zeros(len(temp_vec))
        chi_cf_vec_hsb = np.zeros(len(temp_vec))
        
        invT1T_cc_vec = np.zeros(len(temp_vec))
        invT1T_cf_vec = np.zeros(len(temp_vec))
        invT1T_ff_vec = np.zeros(len(temp_vec))

        for index, temp in enumerate(temp_vec):
            beta = 1.0/(self.KB*temp)

            para = 4*self.v*self.v/self.u
            chi_cc_hsb = ((-1 + np.exp(para*beta) + para*beta)
                      /(2*para*(3 + np.exp(para*beta))))
            chi_cf_hsb = ((1 - np.exp(para*beta) + para*beta)
                      /(2*para*(3 + np.exp(para*beta))))
            chi_cc_vec_hsb[index] = chi_cc_hsb
            chi_cf_vec_hsb[index] = chi_cf_hsb

            term4 = (np.exp(-beta*np.tile(self.evals, (self.dim, 1)))
                     - np.exp(-beta*np.transpose(np.tile(self.evals, (self.dim, 1)))))/term3
            term5 = beta*np.exp(-beta*self.evals)
            term6 = np.exp(-beta*(np.tile(self.evals, (self.dim, 1))
                              + np.transpose(np.tile(self.evals, (self.dim, 1))))/2.0)
            for i in range(self.dim):
                for j in range(self.dim):
                    if np.isnan(term4[i][j]):
                        term4[i][j] = term5[i]
            Z = np.sum(np.exp(-beta*self.evals))
            chi_cc_vec[index] = np.real(np.sum(term4 * term1_c * term2_c)/Z)
            chi_cf_vec[index] = np.real(np.sum(term4 * term1_c * term2_f)/Z)
            chi_ff_vec[index] = np.real(np.sum(term4 * term1_f * term2_f)/Z)

            invT1T_cc_vec[index] = np.real(np.sum(term4 * term1_c * term2_c)/((np.pi**2)*(temp**2)*Z))
            invT1T_cf_vec[index] = np.real(np.sum(term4 * term1_c * term2_f)/((np.pi**2)*(temp**2)*Z))
            invT1T_ff_vec[index] = np.real(np.sum(term4 * term1_f * term2_f)/((np.pi**2)*(temp**2)*Z))

        chi_vec = chi_cc_vec + 2*chi_cf_vec + chi_ff_vec

        return chi_cc_vec, chi_cf_vec, chi_ff_vec, chi_cc_vec_hsb, chi_cf_vec_hsb, invT1T_cc_vec, invT1T_cf_vec, invT1T_ff_vec

# Set parameter values and number of points to sample for both hybridization and
# temperature.
KB = 1.0
filling = 1

A = 0.86
B = 2.86
K0 = -0.056

t = 1.0
u = 4.0
m = 0.0

temp_vec = np.linspace(0.05, 4.0, 200)
v_vec = np.linspace(0.0, 2.0, 9)

# Initializes all vectors.
temp_vec_n = []
chi_cc_vec = []
chi_cc_hsb_vec = []
invT1T_cc_vec = []
chi_cf_vec = []
chi_cf_hsb_vec = []
invT1T_cf_vec = []
chi_ff_vec = []
invT1T_ff_vec = []
chi_vec = []
invT1T_vec = []
K_vec = []

# Meat of the program. This is where the function is called to calculate values
# for chi, K, and 1/T1T. How it works is, all the parameters (t, v, u, m) are
# used as arguments, as well as a _vector_ of temperatures. Then, the function
# returns a _vector_ of chi and 1/T1T corresponding to each temperature in
# the input vector. This is repeated for various values of v, until you are left
# with five vectors: an array of temperatures and hybridization values with
# values of chi and 1/T1T corresponding. So the vectors look like this:
#
#   T      v    chi  1/T1T
# [0.0]  [0.0]  [ ]   [ ]
# [0.5]  [0.0]  [ ]   [ ]
# [1.0]  [0.0]  [ ]   [ ]
# [0.0]  [0.5]  [ ]   [ ]
# [0.5]  [0.5]  [ ]   [ ]
# [1.0]  [0.5]  [ ]   [ ]
# [0.0]  [1.0]  [ ]   [ ]
# [0.5]  [1.0]  [ ]   [ ]
# [1.0]  [1.0]  [ ]   [ ]
model = HubbardModel(KB)
model.operators(filling)
##plt.figure(10, figsize=(8,8), dpi=80)
for i in range(len(v_vec)):
    v = v_vec[i]
    temp_vec_n.extend(temp_vec)
    
    t_list = (t, v, 0, v)
    u_list = (0, 0, u, u)
    m_list = (m, m, m, m)
    param_list = t_list + u_list + m_list
    model.eigen(*param_list)
    
##    plt.subplot(1,5,i+1)
##    plt.hlines(model.evals, 0.0, 1.0, 'r')
##    plt.title('$v={0:.2f}$'.format(v_vec[i]))
##    plt.axis([0.0, 1.0, -10.0, 10.0])
##    plt.tick_params('x', which='both', bottom='off', top='off', labelbottom='off')
##    if i > 0:
##        plt.tick_params('y', which='both', left='off', right='off', labelleft='off')
##    else:
##        plt.ylabel('$E$')
        
    chi_cc, chi_cf, chi_ff, chi_cc_hsb, chi_cf_hsb, invT1T_cc, invT1T_cf, invT1T_ff = model.observables(temp_vec)

    chi_cc_vec.append(chi_cc)
    chi_cc_hsb_vec.append(chi_cc_hsb)
    invT1T_cc_vec.append(invT1T_cc)
    chi_cf_vec.append(chi_cf)
    chi_cf_hsb_vec.append(chi_cf_hsb)
    invT1T_cf_vec.append(invT1T_cf)
    chi_ff_vec.append(chi_ff)
    invT1T_ff_vec.append(invT1T_ff)
    chi_vec.append(chi_cc + 2*chi_cf + chi_ff)
    invT1T_vec.append(invT1T_cc + (A + B)*invT1T_cf + invT1T_ff)
    K_vec.append(A*chi_cc + (A + B)*chi_cf + B*chi_ff)

##plt.tight_layout()
##plt.show()
##plt.savefig('C:\Code\Energy_levels_versus_v.pdf')

# Change python lists to numpy arrays.
temp_vec_n = np.array(temp_vec_n)
chi_cc_vec = np.nan_to_num(np.array(chi_cc_vec))
chi_cf_vec = np.nan_to_num(np.array(chi_cf_vec))
chi_cc_hsb_vec = np.nan_to_num(np.array(chi_cc_hsb_vec))
chi_cf_hsb_vec = np.nan_to_num(np.array(chi_cf_hsb_vec))
chi_ff_vec = np.nan_to_num(np.array(chi_ff_vec))
invT1T_cc_vec = np.nan_to_num(np.array(invT1T_cc_vec))
invT1T_cf_vec = np.nan_to_num(np.array(invT1T_cf_vec))
invT1T_ff_vec = np.nan_to_num(np.array(invT1T_ff_vec))
K_vec = np.nan_to_num(np.array(K_vec))


plot_chi_tot = False
plot_chi = False
plot_T1T = False
plot_K = False
plot_K_vary_B = False
plot_chi_hsb = False
chi_K_comparison = True
save_K = False

# Create plots of all values.
if plot_chi:
    plt.figure(1, figsize=(6,8), dpi=80)

    plt.subplot(3,1,1)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, chi_cc_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.ylabel(r'$\chi_{cc}$', size=18)
    plt.axis([0.0, 2.0, 0.0, 0.3])

    plt.subplot(3,1,2)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, chi_cf_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.ylabel(r'$\chi_{cf}$', size=18)
    plt.axis([0.0, 2.0, -0.3, 0.3])

    plt.subplot(3,1,3)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, chi_ff_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.xlabel('$T$', size=18)
    plt.ylabel(r'$\chi_{ff}$', size=18)
    plt.axis([0.0, 2.0, 0.0, 3.0])

    plt.tight_layout()
    plt.show()
    #plt.savefig('C:\Code\Chi_versus_T_vary_v.pdf')

##    plt.figure(2, figsize=(6,3), dpi=80)
##    for i in range(len(v_vec)):
##        plt.plot(temp_vec, chi_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
##                 linewidth=2)
##    plt.legend(prop={'size':10})
##    plt.xlabel('$T$', size=18)
##    plt.ylabel(r'$\chi_{tot}$', size=18)
##    plt.tight_layout()
##    plt.show()
##    plt.savefig('C:\Code\chi_tot_v.pdf')

if plot_T1T:
    plt.figure(3, figsize=(6,8), dpi=80)

    plt.subplot(3,1,1)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, invT1T_cc_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.ylabel(r'$(T_1 T)^{-1}_{cc}$', size=18)

    plt.subplot(3,1,2)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, invT1T_cf_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.ylabel(r'$(T_1 T)^{-1}_{cf}$', size=18)

    plt.subplot(3,1,3)
    for i in [0, 2, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, invT1T_ff_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.xlabel('$T$', size=18)
    plt.ylabel(r'$(T_1 T)^{-1}_{ff}$', size=18)

    plt.tight_layout()
    plt.show()
    #plt.savefig('C:\Code\invT1T_all_v.pdf')

##    plt.figure(4, figsize=(6,3), dpi=80)
##    for i in range(len(v_vec)):
##        plt.plot(temp_vec, invT1T_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
##                 linewidth=2)
##    plt.legend(prop={'size':10})
##    plt.xlabel('$T$', size=18)
##    plt.ylabel(r'$(T_1 T)^{-1}_{tot}$', size=18)
##    plt.tight_layout()
##    plt.show()
##    #plt.savefig('C:\Code\invT1T_tot_v.pdf')

if plot_K:
    plt.figure(5, figsize=(6,4), dpi=80)
    for i in [4, 5, 6, 7, 8]:
        plt.plot(chi_vec[i], K_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.xlabel(r'$\chi_{tot}$', size=18)
    plt.ylabel('$K$', size=18)
    plt.axis([0.0, 0.7, 0.0, 1.5])
    plt.tight_layout()
    plt.savefig('C:\Code\K_versus_Chi_vary_v.pdf')

if plot_K_vary_B:
    plt.figure(6, figsize=(6,4), dpi=80)
    for B in [-2.0, -1.0, 0.0, 0.5, 0.86, 1.5, 2.86]:
        new_K_vec = A*chi_cc_vec[5] + (A + B)*chi_cf_vec[5] + B*chi_ff_vec[5]
        plt.plot(chi_vec[5], new_K_vec, label='$B={0:.2f}$'.format(B), linewidth=2)
    plt.legend(prop={'size':10})
    plt.xlabel(r'$\chi_{tot}$', size=18)
    plt.ylabel('$K$', size=18)
    plt.tight_layout()
    plt.savefig('C:\Code\K_versus_Chi_vary_B.pdf')

if plot_chi_hsb:
    plt.figure(7, figsize=(6,4), dpi=80)
    plt.plot(temp_vec, chi_cc_vec[4], 'r-', label=r'$\chi_{cc}$', linewidth=2)
    plt.plot(temp_vec, chi_cf_vec[4], 'r--', label=r'$\chi_{cf}$', linewidth=2)
    plt.plot(temp_vec, chi_cc_hsb_vec[4], 'b-', label=r'$\chi_{cc} (hsb)$', linewidth=2)
    plt.plot(temp_vec, chi_cf_hsb_vec[4], 'b--', label=r'$\chi_{cf} (hsb)$', linewidth=2)
    plt.legend(prop={'size':10})
    plt.xlabel('$T$', size=18)
    plt.ylabel(r'$\chi$', size=18)
    #plt.axis([0.0, 1.5, -1.5, 1.5])
    plt.tight_layout()
    plt.show()
    #plt.savefig('C:\Code\Chi_versus_T_hsb.pdf')

if plot_chi_tot:
    plt.figure(8, figsize=(6,4), dpi=80)
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        plt.plot(temp_vec, chi_cc_vec[i] + 2*chi_cf_vec[i] + chi_ff_vec[i], label='$v={0:.2f}$'.format(v_vec[i]),
                 linewidth=2)
    plt.legend(prop={'size':10})
    plt.ylabel(r'$\chi$', size=18)
    plt.xlabel(r'$T$', size=18)
    plt.axis([0.0, 2.0, 0.0, 3.0])
    plt.tight_layout()
    plt.show()

if chi_K_comparison:
    plt.figure(9, figsize=(6,4), dpi=80)
    test_chi = chi_cc_vec[5] + 2*chi_cf_vec[5] + chi_ff_vec[5]
    test_K = 0.86*chi_cc_vec[5] + (0.86 + 2.86)*chi_cf_vec[5] + 2.86*chi_ff_vec[5]
    test_Keff = []
    for i in range(len(test_K)):
        test_Keff.append((test_K[i] - (-0.040369))/(2.1975))
    test_Tstar = 1.7827
    plt.plot(temp_vec, test_chi, 'k-', label='$\chi$', linewidth=2)
    plt.plot(temp_vec, test_Keff, 'ro', label='$\widetilde{K}$')
    #plt.plot([test_Tstar, test_Tstar],[0.0, 0.5], 'k--', label='$T^{*}$', linewidth=2)
    plt.legend(prop={'size':12})
    plt.xlabel('$T$', size=18)
    plt.ylabel('$\chi$', size=18)
    plt.tight_layout()
    #plt.show()
    plt.savefig('C:\Code\K_Chi_versus_T.pdf')


if save_K:
    for i in range(len(v_vec)):
        np.savetxt('C:\Code\New\K_chi_v%03d.txt' % (int(v_vec[i]*100)),
                   np.transpose([temp_vec, chi_vec[i], K_vec[i]]))
