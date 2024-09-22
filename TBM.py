from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, CXGate, QFT
from qiskit.circuit import Parameter, ParameterVector, Barrier
from qiskit.circuit import Gate

from qiskit_algorithms.optimizers import SPSA, COBYLA

from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations

class TightBindingModel_Hamiltonian:
    def __init__(self, num_q_orbital, num_q_site, t_0, t_1):
        self.num_q_orbital=num_q_orbital
        self.num_q_site=num_q_site
        self.t_0=t_0
        self.t_1=t_1
        
    ############################################
    # Useful functions for Hamiltonian mapping #
    ############################################

    def list_to_str(self, list_data):
        str_first = ''
        for str_data in list_data:
            str_first += str_data
        return str_first

    def get_binary_str_desired_ones(self, n, k):
        result = []
        for ones_positions in combinations(range(n), k):
            binary_str = [0] * n
            for position in ones_positions:
                binary_str[position] = 1
            result.append(binary_str)
        return result

    def custom_sparsepauliop(self, input_sparsepauliop, X_boolean, Y_boolean, coeff_multiply):
        if X_boolean==True: add_string='X'
        if Y_boolean==True: add_string='Y'
        if X_boolean==False and Y_boolean==False: add_string=''
        
        paulis_input=input_sparsepauliop.paulis
        coeffs_input=input_sparsepauliop.coeffs
        
        tmp_paulis=paulis_input.to_labels()
        for i in range(len(tmp_paulis)):
            tmp_paulis[i]=add_string+tmp_paulis[i]
        paulis_output=tmp_paulis
        
        coeffs_output=coeff_multiply*coeffs_input
        
        return SparsePauliOp(data=paulis_output, coeffs=coeffs_output)  

    def custom_I(self, input_sparsepauliop, r):
        # add_string='I'*(num_q_orbital+num_q_site-r-1)
        add_string='I'*(self.num_q_site-r-1)
        paulis_custom=input_sparsepauliop.paulis
        coeffs_constant=input_sparsepauliop.coeffs
        
        tmp_paulis=paulis_custom.to_labels()
        for i in range(len(tmp_paulis)):
            tmp_paulis[i]=add_string+tmp_paulis[i]
        paulis_output=tmp_paulis
            
        return SparsePauliOp(data=paulis_output, coeffs=coeffs_constant)

    ########################################
    # Hamiltonian Implementation - Orbital #
    ########################################

    def I_pauli_string(self, hopping_term):
        I_string=[]
        coeff=[]
        for alpha in range(self.num_q_orbital):
            raw_string=['I']*self.num_q_orbital
            raw_I_string=raw_string
            # raw_I_string.append('I'*num_q_site)
            tmp=self.list_to_str(raw_I_string)
            I_string.append(tmp)
            
            array_hop=0.5*np.array(hopping_term)
            coeff.append(array_hop.tolist()[alpha][alpha])
        return SparsePauliOp(data=I_string, coeffs=coeff)

    def Z_pauli_string(self, hopping_term):
        Z_string=[]
        coeff=[]
        for alpha in range(self.num_q_orbital):
            raw_string=['I']*self.num_q_orbital
            raw_Z_string=raw_string
            raw_Z_string[alpha]='Z'
            # raw_Z_string.append('I'*num_q_site)
            tmp=self.list_to_str(raw_Z_string)
            Z_string.append(tmp)
            
            array_hop=(-0.5)*np.array(hopping_term)
            coeff.append(array_hop.tolist()[alpha][alpha])
        return SparsePauliOp(data=Z_string, coeffs=coeff)

    def XX_pauli_string(self, hopping_term):
        XX_string=[]
        coeff=[]
        for alpha in range(self.num_q_orbital):
            for beta in range(self.num_q_orbital):
                if alpha==beta: continue
                else:
                    raw_string=['I']*self.num_q_orbital
                    raw_XX_string=raw_string
                    raw_XX_string[alpha]='X'
                    raw_XX_string[beta]='X'
                    # raw_XX_string.append('I'*num_q_site)
                    tmp=self.list_to_str(raw_XX_string)
                    XX_string.append(tmp)
                    
                    array_hop=0.25*np.array(hopping_term)
                    coeff.append(array_hop.tolist()[alpha][beta])
        return SparsePauliOp(data=XX_string, coeffs=coeff)

    def YY_pauli_string(self, hopping_term):
        YY_string=[]
        coeff=[]
        for alpha in range(self.num_q_orbital):
            for beta in range(self.num_q_orbital):
                if alpha==beta: continue
                else:
                    raw_string=['I']*self.num_q_orbital
                    raw_YY_string=raw_string
                    raw_YY_string[alpha]='Y'
                    raw_YY_string[beta]='Y'
                    # raw_YY_string.append('I'*num_q_site)
                    tmp=self.list_to_str(raw_YY_string)
                    YY_string.append(tmp)
                    
                    array_hop=0.25*np.array(hopping_term)
                    coeff.append(array_hop.tolist()[alpha][beta])
        return SparsePauliOp(data=YY_string, coeffs=coeff)

    def XY_pauli_string(self, hopping_term):
        XY_string=[]
        coeff=[]
        for alpha in range(self.num_q_orbital):
            for beta in range(self.num_q_orbital):
                if alpha==beta: continue
                else:
                    raw_string=['I']*self.num_q_orbital
                    raw_XY_string=raw_string
                    raw_XY_string[alpha]='X'
                    raw_XY_string[beta]='Y'
                    # raw_XY_string.append('I'*num_q_site)
                    # raw_XY_string.insert(0, 'i')
                    tmp=self.list_to_str(raw_XY_string)
                    XY_string.append(tmp)
                    
                    array_hop=0.25*np.array(hopping_term)
                    coeff.append(array_hop.tolist()[alpha][beta])
        return SparsePauliOp(data=XY_string, coeffs=coeff)

    def YX_pauli_string(self, hopping_term):
        YX_string=[]
        coeff=[]
        for alpha in range(self.num_q_orbital):
            for beta in range(self.num_q_orbital):
                if alpha==beta: continue
                else:
                    raw_string=['I']*self.num_q_orbital
                    raw_YX_string=raw_string
                    raw_YX_string[alpha]='Y'
                    raw_YX_string[beta]='X'
                    # raw_YX_string.append('I'*num_q_site)
                    raw_YX_string.insert(0, '-')
                    # raw_YX_string.insert(0, '-i')
                    tmp=self.list_to_str(raw_YX_string)
                    YX_string.append(tmp)
                    
                    array_hop=0.25*np.array(hopping_term)
                    coeff.append(array_hop.tolist()[alpha][beta])
        return SparsePauliOp(data=YX_string, coeffs=coeff)

    def get_A_M(self, t):
        A_M=self.I_pauli_string(t)+self.Z_pauli_string(t)+self.XX_pauli_string(t)+self.YY_pauli_string(t)
        return A_M

    def get_B_M(self, t):
        B_M=self.XY_pauli_string(t)+self.YX_pauli_string(t)
        return B_M
    
    #####################################
    # Hamiltonian Implementation - Site #
    #####################################

    def A_r(self, r):
        A_r_string=[]
        coeff=[]
        for k in range((r//2)+1):
            str_desired_ones=self.get_binary_str_desired_ones(r, 2*k)
            for l in str_desired_ones:
                raw_string=['X']*r
                for idx, bin in enumerate(l):
                    if bin==True: 
                        raw_string[idx]='Y'
                if k%2==1: coeff.append(-1)
                else: coeff.append(1)
                tmp=self.list_to_str(raw_string)
                A_r_string.append(tmp)
        tmp=np.array(coeff)
        coeff=tmp/(2**r)
        return SparsePauliOp(data=A_r_string, coeffs=coeff)

    def B_r(self, r):
        B_r_string=[]
        coeff=[]
        for k in range((r//2)+1):
            if r==0: str_desired_ones=[[]]
            else: str_desired_ones=self.get_binary_str_desired_ones(r, 2*k+1)
            for l in str_desired_ones:
                raw_string=['X']*r
                for idx, bin in enumerate(l):
                    if bin==True:
                        raw_string[idx]='Y'
                        
                if k%2==0: coeff.append(-1)
                else: coeff.append(1)
                
                tmp=self.list_to_str(raw_string)
                B_r_string.append(tmp)
        tmp=np.array(coeff)
        coeff=tmp/(2**r)
        return SparsePauliOp(data=B_r_string, coeffs=coeff)     

    def get_A_N(self):
        sum_list=[]
        
        A_n=self.custom_I(self.A_r(self.num_q_site), r=self.num_q_site-1)
        sum_list.append(A_n)
        
        for i in range(self.num_q_site):
            if i==0:
                tmp_A_0=self.custom_I(self.custom_sparsepauliop(self.A_r(i), X_boolean=True, Y_boolean=False, coeff_multiply=0.5), r=i)
                sum_list.append(tmp_A_0)
                tmp_B_0=self.custom_I(self.custom_sparsepauliop(self.A_r(i), X_boolean=False, Y_boolean=True, coeff_multiply=-0.5), r=i)
                sum_list.append(tmp_B_0)
            else:
                tmp_A=self.custom_I(self.custom_sparsepauliop(self.A_r(i), X_boolean=True, Y_boolean=False, coeff_multiply=0.5), r=i)
                sum_list.append(tmp_A)
                tmp_B=self.custom_I(self.custom_sparsepauliop(self.B_r(i), X_boolean=False, Y_boolean=True, coeff_multiply=-0.5), r=i)
                sum_list.append(tmp_B)
        
        A_N=sum(sum_list)    
        return A_N

    def get_B_N(self):
        sum_list=[]
        
        B_n=self.custom_I(self.B_r(self.num_q_site), r=self.num_q_site-1)
        sum_list.append(B_n)
        
        for i in range(self.num_q_site):
            if i==0:
                tmp_B_0=self.custom_I(self.custom_sparsepauliop(self.B_r(i), X_boolean=False, Y_boolean=True, coeff_multiply=-0.5), r=i)
                sum_list.append(tmp_B_0)
                tmp_A_0=self.custom_I(self.custom_sparsepauliop(self.A_r(i), X_boolean=True, Y_boolean=False, coeff_multiply=0.5), r=i)
                sum_list.append(tmp_A_0)
            else:
                tmp_A=self.custom_I(self.custom_sparsepauliop(self.A_r(i), X_boolean=False, Y_boolean=True, coeff_multiply=0.5), r=i)
                sum_list.append(tmp_A)
                tmp_B=self.custom_I(self.custom_sparsepauliop(self.B_r(i), X_boolean=True, Y_boolean=False, coeff_multiply=0.5), r=i)
                sum_list.append(tmp_B)
        
        B_N=sum(sum_list)    
        return B_N  

    def get_I_N(self):
        I_string=[]
        raw_I_string=['I']*self.num_q_site
        tmp=self.list_to_str(raw_I_string)
        I_string.append(tmp)
        return SparsePauliOp(data=I_string)

    ############################
    # Hamiltonian construction #
    ############################

    def Hamiltonian(self):
        A_M_0=self.get_A_M(self.t_0)
        A_M_1=self.get_A_M(self.t_1)
        B_M_1=self.get_B_M(self.t_1)
        I_N=self.get_I_N()
        A_N=self.get_A_N()
        B_N=self.get_B_N()
        
        term_0=A_M_0.tensor(I_N)
        term_A_1=A_M_1.tensor(A_N)
        term_B_1=self.custom_sparsepauliop(B_M_1.tensor(B_N), X_boolean=False, Y_boolean=False, coeff_multiply=-1)
        
        hamiltonian=term_0+term_A_1+term_A_1+term_B_1+term_B_1
        
        return hamiltonian
    

class TightBindingModel_VQE:
    def __init__(self, num_q_orbital, num_q_site, hamiltonian):
        self.num_q_orbital=num_q_orbital
        self.num_q_site=num_q_site
        self.hamiltonian=hamiltonian
    
    ####################
    # Orbital register #
    ####################

    def A(self, theta, phi):
        # minimum unit of the number preserving ansatze ; included into the qc_orbital
        qc_ansatze_unit=QuantumCircuit(2, name="A") 
        qc_ansatze_unit.cx(0,1)
        qc_ansatze_unit.rz(phi+np.pi, 0).inverse()
        qc_ansatze_unit.ry(theta+np.pi/2, 0).inverse()
        qc_ansatze_unit.cx(1,0)
        qc_ansatze_unit.ry(theta+np.pi/2, 0)
        qc_ansatze_unit.rz(phi+np.pi, 0)
        qc_ansatze_unit.cx(0,1)
        return qc_ansatze_unit

    def ansatze_orbital_encoding(self, energy_level):
        qc_ansatze_o_ecdg=QuantumCircuit(self.num_q_orbital)
        qc_ansatze_o_ecdg.x(energy_level)
        return qc_ansatze_o_ecdg    

    def ansatze_level(self, energy_level, params_level):
        qc_ansatze_E_level=QuantumCircuit(self.num_q_orbital, name='level_'+str(energy_level))
        for i in range(self.num_q_orbital-energy_level-1):
            inst=self.A(theta=params_level[2*i], phi=params_level[2*i+1]).to_instruction()
            qc_ansatze_E_level.append(inst, [i+energy_level, i+energy_level+1])
        return qc_ansatze_E_level
    
    #################
    # Site register #
    #################

    def decimal_to_binary(self, p):
        bin_array=list()
        denominator=p
        for i in reversed(range(self.num_q_site)):
            bit=denominator//(2**i)
            bin_array.append(bit)
            denominator=denominator-bit*(2**i)
        return bin_array

    def QuantumFourierTransform(self):
        qc_QFT=QuantumCircuit(self.num_q_site, name='QFT')
        
        # Before SWAP
        for i in range(self.num_q_site):        
            qc_QFT.h(i)
            for j in range(self.num_q_site-1-i):
                theta=(2*np.pi)/(2**(j+2))
                qc_QFT.crz(theta=theta , control_qubit=j+1+i , target_qubit=i)
            qc_QFT.barrier()
        
        # after SWAP
        for i in range(self.num_q_site//2):
            qc_QFT.swap(i, self.num_q_site-1-i)
        
        return qc_QFT

    def ansatze_site(self, p):
        bin_array=self.decimal_to_binary(p)
        qc_ansatze_site = QuantumCircuit(self.num_q_site, name=f'site_register[p={p}]')
        
        for i, bit in enumerate(bin_array):
            if bit==1:
                qc_ansatze_site.x(i)
        qc_ansatze_site.barrier()
        
        qc_tmp=self.QuantumFourierTransform()
        inst=qc_tmp.to_instruction()
        qc_ansatze_site.append(inst, range(self.num_q_site))
        
        return qc_ansatze_site
    
    def vqe(self, energy_level, previous_params_list, p_list):
        
        optimizer = COBYLA()
        
        # divide quantum circuit into orbital register part and site register part each
        circuit_divider={'orbital':[], 'site':[]}
        for i in range(self.num_q_orbital+self.num_q_site):   
            if i<self.num_q_orbital:
                circuit_divider['orbital'].append(i)
            else:
                circuit_divider['site'].append(i)
        
        if previous_params_list==None: # target energy level is 0th : ground state band
                    
            # parameter list : 2 dimensional list [columns : parameter for each parametric gate, rows : p's(0,1,...,2**n-1)]
            params_fixed_0th=list()
            
            # eigen energy list : 1 dimensional list [row : eigen energy for sequential p's(0,1,...,2**n-1)]
            eigen_energy_0th=list()
            
            # entire ansatze for 0th level and optimization with the for loop containing site ansatze and its corresponding quantum number p
            for p in p_list:
                qc_0th=QuantumCircuit(self.num_q_orbital+self.num_q_site, name='ansatze_0th_level')
            
                # orbital ansatze for 0th level
                qc_orbital_0th=self.ansatze_orbital_encoding(energy_level) # orbital encoding for 0th energy level
                num_params=2*(self.num_q_orbital-energy_level-1)
                params_0th=ParameterVector("θ", num_params)
                qc_tmp=self.ansatze_level(energy_level, params_level=params_0th)
                inst=qc_tmp.to_instruction()
                qc_orbital_0th.append(inst, range(self.num_q_orbital))
                
                # site ansatze for 0th level
                qc_site_0th=self.ansatze_site(p)
                
                # construct entire ansatze
                inst_orbital=qc_orbital_0th.to_instruction()
                inst_site=qc_site_0th.to_instruction()
                qc_0th.append(inst_orbital, circuit_divider['orbital'])
                qc_0th.append(inst_site, circuit_divider['site'])
            
                def loss(x):
                    estimator = Estimator()
                    pub = (qc_0th, self.hamiltonian, x)
                    job = estimator.run([pub])
                    result = job.result()[0]
                    return np.sum(result.data.evs)

                result = optimizer.minimize(loss, x0 = [0.2]*qc_0th.num_parameters)
                print(f'0th eigen energy information (p={p}) : {result}')
                params_fixed_0th.append(result.x)
                eigen_energy_0th.append(result.fun)
            
            return params_fixed_0th, eigen_energy_0th
        
        else: # set ansatze for nonzero energy level 
                    
            # parameter list : 2 dimensional list [columns : parameter for each parametric gate, rows : p's(0,1,...,2**n-1)]
            params_fixed_ith=list()
            
            # eigen energy list : 1 dimensional list [row : eigen energy for sequential p's(0,1,...,2**n-1)]
            eigen_energy_ith=list()
            
            # entire ansatze for ith level with other fixed ansatze's and optimization with the for loop containing site ansatze and its corresponding quantum number p
            for p in p_list:
                qc_ith=QuantumCircuit(self.num_q_orbital+self.num_q_site, name=f'ansatze_{i}th_level')
                
                # current energy level
                qc_orbital_ith=self.ansatze_orbital_encoding(energy_level)    
                num_params=2*(self.num_q_orbital-energy_level-1)
                params=ParameterVector("θ", num_params)
                qc_tmp=self.ansatze_level(energy_level, params_level=params)
                inst_current=qc_tmp.to_instruction()
                qc_orbital_ith.append(inst_current, range(self.num_q_orbital))
                
                # previous energy level with fixed parameters
                for i in reversed(range(energy_level)):
                    qc_orbital_ith.barrier()
                    params=previous_params_list[i][p]
                    previous_level_qc=self.ansatze_level(energy_level=i, params_level=params)
                    inst=previous_level_qc.to_instruction()
                    qc_orbital_ith.append(inst, range(self.num_q_orbital))
                    
                # site ansatze for ith level
                qc_site_ith=self.ansatze_site(p)
                
                # construct entire ansatze
                inst_orbital=qc_orbital_ith.to_instruction()
                inst_site=qc_site_ith.to_instruction()
                qc_ith.append(inst_orbital, circuit_divider['orbital'])
                qc_ith.append(inst_site, circuit_divider['site'])
                
                def loss(x):
                    estimator = Estimator()
                    pub = (qc_ith, self.hamiltonian, x)
                    job = estimator.run([pub])
                    result = job.result()[0]
                    return np.sum(result.data.evs)

                result = optimizer.minimize(loss, x0 = [0.2]*qc_ith.num_parameters)
                print(f'{energy_level}th eigen energy information (p={p}) : {result}')
                params_fixed_ith.append(result.x) # further obtained
                eigen_energy_ith.append(result.fun)
            
            return params_fixed_ith, eigen_energy_ith
        
    def Energy_Momentum_Dispersion(self, upper_bound_level):
        
        ###########
        # Run VQE #
        ###########
        
        params_library = dict()
        eigen_energy_library = dict()
        
        p_list=range(2**self.num_q_site)

        for i in range(upper_bound_level+1):
            if i==0:
                params_fixed_0th, eigen_energy_0th = self.vqe(
                    energy_level=i, 
                    previous_params_list=None,
                    p_list=p_list
                )
                params_library['level_'+str(i)] = params_fixed_0th
                eigen_energy_library['level_'+str(i)]=eigen_energy_0th
            else:
                params_send = [params_library['level_'+str(j)] for j in range(i)]
                params_fixed_ith, eigen_energy_ith = self.vqe(
                    energy_level=i, 
                    previous_params_list=params_send,
                    p_list=p_list
                )
                params_library['level_'+str(i)] = params_fixed_ith
                eigen_energy_library['level_'+str(i)]=eigen_energy_ith
        
        ##################
        # Draw E-k graph #
        ##################
        
        plt.figure(figsize=(8, 5))
        
        x=np.array(p_list)*(2*np.pi)/(2**self.num_q_site)
        for i in range(upper_bound_level+1):
            plt.plot(x, eigen_energy_library[f'level_{i}'], 'o-', markersize=3, label=f'{i}th energy band')        

        x_axis=x.tolist()
        x_axis.append(2*np.pi)

        x_label=list()
        for i in x_axis:
            if (i%(np.pi))==0:
                if i==0: x_label.append('0')
                else : x_label.append(f'{int(i//np.pi)}π/a')
            else:
                x_label.append('')

        plt.xticks(x_axis, labels=x_label)
        plt.xlabel('Momentum (k)')
        plt.ylabel('Energy (E)')
        plt.legend(reverse=True)
        plt.title('Energy-Momentum Dispersion Relation')
        plt.show()