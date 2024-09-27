# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-11 15:58:38
# @Last Modified by:   Joaquin Gazquez
# @Last Modified time: 2024-06-20 16:19:54
                   
import numpy as np
from neuron import h
import sys
sys.path.append(r"C:\Users\jgazquez\RealSONIC")
import tempFunctions as tf

from ..core import PointNeuron, addSonicFeatures

@addSonicFeatures
class Soma(PointNeuron):
    ''' Realistic soma section class '''

    # Neuron name
    name = 'soma'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -75  # Membrane potential (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
		'm_CaHVA' : 'CaHVA activation gate',
		'h_CaHVA' : 'CaHVA inactivation gate',
		'm_Ih' : 'Ih activation gate',
		'm_NaTs2t' : 'NaTs2t activation gate',
		'h_NaTs2t' : 'NaTs2t inactivation gate',
		'm_CaLVAst' : 'CaLVAst activation gate',
		'h_CaLVAst' : 'CaLVAst inactivation gate',
		'm_SKv31' : 'SKv31 activation gate',
	}

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham_CaHVA(cls,Vm):
        v = Vm
        celsius = 37
        gca_hvabar = 0.00001
        if((v == -27) ):
            v = v+0.0001
        malpha =  (0.055*(-27-v))/(np.exp((-27-v)/3.8) - 1)
        mbeta  =  (0.94*np.exp((-75-v)/17))
        minf = malpha/(malpha + mbeta)
        mtau = 1/(malpha + mbeta)
        halpha =  (0.000457*np.exp((-13-v)/50))
        hbeta  =  (0.0065/(np.exp((-v-15)/28)+1))
        hinf = halpha/(halpha + hbeta)
        htau = 1/(halpha + hbeta)

        return malpha

    @classmethod
    def betam_CaHVA(cls,Vm):
        v = Vm
        celsius = 37
        gca_hvabar = 0.00001
        if((v == -27) ):
            v = v+0.0001
        malpha =  (0.055*(-27-v))/(np.exp((-27-v)/3.8) - 1)
        mbeta  =  (0.94*np.exp((-75-v)/17))
        minf = malpha/(malpha + mbeta)
        mtau = 1/(malpha + mbeta)
        halpha =  (0.000457*np.exp((-13-v)/50))
        hbeta  =  (0.0065/(np.exp((-v-15)/28)+1))
        hinf = halpha/(halpha + hbeta)
        htau = 1/(halpha + hbeta)

        return mbeta

    @classmethod
    def alphah_CaHVA(cls,Vm):
        v = Vm
        celsius = 37
        gca_hvabar = 0.00001
        if((v == -27) ):
            v = v+0.0001
        malpha =  (0.055*(-27-v))/(np.exp((-27-v)/3.8) - 1)
        mbeta  =  (0.94*np.exp((-75-v)/17))
        minf = malpha/(malpha + mbeta)
        mtau = 1/(malpha + mbeta)
        halpha =  (0.000457*np.exp((-13-v)/50))
        hbeta  =  (0.0065/(np.exp((-v-15)/28)+1))
        hinf = halpha/(halpha + hbeta)
        htau = 1/(halpha + hbeta)

        return halpha

    @classmethod
    def betah_CaHVA(cls,Vm):
        v = Vm
        celsius = 37
        gca_hvabar = 0.00001
        if((v == -27) ):
            v = v+0.0001
        malpha =  (0.055*(-27-v))/(np.exp((-27-v)/3.8) - 1)
        mbeta  =  (0.94*np.exp((-75-v)/17))
        minf = malpha/(malpha + mbeta)
        mtau = 1/(malpha + mbeta)
        halpha =  (0.000457*np.exp((-13-v)/50))
        hbeta  =  (0.0065/(np.exp((-v-15)/28)+1))
        hinf = halpha/(halpha + hbeta)
        htau = 1/(halpha + hbeta)

        return hbeta




    @classmethod
    def alpham_CaLVAst(cls,Vm):
        v = Vm
        celsius = 37
        gca_lvastbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf = 1.0000/(1+ np.exp((v - -30.000)/-6))
        mtau = (5.0000 + 20.0000/(1+np.exp((v - -25.000)/5)))/qt
        hinf = 1.0000/(1+ np.exp((v - -80.000)/6.4))
        htau = (20.0000 + 50.0000/(1+np.exp((v - -40.000)/7)))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return malpha

    @classmethod
    def betam_CaLVAst(cls,Vm):
        v = Vm
        celsius = 37
        gca_lvastbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf = 1.0000/(1+ np.exp((v - -30.000)/-6))
        mtau = (5.0000 + 20.0000/(1+np.exp((v - -25.000)/5)))/qt
        hinf = 1.0000/(1+ np.exp((v - -80.000)/6.4))
        htau = (20.0000 + 50.0000/(1+np.exp((v - -40.000)/7)))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return mbeta

    @classmethod
    def alphah_CaLVAst(cls,Vm):
        v = Vm
        celsius = 37
        gca_lvastbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf = 1.0000/(1+ np.exp((v - -30.000)/-6))
        mtau = (5.0000 + 20.0000/(1+np.exp((v - -25.000)/5)))/qt
        hinf = 1.0000/(1+ np.exp((v - -80.000)/6.4))
        htau = (20.0000 + 50.0000/(1+np.exp((v - -40.000)/7)))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return halpha

    @classmethod
    def betah_CaLVAst(cls,Vm):
        v = Vm
        celsius = 37
        gca_lvastbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf = 1.0000/(1+ np.exp((v - -30.000)/-6))
        mtau = (5.0000 + 20.0000/(1+np.exp((v - -25.000)/5)))/qt
        hinf = 1.0000/(1+ np.exp((v - -80.000)/6.4))
        htau = (20.0000 + 50.0000/(1+np.exp((v - -40.000)/7)))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return hbeta




    @classmethod
    def alpham_Ih(cls,Vm):
        v = Vm
        celsius = 37
        gihbar = 0.00001
        ehcn =  -45.0
        if(v == -154.9):
            v = v + 0.0001
        malpha =  0.001*6.43*(v+154.9)/(np.exp((v+154.9)/11.9)-1)
        mbeta  =  0.001*193*np.exp(v/33.1)
        minf = malpha/(malpha + mbeta)
        mtau = 1/(malpha + mbeta)

        return malpha

    @classmethod
    def betam_Ih(cls,Vm):
        v = Vm
        celsius = 37
        gihbar = 0.00001
        ehcn =  -45.0
        if(v == -154.9):
            v = v + 0.0001
        malpha =  0.001*6.43*(v+154.9)/(np.exp((v+154.9)/11.9)-1)
        mbeta  =  0.001*193*np.exp(v/33.1)
        minf = malpha/(malpha + mbeta)
        mtau = 1/(malpha + mbeta)

        return mbeta




    @classmethod
    def alpham_NaTs2t(cls,Vm):
        v = Vm
        celsius = 37
        gnats2_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -32):
            v = v+0.0001
        malpha = (0.182 * (v- -32))/(1-(np.exp(-(v- -32)/6)))
        mbeta  = (0.124 * (-v -32))/(1-(np.exp(-(-v -32)/6)))
        minf = malpha/(malpha + mbeta)
        mtau = (1/(malpha + mbeta))/qt
        if(v == -60):
            v = v + 0.0001
        halpha = (-0.015 * (v- -60))/(1-(np.exp((v- -60)/6)))
        hbeta  = (-0.015 * (-v -60))/(1-(np.exp((-v -60)/6)))
        hinf = halpha/(halpha + hbeta)
        htau = (1/(halpha + hbeta))/qt

        return malpha

    @classmethod
    def betam_NaTs2t(cls,Vm):
        v = Vm
        celsius = 37
        gnats2_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -32):
            v = v+0.0001
        malpha = (0.182 * (v- -32))/(1-(np.exp(-(v- -32)/6)))
        mbeta  = (0.124 * (-v -32))/(1-(np.exp(-(-v -32)/6)))
        minf = malpha/(malpha + mbeta)
        mtau = (1/(malpha + mbeta))/qt
        if(v == -60):
            v = v + 0.0001
        halpha = (-0.015 * (v- -60))/(1-(np.exp((v- -60)/6)))
        hbeta  = (-0.015 * (-v -60))/(1-(np.exp((-v -60)/6)))
        hinf = halpha/(halpha + hbeta)
        htau = (1/(halpha + hbeta))/qt

        return mbeta

    @classmethod
    def alphah_NaTs2t(cls,Vm):
        v = Vm
        celsius = 37
        gnats2_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -32):
            v = v+0.0001
        malpha = (0.182 * (v- -32))/(1-(np.exp(-(v- -32)/6)))
        mbeta  = (0.124 * (-v -32))/(1-(np.exp(-(-v -32)/6)))
        minf = malpha/(malpha + mbeta)
        mtau = (1/(malpha + mbeta))/qt
        if(v == -60):
            v = v + 0.0001
        halpha = (-0.015 * (v- -60))/(1-(np.exp((v- -60)/6)))
        hbeta  = (-0.015 * (-v -60))/(1-(np.exp((-v -60)/6)))
        hinf = halpha/(halpha + hbeta)
        htau = (1/(halpha + hbeta))/qt

        return halpha

    @classmethod
    def betah_NaTs2t(cls,Vm):
        v = Vm
        celsius = 37
        gnats2_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -32):
            v = v+0.0001
        malpha = (0.182 * (v- -32))/(1-(np.exp(-(v- -32)/6)))
        mbeta  = (0.124 * (-v -32))/(1-(np.exp(-(-v -32)/6)))
        minf = malpha/(malpha + mbeta)
        mtau = (1/(malpha + mbeta))/qt
        if(v == -60):
            v = v + 0.0001
        halpha = (-0.015 * (v- -60))/(1-(np.exp((v- -60)/6)))
        hbeta  = (-0.015 * (-v -60))/(1-(np.exp((-v -60)/6)))
        hinf = halpha/(halpha + hbeta)
        htau = (1/(halpha + hbeta))/qt

        return hbeta







    @classmethod
    def alpham_SKv31(cls,Vm):
        v = Vm
        celsius = 37
        gskv3_1bar = 0.00001
        minf =  1/(1+np.exp(((v -(18.700))/(-9.700))))
        mtau =  0.2*20.000/(1+np.exp(((v -(-46.560))/(-44.140))))
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file

        return malpha

    @classmethod
    def betam_SKv31(cls,Vm):
        v = Vm
        celsius = 37
        gskv3_1bar = 0.00001
        minf =  1/(1+np.exp(((v -(18.700))/(-9.700))))
        mtau =  0.2*20.000/(1+np.exp(((v -(-46.560))/(-44.140))))
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file

        return mbeta




    # ------------------------------ States derivatives ------------------------------

    @classmethod
    def derStates(cls):
        return {
			'm_CaHVA' : lambda Vm, x: cls.alpham_CaHVA(Vm) * (1 - x['m_CaHVA']) - cls.betam_CaHVA(Vm) * x['m_CaHVA'],
			'h_CaHVA' : lambda Vm, x: cls.alphah_CaHVA(Vm) * (1 - x['h_CaHVA']) - cls.betah_CaHVA(Vm) * x['h_CaHVA'],
			'm_Ih' : lambda Vm, x: cls.alpham_Ih(Vm) * (1 - x['m_Ih']) - cls.betam_Ih(Vm) * x['m_Ih'],
			'm_NaTs2t' : lambda Vm, x: cls.alpham_NaTs2t(Vm) * (1 - x['m_NaTs2t']) - cls.betam_NaTs2t(Vm) * x['m_NaTs2t'],
			'h_NaTs2t' : lambda Vm, x: cls.alphah_NaTs2t(Vm) * (1 - x['h_NaTs2t']) - cls.betah_NaTs2t(Vm) * x['h_NaTs2t'],
			'm_CaLVAst' : lambda Vm, x: cls.alpham_CaLVAst(Vm) * (1 - x['m_CaLVAst']) - cls.betam_CaLVAst(Vm) * x['m_CaLVAst'],
			'h_CaLVAst' : lambda Vm, x: cls.alphah_CaLVAst(Vm) * (1 - x['h_CaLVAst']) - cls.betah_CaLVAst(Vm) * x['h_CaLVAst'],
			'm_SKv31' : lambda Vm, x: cls.alpham_SKv31(Vm) * (1 - x['m_SKv31']) - cls.betam_SKv31(Vm) * x['m_SKv31'],
		}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
			'm_CaHVA' : lambda Vm: cls.alpham_CaHVA(Vm) / (cls.alpham_CaHVA(Vm) + cls.betam_CaHVA(Vm)),
			'h_CaHVA' : lambda Vm: cls.alphah_CaHVA(Vm) / (cls.alphah_CaHVA(Vm) + cls.betah_CaHVA(Vm)),
			'm_Ih' : lambda Vm: cls.alpham_Ih(Vm) / (cls.alpham_Ih(Vm) + cls.betam_Ih(Vm)),
			'm_NaTs2t' : lambda Vm: cls.alpham_NaTs2t(Vm) / (cls.alpham_NaTs2t(Vm) + cls.betam_NaTs2t(Vm)),
			'h_NaTs2t' : lambda Vm: cls.alphah_NaTs2t(Vm) / (cls.alphah_NaTs2t(Vm) + cls.betah_NaTs2t(Vm)),
			'm_CaLVAst' : lambda Vm: cls.alpham_CaLVAst(Vm) / (cls.alpham_CaLVAst(Vm) + cls.betam_CaLVAst(Vm)),
			'h_CaLVAst' : lambda Vm: cls.alphah_CaLVAst(Vm) / (cls.alphah_CaLVAst(Vm) + cls.betah_CaLVAst(Vm)),
			'm_SKv31' : lambda Vm: cls.alpham_SKv31(Vm) / (cls.alpham_SKv31(Vm) + cls.betam_SKv31(Vm)),
		}

    # ------------------------------ Membrane currents ------------------------------
    @classmethod
    def i_CaHVA(cls,m_CaHVA,h_CaHVA,Vm,gca_hvabar = 0.000374 * 1e4):
        ''' iCaHVA current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_CaHVA #
        h = h_CaHVA #
        gca = gca_hvabar*m*m*h
        ica = gca*(v-eca)

        return ica
    
    @classmethod
    def g_CaHVA(cls,m_CaHVA,h_CaHVA,Vm,gca_hvabar = 0.00001):
        ''' gCaHVA conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_CaHVA #
        h = h_CaHVA #
        gca_hvabar = 0.00001
        gca = gca_hvabar*m*m*h
        ica = gca*(v-eca)

        return gca

    @classmethod
    def i_CaLVAst(cls,m_CaLVAst,h_CaLVAst,Vm,gca_lvastbar = 0.000778 * 1e4):
        ''' iCaLVAst current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_CaLVAst #
        h = h_CaLVAst #
        gca_lvast = gca_lvastbar*m*m*h
        ica = gca_lvast*(v-eca)

        return ica
    
    @classmethod
    def g_CaLVAst(cls,m_CaLVAst,h_CaLVAst,Vm,gca_lvastbar = 0.00001):
        ''' gCaLVAst conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_CaLVAst #
        h = h_CaLVAst #
        gca_lvastbar = 0.00001
        gca_lvast = gca_lvastbar*m*m*h
        ica = gca_lvast*(v-eca)

        return gca_lvast

    @classmethod
    def i_Ih(cls,m_Ih,Vm,gihbar = 0.000080 * 1e4):
        ''' iIh current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_Ih #
        ehcn =  -45.0
        gih = gihbar*m
        ihcn = gih*(v-ehcn)

        return ihcn
    
    @classmethod
    def g_Ih(cls,m_Ih,Vm,gihbar = 0.00001):
        ''' gIh conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_Ih #
        gihbar = 0.00001
        ehcn =  -45.0
        gih = gihbar*m
        ihcn = gih*(v-ehcn)

        return gih

    @classmethod
    def i_NaTs2t(cls,m_NaTs2t,h_NaTs2t,Vm,gnats2_tbar = 0.926705 * 1e4):
        ''' iNaTs2t current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_NaTs2t #
        h = h_NaTs2t #
        gnats2_t = gnats2_tbar*m*m*m*h
        ina = gnats2_t*(v-ena)

        return ina
    
    @classmethod
    def g_NaTs2t(cls,m_NaTs2t,h_NaTs2t,Vm,gnats2_tbar = 0.00001):
        ''' gNaTs2t conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_NaTs2t #
        h = h_NaTs2t #
        gnats2_tbar = 0.00001
        gnats2_t = gnats2_tbar*m*m*m*h
        ina = gnats2_t*(v-ena)

        return gnats2_t

    @classmethod
    def i_SKv31(cls,m_SKv31,Vm,gskv3_1bar = 0.102517 * 1e4):
        ''' iSKv31 current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_SKv31 #
        gskv3_1 = gskv3_1bar*m
        ik = gskv3_1*(v-ek)

        return ik
    
    @classmethod
    def g_SKv31(cls,m_SKv31,Vm,gskv3_1bar = 0.00001):
        ''' gSKv31 conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_SKv31 #
        gskv3_1bar = 0.00001
        gskv3_1 = gskv3_1bar*m
        ik = gskv3_1*(v-ek)

        return gskv3_1

    @classmethod
    def i_pas(cls, Vm):
        ''' ipas current '''
        e_pas  = -75
        ra  = 100
        cm  = 1
        g_pas  = 3e-05 * 1e4 #S/cm2 -> S/m2
        ipas = g_pas*(Vm-e_pas)

        return ipas
    
    @classmethod
    def g_pas(cls, Vm):
        ''' gpas conductance '''
        e_pas  = -75
        ra = 100
        cm = 1
        g_pas = 3e-05 * 1e4 #S/cm2 -> S/m2 

        return g_pas

    @classmethod
    def currents(cls):
        return {
			'i_CaHVA': lambda Vm, x: cls.i_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm),
			'i_CaLVAst': lambda Vm, x: cls.i_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm),
			'i_Ih': lambda Vm, x: cls.i_Ih(x['m_Ih'], Vm),
			'i_NaTs2t': lambda Vm, x: cls.i_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm),
			'i_SKv31': lambda Vm, x: cls.i_SKv31(x['m_SKv31'], Vm),
			'i_pas': lambda Vm, x: cls.i_pas(Vm),
        }
    
    @classmethod
    def conductances(cls):
        return {
			'g_CaHVA': lambda Vm, x, g_bar: cls.g_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm) if g_bar is None else cls.g_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm, g_bar),
			'g_CaLVAst': lambda Vm, x, g_bar: cls.g_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm) if g_bar is None else cls.g_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm, g_bar),
			'g_Ih': lambda Vm, x, g_bar: cls.g_Ih(x['m_Ih'], Vm) if g_bar is None else cls.g_Ih(x['m_Ih'], Vm, g_bar),
			'g_NaTs2t': lambda Vm, x, g_bar: cls.g_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm) if g_bar is None else cls.g_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm, g_bar),
			'g_SKv31': lambda Vm, x, g_bar: cls.g_SKv31(x['m_SKv31'], Vm) if g_bar is None else cls.g_SKv31(x['m_SKv31'], Vm, g_bar),
			'g_pas': lambda Vm, x, g_bar: cls.g_pas(Vm)
        }