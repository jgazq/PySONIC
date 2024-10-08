# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-11 15:58:38
# @Last Modified by:   Joaquin Gazquez
# @Last Modified time: 2024-09-25 16:49:15
                   
import numpy as np
from neuron import h
import sys
sys.path.append(r"C:\Users\jgazquez\RealSONIC")
import tempFunctions as tf

from ..core import PointNeuron, addSonicFeatures

@addSonicFeatures
class RealisticNeuron(PointNeuron):
    ''' Realistic neuron class '''

    # Neuron name
    name = 'realneuron'

    # ------------------------------ Biophysical parameters ------------------------------

    # Resting parameters
    Cm0 = 1e-2   # Membrane capacitance (F/m2)
    Vm0 = -75  # Membrane potential (mV)

    # ------------------------------ States names & descriptions ------------------------------
    states = {
		'm_Ca' : 'Ca activation gate',
		'h_Ca' : 'Ca inactivation gate',
		'm_CaHVA' : 'CaHVA activation gate',
		'h_CaHVA' : 'CaHVA inactivation gate',
		'm_Ih' : 'Ih activation gate',
		'm_Im' : 'Im activation gate',
		'm_NapEt2' : 'NapEt2 activation gate',
		'h_NapEt2' : 'NapEt2 inactivation gate',
		'm_NaTat' : 'NaTat activation gate',
		'h_NaTat' : 'NaTat inactivation gate',
		'm_NaTs2t' : 'NaTs2t activation gate',
		'h_NaTs2t' : 'NaTs2t inactivation gate',
		'm_CaLVAst' : 'CaLVAst activation gate',
		'h_CaLVAst' : 'CaLVAst inactivation gate',
		'm_KPst' : 'KPst activation gate',
		'h_KPst' : 'KPst inactivation gate',
		'm_KTst' : 'KTst activation gate',
		'h_KTst' : 'KTst inactivation gate',
		'm_SKv31' : 'SKv31 activation gate',
		'm_KdShu2007' : 'KdShu2007 activation gate',
		'h_KdShu2007' : 'KdShu2007 inactivation gate',
	}

    # ------------------------------ Gating states kinetics ------------------------------

    @classmethod
    def alpham_Ca(cls,Vm):
        v = Vm
        celsius = 37
        gcabar = 0.00001
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
    def betam_Ca(cls,Vm):
        v = Vm
        celsius = 37
        gcabar = 0.00001
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
    def alphah_Ca(cls,Vm):
        v = Vm
        celsius = 37
        gcabar = 0.00001
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
    def betah_Ca(cls,Vm):
        v = Vm
        celsius = 37
        gcabar = 0.00001
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
    def alpham_Im(cls,Vm):
        v = Vm
        celsius = 37
        gimbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        malpha = 3.3e-3*np.exp(2.5*0.04*(v - -35))
        mbeta = 3.3e-3*np.exp(-2.5*0.04*(v - -35))
        minf = malpha/(malpha + mbeta)
        mtau = (1/(malpha + mbeta))/qt

        return malpha

    @classmethod
    def betam_Im(cls,Vm):
        v = Vm
        celsius = 37
        gimbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        malpha = 3.3e-3*np.exp(2.5*0.04*(v - -35))
        mbeta = 3.3e-3*np.exp(-2.5*0.04*(v - -35))
        minf = malpha/(malpha + mbeta)
        mtau = (1/(malpha + mbeta))/qt

        return mbeta




    @classmethod
    def alpham_KdShu2007(cls,Vm):
        v = Vm
        celsius = 37
        gkbar = 0.1
        ek = -100	            
        vhalfm=-43
        km=8
        vhalfh=-67
        kh=7.3
        q10=2.3
        qt=q10**((celsius-22)/10)
        minf=1-1/(1+np.exp((v-vhalfm)/km))
        hinf=1/(1+np.exp((v-vhalfh)/kh))
        mtau = 0.6
        htau = 1500
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return malpha

    @classmethod
    def betam_KdShu2007(cls,Vm):
        v = Vm
        celsius = 37
        gkbar = 0.1
        ek = -100	            
        vhalfm=-43
        km=8
        vhalfh=-67
        kh=7.3
        q10=2.3
        qt=q10**((celsius-22)/10)
        minf=1-1/(1+np.exp((v-vhalfm)/km))
        hinf=1/(1+np.exp((v-vhalfh)/kh))
        mtau = 0.6
        htau = 1500
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return mbeta

    @classmethod
    def alphah_KdShu2007(cls,Vm):
        v = Vm
        celsius = 37
        gkbar = 0.1
        ek = -100	            
        vhalfm=-43
        km=8
        vhalfh=-67
        kh=7.3
        q10=2.3
        qt=q10**((celsius-22)/10)
        minf=1-1/(1+np.exp((v-vhalfm)/km))
        hinf=1/(1+np.exp((v-vhalfh)/kh))
        mtau = 0.6
        htau = 1500
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return halpha

    @classmethod
    def betah_KdShu2007(cls,Vm):
        v = Vm
        celsius = 37
        gkbar = 0.1
        ek = -100	            
        vhalfm=-43
        km=8
        vhalfh=-67
        kh=7.3
        q10=2.3
        qt=q10**((celsius-22)/10)
        minf=1-1/(1+np.exp((v-vhalfm)/km))
        hinf=1/(1+np.exp((v-vhalfh)/kh))
        mtau = 0.6
        htau = 1500
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return hbeta




    @classmethod
    def alpham_KPst(cls,Vm):
        v = Vm
        celsius = 37
        gk_pstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  (1/(1 + np.exp(-(v+1)/12)))
        if(v<-50):
            mtau =  (1.25+175.03*np.exp(-v * -0.026))/qt
        else:
            mtau = ((1.25+13*np.exp(-v*0.026)))/qt
        hinf =  1/(1 + np.exp(-(v+54)/-11))
        htau =  (360+(1010+24*(v+55))*np.exp(-((v+75)/48)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return malpha

    @classmethod
    def betam_KPst(cls,Vm):
        v = Vm
        celsius = 37
        gk_pstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  (1/(1 + np.exp(-(v+1)/12)))
        if(v<-50):
            mtau =  (1.25+175.03*np.exp(-v * -0.026))/qt
        else:
            mtau = ((1.25+13*np.exp(-v*0.026)))/qt
        hinf =  1/(1 + np.exp(-(v+54)/-11))
        htau =  (360+(1010+24*(v+55))*np.exp(-((v+75)/48)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return mbeta

    @classmethod
    def alphah_KPst(cls,Vm):
        v = Vm
        celsius = 37
        gk_pstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  (1/(1 + np.exp(-(v+1)/12)))
        if(v<-50):
            mtau =  (1.25+175.03*np.exp(-v * -0.026))/qt
        else:
            mtau = ((1.25+13*np.exp(-v*0.026)))/qt
        hinf =  1/(1 + np.exp(-(v+54)/-11))
        htau =  (360+(1010+24*(v+55))*np.exp(-((v+75)/48)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return halpha

    @classmethod
    def betah_KPst(cls,Vm):
        v = Vm
        celsius = 37
        gk_pstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  (1/(1 + np.exp(-(v+1)/12)))
        if(v<-50):
            mtau =  (1.25+175.03*np.exp(-v * -0.026))/qt
        else:
            mtau = ((1.25+13*np.exp(-v*0.026)))/qt
        hinf =  1/(1 + np.exp(-(v+54)/-11))
        htau =  (360+(1010+24*(v+55))*np.exp(-((v+75)/48)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return hbeta




    @classmethod
    def alpham_KTst(cls,Vm):
        v = Vm
        celsius = 37
        gk_tstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  1/(1 + np.exp(-(v+0)/19))
        mtau =  (0.34+0.92*np.exp(-((v+71)/59)**2))/qt
        hinf =  1/(1 + np.exp(-(v+66)/-10))
        htau =  (8+49*np.exp(-((v+73)/23)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return malpha

    @classmethod
    def betam_KTst(cls,Vm):
        v = Vm
        celsius = 37
        gk_tstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  1/(1 + np.exp(-(v+0)/19))
        mtau =  (0.34+0.92*np.exp(-((v+71)/59)**2))/qt
        hinf =  1/(1 + np.exp(-(v+66)/-10))
        htau =  (8+49*np.exp(-((v+73)/23)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return mbeta

    @classmethod
    def alphah_KTst(cls,Vm):
        v = Vm
        celsius = 37
        gk_tstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  1/(1 + np.exp(-(v+0)/19))
        mtau =  (0.34+0.92*np.exp(-((v+71)/59)**2))/qt
        hinf =  1/(1 + np.exp(-(v+66)/-10))
        htau =  (8+49*np.exp(-((v+73)/23)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return halpha

    @classmethod
    def betah_KTst(cls,Vm):
        v = Vm
        celsius = 37
        gk_tstbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        v = v + 10
        minf =  1/(1 + np.exp(-(v+0)/19))
        mtau =  (0.34+0.92*np.exp(-((v+71)/59)**2))/qt
        hinf =  1/(1 + np.exp(-(v+66)/-10))
        htau =  (8+49*np.exp(-((v+73)/23)**2))/qt
        v = v - 10
        malpha = minf / mtau #only tau and inf provided in NMODL file
        mbeta = (1 - minf) / mtau #only tau and inf provided in NMODL file
        halpha = hinf / htau #only tau and inf provided in NMODL file
        hbeta = (1 - hinf) / htau #only tau and inf provided in NMODL file

        return hbeta




    @classmethod
    def alpham_NapEt2(cls,Vm):
        v = Vm
        celsius = 37
        gnap_et2bar = 0.00001
        qt = 2.3**((celsius-21)/10)
        minf = 1.0/(1+np.exp((v- -52.6)/-4.6))
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = 6*(1/(malpha + mbeta))/qt
        if(v == -17):
            v = v + 0.0001
        if(v == -64.4):
            v = v+0.0001
        hinf = 1.0/(1+np.exp((v- -48.8)/10))
        halpha = -2.88e-6 * (v + 17) / (1 - np.exp((v + 17)/4.63))
        hbeta = 6.94e-6 * (v + 64.4) / (1 - np.exp(-(v + 64.4)/2.63))
        htau = (1/(halpha + hbeta))/qt

        return malpha

    @classmethod
    def betam_NapEt2(cls,Vm):
        v = Vm
        celsius = 37
        gnap_et2bar = 0.00001
        qt = 2.3**((celsius-21)/10)
        minf = 1.0/(1+np.exp((v- -52.6)/-4.6))
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = 6*(1/(malpha + mbeta))/qt
        if(v == -17):
            v = v + 0.0001
        if(v == -64.4):
            v = v+0.0001
        hinf = 1.0/(1+np.exp((v- -48.8)/10))
        halpha = -2.88e-6 * (v + 17) / (1 - np.exp((v + 17)/4.63))
        hbeta = 6.94e-6 * (v + 64.4) / (1 - np.exp(-(v + 64.4)/2.63))
        htau = (1/(halpha + hbeta))/qt

        return mbeta

    @classmethod
    def alphah_NapEt2(cls,Vm):
        v = Vm
        celsius = 37
        gnap_et2bar = 0.00001
        qt = 2.3**((celsius-21)/10)
        minf = 1.0/(1+np.exp((v- -52.6)/-4.6))
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = 6*(1/(malpha + mbeta))/qt
        if(v == -17):
            v = v + 0.0001
        if(v == -64.4):
            v = v+0.0001
        hinf = 1.0/(1+np.exp((v- -48.8)/10))
        halpha = -2.88e-6 * (v + 17) / (1 - np.exp((v + 17)/4.63))
        hbeta = 6.94e-6 * (v + 64.4) / (1 - np.exp(-(v + 64.4)/2.63))
        htau = (1/(halpha + hbeta))/qt

        return halpha

    @classmethod
    def betah_NapEt2(cls,Vm):
        v = Vm
        celsius = 37
        gnap_et2bar = 0.00001
        qt = 2.3**((celsius-21)/10)
        minf = 1.0/(1+np.exp((v- -52.6)/-4.6))
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = 6*(1/(malpha + mbeta))/qt
        if(v == -17):
            v = v + 0.0001
        if(v == -64.4):
            v = v+0.0001
        hinf = 1.0/(1+np.exp((v- -48.8)/10))
        halpha = -2.88e-6 * (v + 17) / (1 - np.exp((v + 17)/4.63))
        hbeta = 6.94e-6 * (v + 64.4) / (1 - np.exp(-(v + 64.4)/2.63))
        htau = (1/(halpha + hbeta))/qt

        return hbeta




    @classmethod
    def alpham_NaTat(cls,Vm):
        v = Vm
        celsius = 37
        gnata_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = (1/(malpha + mbeta))/qt
        minf = malpha/(malpha + mbeta)
        if(v == -66):
            v = v + 0.0001
        halpha = (-0.015 * (v- -66))/(1-(np.exp((v- -66)/6)))
        hbeta  = (-0.015 * (-v -66))/(1-(np.exp((-v -66)/6)))
        htau = (1/(halpha + hbeta))/qt
        hinf = halpha/(halpha + hbeta)

        return malpha

    @classmethod
    def betam_NaTat(cls,Vm):
        v = Vm
        celsius = 37
        gnata_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = (1/(malpha + mbeta))/qt
        minf = malpha/(malpha + mbeta)
        if(v == -66):
            v = v + 0.0001
        halpha = (-0.015 * (v- -66))/(1-(np.exp((v- -66)/6)))
        hbeta  = (-0.015 * (-v -66))/(1-(np.exp((-v -66)/6)))
        htau = (1/(halpha + hbeta))/qt
        hinf = halpha/(halpha + hbeta)

        return mbeta

    @classmethod
    def alphah_NaTat(cls,Vm):
        v = Vm
        celsius = 37
        gnata_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = (1/(malpha + mbeta))/qt
        minf = malpha/(malpha + mbeta)
        if(v == -66):
            v = v + 0.0001
        halpha = (-0.015 * (v- -66))/(1-(np.exp((v- -66)/6)))
        hbeta  = (-0.015 * (-v -66))/(1-(np.exp((-v -66)/6)))
        htau = (1/(halpha + hbeta))/qt
        hinf = halpha/(halpha + hbeta)

        return halpha

    @classmethod
    def betah_NaTat(cls,Vm):
        v = Vm
        celsius = 37
        gnata_tbar = 0.00001
        qt = 2.3**((celsius-21)/10)
        if(v == -38):
            v = v+0.0001
        malpha = (0.182 * (v- -38))/(1-(np.exp(-(v- -38)/6)))
        mbeta  = (0.124 * (-v -38))/(1-(np.exp(-(-v -38)/6)))
        mtau = (1/(malpha + mbeta))/qt
        minf = malpha/(malpha + mbeta)
        if(v == -66):
            v = v + 0.0001
        halpha = (-0.015 * (v- -66))/(1-(np.exp((v- -66)/6)))
        hbeta  = (-0.015 * (-v -66))/(1-(np.exp((-v -66)/6)))
        htau = (1/(halpha + hbeta))/qt
        hinf = halpha/(halpha + hbeta)

        return hbeta




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
			'm_Ca' : lambda Vm, x: cls.alpham_Ca(Vm) * (1 - x['m_Ca']) - cls.betam_Ca(Vm) * x['m_Ca'],
			'h_Ca' : lambda Vm, x: cls.alphah_Ca(Vm) * (1 - x['h_Ca']) - cls.betah_Ca(Vm) * x['h_Ca'],
			'm_CaHVA' : lambda Vm, x: cls.alpham_CaHVA(Vm) * (1 - x['m_CaHVA']) - cls.betam_CaHVA(Vm) * x['m_CaHVA'],
			'h_CaHVA' : lambda Vm, x: cls.alphah_CaHVA(Vm) * (1 - x['h_CaHVA']) - cls.betah_CaHVA(Vm) * x['h_CaHVA'],
			'm_Ih' : lambda Vm, x: cls.alpham_Ih(Vm) * (1 - x['m_Ih']) - cls.betam_Ih(Vm) * x['m_Ih'],
			'm_Im' : lambda Vm, x: cls.alpham_Im(Vm) * (1 - x['m_Im']) - cls.betam_Im(Vm) * x['m_Im'],
			'm_NapEt2' : lambda Vm, x: cls.alpham_NapEt2(Vm) * (1 - x['m_NapEt2']) - cls.betam_NapEt2(Vm) * x['m_NapEt2'],
			'h_NapEt2' : lambda Vm, x: cls.alphah_NapEt2(Vm) * (1 - x['h_NapEt2']) - cls.betah_NapEt2(Vm) * x['h_NapEt2'],
			'm_NaTat' : lambda Vm, x: cls.alpham_NaTat(Vm) * (1 - x['m_NaTat']) - cls.betam_NaTat(Vm) * x['m_NaTat'],
			'h_NaTat' : lambda Vm, x: cls.alphah_NaTat(Vm) * (1 - x['h_NaTat']) - cls.betah_NaTat(Vm) * x['h_NaTat'],
			'm_NaTs2t' : lambda Vm, x: cls.alpham_NaTs2t(Vm) * (1 - x['m_NaTs2t']) - cls.betam_NaTs2t(Vm) * x['m_NaTs2t'],
			'h_NaTs2t' : lambda Vm, x: cls.alphah_NaTs2t(Vm) * (1 - x['h_NaTs2t']) - cls.betah_NaTs2t(Vm) * x['h_NaTs2t'],
			'm_CaLVAst' : lambda Vm, x: cls.alpham_CaLVAst(Vm) * (1 - x['m_CaLVAst']) - cls.betam_CaLVAst(Vm) * x['m_CaLVAst'],
			'h_CaLVAst' : lambda Vm, x: cls.alphah_CaLVAst(Vm) * (1 - x['h_CaLVAst']) - cls.betah_CaLVAst(Vm) * x['h_CaLVAst'],
			'm_KPst' : lambda Vm, x: cls.alpham_KPst(Vm) * (1 - x['m_KPst']) - cls.betam_KPst(Vm) * x['m_KPst'],
			'h_KPst' : lambda Vm, x: cls.alphah_KPst(Vm) * (1 - x['h_KPst']) - cls.betah_KPst(Vm) * x['h_KPst'],
			'm_KTst' : lambda Vm, x: cls.alpham_KTst(Vm) * (1 - x['m_KTst']) - cls.betam_KTst(Vm) * x['m_KTst'],
			'h_KTst' : lambda Vm, x: cls.alphah_KTst(Vm) * (1 - x['h_KTst']) - cls.betah_KTst(Vm) * x['h_KTst'],
			'm_SKv31' : lambda Vm, x: cls.alpham_SKv31(Vm) * (1 - x['m_SKv31']) - cls.betam_SKv31(Vm) * x['m_SKv31'],
			'm_KdShu2007' : lambda Vm, x: cls.alpham_KdShu2007(Vm) * (1 - x['m_KdShu2007']) - cls.betam_KdShu2007(Vm) * x['m_KdShu2007'],
			'h_KdShu2007' : lambda Vm, x: cls.alphah_KdShu2007(Vm) * (1 - x['h_KdShu2007']) - cls.betah_KdShu2007(Vm) * x['h_KdShu2007'],
		}

    # ------------------------------ Steady states ------------------------------

    @classmethod
    def steadyStates(cls):
        return {
			'm_Ca' : lambda Vm: cls.alpham_Ca(Vm) / (cls.alpham_Ca(Vm) + cls.betam_Ca(Vm)),
			'h_Ca' : lambda Vm: cls.alphah_Ca(Vm) / (cls.alphah_Ca(Vm) + cls.betah_Ca(Vm)),
			'm_CaHVA' : lambda Vm: cls.alpham_CaHVA(Vm) / (cls.alpham_CaHVA(Vm) + cls.betam_CaHVA(Vm)),
			'h_CaHVA' : lambda Vm: cls.alphah_CaHVA(Vm) / (cls.alphah_CaHVA(Vm) + cls.betah_CaHVA(Vm)),
			'm_Ih' : lambda Vm: cls.alpham_Ih(Vm) / (cls.alpham_Ih(Vm) + cls.betam_Ih(Vm)),
			'm_Im' : lambda Vm: cls.alpham_Im(Vm) / (cls.alpham_Im(Vm) + cls.betam_Im(Vm)),
			'm_NapEt2' : lambda Vm: cls.alpham_NapEt2(Vm) / (cls.alpham_NapEt2(Vm) + cls.betam_NapEt2(Vm)),
			'h_NapEt2' : lambda Vm: cls.alphah_NapEt2(Vm) / (cls.alphah_NapEt2(Vm) + cls.betah_NapEt2(Vm)),
			'm_NaTat' : lambda Vm: cls.alpham_NaTat(Vm) / (cls.alpham_NaTat(Vm) + cls.betam_NaTat(Vm)),
			'h_NaTat' : lambda Vm: cls.alphah_NaTat(Vm) / (cls.alphah_NaTat(Vm) + cls.betah_NaTat(Vm)),
			'm_NaTs2t' : lambda Vm: cls.alpham_NaTs2t(Vm) / (cls.alpham_NaTs2t(Vm) + cls.betam_NaTs2t(Vm)),
			'h_NaTs2t' : lambda Vm: cls.alphah_NaTs2t(Vm) / (cls.alphah_NaTs2t(Vm) + cls.betah_NaTs2t(Vm)),
			'm_CaLVAst' : lambda Vm: cls.alpham_CaLVAst(Vm) / (cls.alpham_CaLVAst(Vm) + cls.betam_CaLVAst(Vm)),
			'h_CaLVAst' : lambda Vm: cls.alphah_CaLVAst(Vm) / (cls.alphah_CaLVAst(Vm) + cls.betah_CaLVAst(Vm)),
			'm_KPst' : lambda Vm: cls.alpham_KPst(Vm) / (cls.alpham_KPst(Vm) + cls.betam_KPst(Vm)),
			'h_KPst' : lambda Vm: cls.alphah_KPst(Vm) / (cls.alphah_KPst(Vm) + cls.betah_KPst(Vm)),
			'm_KTst' : lambda Vm: cls.alpham_KTst(Vm) / (cls.alpham_KTst(Vm) + cls.betam_KTst(Vm)),
			'h_KTst' : lambda Vm: cls.alphah_KTst(Vm) / (cls.alphah_KTst(Vm) + cls.betah_KTst(Vm)),
			'm_SKv31' : lambda Vm: cls.alpham_SKv31(Vm) / (cls.alpham_SKv31(Vm) + cls.betam_SKv31(Vm)),
			'm_KdShu2007' : lambda Vm: cls.alpham_KdShu2007(Vm) / (cls.alpham_KdShu2007(Vm) + cls.betam_KdShu2007(Vm)),
			'h_KdShu2007' : lambda Vm: cls.alphah_KdShu2007(Vm) / (cls.alphah_KdShu2007(Vm) + cls.betah_KdShu2007(Vm)),
		}

    # ------------------------------ Membrane currents ------------------------------
    @classmethod
    def i_Ca(cls,m_Ca,h_Ca,Vm,gcabar = 0.00001):
        ''' iCa current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_Ca #
        h = h_Ca #
        gca = gcabar*m*m*h
        ica = gca*(v-eca)

        return ica

    @classmethod
    def g_Ca(cls,m_Ca,h_Ca,Vm,gcabar = 0.00001):
        ''' gCa conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_Ca #
        h = h_Ca #
        gcabar = 0.00001
        gca = gcabar*m*m*h
        ica = gca*(v-eca)

        return gca

    @classmethod
    def i_CaHVA(cls,m_CaHVA,h_CaHVA,Vm,gca_hvabar = 0.00001):
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
    def i_CaLVAst(cls,m_CaLVAst,h_CaLVAst,Vm,gca_lvastbar = 0.00001):
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
    def i_Ih(cls,m_Ih,Vm,gihbar = 0.00001):
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
    def i_Im(cls,m_Im,Vm,gimbar = 0.00001):
        ''' iIm current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_Im #
        gim = gimbar*m
        ik = gim*(v-ek)

        return ik

    @classmethod
    def g_Im(cls,m_Im,Vm,gimbar = 0.00001):
        ''' gIm conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_Im #
        gimbar = 0.00001
        gim = gimbar*m
        ik = gim*(v-ek)

        return gim

    @classmethod
    def i_KdShu2007(cls,m_KdShu2007,h_KdShu2007,Vm,gkbar = 0.1):
        ''' iKdShu2007 current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_KdShu2007 #
        h = h_KdShu2007 #
        ek = -100	            
        vhalfm=-43
        km=8
        vhalfh=-67
        kh=7.3
        q10=2.3
        ik = gkbar * m*h*(v-ek)

        return ik

    @classmethod
    def g_KdShu2007(cls,m_KdShu2007,h_KdShu2007,Vm,gkbar = 0.1):
        ''' gKdShu2007 conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_KdShu2007 #
        h = h_KdShu2007 #
        gkbar = 0.1
        ek = -100	            
        vhalfm=-43
        km=8
        vhalfh=-67
        kh=7.3
        q10=2.3
        ik = gkbar * m*h*(v-ek)

        return None

    @classmethod
    def i_KPst(cls,m_KPst,h_KPst,Vm,gk_pstbar = 0.00001):
        ''' iKPst current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_KPst #
        h = h_KPst #
        gk_pst = gk_pstbar*m*m*h
        ik = gk_pst*(v-ek)

        return ik

    @classmethod
    def g_KPst(cls,m_KPst,h_KPst,Vm,gk_pstbar = 0.00001):
        ''' gKPst conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_KPst #
        h = h_KPst #
        gk_pstbar = 0.00001
        gk_pst = gk_pstbar*m*m*h
        ik = gk_pst*(v-ek)

        return gk_pst

    @classmethod
    def i_KTst(cls,m_KTst,h_KTst,Vm,gk_tstbar = 0.00001):
        ''' iKTst current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_KTst #
        h = h_KTst #
        gk_tst = gk_tstbar*(m**4)*h
        ik = gk_tst*(v-ek)

        return ik

    @classmethod
    def g_KTst(cls,m_KTst,h_KTst,Vm,gk_tstbar = 0.00001):
        ''' gKTst conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_KTst #
        h = h_KTst #
        gk_tstbar = 0.00001
        gk_tst = gk_tstbar*(m**4)*h
        ik = gk_tst*(v-ek)

        return gk_tst

    @classmethod
    def i_NapEt2(cls,m_NapEt2,h_NapEt2,Vm,gnap_et2bar = 0.00001):
        ''' iNapEt2 current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_NapEt2 #
        h = h_NapEt2 #
        gnap_et2 = gnap_et2bar*m*m*m*h
        ina = gnap_et2*(v-ena)

        return ina

    @classmethod
    def g_NapEt2(cls,m_NapEt2,h_NapEt2,Vm,gnap_et2bar = 0.00001):
        ''' gNapEt2 conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_NapEt2 #
        h = h_NapEt2 #
        gnap_et2bar = 0.00001
        gnap_et2 = gnap_et2bar*m*m*m*h
        ina = gnap_et2*(v-ena)

        return gnap_et2

    @classmethod
    def i_NaTat(cls,m_NaTat,h_NaTat,Vm,gnata_tbar = 0.00001):
        ''' iNaTat current '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_NaTat #
        h = h_NaTat #
        gnata_t = gnata_tbar*m*m*m*h
        ina = gnata_t*(v-ena)

        return ina

    @classmethod
    def g_NaTat(cls,m_NaTat,h_NaTat,Vm,gnata_tbar = 0.00001):
        ''' gNaTat conductance '''
        v = Vm
        celsius = 37
        ek = -85.0
        ena = 50.0
        eca = 132.4579341637009
        m = m_NaTat #
        h = h_NaTat #
        gnata_tbar = 0.00001
        gnata_t = gnata_tbar*m*m*m*h
        ina = gnata_t*(v-ena)

        return gnata_t

    @classmethod
    def i_NaTs2t(cls,m_NaTs2t,h_NaTs2t,Vm,gnats2_tbar = 0.00001):
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
    def i_SKv31(cls,m_SKv31,Vm,gskv3_1bar = 0.00001):
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
        ra = 100
        cm = 1
        g_pas = 3e-05 * 1e4 #S/cm2 -> S/m2 
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
			'i_Ca': lambda Vm, x, g_bar: cls.i_Ca(x['m_Ca'], x['h_Ca'], Vm) if g_bar is None else cls.i_Ca(x['m_Ca'], x['h_Ca'], Vm, g_bar),
			'i_CaHVA': lambda Vm, x, g_bar: cls.i_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm) if g_bar is None else cls.i_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm, g_bar),
			'i_CaLVAst': lambda Vm, x, g_bar: cls.i_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm) if g_bar is None else cls.i_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm, g_bar),
			'i_Ih': lambda Vm, x, g_bar: cls.i_Ih(x['m_Ih'], Vm) if g_bar is None else cls.i_Ih(x['m_Ih'], Vm, g_bar),
			'i_Im': lambda Vm, x, g_bar: cls.i_Im(x['m_Im'], Vm) if g_bar is None else cls.i_Im(x['m_Im'], Vm, g_bar),
			'i_KdShu2007': lambda Vm, x, g_bar: cls.i_KdShu2007(x['m_KdShu2007'], x['h_KdShu2007'], Vm) if g_bar is None else cls.i_KdShu2007(x['m_KdShu2007'], x['h_KdShu2007'], Vm, g_bar),
			'i_KPst': lambda Vm, x, g_bar: cls.i_KPst(x['m_KPst'], x['h_KPst'], Vm) if g_bar is None else cls.i_KPst(x['m_KPst'], x['h_KPst'], Vm, g_bar),
			'i_KTst': lambda Vm, x, g_bar: cls.i_KTst(x['m_KTst'], x['h_KTst'], Vm) if g_bar is None else cls.i_KTst(x['m_KTst'], x['h_KTst'], Vm, g_bar),
			'i_NapEt2': lambda Vm, x, g_bar: cls.i_NapEt2(x['m_NapEt2'], x['h_NapEt2'], Vm) if g_bar is None else cls.i_NapEt2(x['m_NapEt2'], x['h_NapEt2'], Vm, g_bar),
			'i_NaTat': lambda Vm, x, g_bar: cls.i_NaTat(x['m_NaTat'], x['h_NaTat'], Vm) if g_bar is None else cls.i_NaTat(x['m_NaTat'], x['h_NaTat'], Vm, g_bar),
			'i_NaTs2t': lambda Vm, x, g_bar: cls.i_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm) if g_bar is None else cls.i_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm, g_bar),
			'i_SKv31': lambda Vm, x, g_bar: cls.i_SKv31(x['m_SKv31'], Vm) if g_bar is None else cls.i_SKv31(x['m_SKv31'], Vm, g_bar),
			'i_pas': lambda Vm, x, g_bar: cls.i_pas(Vm)
        }

    @classmethod
    def conductances(cls):
        return {
			'g_Ca': lambda Vm, x, g_bar: cls.g_Ca(x['m_Ca'], x['h_Ca'], Vm) if g_bar is None else cls.g_Ca(x['m_Ca'], x['h_Ca'], Vm, g_bar),
			'g_CaHVA': lambda Vm, x, g_bar: cls.g_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm) if g_bar is None else cls.g_CaHVA(x['m_CaHVA'], x['h_CaHVA'], Vm, g_bar),
			'g_CaLVAst': lambda Vm, x, g_bar: cls.g_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm) if g_bar is None else cls.g_CaLVAst(x['m_CaLVAst'], x['h_CaLVAst'], Vm, g_bar),
			'g_Ih': lambda Vm, x, g_bar: cls.g_Ih(x['m_Ih'], Vm) if g_bar is None else cls.g_Ih(x['m_Ih'], Vm, g_bar),
			'g_Im': lambda Vm, x, g_bar: cls.g_Im(x['m_Im'], Vm) if g_bar is None else cls.g_Im(x['m_Im'], Vm, g_bar),
			'g_KdShu2007': lambda Vm, x, g_bar: cls.g_KdShu2007(x['m_KdShu2007'], x['h_KdShu2007'], Vm) if g_bar is None else cls.g_KdShu2007(x['m_KdShu2007'], x['h_KdShu2007'], Vm, g_bar),
			'g_KPst': lambda Vm, x, g_bar: cls.g_KPst(x['m_KPst'], x['h_KPst'], Vm) if g_bar is None else cls.g_KPst(x['m_KPst'], x['h_KPst'], Vm, g_bar),
			'g_KTst': lambda Vm, x, g_bar: cls.g_KTst(x['m_KTst'], x['h_KTst'], Vm) if g_bar is None else cls.g_KTst(x['m_KTst'], x['h_KTst'], Vm, g_bar),
			'g_NapEt2': lambda Vm, x, g_bar: cls.g_NapEt2(x['m_NapEt2'], x['h_NapEt2'], Vm) if g_bar is None else cls.g_NapEt2(x['m_NapEt2'], x['h_NapEt2'], Vm, g_bar),
			'g_NaTat': lambda Vm, x, g_bar: cls.g_NaTat(x['m_NaTat'], x['h_NaTat'], Vm) if g_bar is None else cls.g_NaTat(x['m_NaTat'], x['h_NaTat'], Vm, g_bar),
			'g_NaTs2t': lambda Vm, x, g_bar: cls.g_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm) if g_bar is None else cls.g_NaTs2t(x['m_NaTs2t'], x['h_NaTs2t'], Vm, g_bar),
			'g_SKv31': lambda Vm, x, g_bar: cls.g_SKv31(x['m_SKv31'], Vm) if g_bar is None else cls.g_SKv31(x['m_SKv31'], Vm, g_bar),
			'g_pas': lambda Vm, x, g_bar: cls.g_pas(Vm)
        }