import pandas as pd
import numpy as np
import hysys_python.hysys_util as hu


# Class for calculating all the purchased equipment costs

class PEC():
    def __init__(self, fsheet, cepci, custom_units={}, ignore=[]):
        self.cc = self.calc_cost_curves()
        # self.fsheet = fsheet
        self.cepci = cepci
        self.calc_unit_pec(fsheet, ignore=ignore, special_list=custom_units)

    # reading in the cost curve csv file
    def calc_cost_curves(self):
        cc = pd.read_csv('hysys_python/reference/cost_curves.csv',index_col=0,header=0,usecols=[0,4,5,6])
        return cc

    # input is the unit type as described in hysys, the size S as specified in Towler and the cost curve pd dataframe
    def calc_pec(self,unit,S,cc):
        equip = cc.loc[unit]
        a = equip.at['a']
        b = equip.at['b']
        n = equip.at['n']

        pec = a + b*S**n

        return pec

    # for distillation columns
    def calc_shell_thickness(self,p,d):
        # inputs are pressure = p [bara] and diameter = d [m]
        # calculation from turton chp 7.3.4
        
        if d != 0:
            CA = 0.00315 # corrosion allowance
            S = 944 # maximum allowable stress for carbon steel [bara]
            E = 0.9 # weld efficiency
            p = 1.1*p # must be 10% above design

            t = p*d/(2*S*E - 1.2*p) + CA # thickness [m]

            # comparing to the minimum thickness
            min = self.calc_min_thickness(d)
            if t < min:
                t = min
        else:
            t = 0

        return t

    # get the minimum shell thickness
    def calc_min_thickness(self,d):
        # From minimum thicknesses specified by towler2022
        if d < 1:
            min = 0.005
        elif d < 2:
            min = 0.007
        elif d < 2.5:
            min = 0.009
        elif d < 3:
            min = 0.01
        else:
            min = 0.012

        return min

    # Calculating the mass of a vertical cylindrical shell with hemispherical heads
    def calc_shell_mass(self,t,d,h,col=False):
        density = 7850 # carbon steel wikipedia

        # volume of cylindrical section
        vol_body = np.pi/4*h*((d+t)**2 - d**2)

        # volume of heads i.e. sphere
        vol_head = 4/3*np.pi*((d+t)**3 - d**3)
        if col:
            vol_head *= 0.5 # only want top/bottom head for rectifying and stripping section

        # total shell mass
        mass = density * (vol_body + vol_head)

        return mass

    # Pump
    def calc_pump_pec(self,hysys_unit,cc,cepci):

        flow = hysys_unit.VolumeFlowValue*1000 # m3/s to l/s
        power = hysys_unit.WorkValue # kW

        pump = self.calc_pec('Pump (centrifugal)',flow,cc)
        pump_driver = self.calc_pec('Pump motor',power,cc)

        pump_pec = round((pump + pump_driver)*cepci/532.9)
        isbl_hand = 4
        pump_isbl = round(isbl_hand*pump_pec)

        return pump_pec, pump_isbl

    # Compressor
    def calc_comp_pec(self,hysys_unit,cc,cepci):
        power = hysys_unit.EnergyValue # kW

        if power <= 0:
            comp_pec = 0
            comp_isbl = 0
        else:
            comp = self.calc_pec('Compressor',power,cc)

            comp_pec = round(comp*cepci/532.9)
            isbl_hand = 2.5
            comp_isbl = round(comp_pec*isbl_hand)

        return comp_pec, comp_isbl

    def calc_exp_pec(self,hysys_unit,cc,cepci):
        try:
            power = hysys_unit.EnergyValue # kW
        except:
            power = 0
        
        if power <= 0:
            exp = 0
        else:
            exp = self.calc_pec('Turbine',power,cc)

        exp_pec = round(exp*cepci/532.9)
        isbl_fac = 3.5 # turton2012
        exp_isbl = round(exp_pec*isbl_fac)

        return exp_pec, exp_isbl


    # Flash/Empty Vessel
    def calc_vessel_pec(self, fsheet, hysys_unit, cc, cepci):
        d = hu.get_diameter_separator(fsheet, hysys_unit)
        h = 1.5*d # from aspen hysys

        if d != 0: # i.e. does not exist
            p = hysys_unit.VesselPressureValue # kPa
            t = self.calc_shell_thickness(p/100,d)
            mass = self.calc_shell_mass(t,d,h)

            vessel = self.calc_pec('Vessel (vertical)',mass,cc)

            vessel_pec = round(vessel*cepci/532.9)
            isbl_hand = 4
            vessel_isbl = round(vessel*isbl_hand)
        else:
            vessel_pec = 0
            vessel_isbl = 0

        return vessel_pec, vessel_isbl

    # PFR Reactor (assume pressure vessel)
    def calc_pfr_pec(self, hysys_unit,cc,cepci,hysys_sim): # have to input simulation as well for feed
        vol = hysys_unit.TotalVolumeValue # m3
        feed_name = hysys_unit.AttachedFeeds.Names[0]
        feed = hysys_sim.MaterialStreams(feed_name)

        p = feed.PressureValue

        # assume 5.5 ratio for height to diameter (from aspen hysys)
        # d = (4V/5.5pi)^0.333333
        d = (4*vol/(5.5*np.pi))**0.333333333
        
        t = self.calc_shell_thickness(p/100,d)

        mass = self.calc_shell_mass(t,d,5.5*d)
        # print(mass)

        vessel = self.calc_pec('Vessel (vertical)',mass,cc)

        vessel_pec = round(vessel*cepci/532.9)
        isbl_hand = 4
        vessel_isbl = round(vessel*isbl_hand)

        return vessel_pec, vessel_isbl

    # Equation for an MTO or OCP reactor section (from hannula2015)
    def calc_mto_pec(self, hysys_unit,cc, cepci,hysys_sim):
        S_0 = 86 # t/h
        C_0 = 90e6 # EUR

        feed = hysys_unit.AttachedFeeds.Names[0]
        try:
            S = hysys_sim.ms[feed].MassFlowValue * 3.6 # t/h
        except:
            S = hysys_sim.MaterialStreams(feed).MassFlowValue * 3.6

        C = C_0 * (S/S_0)**0.85 # EUR
        C = C/1.11 # average EUR/USD exchange rate 2015

        mto_tci = round(C*cepci/556.8) # paper gives total investment cost, not pec/isbl
        # Back-calculating using Sinnott factors for equivalent pec and isbl
        mto_pec = mto_tci/1.82 # osbl, e/c and contingency factors
        isbl_factor = 1 # unknown from paper
        mto_isbl = mto_pec*isbl_factor

        return mto_pec, mto_isbl

    # Function for furnaces (also can be used for some reactors)
    def calc_furnace_pec(self, hysys_unit,cc,cepci,hysys_sim):
        duty = np.absolute(hysys_unit.HeatFlowValue/1000) # kW to MW - absolute needed if different heat direction e.g. pfr vs gibbs

        furnace = self.calc_pec('Furnace',duty,cc)

        furnace_pec = round(furnace*cepci/532.9)
        isbl_hand = 2
        furnace_isbl = furnace_pec*isbl_hand

        return furnace_pec, furnace_isbl

    # Function for calculating the cost of an ATR or TR reactor - from baliban2010
    def calc_atr_pec(self, hysys_unit,cc,cepci,hysys_sim):
        S_0 = 430639
        C_0 = 3.18e6 # MM$

        # Getting output flowrate in Nm3/h - only have standard but close enough
        prod = hysys_unit.AttachedProducts.Names[1]
        try:
            S = hysys_sim.ms[prod].StdGasFlowValue * 3600 # m3/h
        except:
            S = hysys_sim.MaterialStreams(prod).StdGasFlowValue * 3600

        C = C_0 * (S/S_0)**0.67 # USD 2010

        cepci_2010 = 550.8
        pec = round(C*cepci/cepci_2010)
        isbl_factor = 1
        isbl = pec*isbl_factor

        return pec, isbl

    # Function for calculating the cost of an OCM reactor
    # Used stansch kinetics and found volume where O2 fraction = 0.01 at outlet
    def calc_ocm_pec(self, hysys_unit, cc, cepci, hysys_sim):
        # Getting the feed pressure
        
        try:
            feed = hysys_unit.AttachedFeeds[0]
            p = hysys_sim.ms[feed].PressureValue # kPa
            feed_flow = hysys_sim.ms[feed].MassFlowValue
        except:
            feed = hysys_unit.AttachedFeeds.Names[0]
            p = hysys_sim.MaterialStreams(feed).PressureValue
            feed_flow = hysys_sim.MaterialStreams(feed).MassFlowValue

        # Length of OCM reactor = 6.7382m with d = 0.0254m 
        # at Tin = 750 degC, CH4:O2 ratio = 10, and feed flow = 9595.67555707372 kg/h
        d_ref = 0.0254
        l_ref = 6.7382
        feed_ref = 9595.67555707372
        vol_ref = np.pi/4*l_ref*d_ref**2
        vol = vol_ref*(feed_flow*3600/feed_ref)

        # assume 5.5 ratio for height to diameter (from aspen hysys)
        # d = (4V/5.5pi)^0.333333
        d = (4*vol/(5.5*np.pi))**0.333333333
        
        t = self.calc_shell_thickness(p/100,d)

        mass = self.calc_shell_mass(t,d,5.5*d)
        # print(mass)

        vessel = self.calc_pec('Vessel (vertical)',mass,cc)

        vessel_pec = round(vessel*cepci/532.9)
        isbl_hand = 4
        vessel_isbl = round(vessel*isbl_hand)

        return vessel_pec, vessel_isbl
    
    # Function for calculating cost of a methanator reactor section from spallina2017
    def calc_methanator_pec(self, hysys_unit, cc, cepci, hysys_sim):
        # Get the volume flow of the feed
        feed = hysys_unit.AttachedFeeds.Names[0]
        feed_flow = hysys_sim.MaterialStreams(feed).ActualVolumeFlowValue # m3/s

        pec_EUR_2017 = 64.6e6*(feed_flow/6.96)**0.67
        USD_EUR = 0.849259 # exchange rate on date of publication (2017/12/15)
        cepci_2017 = 567.5

        pec = pec_EUR_2017*USD_EUR*cepci/cepci_2017
        isbl_factor = 2 # from the original source braington2011, bare erected cost is twice equipment cost
        isbl = pec*isbl_factor

        return pec, isbl
    
    # Calculating cost of ocm reactor as PBR from spreadsheet info
    def calc_ocm_ss_pec(self, hysys_unit, cc, cepci, hysys_sim):
        vol = hysys_unit.Cell(1,1).CellValue # m3
        p = hysys_unit.Cell(1,3).CellValue # kPa

        # assume 5.5 ratio for height to diameter (from aspen hysys)
        # d = (4V/5.5pi)^0.333333
        d = (4*vol/(5.5*np.pi))**0.333333333
        
        t = self.calc_shell_thickness(p/100,d)

        mass = self.calc_shell_mass(t,d,5.5*d)
        # print(mass)

        vessel = self.calc_pec('Vessel (vertical)',mass,cc)

        vessel_pec = round(vessel*cepci/532.9)
        isbl_hand = 4
        vessel_isbl = round(vessel*isbl_hand)

        return vessel_pec, vessel_isbl

    # Function for calculating the cost of the HEN (assume HEX U-tube)
    def add_hen_pec(self, hen):
        area = hen.area
        nmin = hen.Nmin
        outside_range = True

        while outside_range:
            area_per_hex = area/nmin
            if area_per_hex < 1000:
                outside_range = False
            else:
                nmin *= 2

        hex = self.calc_pec('HEX (U-tube)',area_per_hex,self.cc)
        hen_pec = round(nmin*hex*self.cepci/532.9)
        isbl_hand = 3.5
        hen_isbl = hen_pec*isbl_hand

        hen_df = pd.DataFrame({'Name':'HEN','PEC':hen_pec,'ISBL':hen_isbl}, [0])
        self.PEC_df = pd.concat([self.PEC_df, hen_df], ignore_index=True)

    # Function that uses the DisCol class from hysys_util to cost distillation columns
    def calc_col_pec(self, dc):
        cc = self.cc
        cepci = self.cepci

        # First need to get the shell thickness and mass
        rec_thickness = self.calc_shell_thickness(dc.Pressure/100, dc.DiameterRectifying)
        str_thickness = self.calc_shell_thickness(dc.Pressure/100, dc.DiameterStripping)
        rec_mass = self.calc_shell_mass(rec_thickness, dc.DiameterRectifying, dc.HeightRectifying, col=True)
        str_mass = self.calc_shell_mass(str_thickness, dc.DiameterStripping, dc.HeightStripping, col=True)

        tot_mass = rec_mass + str_mass

        # Get PEC of the column
        vessel = self.calc_pec('Vessel (vertical)',tot_mass,cc)

        # Add on the trays
        no_trays_rec = dc.FeedStage
        no_trays_str = dc.NoStages - dc.FeedStage
        tray_rec = no_trays_rec*self.calc_pec('Sieve trays', dc.DiameterRectifying, cc)
        tray_str = no_trays_str*self.calc_pec('Sieve trays', dc.DiameterStripping, cc)

        col_cost = vessel + tray_str + tray_rec

        # total cost
        vessel_pec = round(col_cost*cepci/532.9)
        isbl_hand = 4
        vessel_isbl = round(vessel*isbl_hand)

        return vessel_pec, vessel_isbl

    # Looping through all unit ops to get the total pec
    def calc_unit_pec(self, hysys_sim, ignore=[], special_list={}):
        op_names = []
        op_pec = []
        op_isbl = []
        # hysys_sim = self.fsheet
        cc = self.cc
        cepci = self.cepci

        # Adding a check is it's acting on the json file or the hysys simulation
        try:
            operations = hysys_sim.op
        except:
            operations = hysys_sim.Operations.Names

        for op_name in operations:
            if op_name in ignore:
                continue

            try:
                op = hysys_sim.op[op_name]
            except:
                op = hysys_sim.Operations(op_name)

            if op_name in special_list.keys():
                # # Getting correct method
                method_name = 'calc_' + special_list[op_name]
                pec, isbl = getattr(self, method_name)(op,cc,cepci,hysys_sim)

            elif op.TypeName == 'compressor':
                pec, isbl = self.calc_comp_pec(op,cc,cepci)
            elif op.TypeName == 'pumpop':
                pec, isbl = self.calc_pump_pec(op,cc,cepci)
            elif op.TypeName == 'flashtank':
                pec, isbl = self.calc_vessel_pec(hysys_sim, op,cc,cepci)
            elif op.TypeName == 'expandop':
                pec, isbl = self.calc_exp_pec(op,cc,cepci)
            elif op.TypeName == 'distillation':
                dc = hu.DisCol(op, hysys_sim)
                dc.size()
                pec, isbl = self.calc_col_pec(dc)
            else:
                continue
            
            # if op_found == True:
            op_names.append(op_name)
            op_pec.append(pec)
            op_isbl.append(isbl)

        # Putting results into a dataframe
        costs = {'Name':op_names,'PEC':op_pec,'ISBL':op_isbl}
        cost_df = pd.DataFrame(costs)

        self.PEC_df = cost_df
