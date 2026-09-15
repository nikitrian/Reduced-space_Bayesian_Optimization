# Code for getting the MER (and eventually the HEN design)

# Pandas dataframe as basis for calculation for MER

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class HEN():
    # Overall calculation
    def __init__(self, hysys_sim, rct_list='', fluids='',htc_spec={}, ignore_hex=[]):
        # First calculate the MER
        self.hex_data, self.htc_spec = self.problem_table(hysys_sim, rct_list, htc_spec=htc_spec, ignore_hex=ignore_hex)
        self.t_intervals = self.temp_intervals(self.hex_data)
        self.temp_table, self.shift_cascade = self.interval_table(self.hex_data, self.t_intervals)

        self.Qh = self.shift_cascade['H'].iloc[0]
        self.Qc = self.shift_cascade['H'].iloc[-1]

        # Seeing if there's a pinch
        pinch_idx = self.shift_cascade.loc[self.shift_cascade.H < 1e-6].index.values[0]
        if pinch_idx in [0,len(self.shift_cascade)]: # if pinch at start or end i.e. no hot or cold utility
            self.pinch_temp = np.NaN
            no_utils = 1
        else:
            self.pinch_temp = self.shift_cascade['Tshift'][pinch_idx]
            no_utils = 2

        # Getting the number of units
        self.Nmin = self.min_units(self.hex_data, self.temp_table, self.pinch_temp) + no_utils

        # Adding in the utility streams and htc values
        self.hex_data = self.add_htc(hysys_sim, self.hex_data, fluids, htc_spec=self.htc_spec)
        h_info = ['fh', self.Qh]
        c_info = ['cw', self.Qc]
        self.hex_data = self.hen_utility(self.hex_data, h_info, c_info)

        # Getting the enthalpy invervals and calculating the area
        self.enthalpy_table = self.enthalpy_intervals(self.hex_data)
        self.enthalpy_interval_table = self.enthalpy_interval_area(self.enthalpy_table, self.hex_data)
        self.area = self.enthalpy_interval_table.Area.sum()

    # Function to plot the grand composite curve
    def plot_gcc(self):
        fig, ax = plt.subplots()
        self.shift_cascade.plot(x='H', y='Tshift', ax=ax)
        ax.set_xlim([0,None])
        plt.show()

    # Function to loop through and add areas of multistream/LNG exchangers e.g. as used in refrigeration cycle
    def add_multistream(self, fsheet, u_values):
        # op_names = fsheet.Operations.Names
        area = 0
        units = 0
        for op_name in u_values.keys():
            op = fsheet.Operations(op_name)
            # if op.TypeName in ['lngop', 'heatexop']:
            u = u_values[op_name] # in kW/m2 K
            ua = op.UAValue
            area += ua/u
            units += 1
        
        # update results
        self.area += area
        self.Nmin += units



    '''
    Need a function to check that hot and cold streams have been correctly assigned - e.g. sometimes fluid package shows stream decreasing in temperature during pure component phase change in a heater
    Function will return an adjusted feed temperature for the HEN - want to keep target temperature the same
    '''
    def check_correct_hex_temps(self, hex):
        # get heater type
        hex_type = hex.TypeName
        feed_temp = hex.FeedTemperatureValue
        prod_temp = hex.ProductTemperatureValue

        # Check for inconsistencies
        if hex_type == 'heaterop' and (prod_temp <= feed_temp):
            feed_temp = prod_temp - 0.001
        elif hex_type == 'coolerop' and (prod_temp >= feed_temp):
            feed_temp = prod_temp + 0.001
        
        return feed_temp

    # Function to generate the initial problem table
    def problem_table(self, hysys_sim, rct_list=[],deltaT=10, htc_spec={}, ignore_hex=[]):
        operations = hysys_sim.Operations.Names
        names = []
        feed_temps = []
        prod_temps = []
        duties = []
        types = []
        shifts = []

        # Looping through and getting data on heat exchangers
        for op_name in operations:
            # if op_name in ignore_hex:
            #     continue
            op = hysys_sim.Operations(op_name)

            if op.TypeName in ['heaterop','coolerop'] and op.DutyValue != 0 and op.name not in ignore_hex:
                names.append(op.name)
                feed_temps.append(self.check_correct_hex_temps(op))
                prod_temps.append(op.ProductTemperatureValue)
                duties.append(op.DutyValue)
                types.append(op.TypeName)

                # Calculating correct shift temp
                if op.TypeName == 'coolerop':
                    Tshift = -deltaT/2
                else:
                    Tshift = deltaT/2
                shifts.append(Tshift)

            # Section to include reboilers and condensers
            elif op.TypeName == 'distillation':
                if op.name in ignore_hex:
                    continue

                if op.name + '_R' not in ignore_hex:
                    names.append(op.name + '_Reboiler')
                    feed_temps.append(op.ColumnFlowsheet.MaterialStreams('To Reboiler').TemperatureValue)
                    prod_temps.append(op.ColumnFlowsheet.MaterialStreams('Boilup').TemperatureValue)
                    duties.append(op.ColumnFlowsheet.Operations('Reboiler').HeatFlowValue)
                    types.append('heaterop')
                    shifts.append(deltaT/2)

                # Condensers may be ignored if they're already considered in the refrigeration cycle
                if op.name + '_C' not in ignore_hex:
                    names.append(op.name + '_Condenser')
                    feed_temps.append(op.ColumnFlowsheet.MaterialStreams('To Condenser').TemperatureValue)
                    prod_temps.append(op.ColumnFlowsheet.MaterialStreams('Reflux').TemperatureValue)
                    duties.append(op.ColumnFlowsheet.Operations('Condenser').HeatFlowValue)
                    types.append('coolerop')
                    shifts.append(-deltaT/2)


        # Adding reactor data
        if rct_list != []:
            rct_names, rct_feed_temps, rct_prod_temps, rct_duties, rct_shifts, rct_types, htc_spec = self.create_reactor_duty_streams(hysys_sim, rct_list, htc_spec)

            names += rct_names
            feed_temps += rct_feed_temps
            prod_temps += rct_prod_temps
            duties += rct_duties
            shifts += rct_shifts
            types += rct_types
    
        # Making the dataframe
        hex_data = {'Name':names,'Type':types,'Duty':duties,'Tin':feed_temps,'Tout':prod_temps,'Tshift':shifts}
        hex_data = pd.DataFrame(hex_data)

        # Adding shifted temps
        hex_data['Tin_shift'] = hex_data['Tin'] + hex_data['Tshift']
        hex_data['Tout_shift'] = hex_data['Tout'] + hex_data['Tshift']
        hex_data = hex_data.drop('Tshift',axis=1)

        # Adding Fcp with signs for heat demand/cooling
        hex_data['FCp'] = hex_data['Duty'] / abs(hex_data['Tin'] - hex_data['Tout'])

        return hex_data, htc_spec

    # # Making a function to create heating/cooling streams that will connect with reactors
    def get_rct_type(self, hysys_sim, rct_name):
        # Getting relevant information
        rct = hysys_sim.Operations(rct_name)
        feed_name = rct.AttachedFeeds.Names[0]
        prod_name = rct.AttachedProducts.Names[0]
        feed_t = hysys_sim.MaterialStreams(feed_name).TemperatureValue
        prod_t = hysys_sim.MaterialStreams(prod_name).TemperatureValue

        # First determining if its exo or endothermic
        duty = rct.HeatFlowValue
        if rct.TypeName in ['conversionreactorop','gibbsreactorop']:
            if duty > 0:
                rct_type = 'endo'
            elif duty < 0:
                rct_type = 'exo'
        elif rct.TypeName in ['pfreactorop']:
            if duty > 0:
                rct_type = 'exo'
            elif duty < 0:
                rct_type = 'endo'
        else:
            print('Reactor type not defined')

        if rct_type == 'exo':
            T = min(feed_t, prod_t) - 5 # adjust by Tmin/2
        elif rct_type == 'endo':
            T = max(feed_t, prod_t) + 5

        return rct_type, T, duty

            
    # Another function to create the stream info for the reactor heat integration
    def reactor_duty_stream(self, T,rct_type):
        if rct_type == 'exo': # need to cool stream
            if T > 250: # Use HP steam
                Tin = 250
                Tout = 249
                htc = 21600/3.6 # W/m2 K
                stream_name = 'HP_gen'
            elif T > 175 and T < 250: # Use MP steam
                Tin = 175
                Tout = 174
                htc = 21600/3.6 # W/m2 K
                stream_name = 'MP_gen'
            elif T > 125 and T < 175: # use LP steam
                Tin = 125
                Tout = 124
                htc = 21600/3.6 # W/m2 K
                stream_name = 'LP_gen'
            else: # cooling water
                Tin = 30
                Tout = 35
                htc = 13500/3.6 # W/m2 K
                stream_name = 'cw'
            stream_type = 'coolerop'
        elif rct_type == 'endo': # need to heat stream
            if T < 124: # LP steam
                Tin = 124
                Tout = 125
                htc = 21600/3.6 # W/m2 K
                stream_name = 'LP'
            elif T > 124 and T < 174:
                Tin = 174
                Tout = 175
                htc = 21600/3.6 # W/m2 K
                stream_name = 'MP'
            elif T > 174 and T < 249:
                Tin = 249
                Tout = 250
                htc = 21600/3.6 # W/m2 K
                stream_name = 'HP'
            stream_type = 'heaterop'

        return Tin, Tout, htc, stream_name, stream_type

    # Function to loop through reactors in input list and get the equivalent cooling/heating stream info as dataframe
    def create_reactor_duty_streams(self, hysys_sim, rct_list,htc_spec={}):
        # Initialising lists
        names = []
        feed_temps = []
        prod_temps = []
        duties = []
        shifts = []
        types = []

        # Looping through all reactors in list
        for rct_name in rct_list:
            rct_type, T_rct, duty = self.get_rct_type(hysys_sim, rct_name)
            Tin, Tout, htc, stream_name, stream_type = self.reactor_duty_stream(T_rct, rct_type)

            # Compiling results
            duty_name = rct_name + '_' + stream_name
            if stream_type == 'heaterop':
                Tshift = 5
            else:
                Tshift = -5

            htc_spec[duty_name] = htc

            names.append(duty_name)
            feed_temps.append(Tin)
            prod_temps.append(Tout)
            duties.append(abs(duty))
            shifts.append(Tshift)
            types.append(stream_type)

        return names, feed_temps, prod_temps, duties, shifts, types, htc_spec

    # Function to get the unique shifted temperatures
    def temp_intervals(self, hex_data):
        Tin_shift = hex_data['Tin_shift']
        Tout_shift = hex_data['Tout_shift']

        T_shift = pd.concat([Tin_shift,Tout_shift],ignore_index=True)
        
        T_intervals = T_shift.unique()
        T_intervals = np.sort(T_intervals)[::-1]

        return T_intervals

    # Function to make the temperature interval table
    def interval_table(self, hex_data,T_intervals):

        T1 = T_intervals[:-1]
        T2 = T_intervals[1:]

        int_table = pd.DataFrame({'Tin':T1,'Tout':T2})
        int_table['deltaT'] = int_table['Tin'] - int_table['Tout'] 

        # Looping to see if the streams exist in the interval
        Tin = int_table['Tin'].to_numpy()
        Tout = int_table['Tout'].to_numpy()

        for i in hex_data.index:
            Tmin = min(hex_data['Tin_shift'][i],hex_data['Tout_shift'][i])
            Tmax = max(hex_data['Tin_shift'][i],hex_data['Tout_shift'][i])
            hex_exist = []
            for j in int_table.index:
                Tint_low = Tin[j]
                Tint_high = Tout[j]
                if (Tmin <= Tint_low <= Tmax) and (Tmin <= Tint_high <= Tmax):
                    if hex_data['Type'][i] == 'heaterop':
                        hex_exist.append(-1)
                    else:
                        hex_exist.append(1)
                else:
                    hex_exist.append(0)

            int_table[hex_data['Name'][i]] = hex_exist

        # Getting the FCp balance at each interval
        int_table['Balance'] = np.dot(int_table.to_numpy()[:,3:],hex_data['FCp'].to_numpy()) * int_table['deltaT']

        # Getting the cascade
        cascade = []
        cas_val = 0
        for i in int_table.index:
            bal = int_table['Balance'][i]
            cas_val += bal
            cascade.append(cas_val)

        int_table['Cascade'] = cascade

        # Shifted cascade
        shift_cascade = []
        for i in range(len(T_intervals)):
            if i == 0:
                if min(int_table['Cascade'].values) < 0:
                    new_val = abs(min(int_table['Cascade'].values))
                else:
                    new_val = 0
            else:
                new_val = shift_cascade[i-1] + int_table['Balance'][i-1]

            shift_cascade.append(new_val)

        # returning the shift table
        shift_table = pd.DataFrame()
        shift_table['Tshift'] = T_intervals
        shift_table['H'] = shift_cascade

        return int_table, shift_table

    '''
    ====================================================
        Section on area calculation of the HTC
    ====================================================
    '''
    # First need to calculate the HTC of the streams using the correlations in AEE
    def htc(self, prop_list,type):
        k = prop_list[0]
        p = prop_list[1]
        mu = prop_list[2]
        cp = prop_list[3]

        # First convert to SI units
        mu *= 0.001 # cP to Pa S
        cp *= 1000 # kJ/kg C to J/kg C

        # Calculating Re and Pr (assume v = 1 m/s and d = 0.0254 m, same as AEE
        v = 1
        d = 0.0254
        Re = d*v*p/mu
        Pr = cp*mu/k

        # Different Re term depending on stream type
        if type == 'heaterop':
            Re_term = 0.023*Re**0.8
        else:
            Re_term = 0.36*Re**0.55

        Pr_term = Pr**(1/3)

        h = Re_term*Pr_term*k/d # W/m2 K

        return h

    ''' 
    Back calculating the heat transfer coefficient of condensing stream using water and tabular U values from towler2022
    Condensing Fluid    Overall U [W/m2 K]
    ----------------    ------------------
    Aqueous Vapours     1000-1500
    Organic Vapours     700-1000
    Organics with NCs   500-700
    '''
    def htc_cond(self, fluid):
        if fluid == 'aq_vap':
            U = 1250
        elif fluid == 'org_vap':
            U = 850
        elif fluid == 'org_nc':
            U = 600
        else:
            print('Specify correct fluid name: aq_vap, org_vap or org_nc')

        hw = 13500/3.6 # from cooling water as defined in AEE
        h_cond = (1/U - 1/hw)**(-1)

        return h_cond

    ''' 
    Same as above but for evaporators
    Evaporating Fluid   Overall U [W/m2 K]
    -----------------   ------------------
    Aqeuous solutions   1000-1500
    Light organics      900-1200
    Heavy organics      600-900
    '''
    def htc_evap(self, fluid):
        if fluid == 'aq_sol':
            U = 1250
        elif fluid == 'lht_org':
            U = 1050
        elif fluid == 'hvy_org':
            U = 750
        else:
            print('Specify correct fluid name: aq_sol, lht_org or hvy_org')

        hs = 21600/3.6 # from steam as defined in AEE
        h_evap = (1/U - 1/hs)**(-1)

        return h_evap

    # Function to get the required properties for htc calculation from a material stream
    def get_htc_properties(self, hysys_st):
        k = hysys_st.ThermalConductivityValue
        p = hysys_st.MassDensityValue
        mu = hysys_st.ViscosityValue
        cp = hysys_st.MassHeatCapacityValue

        return [k, p, mu, cp]
    
    def mass_average(self, prop1, prop2, flow1, flow2):
        return (prop1*flow1 + prop2*flow2)/(flow1+flow2)

    # two phase streams have a mass weighted average of the thermal properties between the phases when considered in AEE
    # Can access the phase compositions via a duplicate fluid
    def get_two_phase_properties(self, fsheet, hysys_st):
        # make the stream copy
        dup = hysys_st.DuplicateFluid()
        fp_name = str(hysys_st.FluidPackage)
        phase_dummy = fsheet.MaterialStreams(fp_name)

        # Reset parameters
        phase_dummy.Temperature.Erase()
        phase_dummy.VapourFraction.Erase()

        phase_dummy.TemperatureValue = hysys_st.TemperatureValue
        phase_dummy.PressureValue = hysys_st.PressureValue

        # try either light of heavy phase for property setting
        try:
            phase_dummy.ComponentMolarFlowValue = dup.LightLiquidPhase.MolarFlowsValue
        except:
            phase_dummy.ComponentMolarFlowValue = dup.HeavyLiquidPhase.MolarFlowsValue

        # Check that copying compositions gave a saturated stream, adjusting if not (temperature difference should be negligible)
        if phase_dummy.ThermalConductivity.IsKnown == False:
            phase_dummy.Temperature.Erase()
            phase_dummy.VapourFractionValue = 0

        k_liq = phase_dummy.ThermalConductivityValue
        mu_liq = phase_dummy.ViscosityValue
        p_liq = phase_dummy.MassDensityValue
        flow_liq = phase_dummy.MassFlowValue

        # Get vapour phase info
        phase_dummy.ComponentMolarFlowValue = dup.VapourPhase.MolarFlowsValue

        # Check that copying compositions gave a saturated stream, adjusting if not (temperature difference should be negligible)
        if phase_dummy.ThermalConductivity.IsKnown == False:
            phase_dummy.Temperature.Erase()
            phase_dummy.VapourFractionValue = 1

        k_vap = phase_dummy.ThermalConductivityValue
        mu_vap = phase_dummy.ViscosityValue
        p_vap = phase_dummy.MassDensityValue
        flow_vap = phase_dummy.MassFlowValue

        k_avg = self.mass_average(k_liq, k_vap, flow_liq, flow_vap)
        mu_avg = self.mass_average(mu_liq, mu_vap, flow_liq, flow_vap)
        p_avg = self.mass_average(p_liq, p_vap, flow_liq, flow_vap)

        # don't actually need cp - need effective one, use temp value here
        cp = 1

        return [k_avg, p_avg, mu_avg, cp]


    # Calculating the htc from an input heat exchanger name
    # Also inputting 
    def calc_htc(self, hysys_sim, hex_name, fluid_types={}):
        # Getting the feed and product streams
        hex_op = hysys_sim.Operations(hex_name)
        feed_name = hex_op.AttachedFeeds.Names[0]
        prod_name = hex_op.AttachedProducts.Names[0]
        feed = hysys_sim.MaterialStreams.Item(feed_name)
        prod = hysys_sim.MaterialStreams.Item(prod_name)

        # First check if either stream is two phase flow - if it is, goes through separate calc method
        two_phase = False
        tol = 1e-5
        if feed.VapourFractionValue <= tol or feed.VapourFractionValue >= (1-tol):
            feed_prop = self.get_htc_properties(feed)
        else:
            feed_prop = self.get_two_phase_properties(hysys_sim, feed)
            two_phase = True

        if prod.VapourFractionValue <= tol or prod.VapourFractionValue >= (1-tol):
            prod_prop = self.get_htc_properties(prod)
        else:
            prod_prop = self.get_two_phase_properties(hysys_sim, prod)
            two_phase = True

        # Getting the average of the thermal properties
        avg_prop = []
        for i in range(len(feed_prop)):
            prop = (feed_prop[i] + prod_prop[i]) / 2
            avg_prop.append(prop)

        # adjusting cp value if it's a phase change
        if two_phase:
            cp = hex_op.DutyValue/(feed.MassFlowValue*abs(feed.TemperatureValue - prod.TemperatureValue))
            avg_prop[3] = cp

        h = self.htc(avg_prop,hex_op.TypeName)


        return h

    # Adding the HTC to the hex data table
    def add_htc(self, hysys_sim, hex_data,fluids={},htc_spec={}):
        htc = []
        for name in hex_data['Name']:
            # include if specified as special htc
            if name in htc_spec.keys():
                h = htc_spec[name]

            # methods for reboilers and condensers
            elif 'Condenser' in name or 'Reboiler' in name:
                col_name = name.split('_')[0]
                hex_type = name.split('_')[1]
                h = self.calc_reb_cond_htc(hysys_sim, col_name, hex_type)

            # normal heat exchangers
            else:
                h = self.calc_htc(hysys_sim, name, fluids)

            htc.append(h)

        hex_data['HTC'] = htc

        return hex_data

        # function for getting mass weighted average of products in condenser and reboiler - matches AEE methodology
    def average_htc_prop(self, fsheet, col_name, hex_type, attr):
        col_fsheet = fsheet.Operations(col_name).ColumnFlowsheet
        reb = col_fsheet.Operations(hex_type)
        feed_name = reb.AttachedFeeds.Names[0]
        prod_name = reb.AttachedProducts.Names[0]
        boil_name = reb.AttachedProducts.Names[1]

        f = col_fsheet.MaterialStreams(feed_name)
        p = col_fsheet.MaterialStreams(prod_name)
        b = col_fsheet.MaterialStreams(boil_name)

        f_prop = getattr(f, attr)
        prod_average = (getattr(p, attr)*p.MassFlowValue + getattr(b, attr)*b.MassFlowValue)/f.MassFlowValue
        average_prop = np.mean([f_prop, prod_average])

        return average_prop
    
    def calc_reb_cond_htc(self, fsheet, col_name, hex_type):
        k = self.average_htc_prop(fsheet, col_name, hex_type, 'ThermalConductivityValue')
        p = self.average_htc_prop(fsheet, col_name, hex_type, 'MassDensityValue')
        mu = self.average_htc_prop(fsheet, col_name, hex_type, 'ViscosityValue')
        
        # Need to use the effective cp
        col_fsheet = fsheet.Operations(col_name).ColumnFlowsheet
        hex = col_fsheet.Operations(hex_type)
        feed = col_fsheet.MaterialStreams(hex.AttachedFeeds.Names[0])
        prod = col_fsheet.MaterialStreams(hex.AttachedProducts.Names[0])
        duty = hex.HeatFlowValue
        m = feed.MassFlowValue
        Tin = feed.TemperatureValue
        Tout = prod.TemperatureValue

        cp = duty/(m*abs(Tin-Tout))

        if hex_type == 'Reboiler':
            hex_type = 'heaterop'
        else:
            hex_type = 'coolerop'

        h = self.htc([k, p, mu, cp], hex_type)

        return h

    '''
    =============================================================
    Area calculation via composite curves and enthalpy intervals
    =============================================================
    '''
    # Making a dataframe reference for utilities
    def hen_utility_ref(self):
        utilities = ['cw','fh','hp','mp','lp']
        Tin = [20,2500,250,175,125]
        Tout = [25,2499,249,174,124]
        htc = np.array([13500,399.6,21600,21600,21600])
        htc /= 3.6 # conversion to W/m2 K

        util = {'Name':utilities,'Tin':Tin,'Tout':Tout,'HTC':htc}
        util = pd.DataFrame(util)

        return util

    # Adding utility requirement to the hex data (list utility type and value)
    def hen_utility(self, hex_data, h_util, c_util):
        Qh = h_util[1]
        Qc = c_util[1]

        # get ref utilities
        util = self.hen_utility_ref()
        Thin = util.loc[util['Name'] == h_util[0],'Tin'].iloc[0]
        Thout = util.loc[util['Name'] == h_util[0],'Tout'].iloc[0]
        h_htc = util.loc[util['Name'] == h_util[0],'HTC'].iloc[0]

        Tcin = util.loc[util['Name'] == c_util[0],'Tin'].iloc[0]
        Tcout = util.loc[util['Name'] == c_util[0],'Tout'].iloc[0]
        c_htc = util.loc[util['Name'] == c_util[0],'HTC'].iloc[0]

        # don't add utility if duty is 0
        if Qh == 0:
            names = c_util[0]
            types = 'heaterop'
            duties = Qc
            Tin = Tcin
            Tout = Tcout
            htc = c_htc
        elif Qc == 0:
            names = h_util[0]
            types = 'coolerop'
            duties = Qh
            Tin = Thin
            Tout = Thout
            htc = h_htc
        else:
            names = [h_util[0],c_util[0]]
            types = ['coolerop','heaterop']
            duties = [Qh, Qc]
            Tin = [Thin, Tcin]
            Tout = [Thout, Tcout]
            htc = [h_htc,c_htc]

        util_data = {   'Name':names,
                        'Type':types,
                        'Duty':duties,
                        'Tin':Tin,
                        'Tout':Tout,
                        'HTC':htc}

        if Qh == 0 or Qc == 0:
            util_data = pd.DataFrame(util_data,index=[0])
        else:
            util_data = pd.DataFrame(util_data)

        # Calculating the FCp
        util_data['FCp'] = util_data['Duty'] / abs(util_data['Tin'] - util_data['Tout'])

        # Adding the temperature shift
        util_data['Tin_shift'] = util_data['Tin']
        util_data['Tout_shift'] = util_data['Tout']

        util_data.loc[util_data['Type'] == 'coolerop','Tin_shift'] -= 5
        util_data.loc[util_data['Type'] == 'coolerop','Tout_shift'] -= 5 

        util_data.loc[util_data['Type'] == 'heaterop','Tin_shift'] += 5 
        util_data.loc[util_data['Type'] == 'heaterop','Tout_shift'] += 5 

        hex_data = pd.concat([hex_data,util_data],ignore_index=True)

        return hex_data

    # Creating the composite curve table
    def comp_curves(self, hex_data, hex_type):
        # Separating the hex data by coolerop or heaterop
        comp = hex_data[hex_data.Type == hex_type].reset_index(drop=True)

        # Getting unique temperature values
        Tin = comp['Tin']
        Tout = comp['Tout']
        T = pd.concat([Tin,Tout],ignore_index=True)
        T_intervals = T.unique()
        T_intervals = np.sort(T_intervals)
        # Getting numpy array of Tin/Tout values
        Tin = comp['Tin'].to_numpy()
        Tout = comp['Tout'].to_numpy()

        # Getting the enthalpy values at each temperature level
        H = []
        for i in range(len(T_intervals)):
            if i == 0:
                H.append(0)
            else:
                delta_T = T_intervals[i] - T_intervals[i-1]
                total_FCp = 0
                for name in comp.Name:
                    if hex_type == 'coolerop':
                        Tmin = Tout[comp['Name'].to_numpy() == name].item()
                        Tmax = Tin[comp['Name'].to_numpy() == name].item()
                    else:
                        Tmin = Tin[comp['Name'].to_numpy() == name].item()
                        Tmax = Tout[comp['Name'].to_numpy() == name].item()

                    if (Tmin <= T_intervals[i] <= Tmax) and (Tmin <= T_intervals[i-1] <= Tmax):
                        # total_FCp += comp.loc[comp.Name == name, 'FCp'].iloc[0]
                        total_FCp += comp['FCp'].to_numpy()[comp['Name'].to_numpy() == name].item()

                H.append(total_FCp*delta_T + H[i-1])

        comp_curve = pd.DataFrame({'T':T_intervals,'H':H})

        # Adjusting enthalpy values if there is an 'temperature jump' 
        # i.e. if there is a temperature range where cold or hot streams don't exist, 
        # the same enthalpy value will have two different temperature values which ruins the interpolation of the temperatures in the below function
        for i in comp_curve.index:
            if i == 0:
                continue
            else:
                H1 = comp_curve.H[i-1]
                H2 = comp_curve.H[i]
                if H1 == H2:
                    comp_curve.H[i] += 1e-8

        return comp_curve

    # Function to get the temperatures in the enthalpy levels
    def enthalpy_intervals(self, hex_data):
        # First get the hot and cold composite curves
        hot = self.comp_curves(hex_data, 'coolerop')
        cold = self.comp_curves(hex_data, 'heaterop')

        # Getting the unique enthalpy levels
        Hh = hot['H']
        Hc = cold['H']
        Hh[len(Hh)-1] = Hc[len(Hc)-1] # force end enthalpy values to be equal to account for rounding errors
        H = pd.concat([Hh,Hc],ignore_index=True)
        H_intervals = H.unique()
        H_intervals = np.sort(H_intervals)

        # Setting up a dataframe with the hot and cold temps at each enthalpy level
        enthalpy = pd.DataFrame({'H':H_intervals})
        Th = []
        Tc = []
        for H in enthalpy.H:
            # Linear interpolation
            Th.append(np.interp(H, hot['H'].values.flatten(), hot['T'].values.flatten()))
            Tc.append(np.interp(H, cold['H'].values.flatten(), cold['T'].values.flatten()))

        enthalpy['Th'] = Th
        enthalpy['Tc'] = Tc

        return enthalpy


    # Function to calculate area in each enthalpy interval
    def enthalpy_interval_area(self, enthalpy, hex_data):
        # First making the interval dataframe
        Hin = enthalpy.H.values[:-1]
        Hout = enthalpy.H.values[1:]
        Thin = enthalpy.Th.values[1:]
        Thout = enthalpy.Th.values[:-1]
        Tcin = enthalpy.Tc.values[:-1]
        Tcout = enthalpy.Tc.values[1:]

        intervals = pd.DataFrame({'Hin':Hin,'Hout':Hout,'Thin':Thin,'Thout':Thout,'Tcin':Tcin,'Tcout':Tcout})

        # Calculating the delta H and LMTD
        intervals['delta_H'] = intervals['Hout'] - intervals['Hin']
        intervals['dT1'] = abs(intervals['Thin'] - intervals['Tcout'])
        intervals['dT2'] = abs(intervals['Thout'] - intervals['Tcin'])
        intervals['dTh'] = intervals['Thin'] - intervals['Thout']
        intervals['dTc'] = intervals['Tcout'] - intervals['Tcin']
        intervals['LMTD'] = (intervals['dT1'] - intervals['dT2'])/np.log(intervals['dT1']/intervals['dT2'])

        # Looping to see if the streams exist in the enthalpy interval
        for i in hex_data.index:
            Tmin = min(hex_data['Tin'][i],hex_data['Tout'][i])
            Tmax = max(hex_data['Tin'][i],hex_data['Tout'][i])
            hex_exist = []
            for j in intervals.index:
                if hex_data['Type'].to_numpy()[i] == 'coolerop':
                    Tint_low = intervals['Thin'].to_numpy()[j]
                    Tint_high = intervals['Thout'].to_numpy()[j]
                else:
                    Tint_low = intervals['Tcin'].to_numpy()[j]
                    Tint_high = intervals['Tcout'].to_numpy()[j]

                # checking if stream exists at enthalpy interval
                if (Tmin <= Tint_low <= Tmax) and (Tmin <= Tint_high <= Tmax):
                    if hex_data['Type'].to_numpy()[i] == 'heaterop': # instead of binary 1/0 existence, use the appropriate temperature difference depending on hot or cold stream
                        hex_exist.append(intervals['dTc'][j])
                    else:
                        hex_exist.append(intervals['dTh'][j])
                else:
                    hex_exist.append(0)

            intervals[hex_data['Name'][i]] = hex_exist

        # Calculating m2 K for each interval
        htc_balance = []
        fcp = hex_data['FCp'].values
        htc_inverse = 1/hex_data['HTC'].values

        for i in intervals.index:
            T_diff = intervals.values[i][12:]
            htc_total = np.dot(T_diff,htc_inverse*fcp*1000) # from kW to W
            htc_balance.append(htc_total)

        # Getting area of each interval
        intervals['m2K'] = htc_balance
        intervals['Area'] = intervals['m2K'] / intervals['LMTD']

        return intervals

    '''
    =============================================================
    Determining the minimum number of units and overall area
    =============================================================
    '''
    # Getting the minimum number of units - ***NEED TO DO THIS BEFORE ADDING UTILITIES***
    def min_units(self, hex_data, temp_table, pinch_temp):
        if np.isnan(pinch_temp):
            Nmin = len(hex_data) - 1
        else:
            above_pinch = temp_table[temp_table.Tout >= pinch_temp]
            below_pinch = temp_table[temp_table.Tin <= pinch_temp]

            above_units = 0
            below_units = 0
            # Looping through each hex name to see if it exists in the pinch range by having values > 0
            for name in hex_data.Name:
                if abs(above_pinch[name].sum()) > 0:
                    above_units += 1

                if abs(below_pinch[name].sum()) > 0:
                    below_units += 1
            
            Nmin = (above_units - 1) + (below_units - 1)

        return Nmin
