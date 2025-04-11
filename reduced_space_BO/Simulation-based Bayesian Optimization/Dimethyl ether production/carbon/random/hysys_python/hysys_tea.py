# TEA costing for HYSYS files saved in json

# set of functions that perform TEA on either deserialized json file or directly on the hysys simulation

import pandas as pd
import numpy as np
import math as m
import hysys_python.hysys_object_persistence as hop
import hysys_python.hysys_pec as hp

# Overall TEA class 
class TEA():
    def __init__(self, hysys_sim, cepci=600, plant_type='f', years=20, interest=0.1, hrs=8000):
        # Saving the inputs
        self.name = hysys_sim.name
        self.plant_type = plant_type
        self.yrs = years
        self.int = interest
        self.hrs = hrs
        self.cepci = cepci
        # self.fsheet = hysys_sim

        # Initialize the LCP section
        self.lev = self.LCP()

    def get_pec(self, fsheet, custom_units={}, ignore=[]):
        self.PEC = hp.PEC(fsheet, self.cepci, custom_units=custom_units, ignore=ignore)

    # Function to calculate the levelized profit or cost of production
    def calc_lcp(self, hysys_sim, ref_streams, calc_type='lcp'):
        # Getting the reference flow
        if type(ref_streams) is list:
            ref_streams_df = self.calc_revenue_cost(hysys_sim, ref_streams)
            flow = ref_streams_df.Flow.sum()*self.hrs
        else:
            flow = ref_streams
        
        

        # Switch the multipliers depending if want profit or lcp
        if calc_type == 'lcp':
            title = 'LCP'
            multi = 1
            self.lev.type = title
        else:
            title = 'LP'
            multi = -1
            self.lev.type = title

        self.lev.flow = flow
        self.lev.multi = multi

        # Calculating levelised versions
        self.lev.acc = self.acc/flow*multi
        self.lev.rev = -(self.rev/flow)*multi
        self.lev.rm = self.rm/flow*multi
        self.lev.wt = self.wt/flow*multi
        self.lev.ut = self.ut/flow*multi
        self.lev.fcp = self.fcp/flow*multi

        self.lev.lcp = self.lev.acc + self.lev.rev + self.lev.rm + self.lev.wt + self.lev.ut + self.lev.fcp

        # Saving results in dataframe if want quick plotting
        df = pd.DataFrame({ 'Name':self.name,
                            title:self.lev.lcp, 
                            'ACC':self.lev.acc, 
                            'REV':self.lev.rev, 
                            'RM':self.lev.rm, 
                            'UT':self.lev.ut, 
                            'WT':self.lev.wt,
                            'FCP':self.lev.fcp},index=[0])

        self.lev.df = pd.concat([self.lev.df, df],ignore_index=True)

    # Adding columns with source split to the levelised dataframe
    def add_source_split(self):
        # First we do the revenue sources
        if self.rev != 0:
            grouped = self.Revenue.groupby('Name').sum(numeric_only=True)
            for idx in grouped.index:
                self.lev.df[idx] = -(grouped.Revenue[idx]*self.hrs/self.lev.flow)*self.lev.multi

        # Then the raw materials
        grouped = self.RawMaterials.groupby('Name').sum(numeric_only=True)
        for idx in grouped.index:
            self.lev.df[idx] = grouped.Revenue[idx]*self.hrs/self.lev.flow*self.lev.multi

        # Finally adding the utility split - cooling, heating and electricity
        self.lev.df['CU'] = self.CoolingUtility*self.hrs/self.lev.flow*self.lev.multi
        self.lev.df['HU'] = self.HeatingUtility*self.hrs/self.lev.flow*self.lev.multi
        self.lev.df['Elec'] = self.Electricity['Cost'].sum()*self.hrs/self.lev.flow*self.lev.multi

    # Class to store the LCP results
    class LCP:
        def __init__(self):
            self.lcp = 0
            self.acc = 0
            self.rev = 0
            self.rm = 0
            self.wt = 0
            self.ut = 0
            self.fcp = 0
            self.type = ''
            self.df = pd.DataFrame()

        # Function to quickly plot the levelised results
        def lcp_plot(self):
            ax = self.df.plot.bar(x='Name',y=['ACC','REV','RM','UT','WT','FCP'],stacked=True)
            self.df.plot.scatter(x='Name',y=self.type,ax = ax,c='w',edgecolors='k')



    # Function to calculate the total capex using factors from Sinnott
    def calc_capex(self, return_bl=False):
        isbl = self.PEC.PEC_df.ISBL.sum()
        x_f = 0.1
        type = self.plant_type

        # Getting correct factors
        if type == 'f': # fluids
            osbl_f = 0.3
            de_f = 0.3
        elif type == 'fs': # fluid-solids
            osbl_f = 0.4
            de_f = 0.25
        elif type == 's': # solids
            osbl_f = 0.4
            de_f = 0.2

        # Calculating CAPEX
        osbl = osbl_f * isbl
        de = de_f * (isbl + osbl)
        x = x_f * (isbl + osbl)

        capex = (isbl + osbl + de + x)

        self.isbl = isbl
        self.osbl = osbl
        self.capex = capex

        self.calc_acc()



    # Function for getting the annualised capital charge
    def calc_acc(self):
        i = self.int
        n = self.yrs
        capex = self.capex

        accr = (i*(1+i)**n) / ((1+i)**n - 1)

        acc = accr * capex

        self.acc = acc

    # Looping through a list of Stream objects to calculate the total revenue/cost
    def calc_revenue_cost(self, hysys_sim, stream_list):
        rev = pd.DataFrame()
        for st in stream_list:
            st.df = st.stream_price(hysys_sim)
            rev = pd.concat([rev, st.df], ignore_index=True)

        return rev

    # Adding the revenue
    def calc_revenue(self, hysys_sim, stream_list):
        rev_df = self.calc_revenue_cost(hysys_sim, stream_list)
        self.rev = rev_df['Revenue'].sum()*self.hrs
        self.Revenue = rev_df

    # Function for electricity (excluding the cooling water pump costs)
    def calc_electricity(self, hysys_sim, cost):
        if isinstance(hysys_sim, hop.Hysys_Flowsheet):
            operations = hysys_sim.op
        else:
            operations = hysys_sim.Operations.Names

        duties = []
        costs = []
        names = []

        for op_name in operations:
            if isinstance(hysys_sim, hop.Hysys_Flowsheet):
                op = operations[op_name]
            else:
                op = hysys_sim.Operations(op_name)

            if op.TypeName == 'pumpop':
                duty = op.WorkValue
            elif op.TypeName == 'compressor':
                duty = op.EnergyValue
            elif op.TypeName == 'expandop':
                duty = -op.EnergyValue
            else:
                continue

            duties.append(duty)
            costs.append(duty * cost)
            names.append(op.name)

        rev = {'Name':names,'Duty':duties,'Cost':costs}
        rev_df = pd.DataFrame(rev)

        return rev_df

    # Function for calculating the cost of cooling water - from turton2012 chp 8
    def calc_cooling_water(self, cw_duty, elec_cost): # duty in kW; cost in USD/kWh
        if cw_duty != 0:
            Tin = 20
            Tout = 25
            cp = 4.183 # kJ/kg C

            cw_duty *= 3600 # kJ/h
            loop_flow = cw_duty/((Tout-Tin)*cp)

            Tavg = (Tin+Tout)/2
            # below from trendline of nist data at different temps and 1 bar [kJ/kg]
            latent_heat = -0.00000005*Tavg**4 + 0.000004*Tavg**3 - 0.0014*Tavg**2 - 2.2998*Tavg + 2500.3

            water_evap = cw_duty/latent_heat # kg/h
            water_evap_pct = water_evap/loop_flow*100

            windage_loss_pct = 0.2
            purge_loss_pct = water_evap_pct/4 - windage_loss_pct

            water_makeup = (water_evap_pct + windage_loss_pct + purge_loss_pct)*loop_flow/100 # kg/h

            pump_power = loop_flow*266.7/(3600*1000*0.75)
            fan_power =(loop_flow*2.2048*0.5*0.041)/(60*8.337)
            total_elec = pump_power + fan_power # kW

            makeup_cost = 0.223/1000 # USD/kg for chemicals and process water
            
            total_cost = makeup_cost*water_makeup + total_elec*elec_cost
            if total_cost <= 0 or total_cost == np.nan or m.isnan(total_cost):
                total_cost = 0
        else:
            total_cost = 0

        return total_cost # USD/h

    # Function for calculating the cost of operating labour from turton2012 Chp8
    def calc_labour(self, pec_df, hen_nmin, solid_steps=0):
        # First get number of units
        n_units = 0
        for unit in pec_df.Name:
            if unit == 'HEN':
                continue

            # if isinstance(hysys_sim, hop.Hysys_Flowsheet):
            #     op = hysys_sim.op[unit]
            # else:
            #     op = hysys_sim.Operations(unit)

            # if op.TypeName not in ['pumpop','flashtank']: 
            if unit.split('-')[0] not in ['P','V']: # ignore pumps and flash tanks
                n_units += 1

        n_units += hen_nmin

        if self.plant_type == 'fs':
            solid_steps = 1
        elif self.plant_type == 's':
            solid_steps = 2
        
        # Use equation 8.3
        op_per_shift = (6.29 + 31.7*solid_steps**2 + 0.23*n_units)**0.5
        shifts = 4.5
        no_ops = m.ceil(shifts*op_per_shift)
        op_sal = 50000 # USD/y
        lab = op_sal*no_ops

        return lab

    # Function to determine the variable cost of production i.e. sum of raw materials, utilities, and waste treatment
    # Inputs is a dictionary containing the name of the rm streams, the minimum cooling and heating utility and the cost of heating (assuming cooling is provided by water)
    def calc_vcp(self, hysys_sim, rm_df, wt_df, elec_cost, hen_results, Qh_cost):
        # Utilities
        elec = self.calc_electricity(hysys_sim, elec_cost)
        tot_elec = elec.Cost.sum()
        cw = self.calc_cooling_water(hen_results.Qc, elec_cost)
        fh = hen_results.Qh*Qh_cost
        
        # # Final results
        if rm_df.empty:
            rm_cost = 0
        else:
            rm_cost = rm_df.Revenue.sum()*self.hrs

        if wt_df.empty:
            wt_cost = 0
        else:
            wt_cost = wt_df.Revenue.sum()*self.hrs

        ut_cost = (cw + fh + tot_elec)*self.hrs

        self.CoolingUtility = cw
        self.HeatingUtility = fh
        self.Electricity = elec
        self.RawMaterials = rm_df
        self.WasteTreatment = wt_df
        self.HEN = hen_results

        self.rm = rm_cost
        self.ut = ut_cost
        self.wt = wt_cost
        self.vcp = rm_cost + ut_cost + wt_cost

    # Function to calculate the fixed cost of production (from towler2022)
    def calc_fcp(self, hysys_sim, hen_results, show_breakdown=False):
        pec_df = self.PEC.PEC_df
        vcp = self.vcp
        rev = self.rev

        # Labour, supervision and salary overhead
        lab = self.calc_labour(pec_df, hen_results.Nmin)
        sup = 0.25*lab
        ovh = 0.4*(lab + sup)
        tot_lab = lab + sup + ovh

        # Maintenance, land, taxes and insurance
        isbl  = self.isbl
        osbl = self.osbl
        if self.plant_type == 'f':
            main = 0.03*isbl
        else:
            main = 0.05*isbl
        land = 0.01*(isbl + osbl)
        tax = 0.01*(isbl + osbl)
        ins = 0.01*(isbl + osbl)

        # R&D, sales/marketing and G&A
        rd = 0.01*rev
        sm = 0.01 # portion of overall opex
        ga = 0.65*(tot_lab)

        # Overall FCP
        fcp = (sm*vcp + tot_lab + main + land + tax + ins + rd + ga)/(1-sm)

        if show_breakdown:
            print('Labour: {:.2f}'.format(tot_lab/fcp*100))
            print('Maintenance: {:.2f}'.format(main/fcp*100))
            print('Land: {:.2f}'.format(land/fcp*100))
            print('Taxes: {:.2f}'.format(tax/fcp*100))
            print('Insurance: {:.2f}'.format(ins/fcp*100))
            print('R&D: {:.2f}'.format(rd/fcp*100))
            print('G&A: {:.2f}'.format(ga/fcp*100))
            print('S&M: {:.2f}'.format(sm/fcp*100))

        # return fcp # USD/y
        self.fcp = fcp # USD/y

    def calc_opex(self):
        self.opex = self.fcp + self.vcp

    # Get a list of all material streams
    def print_stream_names(self):
        dfs = [self.Revenue, self.RawMaterials, self.WasteTreatment]
        for df in dfs:
            if type(df) == pd.DataFrame:
                names = df.Name.unique()
                for name in names:
                    print(name)

    # Functiont to recalculate the full economics
    def recalculate(self, prices={}, elec=None, heat=None):
        # redo the capex - must have updated i, yrs etc beforehand
        self.calc_capex()

        # redo the revenue, waste treatment and raw materials
        dfs = [self.Revenue, self.RawMaterials, self.WasteTreatment]
        for df in dfs:
            if type(df) == pd.DataFrame:
                for idx in df.index:
                    name = df.Name[idx]
                    if name in prices:
                        # df['Value [USD/kg]'][idx] = replaceprices[name]
                        # df.Revenue[idx] = df.Flow[idx]*prices[name]
                        df.at[idx, 'Value [USD/kg]'] = prices[name]
                        df.at[idx, 'Revenue'] = df.Flow[idx]*prices[name]

        # redo the utility costs
        if elec != None:
            self.Electricity['Cost'] = self.Electricity.Duty*elec
            self.CoolingUtility = self.calc_cooling_water(self.HEN.Qc, elec)

        if heat != None:
            self.HeatingUtility = self.HEN.Qh*heat

        self.ut = (self.CoolingUtility + 
                   self.HeatingUtility + 
                   self.Electricity['Cost'].sum()) * self.hrs

        # redo the vcp
        self.rev = self.Revenue.Revenue.sum()*self.hrs
        self.rm = self.RawMaterials.Revenue.sum()*self.hrs
        self.wt = self.WasteTreatment.Revenue.sum()*self.hrs
        self.vcp = self.rm + self.ut + self.wt

        # redo the fcp
        self.calc_fcp(None, self.HEN)

        # reset the levelised results
        self.lev = self.LCP()


# Trialling new object for stream price calcs (raw materials, waste, products)
class Stream:
    def __init__(self, names, value, comp=''):
        self.names = names # list of streams
        self.value = value # USD/kg of stream/component
        self.comp = comp # component of interest
        self.df = ''

    # Function for calculating steam revenues/costs using Stream class
    def stream_price(self, hysys_sim):
        # Initialising
        prices = []
        flows = []

        # Looping through all streams
        for name in self.names:
            # Getting either total or component flow
            if self.comp == '':
                if isinstance(hysys_sim, hop.Hysys_Flowsheet):
                    flow = hysys_sim.ms[name].MassFlowValue * 3600
                else:
                    flow = hysys_sim.MaterialStreams(name).MassFlowValue * 3600
            else: # Component flow
                if isinstance(hysys_sim, hop.Hysys_Flowsheet):
                    comp_idx = hysys_sim.comp.index(self.comp)
                    flow = hysys_sim.ms[name].ComponentMassFlowValue[comp_idx] * 3600
                else:
                    try:
                        comps = hysys_sim.Parent.BasisManager.ComponentLists[0].Components.Names
                    except:
                        comps = hysys_sim.Parent.Parent.Parent.BasisManager.ComponentLists[0].Components.Names  # if its a subflowsheet
                    comp_idx = comps.index(self.comp)
                    flow = hysys_sim.MaterialStreams(name).ComponentMassFlowValue[comp_idx] * 3600 
                
            price = flow * self.value # USD/h
            prices.append(price)
            flows.append(flow)

        # Putting results into a dataframe
        revenues = {'Name':self.names,'Component':self.comp,'Value [USD/kg]':self.value,'Flow':flows,'Revenue':prices}
        rev_df = pd.DataFrame(revenues)

        return rev_df