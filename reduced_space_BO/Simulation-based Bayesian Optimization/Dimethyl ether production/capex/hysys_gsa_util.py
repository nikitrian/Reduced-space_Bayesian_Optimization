import hysys_python.hysys_util as hu
import hysys_python.hysys_hen as hh
import hysys_python.hysys_tea as ht
import time as ti
import numpy as np
from scipy.integrate import solve_ivp

# Function to determine the heat of reaction at certain T and P from hysys
def calc_hrxn_cp(cp_st, st1, st2, solver, T, P, F):
    solver.CanSolve = False

    # Setting temperatures and pressures
    cp_st.TemperatureValue = T
    cp_st.PressureValue = P
    st1.TemperatureValue = T
    st1.PressureValue = P
    st2.TemperatureValue = T
    st2.PressureValue = P

    # First set the cp dummy stream to find the cp and vol flow of the mixture
    cp_st.ComponentMolarFlowValue = F
    
    # Stream to calculate enthalpy of reactants
    F = (np.zeros(9))
    F[5] = 2 # 2 kmol/h of MeOH
    st1.ComponentMolarFlowValue = tuple(F/3600)

    # Now define for the products
    F = np.zeros(9)
    F[3] = 1 # 1 kmol/h of DME
    F[4] = 1 # 1 kmol/h of H2O
    st2.ComponentMolarFlowValue = tuple(F/3600)

    solver.CanSolve = True
    H_reactants = st1.MolarEnthalpyValue
    H_products = st2.MolarEnthalpyValue
    cp = cp_st.MolarHeatCapacityValue
    vol_flow = cp_st.ActualVolumeFlowValue

    # Calculating the heat of reaction
    hrxn = H_products - H_reactants

    return hrxn, cp, vol_flow

# DME dehydration kinetics ODE - copied from the HYSYS file
def dme_dehydration(V, F, P, dummy_sts, solver, Finit, Xtarget):
    # species = H2, CO, CO2, DME, H2O, Meoh, o2, n2, ch4
    T = F[-1]
    flows = F[:-1]

    cp_st = dummy_sts[0]
    hrxn1_st = dummy_sts[1]
    hrxn2_st = dummy_sts[2]

    # get properties of the stream
    F_tuple = tuple(flows)
    T_C = T - 273.15
    hrxn, cp, vol_flow = calc_hrxn_cp(cp_st, hrxn1_st, hrxn2_st, solver, T_C, P, F_tuple)

    # Determine the molar concentration of each species
    C = flows / vol_flow

    # forward basis
    Af = 1059000
    Ef = 2544.084
    R = 8.314
    kf = Af * np.exp(-Ef / (R * T))

    # reverse basis
    Ar = 10000000
    Er = 24402.084
    kr = Ar * np.exp(-Er / (R * T))

    # numerator
    numerator = kf * C[5] **2 - kr * C[4] * C[3]

    # Adsorption terms
    Ai = np.array([5.62138770000000e-002, 8.47000000000000e-002])
    Ei = np.array([-35280.0000000000, -5070.00000000000])
    Ki = Ai * np.exp(-Ei / (R * T))

    # Denominator
    denominator = (1 + Ki[0]*C[5]**0.5 + Ki[1]*C[4])**4

    # rate of reaction
    r = numerator / denominator

    # rate of change of each species
    dFdV = np.zeros_like(F)
    dFdV[3] = 0.5*r
    dFdV[4] = 0.5*r
    dFdV[5] = -r

    # heat of reaction
    # hrxn = calc_hrxn(hrxn1_dummy, hrxn2_dummy, solver, T_C, P) # kJ/kmol
    Qrxn = hrxn * r # kJ/s
    dFdV[-1] = -Qrxn / (cp * sum(flows))
        
    return dFdV

# Stop event for the integration to match the target conversion
def stop_event(V, F, P, dummy_sts, solver, Finit, Xtarget):
    X = (Finit[5] - F[5]) / Finit[5] * 100 # % conversion of MeOH
    return X - Xtarget

# Function to calculate the economics of the flowsheet
def calc_economics(fsheet, h2_price=3, co2_price=0.08, elec_price=0.1):
    # First determine the HEN
    rct_names = ['Reactor100','Reactor101']
    included_rcts = []
    for rct in rct_names:
        duty = fsheet.Operations(rct).HeatFlowValue
        if duty > 0:
            included_rcts.append(rct)
    
    # estimates the cost of the HEN 
    hen = hh.HEN(fsheet, rct_list = included_rcts)

    # Initialize TEA class
    cepci = 816 # 2022 inflation index
    plant_type = 'f' # fluid
    tea = ht.TEA(fsheet, cepci=cepci, plant_type=plant_type)

    # Creating PEC table and adding reactor costing
    reactors = {'Reactor100':'pfr_pec',
                'Reactor101':'pfr_pec'}
    tea.get_pec(fsheet, custom_units=reactors)
    tea.PEC.add_hen_pec(hen)

    # Adding percentage of total ISBL to each entry and sorting from smallest to largest
    tea.PEC.PEC_df['%'] = tea.PEC.PEC_df['ISBL']/tea.PEC.PEC_df['ISBL'].sum()*100
    tea.PEC.PEC_df = tea.PEC.PEC_df.sort_values('ISBL', ascending=False)

    # Calculating the total CAPEX and ACC
    tea.calc_capex()

    # Adding the raw materials
    co2 = ht.Stream(['co2_1atm'], co2_price)
    h2 = ht.Stream(['h2'], h2_price)
    rm_df = tea.calc_revenue_cost(fsheet, [co2, h2])

    # Waste treatment
    ww_cost = 1.5/1000 # USD/kg, towler2022 Chp 8.4.4
    ww_st = ['WasteWater']
    ww_st = ht.Stream(ww_st, ww_cost)
    ww_df = tea.calc_revenue_cost(fsheet, [ww_st])

    # Utilities cost
    # elec_price = 114.26 # GBP/MWh on 6 June 2022 UK week ahead wholesale https://www.ofgem.gov.uk/energy-data-and-research/data-portal/wholesale-market-indicators
    # USD_GBP = 0.804354 # USD/GBP rate 6th June xe.com
    # elec_price *= USD_GBP # USD/MWh
    # elec_price /= 1000 # USD/kWh
    heating_cost = elec_price # electric heating?

    # Now getting the variable and fixed costs of production
    tea.calc_vcp(fsheet, rm_df, ww_df, elec_price, hen, heating_cost)
    tea.rev = 0 # no additional revenue as DME only product
    tea.calc_fcp(fsheet, hen)
    tea.calc_opex()

    # Now getting the levelised cost of production
    ref_flow = ht.Stream(['DME-prod'], 0)
    tea.calc_lcp(fsheet, [ref_flow], calc_type='lcp')
    tea.add_source_split()


    return tea

def calc_energy_efficiency(fsheet, tea):
    # net electricity input kW
    elec = tea.Electricity['Duty'].sum()

    # net heat input kW from pinch
    heat = tea.HEN.Qh

    # chemical energy input kW from h2
    h2_flow = fsheet.MaterialStreams('h2').MolarFlowValue
    h2_lhv = fsheet.MaterialStreams('h2').LowerHeatValueValue
    h2_energy = h2_flow*h2_lhv # kW

    # chemical energy output kW from DME
    dme_out = fsheet.MaterialStreams('DME-prod').MolarFlowValue
    dme_lhv = fsheet.MaterialStreams('DME-prod').LowerHeatValueValue
    dme_energy = dme_out*dme_lhv # kW

    # energy efficiency
    energy_efficiency = dme_energy/(elec + heat + h2_energy)*100
    
    return energy_efficiency

def calc_ouputs(fsheet, inputs):
    # DME production
    prod = fsheet.MaterialStreams('DME-prod')
    dme_prod = prod.MassFlowValue

    # Carbon efficiency
    feed = fsheet.MaterialStreams('syngas-dry')
    co2_idx = feed.ComponentIndices('CO2')[0]
    c_in = feed.ComponentMolarFlowValue[co2_idx]

    dme_idx = prod.ComponentIndices('diM-Ether')[0]
    c_out = prod.ComponentMolarFlowValue[dme_idx]*2

    c_eff = c_out/c_in*100

    # Economic
    h2_price = 5.24 # USD/kg, parkinson2019 - literature central estimate for wind h2
    co2_price = 0.082 # USD/kg, industry CO2 capture cost from iea (midpoint)
    elec_nom = 89.2 # GBP/MWh - 02 Jan 2024
    elec_nom *= 0.804354 # USD/MWh
    elec_nom /= 1000 # USD/kWh
    
    tea_results = calc_economics(fsheet, h2_price=h2_price, co2_price=co2_price,
                                 elec_price=elec_nom)
    capex = tea_results.capex
    opex = tea_results.opex
    lcp = tea_results.lev.lcp

    # Energy efficiency
    energy_efficiency = calc_energy_efficiency(fsheet, tea_results)

    outputs = np.array([dme_prod, c_eff, capex, opex, lcp, energy_efficiency])

    return outputs

# Making the reactor either adiabatic (Q=0) or isothermal (Tout = Tin)
def set_reactor_mode(fsheet, outlet_stream, energy_stream, mode, Tin):
    fsheet.EnergyStreams(energy_stream).HeatFlow.Erase()
    fsheet.MaterialStreams(outlet_stream).Temperature.Erase()
    ti.sleep(0.1)
    if mode >= 0.5:
        fsheet.MaterialStreams(outlet_stream).TemperatureValue = Tin # isothermal
    else:
        fsheet.EnergyStreams(energy_stream).HeatFlowValue = 0 # adiabatic

# Function to set the number of trays and feed location of the distillation column
def set_dc_trays_feed(fsheet, solver, col, no_trays, feed_loc, sc_col, ss):
    # First check the minimum trays
    # Get the column feed stream
    col_feed_name = col.AttachedFeeds.Names[0]
    col_feed = fsheet.MaterialStreams(col_feed_name)

    # Shortcut feed
    sc_feed_name = sc_col.AttachedFeeds.Names[0]
    sc_feed = fsheet.MaterialStreams(sc_feed_name)

    # Copy the feed from the main to the shortcut
    hu.update_stream(sc_feed,col_feed)

    # Solve the shortcut and extract the min trays
    solver.CanSolve = True
    min_trays = int(np.ceil(ss.Cell(0,3).CellValue))
    solver.CanSolve = False

    if no_trays < min_trays:
        print(f"**WARNING**: {no_trays} trays is less than the minimum of {min_trays} trays")
        print(f"Setting number of trays to {min_trays}")
        no_trays = min_trays

    # Check which attribute we need to set in the feed stream when we add it back in
    attrs = ['VapourFraction','Temperature','Pressure']
    attr_to_set = ''
    for attr in attrs:
        if getattr(getattr(col_feed, attr), 'CanModify'): # i.e. is the can modify section true or false
            attr_to_set = attr + 'Value'
            attr_val = getattr(col_feed, attr_to_set)
            break

    # Set number of trays
    solver.CanSolve = True
    col.EnterColumnEnvironment() # need to enter and leave to avoid crashed for high # of trays
    col.ColumnFlowsheet.Reset()
    col.ColumnFlowsheet.Operations('Main Tower').NumberOfTrays = no_trays
    col.ColumnFlowsheet.Run()
    col.EnterParentEnvironment()

    # Now convoluted method to set feed stage location (have to delete and recreate stream)
    '''
    HOW TO SOLVE COLUMN
    1) Set number of trays as already defined
    2) Get feed name and index in material stream
    3) Remove feed using MaterialStreams.Remove(idx)
    4) col.ColumnFlowsheet.Operations('Main Tower').AddFeedStream(feed_name, feed_loc, False)
    5) Add to productstream of upstream operation
    6) apply required upstream change (temperature or pressure spec from shortcut value)
    7) col.ColumnFlowhsheet.Run()
    '''
    feed_idx = fsheet.MaterialStreams.index(col_feed_name)
    upstream_oper = fsheet.MaterialStreams(col_feed_name).UpstreamOpers.Names[0]
    fsheet.MaterialStreams.Remove(feed_idx)
    col.ColumnFlowsheet.Operations('Main Tower').AddFeedStream(col_feed_name, feed_loc, False)

    # Most units have ProductStream attribute, but some don't e.g. mixer, component splitter
    try:
        fsheet.Operations(upstream_oper).ProductStream = fsheet.MaterialStreams(col_feed_name)
    except:
        fsheet.Operations(upstream_oper).Product = fsheet.MaterialStreams(col_feed_name)

    if attr_to_set != '':
        setattr(fsheet.MaterialStreams(col_feed_name), attr_to_set, attr_val)
    col.ColumnFlowsheet.Run()
    col.IsIgnored = False

    solver.CanSolve = False

# Setting all of the inputs
def set_inputs(fsheet, solver, inputs):
    fsheet.Operations('co2:h2_ratio_equals_3').Cell(2,1).CellValue = inputs[0] # h2 ratio
    fsheet.Operations('MeOH Pressure').Cell(0,0).CellValue = inputs[1] # meoh feed pressure
    fsheet.MaterialStreams('RinV').TemperatureValue = inputs[2] # meoh Tin

    # meoh reactor heat flow
    set_reactor_mode(fsheet, 'RoutV', 'Q_MeOH', inputs[3], inputs[2])

    fsheet.Operations('TEE-100').SplitsValue = (1-inputs[4], inputs[4]) # purge ratio
    fsheet.Operations('Reactor100').TotalVolumeValue = inputs[5] # meoh reactor volume
    fsheet.MaterialStreams('Rinv-101').TemperatureValue = inputs[6] # dme feed temp

    # dme reactor pressure - have to set for feed, recycle and distillation column
    dme_pres = inputs[7]
    fsheet.MaterialStreams('MeOH-12bar-').PressureValue = dme_pres
    fsheet.MaterialStreams('met-rec-1b').PressureValue = dme_pres
    fsheet.Operations('Twb102 Pressures').Cell(1,0).CellValue = dme_pres
    col_fsheet = fsheet.Operations('Twb102').ColumnFlowsheet
    col_fsheet.MaterialStreams('Cond-T102').PressureValue = dme_pres - 200
    col_fsheet.MaterialStreams('Reb-T102').PressureValue = dme_pres - 100

    # column feed vapour fractions
    fsheet.MaterialStreams('meoh_sep_feed').VapourFractionValue = inputs[8]
    fsheet.MaterialStreams('reac-out-cool').VapourFractionValue = inputs[9]

# Function to size the DME dehydration reactor based on 95% equilbrium conversion
def size_dme_reactor(fsheet, solver):
    # Equilbrium conversion
    eq_reactor = fsheet.Operations('ERV-100')
    X_target = eq_reactor.RxnPercentConversionValue[0]

    # Setting inputs to ODE solver
    tot_vol = 500 # upper bound
    void = 0.4 # reactor voidage
    void_vol = tot_vol*void
    
    # Initial conditions
    feed = fsheet.MaterialStreams('Rin_ref')
    F0 = feed.ComponentMolarFlowValue
    T0 = feed.TemperatureValue
    P0 = feed.PressureValue
    F0 = np.append(F0, T0+273.15)

    # hysys dummy streams for calculating properties
    cp_dummy = fsheet.MaterialStreams('cp_dummy')
    hrxn1_dummy = fsheet.MaterialStreams('hrxn1_dummy')
    hrxn2_dummy = fsheet.MaterialStreams('hrxn2_dummy')
    dummy_sts = [cp_dummy, hrxn1_dummy, hrxn2_dummy]
    stop_event.terminal = True # will stop ode when condition is met
    events = [stop_event]

    # Solving the ODE
    V_span = [0, void_vol]
    sol = solve_ivp(dme_dehydration, V_span, F0, 
                    args=(P0, dummy_sts, solver, F0, X_target), 
                    dense_output=True, method='LSODA', events=events)
    
    # Getting the volume of the reactor
    pfr_vol = sol.t[-1]/void

    # Setting the volume of the reactor
    fsheet.Operations('Reactor101').TotalVolumeValue = pfr_vol

# Using alternative version of the ignore solver function for dcs
def solve_fsheet(fsheet, solver, ignored_ops, sleep_timer=0.01, sc_cols={}, inputs=[]):
    solver.CanSolve = False
    overall_tic = ti.perf_counter()
    # First ignore relevant operations to make convergence easier
    for op in ignored_ops:
        if op not in fsheet.MaterialStreams.Names: # checking if it's a recycle stream
            fsheet.Operations(op).IsIgnored = True
        else:
            # Find downstream mixer
            mix_name = fsheet.MaterialStreams(op).DownstreamOpers.Names[0]
            mix = fsheet.Operations(mix_name)
            # Find index of stream and detach
            idx = mix.Feeds.Names.index(op)
            mix.Feeds.Remove(idx)

    # Initial solve
    tic = ti.perf_counter()
    solver.CanSolve = True
    ti.sleep(sleep_timer)
    if solver.IsSolving:
        ti.sleep(sleep_timer/10)
        print('waiting')
    toc = ti.perf_counter()
    formatted_elapsed_time = "{:.2f}".format(toc-tic)
    print(f"Initial solve up to {ignored_ops[0]} in {formatted_elapsed_time}s")

    # Now solve one by one
    dis_count = 0
    for op in ignored_ops:
        solver.CanSolve = False
        tic = ti.perf_counter()
        # Check if it's a recycle stream or an operation
        if op not in fsheet.MaterialStreams.Names: # checking if it's a recycle stream
            fsheet.Operations(op).IsIgnored = False
            # solve distillation column 
            if fsheet.Operations(op).TypeName == 'distillation':
                col = fsheet.Operations(op)
                sc = fsheet.Operations(sc_cols[op])
                ss = fsheet.Operations('SS:' + sc_cols[op]) # named after appropriate shortcut column
                set_dc_trays_feed(fsheet, solver, col, int(round(inputs[10+dis_count])), \
                                  int(round(inputs[12+dis_count]*inputs[10+dis_count])), sc, ss)
                dis_count += 1
        else:
            # Get name of mixer
            mix_name = op[0:7]
            mix = fsheet.Operations(mix_name)
            rcy_stream = fsheet.MaterialStreams(op)
            mix.Feeds.Add(rcy_stream)

        solver.CanSolve = True
        ti.sleep(sleep_timer)
        if solver.IsSolving:
            ti.sleep(sleep_timer/10)
            print('waiting')
        toc = ti.perf_counter()
        formatted_elapsed_time = "{:.2f}".format(toc-tic)
        print(f"Solved {op} in {formatted_elapsed_time}s")

    overall_toc = ti.perf_counter()
    formatted_elapsed_time = "{:.2f}".format(overall_toc-overall_tic)
    print('*** COMPLETE ***')
    print(f"Solved flowsheet in {formatted_elapsed_time}s\n")
    
# function to pass the gsa sample points to the flowsheet, solve it and then calculate the outputs
def solve_calc_flowsheet(fsheet, solver, inputs):
    solver.CanSolve = False

    # Passing the GSA inputs
    tic = ti.perf_counter()
    set_inputs(fsheet, solver, inputs)
    toc = ti.perf_counter()
    formatted_elapsed_time = "{:.2f}".format(toc-tic)
    print(f"Set inputs in {formatted_elapsed_time}s")
    
    # Solving the flowsheet
    ignored_for_solve = ['Reactor100','MIX-101_meoh_rcy','K-103', 'Vessel 101', 'Twp101', \
                          'V-100','CRV-100','Twb102','MIX-100_met_rcy']
    sc_cols = {'Twp101':'T-101',
               'Twb102':'T-100'}
    fsheet.Operations('ADJ-1').IsIgnored = True # need to ignore outside of the solve function

    solve_fsheet(fsheet, solver, ignored_for_solve, sc_cols=sc_cols, sleep_timer=0.1, inputs=inputs)

    # now update the reactor flow pfr so we can get the volume
    solver.CanSolve = False
    equil_in = fsheet.MaterialStreams('Rinv-101')
    pfr_in = fsheet.MaterialStreams('Rin_ref')
    hu.update_stream(pfr_in, equil_in)

    # Sizing the reactor based on equilibrium reactor conversion
    tic = ti.perf_counter()
    size_dme_reactor(fsheet, solver)
    solver.CanSolve = True
    toc = ti.perf_counter()
    formatted_elapsed_time = "{:.2f}".format(toc-tic)
    print(f"Sized DME reactor in {formatted_elapsed_time}s")

    # Check for convergence based on outlet streams of columns
    meoh = fsheet.MaterialStreams('MeOH').MolarFlow
    dme = fsheet.MaterialStreams('DME-prod').MolarFlow
    meth_rec = fsheet.MaterialStreams('meth-rec-1').MolarFlow
    
    # Is known is false if the column hasn't converged
    if meoh.IsKnown and dme.IsKnown and meth_rec.IsKnown:
        # Calculating the outputs
        outputs = calc_ouputs(fsheet, inputs)
    else:
        print('Column not converged')
        outputs = [None, None, None, None, None, None]
    
    return outputs

# Making random inputs for testing purposes
def random_inputs(nominal=False):
    # electricity prices - 2018-2023 https://tradingeconomics.com/united-kingdom/electricity-price
    elec_nom = 89.2 # GBP/MWh - 02 Jan 2024
    elec_nom *= 0.804354 # USD/MWh
    elec_nom /= 1000 # USD/kWh

    # For the nominal/range of the MeOH reactor size from van Dal 2013
    ref_co2_flow = 88e3 # kg/h CO2
    co2_flow = 28333.3620500565 # kg/h in simulation
    co2_ratio = co2_flow/ref_co2_flow
    ref_cat_mass = 44500 # kg catalyst
    cat_mass = ref_cat_mass*co2_ratio # kg catalyst in simulation
    void = 0.5
    density = 1775
    meoh_nominal_vol = cat_mass * (1/density) * (1/(1-void)) # m3

    bounds = [[2.4, 3.6],       # h2 ratio - +/-20% of stoich
            [5000, 10000],    # meoh pressure - van-dal2013
            [210, 270],       # meoh feed temp - van-dal2013
            [0, 1],           # adiabatic/isothermal meoh
            [0.95, 0.99],      # recycle ratio
            [0.8*meoh_nominal_vol, 1.2*meoh_nominal_vol], # meoh reactor volume - van-dal2013 +/- 20%
            [250, 300],       # dme feed temperature - peinado2020    
            [1000, 2000],     # dme reaction pressure - peinado2020
            [0,1],            # feed vapour fraction meoh column
            [0,1],            # feed vapour fraction dme-meoh column
            [57*0.8, 57*1.2], # trays col 1 +/- 20% of nominal
            [17*0.8, 17*1.2], # trays col 2 +/- 20% of nominal
            # [1.3, 4.5],       # green h2 price USD/kg - https://www.iea.org/reports/global-hydrogen-review-2022/executive-summary
            # [0.079, 0.085],   # co2 price USD/tonne - industry co2 capture cost https://www.iea.org/data-and-statistics/charts/current-cost-of-co2-capture-for-carbon-removal-technologies-by-sector
            # [0.8*elec_nom, 1.2*elec_nom],      # electricity price USD/kWh +/- 20% of nominal
            [44/57*0.8, 44/57*1.2], # relative feed location col 1 +/- 20% of nominal
            [10/17*0.8, 10/17*1.2]  # relative feed location col 2 +/- 20% of nominal
            ]

    # generating random inputs
    inputs = np.zeros(len(bounds))
    for i, bound in enumerate(bounds):
        # input = min + rand*(max-min)
        inputs[i] = bound[0] + np.random.rand()*(bound[1] - bound[0])

    if nominal:
        # nominal inputs
        inputs = [3, # stoi
                  7800, # van-dal2013
                  210, # van-dal2013
                  0, # van-dal2013
                  0.99, # van-dal2013
                  meoh_nominal_vol, # midpoint
                  275, # bernardi2019
                  1500, # midpoint
                  0.5, # van-dal2013/perez-fortes2016
                  0.5, # midpoint
                  57, # van-dal2013 column no trays
                  17, # luyben2017 column no trays
                #   3,   # iea
                #   0.082, # midpoint
                #   elec_nom, # midpoint of electricity prices
                  44/57, # van dal 2013
                  10/17, # luyben2017
        ]

    return inputs

# Main running
if __name__ == '__main__':
    import os
    import hysys_python.hysys_object_persistence as hop
    import _archive.hysys_distillation as hd
    import numpy as np
    import sys
    os.system('cls')

    filepath = r"C:\Users\nt320\OneDrive - Imperial College London\Niki GSA bayesian optimisation\Submission 2"
    file = filepath + r"\i-dme-complete-gsa-equil.hsc"

    # Creating and connecting to the hysys flowsheet and solve
    try:
        sim = hop.hysys_connection(file, active=1)
    except:
        sim = hop.hysys_connection(file, active=0)
    fsheet = sim.Flowsheet
    solver = sim.Solver

    # Running the flowsheet
    tic = ti.perf_counter()
    sim.Visible = False
    inputs = random_inputs(nominal=False)
    out = solve_calc_flowsheet(fsheet, solver, inputs)

    if type(out) != type(None):
        print('DME flowrate         = {:.2f} kg/s'.format(out[0]))
        print('Carbon efficiency    = {:.1f}%'.format(out[1]))
        print('Energy efficiency    = {:.1f}%'.format(out[5]))
        print('CAPEX                = {:.0f} M USD'.format(out[2]/1e6))
        print('OPEX                 = {:.0f} M USD'.format(out[3]/1e6))
        print('LCP                  = {:.2f} USD/kg'.format(out[4]))
    toc = ti.perf_counter()
    formatted_elapsed_time = "{:.2f}".format(toc-tic)
    print(f"Solved script without unhiding in {formatted_elapsed_time}s")
    sim.Visible = True
