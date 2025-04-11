# Functions that are useful when working with hysys flowsheets from python

import time as ti
import math as m
import numpy as np

# Copies all conditions from one stream to another
def update_stream(target, source):
    # Need the pressure, temperature and molar flowrates
    target.TemperatureValue = source.TemperatureValue
    target.PressureValue = source.PressureValue
    target.ComponentMolarFlowValue = source.ComponentMolarFlowValue

# Copies the temperature from one stream to another (useful for refrigeration design)
def update_temperature(target, source):
    target.TemperatureValue = source.TemperatureValue

'''
Function to update the refrigeration cycle
Used when a refrigeration cycle using LNG exchangers needs updating
Pass the full copy streams and temperature update streams as a nested list
e.g. 
full_update = [['T-109_cond',             'DC:T-109:To Condenser'],
               ['Cb_ref_in',              'cb_sat_1']]
'''
def update_refrigeration(fsheet, solver, full_update_streams, temp_update_streams):
    solver.CanSolve = False

    # Full updates first
    for i in full_update_streams:
        target = fsheet.MaterialStreams(i[0])
        # Add check name if the source stream is within a distillation column flowsheet, will be called 'DC:T-XXX:Stream Name'
        if 'DC' in i[1]:
            _, col, stream_name = i[1].split(':')
            source = fsheet.Operations(col).ColumnFlowsheet.MaterialStreams(stream_name)
        else:
            source = fsheet.MaterialStreams(i[1])

        update_stream(target, source)

    # Now the temperature updates
    for i in temp_update_streams:
        target = fsheet.MaterialStreams(i[0])
        source = fsheet.MaterialStreams(i[1])
        update_temperature(target, source)

    solver.CanSolve = True

# Generic code to solve a flowsheet - ignores certain ops to make convergence quicker e.g. recycle section, distillation columns
def solve_fsheet(fsheet, solver, ignored_ops, sleep_timer=0.01, sc_cols={}):
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
                solve_distillation(fsheet, solver, col, sc, ss, sleep_timer=sleep_timer)
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

# Code to solve distillation columns based on shortcuts dynamically
# fsheet = flowsheet object, solver = solver, col = full distillation column, sc = shortcut, ss = spreadsheet w/ sc info
def solve_distillation(fsheet, solver, col, sc, ss, sleep_timer=0.1):
    # First get the column feeds
    col_feed_name = col.AttachedFeeds.Names[0]
    col_feed = fsheet.MaterialStreams(col_feed_name)

    sc_feed_name = sc.AttachedFeeds.Names[0]
    sc_feed = fsheet.MaterialStreams(sc_feed_name)

    # Copy the feed from the main to the shortcut
    update_stream(sc_feed,col_feed)

    # Unignore the column to let it solve so we can extract the number of trays and feed
    # Can't leave it running as it's a circular reference
    ss.Cell(0,4).CellValue = 50 # so we ignore external reflux errors, remember to set solver level for shortcuts at 600
    sc.IsIgnored = False
    solver.CanSolve = True
    Rmin = ss.Cell(0,2).CellValue
    R = 1.2*Rmin
    ss.Cell(0,4).CellValue = R
    # ti.sleep(sleep_timer)
    solver.CanSolve = True
    ti.sleep(sleep_timer)
    if solver.IsSolving:
        ti.sleep(sleep_timer/10)
        print('waiting')

    no_trays = m.ceil(ss.Cell(0,0).CellValue) # MAKE SURE THIS ADDRESS IS CONSISTENT
    feed_loc = round(ss.Cell(0,1).CellValue)

    # Check which attribute we need to set in the feed stream when we add it back in
    attrs = ['VapourFraction','Temperature','Pressure']
    attr_to_set = ''
    for attr in attrs:
        if getattr(getattr(col_feed, attr), 'CanModify'): # i.e. is the can modify section true or false
            attr_to_set = attr + 'Value'
            break

    # Set number of trays
    solver.CanSolve = True
    col.ColumnFlowsheet.Operations('Main Tower').NumberOfTrays = no_trays 

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
        attr_val = getattr(sc_feed, attr_to_set)
        setattr(fsheet.MaterialStreams(col_feed_name), attr_to_set, attr_val)
    col.ColumnFlowsheet.Run()
    col.IsIgnored = False

    solver.CanSolve = False

# Function to size a two phase separator (from towler2022, Section 16.3)
def get_diameter_separator(fsheet, sep):
    feed_name = sep.AttachedFeeds.Names[0]
    feed = fsheet.MaterialStreams(feed_name)

    dup = feed.DuplicateFluid()
    fp_name = str(feed.FluidPackage)
    phase_dummy = fsheet.MaterialStreams(fp_name)

    # Reset parameters
    phase_dummy.Temperature.Erase()
    phase_dummy.VapourFraction.Erase()

    phase_dummy.TemperatureValue = feed.TemperatureValue
    phase_dummy.PressureValue = feed.PressureValue

    phase_found = True
    # try either light of heavy phase for property setting
    try:
        phase_dummy.ComponentMolarFlowValue = dup.LightLiquidPhase.MolarFlowsValue
    except:
        try:
            phase_dummy.ComponentMolarFlowValue = dup.HeavyLiquidPhase.MolarFlowsValue
        except:
            phase_found = False

    # Check that copying compositions gave a saturated stream, adjusting if not (temperature difference should be negligible)
    if phase_found == True:
        if phase_dummy.MassDensity.IsKnown == False or phase_dummy.VapourFractionValue != 0.0:
            phase_dummy.Temperature.Erase()
            phase_dummy.VapourFractionValue = 0

        p_liq = phase_dummy.MassDensityValue

        # Get vapour phase info
        phase_dummy.ComponentMolarFlowValue = dup.VapourPhase.MolarFlowsValue

        # Check that copying compositions gave a saturated stream, adjusting if not (temperature difference should be negligible)
        if phase_dummy.ThermalConductivity.IsKnown == False or phase_dummy.VapourFractionValue != 1.0:
            phase_dummy.Temperature.Erase()
            phase_dummy.VapourFractionValue = 1

        p_vap = phase_dummy.MassDensityValue
        Vv = phase_dummy.ActualGasFlowValue

        ut = 0.07*((p_liq - p_vap)/p_vap)**0.5
        # ut *= 0.15 # no demister pad

        Dv = ((4*Vv)/(np.pi*ut))**0.5
    else:
        Dv = 0 # not two phase so no separation, can ignore the unit

    return Dv


# Class that calculates and saves different parameters needed for sizing/costing a distillation column
# Method from kister Distillation Design for densities and max loading and towler2022 for uv and diameter calcs
class Stage():
    def __init__(self):
        self.Flow = 0
        self.StageNumber = 0

class DisCol():
    def __init__(self, col, fsheet):
        self.Column = col
        self.Tower = self.Column.ColumnFlowsheet.Operations('Main Tower')
        self.Fsheet = fsheet
        self.DiameterRectifying = 0
        self.DiameterStripping = 0
        self.FeedStage = 0
        self.NoStages = self.Tower.NumberOfTrays
        self.TopStage = Stage()
        self.BotStage = Stage()
        self.TraySpacing = 0
        self.HeightRectifying = 0.0
        self.HeightStripping = 0.0
        self.Pressure = max(self.Column.ColumnFlowsheet.PressuresValue)


    def get_feed_stage(self):
        feed_name = self.Tower.FeedStages.Names[0]
        self.FeedStage = int(feed_name.split('__')[0]) # Can use directly as index since Condenser stage = 0

    def get_max_flows(self):
        vapour_flows = self.Column.ColumnFlowsheet.NetMassVapourFlowsValue
        self.TopStage.Flow = max(vapour_flows[0:self.FeedStage-1])
        self.TopStage.StageNumber = vapour_flows.index(self.TopStage.Flow)

        # ignore the last stage as it's the reboiler
        self.BotStage.Flow = max(vapour_flows[self.FeedStage+1:-1])
        self.BotStage.StageNumber = vapour_flows.index(self.BotStage.Flow)

    def get_stage_info(self, stage_no):
        stage = self.Column.ColumnFlowsheet.ColumnStages(str(stage_no)+'__Main Tower').SeparationStage
        temp = stage.TemperatureValue
        pres = stage.PressureValue

        vap_flows = []
        liq_flows = []
        for comp in self.Column.ColumnFlowsheet.ComponentMolarVapourFlowsValue:
            # comp is the flow of component in all stages
            vap_flows.append(comp[stage_no])

        for comp in self.Column.ColumnFlowsheet.ComponentMolarLiquidFlowsValue:
            liq_flows.append(comp[stage_no])

        return temp, pres, vap_flows, liq_flows
    
    def get_fluid_densities(self, t, p, vap_flows, liq_flows):
        dummy_name = 'column_dummy'
        self.Fsheet.MaterialStreams.Add(dummy_name)
        d = self.Fsheet.MaterialStreams(dummy_name)
        d.TemperatureValue = t
        d.PressureValue = p
        d.ComponentMolarFlowValue = vap_flows
        vap_den = d.MassDensityValue

        d.ComponentMolarFlowValue = liq_flows
        liq_den = d.MassDensityValue

        return vap_den, liq_den
    
    # Using the Sinnott and Towler correlation
    def calc_diameter(self, vap_den, liq_den, Vw, lt):
        uv = (-0.171*lt**2 + 0.27*lt - 0.047)*((liq_den - vap_den)/vap_den)**0.5

        diameter = ((4*Vw)/(np.pi*vap_den*uv))**0.5

        return diameter

    # Overall sizing function
    def size(self, lt=0.5):
        # assume 0.5m for tray spacing
        self.TraySpacing = lt

        # Get basic info
        self.get_feed_stage()
        self.get_max_flows()

        # Get section heights
        self.HeightRectifying = lt*self.FeedStage
        self.HeightStripping = lt*(self.NoStages - self.FeedStage)

        # Loop for both column sections
        stage_nos = [self.TopStage, self.BotStage]
        diameters = []
        for stage in stage_nos:
            stage_no = stage.StageNumber
            t, p, v, l = self.get_stage_info(stage_no)
            vap_den, liq_den = self.get_fluid_densities(t, p, v, l)
            d = self.calc_diameter(vap_den, liq_den, stage.Flow, lt)
            diameters.append(d)

        self.DiameterRectifying = diameters[0]
        self.DiameterStripping = diameters[1]

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(attribute, ":", value)