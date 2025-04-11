# Testing out object persistence with pickle for hysys case

# Need material streams, energy streams, operations and component list

# Material streams first - use setattr from here: https://www.moonbooks.org/Articles/How-to-dynamically-add-new-attributes-to-a-class-in-python-/

import math as m
import os

if os.name == "nt":
    import win32com.client as win32

import jsonpickle

# Object that can add attributes
class Hysys_Ob:
    def __init__(self, name):
        self.name = name

    def adding_new_attr(self, attr):
        setattr(self, attr, attr)


# Need to loop through dir to find the attributes that have values
def make_hysys_ob(original_ob, name, attr_list=''):
    # Get directory of attributes and create target/new object
    new_ob = Hysys_Ob(name)
    if attr_list == '':
        ob_dir = dir(original_ob)
    else:
        ob_dir = attr_list 

    for attr_name in ob_dir:
        # skipping things that start with _
        # if attr_name[0] == "_":
        #     continue

        # Getting attribute value with exception in case it isn't defined
        try:
            a = getattr(original_ob, attr_name)
        except:
            continue

        # if it's a value or list, include it
        if isinstance(a, (int, str, tuple)):
            setattr(new_ob, attr_name, a)
        elif (isinstance(a, float) and m.isnan(a) == False):  # ignore Nan numbers that break JSON
            setattr(new_ob, attr_name, a)
        elif attr_name in ["Feeds","AttachedOpers","AttachedFeeds","AttachedProducts"]:
            setattr(new_ob, attr_name, a.Names)

    return new_ob


# Creating a class for a hysys simulation
class Hysys_Flowsheet:

    # Creating function to loop through and add material streams
    def add_material(self, hysys_sim, attr_list=''):
        print("    Adding material streams...")
        names = hysys_sim.MaterialStreams.Names

        for name in names:
            self.progress(name, names)
            hysys_stream = hysys_sim.MaterialStreams.Item(name)
            new_stream = make_hysys_ob(hysys_stream, name, attr_list)

            # Add to MaterialStreams attribute
            self.ms[name] = new_stream

    # Similar function for energy streams
    def add_energy(self, hysys_sim, attr_list=''):
        print("\r", "   Adding energy streams...")
        names = hysys_sim.EnergyStreams.Names

        for name in names:
            self.progress(name, names)
            hysys_stream = hysys_sim.EnergyStreams.Item(name)
            new_stream = make_hysys_ob(hysys_stream, name, attr_list)

            # Add to MaterialStreams attribute
            self.es[name] = new_stream

    # Final function for operations
    def add_operation(self, hysys_sim):
        print("\r", "   Adding unit operations...")
        names = hysys_sim.Operations.Names

        for name in names:
            self.progress(name, names)
            hysys_ob = hysys_sim.Operations.Item(name)
            new_ob = make_hysys_ob(hysys_ob, name)

            # Add to MaterialStreams attribute
            self.op[name] = new_ob

    # Getting the list of components
    def add_components(self, hysys_sim):
        try:
            comps = hysys_sim.Parent.BasisManager.ComponentLists[0].Components.Names
        except:
            comps = hysys_sim.Parent.Parent.Parent.BasisManager.ComponentLists[0].Components.Names  # if its a subflowsheet

        # Add to Component attribute
        self.comp = comps

    # Progress indicator
    def progress(self, name, names):
        total = len(names)
        current = names.index(name) + 1
        prog = round(current / total * 10)  # nearest 10%
        prog_bar = "[" + prog * "=" + (10 - prog) * " " + "]"

        # print('\r','  ',str(prog)+'%',end='')
        print("\r", "  ", prog_bar, end="")

    # Initialising the class
    def __init__(self, hysys_sim):

        self.name = hysys_sim.name
        self.comp = []  # list of components
        self.ms = {}  # material streams
        self.es = {}  # energy streams
        self.op = {}  # operations/units

        print("Creating", self.name)

        # Getting the required list of attributes
        with open('hysys_python/reference/attr_list_material.json', "r") as f:
            material_json = f.readline()
        with open('hysys_python/reference/attr_list_energy.json', "r") as f:
            energy_json = f.readline()

        material_list = jsonpickle.decode(material_json)
        energy_list = jsonpickle.decode(energy_json)

        self.add_components(hysys_sim)
        self.add_material(hysys_sim,material_list)
        self.add_energy(hysys_sim,energy_list)
        self.add_operation(hysys_sim)
        print("\r", "   Complete     ")


# Function to connect to correct aspen flowsheet
def hysys_connection(file_name, active=1):

    # print("Connecting to Aspen...")

    # Path to correct Aspen File
    hyFilePath = os.path.abspath(file_name)

    # Starting HYSYS - if else just to hide errors on MacOS
    if os.name == "nt":
        # hysys_app = win32.Dispatch("HYSYS.Application")
        hysys_app = win32.gencache.EnsureDispatch("HYSYS.Application") # for some reason the above doesn't work with adding feed streams to distillation columns
    else:
        hysys_app = []

    # Opening hysys document
    if active == 0:
        hysys_case = hysys_app.SimulationCases.Open(hyFilePath)
    elif active == 1:
        hysys_case = hysys_app.ActiveDocument
    else:
        raise Exception('Argument for input variable "active" is not valid')

    # Setting if want hysys to be visible
    hysys_case.Visible = 1

    # Returning hysys object
    return hysys_case


# Creating a function to connect and create JSON file
def create_json(hysys_file, flowsheet="", json_file=""):
    # Get connection to hysys object
    if flowsheet == "":
        main_sim = hysys_connection(hysys_file)
        hysys_sim = main_sim.Flowsheet
        space = 0
    else:
        main_sim = hysys_connection(hysys_file)
        hysys_sim = main_sim.Flowsheet.Flowsheets(flowsheet)
        space = 1

    # Creating the JSON file
    hysys_ob = Hysys_Flowsheet(hysys_sim)
    if json_file == "":
        json_file = (
            hysys_file[: (len(hysys_file) - 4)] + "_" * space + flowsheet + ".json"
        )

    hysys_json = jsonpickle.encode(hysys_ob)
    with open(json_file, "w") as f:
        f.write(hysys_json)

    print(json_file, "saved\n")


# Loading the json file
def load_json(json_file):
    # path = os.path
    with open(json_file, "r") as f:
        hysys_json = f.readline()

    hysys = jsonpickle.decode(hysys_json)

    return hysys
