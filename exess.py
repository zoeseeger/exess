#!/usr/bin/env python3
"""Working with exess files."""

import re
import os
import sys
import glob
import math
import json
import tqdm
import numpy as np
import pandas as pd
#import matplotlib.style
#import matplotlib as mpl
#import matplotlib.pyplot as plt
sys.path.append("/Users/zoes/apps/qcp-python-app/qcp")
# sys.path.append("/g/data/k96/apps/qcp/qcp")

### GENERAL FUNCTIONS --------------------------------------------

class Pprint:
    """Print when to_print is set to True."""

    def __init__(self, to_print):
        self.to_print = to_print

    def print_(self, name, value, still_print=True):
        if self.to_print and still_print:
            print('%-20s %s' % (name, value))


def chunk(list_, n):
    """Turn list into list of lists where inner list has length n."""

    for i in range(0, len(list_), n):
        yield list_[i:i + n]


def eof(path, File, percFile):
    """Return percentage end of file as list of lines."""

    # OPEN IN BYTES
    with open(path + File, "rb") as f:
        f.seek(0, 2)                      # Seek @ EOF
        fsize = f.tell()                  # Get size
        Dsize = int(percFile * fsize)
        f.seek (max (fsize-Dsize, 0), 0)  # Set pos @ last n chars lines
        lines = f.readlines()             # Read to end
    # RETURN DECODED LINES
    for i in range(len(lines)):
        try:
            lines[i] = lines[i].decode("utf-8")
        except:
            lines[i] = "CORRUPTLINE"
            print("eof function passed a corrupt line in file ", File)
    return lines


def periodicTable():
    """Periodic table."""

    return {
        "H"    :   [1.0, 1.007825       , 0.430],
        "He"   :   [2.0, 4.0026022      , 0.741],
        "Li"   :   [3.0, 6.938          , 0.880],
        "Be"   :   [4.0, 9.01218315     , 0.550],
        "B"    :   [5.0, 10.806         , 1.030],
        "C"    :   [6.0, 12.0096        , 0.900],
        "N"    :   [7.0, 14.00643       , 0.880],
        "O"    :   [8.0, 15.99491       , 0.880],
        "F"    :   [9.0, 18.99840316    , 0.840],
        "Ne"   :   [10.0, 20.17976      , 0.815],
        "Na"   :   [11.0, 22.98976928   , 1.170],
        "Mg"   :   [12.0, 24.304        , 1.300],
        "Al"   :   [13.0, 26.98153857   , 1.550],
        "Si"   :   [14.0, 28.084        , 1.400],
        "P"    :   [15.0, 30.973762     , 1.250],
        "S"    :   [16.0, 32.059        , 1.220],
        "Cl"   :   [17.0, 35.446        , 1.190],
        "Ar"   :   [18.0, 39.9481       , 0.995],
        "K"    :   [19.0, 39.09831      , 1.530],
        "Ca"   :   [20.0, 40.0784       , 1.190],
        "Sc"   :   [21.0, 44.9559085    , 1.640],
        "Ti"   :   [22.0, 47.8671       , 1.670],
        "V"    :   [23.0, 50.94151      , 1.530],
        "Cr"   :   [24.0, 51.99616      , 1.550],
        "Mn"   :   [25.0, 54.9380443    , 1.555],
        "Fe"   :   [26.0, 55.8452       , 1.540],
        "Co"   :   [27.0, 58.9331944    , 1.530],
        "Ni"   :   [28.0, 58.69344      , 1.700],
        "Cu"   :   [29.0, 63.5463       , 1.720],
        "Zn"   :   [30.0, 65.382        , 1.650],
        "Ga"   :   [31.0, 69.7231       , 1.420],
        "Ge"   :   [32.0, 72.6308       , 1.370],
        "As"   :   [33.0, 74.9215956    , 1.410],
        "Se"   :   [34.0, 78.9718       , 1.420],
        "Br"   :   [35.0, 79.901        , 1.410],
        "Kr"   :   [36.0, 83.7982       , 1.069],
        "Rb"   :   [37.0, 85.46783      , 1.670],
        "Sr"   :   [38.0, 87.621        , 1.320],
        "Y"    :   [39.0, 88.905842     , 1.980],
        "Zr"   :   [40.0, 91.2242       , 1.760],
        "Nb"   :   [41.0, 92.906372     , 1.680],
        "Mo"   :   [42.0, 95.951        , 1.670],
        "Tc"   :   [43.0, 98            , 1.550],
        "Ru"   :   [44.0, 101.072       , 1.600],
        "Rh"   :   [45.0, 102.905502    , 1.650],
        "Pd"   :   [46.0, 106.421       , 1.700],
        "Ag"   :   [47.0, 107.86822     , 1.790],
        "Cd"   :   [48.0, 112.4144      , 1.890],
        "In"   :   [49.0, 114.8181      , 1.830],
        "Sn"   :   [50.0, 118.7107      , 1.660],
        "Sb"   :   [51.0, 121.7601      , 1.660],
        "Te"   :   [52.0, 127.603       , 1.670],
        "I"    :   [53.0, 126.9045      , 1.600],
        "Xe"   :   [54.0, 131.2936      , 1.750],
        "Cs"   :   [55.0, 132.90545     , 1.870],
        "Ba"   :   [56.0, 137.3277      , 1.540],
        "La"   :   [57.0, 138.9055      , 2.070],
        "Ce"   :   [58.0, 140.1161      , 2.030],
        "Pr"   :   [59.0, 140.9077      , 2.020],
        "Nd"   :   [60.0, 144.242       , 2.010],
        "Pm"   :   [61.0, 145           , 2.000],
        "Sm"   :   [62.0, 150.362       , 2.000],
        "Eu"   :   [63.0, 151.9641      , 2.190],
        "Gd"   :   [64.0, 157.253       , 1.990],
        "Tb"   :   [65.0, 158.9253      , 1.960],
        "Dy"   :   [66.0, 162.5001      , 1.950],
        "Ho"   :   [67.0, 164.930       , 1.940],
        "Er"   :   [68.0, 167.2593      , 1.930],
        "Tm"   :   [69.0, 00.0000       , 1.920],
        "Yb"   :   [70.0, 00.0000       , 2.140],
        "Lu"   :   [71.0, 00.0000       , 1.920],
        "Hf"   :   [72.0, 00.0000       , 1.770],
        "Ta"   :   [73.0, 00.0000       , 1.630],
        "W"    :   [74.0, 00.0000       , 1.570],
        "Re"   :   [75.0, 00.0000       , 1.550],
        "Os"   :   [76.0, 00.0000       , 1.570],
        "Ir"   :   [77.0, 00.0000       , 1.520],
        "Pt"   :   [78.0, 00.0000       , 1.700],
        "Au"   :   [79.0, 00.0000       , 1.700],
        "Hg"   :   [80.0,200.5923       , 1.900],
        "Tl"   :   [81.0, 00.0000       , 1.750],
        "Pb"   :   [82.0, 00.0000       , 1.740],
        "Bi"   :   [83.0, 00.0000       , 1.740],
        "Po"   :   [84.0, 00.0000       , 1.880],
        "At"   :   [85.0, 00.0000       , 0.200],
        "Rn"   :   [86.0, 00.0000       , 0.200],
        "Fr"   :   [87.0, 00.0000       , 0.200],
        "Ra"   :   [88.0, 00.0000       , 2.100],
        "Ac"   :   [89.0, 00.0000       , 2.080],
        "Th"   :   [90.0, 00.0000       , 1.990],
        "Pa"   :   [91.0, 00.0000       , 1.810],
        "U"    :   [92.0, 00.0000       , 1.780],
        "Np"   :   [93.0, 00.0000       , 1.750],
        "Pu"   :   [94.0, 00.0000       , 0.200],
        "Am"   :   [95.0, 00.0000       , 1.710],
        "Cm"   :   [96.0, 00.0000       , 0.200],
        "Bk"   :   [97.0, 00.0000       , 0.200],
        "Cf"   :   [98.0, 00.0000       , 1.730],
        "Es"   :   [99.0, 00.0000       , 0.100],
        "Fm"   :   [100.0, 00.0000      , 0.200],
    }


def keyName(*args):
    """Sort values in order and separate by hyphen."""

    a = [int(i) for i in [*args]] # make sure all ints
    a = [str(i) for i in sorted(a)] # sort ints and then return strings
    return '-'.join(a)


### COORD MANIPULATION ------------------------------------------


def midpoint(list_):
    """Return midpoint between a list of values in 1D."""

    return np.min(list_) + (np.max(list_) - np.min(list_))/2


def distance(x1, y1, z1, x2, y2, z2):
    """Return distance between 2 points. Inputs are 6 floats."""

    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def add_center_of_mass(frag_list, atm_list):
    """Add center of mass to each frag."""

    for frag in frag_list:
        m, x, y, z = 0, 0, 0, 0
        for id in frag['ids']:
            atm = atm_list[id]
            x += atm["x"] * atm['mas']
            y += atm["y"] * atm['mas']
            z += atm["z"] * atm['mas']
            m += atm['mas']
        frag['comx'] = x/m
        frag['comy'] = y/m
        frag['comz'] = z/m

    return frag_list


def coords_midpoint(atom_list):
    """Midpoint of all points of xyz."""

    listx, listy, listz = [], [], []
    for atm in atom_list:
        listx.append(atm["x"])
        listy.append(atm["y"])
        listz.append(atm["z"])
    return midpoint(listx), midpoint(listy), midpoint(listz)


def central_frag(frag_list, midpointx, midpointy, midpointz):
    """Returns the frag_id/grp of the central frag by finding the average
    distance to the midpoint for each fragment."""

    min_dist = 10000
    min_ion  = None
    for frag in frag_list:
        dist = distance(midpointx, midpointy, midpointz,
                         frag['comx'], frag['comy'], frag['comz'])
        if dist < min_dist:
            min_dist  = dist
            min_ion   = frag['grp']
    return min_ion


def distances_to_central_frag(fragList, atmList, center_ip_id, cutoff, mers="dimers"):
    """Find the distance between each frag and the central ion pair."""

    dists_list = []
    central = fragList[center_ip_id]

    # FRAG INDICES THOSE BELOW CUTOFFS
    frags_cutoff = []
    for i in range(len(fragList)):
        if fragList[i]["dist"] < cutoff and i != center_ip_id:
            frags_cutoff.append(i)

    if mers == "dimers":
        for i in frags_cutoff:
            # dist = 0
            # for id in frag['ids']:
                # DIST
                # dist += distance(central['comx'], central['comy'], central['comz'],
                #                  atmList[id]['x'], atmList[id]['y'], atmList[id]['z'])
            # AVERAGE
            # dist = dist / len(frag['ids'])

            dist = distance(central['comx'], central['comy'], central['comz'],
                            fragList[i]['comx'], fragList[i]['comy'], fragList[i]['comz'])

            # ADD TO LIST [dist, grp]
            dists_list.append([dist, None, None, None, None, None, None, dist, keyName(fragList[i]["grp"], center_ip_id)])

    elif mers == "trimers":

        for i in range(len(frags_cutoff)):
            frag1 = fragList[frags_cutoff[i]]
            for j in range(i+1, len(frags_cutoff)):
                frag2 = fragList[frags_cutoff[j]]

                r1 = distance(central['comx'], central['comy'], central['comz'],
                        frag1['comx'], frag1['comy'], frag1['comz'])
                r2 = distance(central['comx'], central['comy'], central['comz'],
                        frag2['comx'], frag2['comy'], frag2['comz'])
                r3 = distance(frag1['comx'], frag1['comy'], frag1['comz'],
                        frag2['comx'], frag2['comy'], frag2['comz'])
                dist = (r1 + r2 + r3) / 3
                dmax = max(r1, r2)
                dists_list.append([dist, r1, r2, r3, None, None, None, dmax, keyName(frag1["grp"], frag2["grp"], center_ip_id)])

    elif mers == "tetramers":

        for i in range(len(frags_cutoff)):
            frag1 = fragList[frags_cutoff[i]]
            for j in range(i+1, len(frags_cutoff)):
                frag2 = fragList[frags_cutoff[j]]
                for k in range(j+1, len(frags_cutoff)):
                    frag3 = fragList[frags_cutoff[k]]

                    r1 = distance(central['comx'], central['comy'], central['comz'],
                                  frag1['comx'], frag1['comy'], frag1['comz'])
                    r2 = distance(central['comx'], central['comy'], central['comz'],
                                  frag2['comx'], frag2['comy'], frag2['comz'])
                    r3 = distance(central['comx'], central['comy'], central['comz'],
                                  frag3['comx'], frag3['comy'], frag3['comz'])
                    r4 = distance(frag1['comx'], frag1['comy'], frag1['comz'],
                                  frag2['comx'], frag2['comy'], frag2['comz'])
                    r5 = distance(frag1['comx'], frag1['comy'], frag1['comz'],
                                  frag3['comx'], frag3['comy'], frag3['comz'])
                    r6 = distance(frag2['comx'], frag2['comy'], frag2['comz'],
                                  frag3['comx'], frag3['comy'], frag3['comz'])
                    dist = ( r1 + r2 + r3 + r4 + r5 + r6 ) / 6
                    dmax = max(r1, r2, r3)
                    dists_list.append([dist, r1, r2, r3, r4, r5, r6, dmax, keyName(frag1["grp"], frag2["grp"], frag3["grp"], center_ip_id)])

    return sorted(dists_list, key=lambda x:x[0])


def central_frag_with_charge(frag_list, atmList, midpointx, midpointy, midpointz, charge=0):
    """Returns the frag_id/grp of the central fragment with charge=charge by finding the average
    distance to the midpoint for each fragment."""

    min_dist = 10000
    min_ion  = None
    for frag in frag_list:
        if frag['chrg'] == charge:
            dist = 0
            for id in frag['ids']:
                # DIST
                dist += distance(midpointx, midpointy, midpointz,
                                 atmList[id]['x'], atmList[id]['y'], atmList[id]['z'])
            # AVERAGE
            dist = dist / len(frag['ids'])
            # IF SMALLEST DIST
            if dist < min_dist:
                min_dist  = dist
                min_ion   = frag['grp']
    return min_ion


def add_two_frags_together(fragList, atm_list, frag1_id, frag2_id):
    """Combine two fragments in fragList."""

    new_id = min(frag1_id, frag2_id)
    other_id = max(frag1_id, frag2_id)
    new_fragList = fragList[:new_id] # copy up to the combined one

    new_frag = { # combined frag
        'ids': fragList[frag1_id]['ids'] + fragList[frag2_id]['ids'],
        'syms': fragList[frag1_id]['syms'] + fragList[frag2_id]['syms'],
        'grp': new_id,
        'chrg': fragList[frag1_id]['chrg'] + fragList[frag2_id]['chrg'],
        'mult': fragList[frag1_id]['mult'] + fragList[frag2_id]['mult'] - 1,
        'name': fragList[new_id]['name'],
    }

    new_frag = add_center_of_mass([new_frag], atm_list)

    new_fragList.extend(new_frag) # add new frag

    # add up to removed frag
    new_fragList.extend(fragList[new_id+1:other_id])

    # change rest of values
    for i in range(other_id+1,len(fragList)):
        fragList[i]['grp'] = i-1
        fragList[i]['name'] = f"frag{i-1}"
        new_fragList.append(fragList[i])

    for i in range(len(new_fragList)):
        if i != new_fragList[i]["grp"]:
            print(i, "does not")

    return new_fragList, new_id


def add_dist_from_central_ip(fragList, center_ip_id):
    """Add distance from center_ip which should be first in list."""

    for frag in fragList:
        frag['dist'] = distance(fragList[center_ip_id]['comx'], fragList[center_ip_id]['comy'],
                                fragList[center_ip_id]['comz'],
                                frag['comx'], frag['comy'], frag['comz'])
    return fragList


def frags_in_cutoff(fragList, cutoff, center_ip_id):
    """Return list of indices of frags within cutoff from central ion pair excluding central ion pair."""

    indexes = []
    for i in range(len(fragList)):
        if fragList[i]["dist"] < cutoff and i != center_ip_id:
            indexes.append(i)
    return indexes


### JSON --------------------------------------------


def read_json(filename):
    """Read json file and return contents as dict."""

    with open(filename) as f:
        return json.load(f)


def json_to_frags(json_data):
    """Convert exess input to atoms and frags data."""

    atmList  = []
    fragList = []
    totChrg  = 0
    totMult  = 0
    pTable   = periodicTable()
    mbe      = False

    if "MBE" in json_data['model']['method']:
        mbe = True

    # FROM JSON
    symbols  = json_data["molecule"]["symbols"]
    geometry = json_data["molecule"]["geometry"]

    if mbe:
        frag_ids = json_data["molecule"]["fragments"]["fragid"]
        nfrags   = json_data["molecule"]["fragments"]["nfrag"]
        charges = json_data["molecule"]["fragments"]["fragment_charges"]
        # broken  = json_data["molecule"]["fragments"]["broken_bonds"]
    else:
        frag_ids = [1] * len(symbols)
        nfrags   = 1
        charges  = [0]

    # SPLIT GEOMETRY INTO LIST OF [X, Y, Z]
    coords   = list(chunk(geometry, 3))

    # MAKE EMPTY fragList ORDERED BY GROUP
    for i in range(nfrags):
        fragList.append({
            'ids'  : [],
            'syms' : [],
            'grp'  : i,
            'chrg' : charges[i],
            'mult' : 1,
            'name' : "frag"+str(i),
        })
        totChrg += charges[i]

    # MAKE atmList ORDERED BY APPEARANCE
    for i in range(len(frag_ids)):
        grp = int(frag_ids[i])-1
        atmDict = {
            # 'id'  : i,
            'x'   : float(coords[i][0]),
            'y'   : float(coords[i][1]),
            'z'   : float(coords[i][2]),
            'sym' : symbols[i],
            'grp' : grp,
        }
        for sym, data in pTable.items():
            if atmDict["sym"]  == sym:
                atmDict["nu"]  = data[0]
                atmDict["mas"] = data[1]
                atmDict["vdw"] = data[2]
        atmList.append(atmDict)
        fragList[grp]['ids'].append(i)
        fragList[grp]['syms'].append(symbols[i])

    return fragList, atmList, totChrg, totMult, mbe


def exess_mbe_template(frag_ids, frag_charges, symbols, geometry, method="RIMP2", nfrag_stop=None, basis="cc-pVDZ", auxbasis="cc-pVDZ-RIFIT", number_checkpoints=3):
    """Json many body energy exess template."""

    # FRAGS
    mons = len(frag_charges)
    total_frags = int(mons+mons*(mons-1)/2)

    if not nfrag_stop:
        nfrag_stop = total_frags

    # CHECKPOINTING
    ncheck = number_checkpoints + 1
    ncheck = int((mons+ncheck)/ncheck)

    # METHOD
    method = "MBE-" + method

    dict_ = {
        "driver"    : "energy",
        "model"     : {
            "method": method,
            "basis"     : basis,
            "aux_basis" : auxbasis,
        },
        "keywords"  : {
            "scf"           : {
                "niter"             : 100,
                "ndiis"             : 10,
                "dele"              : 1E-8,
                "rmsd"              : 1E-8,
                "debug"             : False,
            },
            "mbe"       : {
                "ngpus_per_group"   : 4,
            },
            "check_rst": {
                "checkpoint": True,
                "restart": False,
                #"nfrag_check": int((m+ncheck)/ncheck),
                #"nfrag_stop": m # total number of frags
                "nfrag_check": min(ncheck, total_frags),
                "nfrag_stop": min(nfrag_stop, total_frags)
            }
        },
        "molecule"  : {
            "fragments"     : {
                "nfrag"             : len(frag_charges),
                "fragid"            : frag_ids,
                "fragment_charges"  : frag_charges,
                "broken_bonds"      : [],
            },
            "symbols"       : symbols,
            "geometry"      : geometry,
        },
    }

    if number_checkpoints == 0:
        del dict_["keywords"]["check_rst"]

    return dict_


def exess_template(symbols, geometry, method="RIMP2", basis="cc-pVDZ", auxbasis="cc-pVDZ-RIFIT"):
    """RIMP2 template for exess."""

    dict_ = {
        "driver"    : "energy",
        "model"     : {
            "method": method,
            "basis"     : basis,
            "aux_basis" : auxbasis,
        },
        "keywords"  : {
            "scf"           : {
                "niter"             : 100,
                "ndiis"             : 10,
                "dele"              : 1E-8,
                "rmsd"              : 1E-8,
                "debug"             : False,
            },
        },
        "molecule"  : {
            "symbols"       : symbols,
            "geometry"      : geometry,
        },
    }

    return dict_


def format_json_input_file(dict_):
    """Put 5 items per line and 3 coords per line."""

    # GET JSON LINES
    lines = json.dumps(dict_, indent=4)
    # COMPACT LISTS
    newlines = []
    list_ = False
    geometry_ = False
    list_lines =  []
    for line in lines.split('\n'):

        if "]" in line and not '[]' in line:
            list_ = False

            # LIST OF STRINGS - 5 PER LINE
            if not geometry_:
                newline = ""
                for i in range(len(list_lines)):
                    if i % 5 == 0:
                        newline += list_lines[i]
                    elif i % 5 == 4:
                        newline += " " + list_lines[i].strip()
                        newlines.append(newline)
                        newline = ""
                    else:
                        newline += " " + list_lines[i].strip()
                newlines.append(newline)
                newline = ""

            # LIST OF NUMBERS THREE PER LINE
            else:
                newline = ""
                for i in range(len(list_lines)):
                    if i % 3 == 0:
                        newline += list_lines[i]
                    elif i % 3 == 2:
                        newline += " " + list_lines[i].strip()
                        newlines.append(newline)
                        newline = ""
                    else:
                        newline += " " + list_lines[i].strip()
                newlines.append(newline)
                newline = ""

            list_lines = []
            geometry_ = False

        if ": [" in line and not '[]' in line:
            newlines.append(line)
            list_ = True
            if "geometry" in line:
                geometry_ = True

        elif list_:
            list_lines.append(line)

        else:
            newlines.append(line)

    return newlines


def make_exess_input_from_frag_ids(frag_indexs, fragList, atmList, method="RIMP2", nfrag_stop=None, basis="cc-pVDZ", auxbasis="cc-pVDZ-RIFIT", number_checkpoints=3, mbe=False):
    """Make exess input from frag indexes and fraglist."""
    symbols      = []
    frag_ids     = []
    frag_charges = []
    geometry     = []
    xyz_lines    = []
    num          = 0
    # FOR EACH FRAGMENT
    for index in frag_indexs:
        num += 1
        frag_charges.append(fragList[index]['chrg'])
        # FOR EACH ATOM OF THAT FRAG
        for id in fragList[index]['ids']:
            symbols.append(atmList[id]['sym'])
            frag_ids.append(num)
            geometry.extend([atmList[id]['x'], atmList[id]['y'], atmList[id]['z']])
            xyz_lines.append(f"{atmList[id]['sym']} {atmList[id]['x']} {atmList[id]['y']} {atmList[id]['z']}\n")
    # TO JSON
    if mbe:
        json_dict = exess_mbe_template(frag_ids, frag_charges, symbols, geometry, method, nfrag_stop, basis, auxbasis, number_checkpoints)
    else:
        json_dict = exess_template(symbols, geometry, method, basis, auxbasis)
    json_lines = format_json_input_file(json_dict)
    return json_lines, xyz_lines


### WRITE --------------------------------------------


def write_xyz(filename, lines, atmList=None):
    """Write lines to xyz file."""

    if atmList:
        for i in atmList:
            lines.append(f"{i['sym']} {i['x']} {i['y']} {i['z']}\n")

    with open(filename, 'w') as w:
        w.write(f"{len(lines)}\n\n")
        w.writelines(lines)


def write_central_ip(fragList, atmList, center_ip_id, mx, my, mz):
    """WRITE XYZS, COMS, MIDPOINT AND CENTRAL IP TO XYZ"""

    lines = []

    for val, atm in enumerate(atmList):
        # WRITE
        lines.append(f"Cl {mx} {my} {mz}\n")
        if val in fragList[center_ip_id]['ids']:
            lines.append(f"N {atm['x']} {atm['y']} {atm['z']}\n")
        else:
            lines.append(f"H {atm['x']} {atm['y']} {atm['z']}\n")

    write_xyz("../json_sep_frag_calcs/central.xyz", lines)


def write_file(filename, lines):
    """Write any filetype given as list of lines."""

    with open(filename, 'w') as w:
        for line in lines:
            w.write(line + '\n')


def write_job_from_list(path, name, inputfile_list):
    """Write job with for list input files."""

    lines = [
        "#!/bin/bash",
        "#PBS -l walltime=05:00:00",
        "#PBS -l ncpus=48",
        "#PBS -l ngpus=4",
        "#PBS -l mem=384GB",
        "#PBS -l jobfs=100GB",
        "#PBS -q gpuvolta",
        "#PBS -P kv03",
        "#PBS -l storage=gdata/k96+scratch/k96",
        "#PBS -l wd",
        "",
        "# PATH TO EXESS",
        "path_exe=/g/data/k96/apps/EXESS-dev",
        "",
        "# LOAD MODULES",
        "source ~/exess/my_modules.sh",
        "",
        "# UNTAR INPUT FILES",
        f"tar -xzf {name}.tar.gz",
        f"rm {name}.tar.gz",
        "",
        "# RUN",
        "cd $path_exe",
    ]

    tar_line = f"tar -czf {name}.tar.gz "
    rm_line = f"rm "
    for inputfile in inputfile_list:
        outputfile = inputfile.replace('.json', '.log')
        lines.append(f"./run.sh {inputfile} 6 &> $PBS_O_WORKDIR/{outputfile}")
        tar_line += f" {inputfile} {outputfile}"
        rm_line += f" {inputfile} {outputfile}"

    lines += [
        "",
        "# CLEAN UP",
        "cd $PBS_O_WORKDIR",
        tar_line,
        rm_line,
    ]

    write_file(f"{path}/{name}.job", lines)


def write_xxmers(fragList, atmList, center_ip_id, method="RIMP2", typ="dimers", num_json_per_job=120, cutoff_central=None, cutoff_all=None):
    """Write json for each dimer/trimer with central ion pair."""

    dry_run = False

    if cutoff_central == None:
        cutoff_central = 10000
    if cutoff_all == None:
        cutoff_all = 10000

    try:
        os.mkdir(typ)
    except:
        pass

    def write_json(indexes, central, fragList, atmList, typ, inputs, mbe):
        """Write json from indexes."""

        if central:
            name = f"cntr-{keyName(*indexes)}"
        else:
            name = f"add-{keyName(*indexes)}"
        if not dry_run:
            json_lines, lines = make_exess_input_from_frag_ids(indexes, fragList, atmList, method=method, number_checkpoints=0, mbe=mbe)
            write_file(f"{typ}/{name}.json", json_lines)
            # write_xyz(f"{typ}/{name}.xyz", lines)
        inputs.append(f"{name}.json")
        return inputs

    # FRAG INDICES THOSE BELOW CUTOFFS
    frags_cutoff_central = []
    frags_cutoff_all = []
    for i in range(len(fragList)):
        if fragList[i]["dist"] < cutoff_central and i != center_ip_id:
            frags_cutoff_central.append(i)
        if fragList[i]["dist"] < cutoff_all and i != center_ip_id:
            frags_cutoff_all.append(i)

    print("Type:", typ)
    print("Ion pairs in", cutoff_central, ":", len(frags_cutoff_central)+1)
    print("Ion pairs in", cutoff_all, ":", len(frags_cutoff_all)+1)
    print("Ion pairs in cutoff_central", ":", frags_cutoff_central)
    print("Ion pairs in cutoff_all", ":", frags_cutoff_all)


    inputs = []
    if typ == "dimers":

        print("Dimer input files ...")

        # DIMERS WITH CENTRAL IP
        for i in tqdm.tqdm(frags_cutoff_central):
            inputs = write_json([i, center_ip_id], True, fragList, atmList, typ, inputs, True)

        # DIMERS WITH ALL
        for i in tqdm.tqdm(range(len(frags_cutoff_all))):
            for j in range(i+1, len(frags_cutoff_all)):
                inputs = write_json([frags_cutoff_all[i], frags_cutoff_all[j]], False, fragList, atmList, typ, inputs, True)

    elif typ == "trimers":

        print("Trimer json files ...")

        # TRIMERS WITH CENTRAL IP
        for i in tqdm.tqdm(range(len(frags_cutoff_central))):
            for j in range(i+1, len(frags_cutoff_central)):
                inputs = write_json([frags_cutoff_central[i], frags_cutoff_central[j], center_ip_id], True, fragList, atmList, typ, inputs, False)

        # TRIMERS WITH ALL
        for i in tqdm.tqdm(range(len(frags_cutoff_all))):
            for j in range(i+1, len(frags_cutoff_all)):
                for k in range(j+1, len(frags_cutoff_all)):
                    inputs = write_json([frags_cutoff_all[i], frags_cutoff_all[j], frags_cutoff_all[k]], False, fragList, atmList, typ, inputs, False)

    elif typ == "tetramers":

        print("Tetramer json files ...")

        # TETRAMERS WITH CENTRAL IP
        for i in tqdm.tqdm(range(len(frags_cutoff_central))):
            for j in range(i+1, len(frags_cutoff_central)):
                for k in range(j+1, len(frags_cutoff_central)):
                    inputs = write_json([frags_cutoff_central[i], frags_cutoff_central[j], frags_cutoff_central[k], center_ip_id], True, fragList, atmList, typ, inputs, False)
                    # print(keyName(frags_cutoff_central[i], frags_cutoff_central[j], frags_cutoff_central[k], center_ip_id))

        # TETRAMERS WITH ALL
        for i in tqdm.tqdm(range(len(frags_cutoff_all))):
            for j in range(i+1, len(frags_cutoff_all)):
                for k in range(j+1, len(frags_cutoff_all)):
                    for l in range(k+1, len(frags_cutoff_all)):
                        inputs = write_json([frags_cutoff_all[i], frags_cutoff_all[j], frags_cutoff_all[k], frags_cutoff_all[l]], False, fragList, atmList, typ, inputs, False)

    # JOB FILES
    if dry_run:
        write_file(f"{typ}-files.txt", inputs)
    else:
        print("Making jobs and tarring ...")
        input_lists = list(chunk(inputs, num_json_per_job))
        for val, input_list in enumerate(input_lists):
            create_tar(typ, val, input_list, True)
            write_job_from_list(typ, val, input_list)


def create_tar(path, name, files, delete):
    """Create .tar.gz from list of files in path."""

    files_ = ' '.join(files)
    os.system(f"tar -czf {path}/{name}.tar.gz -C {path} {files_}")
    if delete:
        line = "rm"
        for f in files:
            line += f" {path}/{f}"
        os.system(line)


### GRAPHS --------------------------------------------


def plot_dist_v_acc_energy(df, savefile):
    """Plot distance versus accumulated difference in energy."""

    #https://towardsdatascience.com/styling-pandas-plots-and-charts-9b4721e9e597
    mpl.style.use('ggplot')
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['scatter.marker'] = '*'
    ax = df.plot(x='dists', y=['acc_diff_tot', 'acc_diff_hf', 'acc_diff_srs'], style='.-', ms=8)
    ax.set_xlabel("Distance, A")
    ax.set_ylabel("Energy, kJ/mol")
    ax.legend(["Total", "HF", "SRS corr"])
    fig = ax.get_figure()
    fig.savefig(savefile)


def plot_dist_v_energy(df, savefile):
    """Plot distance versus energy of single monomer/dimers."""

    #https://towardsdatascience.com/styling-pandas-plots-and-charts-9b4721e9e597
    try:
        mpl.style.use('ggplot')
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['scatter.marker'] = '*'
        ax = df.plot(x='dists', y=['tot', 'hf', 'srs'], style='.-', ms=8)
        ax.set_xlabel("Distance, A")
        ax.set_ylabel("Energy, kJ/mol")
        ax.legend(["Total", "HF", "SRS corr"])
        fig = ax.get_figure()
        fig.savefig(savefile)
    except Exception as e:
        print("ERROR PLOTTING:", e)


def separated_plot(df):
    mpl.style.use('ggplot')
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['scatter.marker'] = '*'
    fig, axes = plt.subplots(nrows=3, ncols=1)
    ax1 = df.plot(x='dists', y='acc_diff_hf', style='.-', ms=8, ax=axes[0])
    ax2 = df.plot(x='dists', y='acc_diff_srs', style='.-', ms=8, ax=axes[1])
    ax3 = df.plot(x='dists', y='acc_diff_tot', style='.-', ms=8, ax=axes[2])
    plt.xlabel('Distance, A')
    ax2.set_ylabel('Energy, kJ/mol')
    # ax.legend(["Total", "HF", "SRS corr"])
    # fig = ax.get_figure()
    fig.savefig("input-files.pdf")


def single_plot(df, y_column, savefile):
    mpl.style.use('ggplot')
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['scatter.marker'] = '*'
    ax = df.plot(x='dists', y=y_column, style='.-', ms=8)
    ax.set_xlabel('Distance, A')
    ax.set_ylabel('Energy, kJ/mol')
    fig = ax.get_figure()
    fig.savefig(savefile)


### ENERGIES --------------------------------------------


def energies_from_mbe_log(filename):
    """Monomer dimer energies from log file."""

    monomers, dimers = {}, {}
    hf, os_, ss_ = True, False, False
    mons, dims = True, False
    energies = False

    dir, File = os.path.split(filename)
    lines = eof(dir+'/', File, 0.15)

    for line in lines:

        if '-----ENERGIES OF MONOMERS------' in line:
            energies = True
            dims = False
            mons = True

        elif not energies or not line.strip():
            continue

        elif 'Final E(HF) =' in line:
            break

        elif 'DIMER ENERGY CORRECTION' in line:
            dims = True
            mons = False

        elif 'Summary of MBE RI-MP2 OS energies' in line:
            ss_ = False
            os_ = True
            hf = False

        elif 'Summary of MBE RI-MP2 SS energies' in line:
            ss_ = True
            os_ = False
            hf = False

        # ENERGIES
        else:

            # IF ENERGIES IN LINE
            if re.search('^[0-9]', line) or line.startswith('('):
                if mons:
                    id, e = line.split()
                    e = float(e)
                    if hf:
                        monomers[id] = {'hf': e, 'os': np.nan, 'ss': np.nan}
                    elif os_:
                        monomers[id]['os'] = e
                    elif ss_:
                        monomers[id]['ss'] = e

                elif dims:
                    id1, id2, e = line.split()
                    e = float(e)
                    key = keyName(id1, id2)
                    if hf:
                        dimers[key] = {'hf': e, 'os': np.nan, 'ss': np.nan}

                    elif os_:
                        dimers[key]['os'] = e

                    elif ss_:
                        dimers[key]['ss'] = e

    return monomers, dimers


def energies_from_log(filename):
    """Energies from non-mbe RIMP2 or RHF log file."""

    hf, hf2, os_, ss_ = None, None, None, None
    dir, File = os.path.split(filename)
    lines = eof(dir+'/', File, 0.15)
    for line in lines:
        if "Final E(HF)" in line: # MP2
            hf = line.split()[3]
        elif "Final energy is:" in line: # RHF
            hf2 = line.split()[3]
        elif "E(RIMP2 (OS))" in line:
            os_ = line.split()[3]
        elif "E(RIMP2 (SS))" in line:
            ss_ = line.split()[3]
    try:
        # MP2 CALC
        hf, os_, ss_ = float(hf), float(os_), float(ss_)
    except TypeError:
        try:
            # RHF CALC
            hf, os_, ss_ = float(hf2), np.nan, np.nan
        except TypeError:
            print('LOG NOT SUCCESSFUL:', filename)
    return hf, os_, ss_


def distance_energy_df(dimer_dists, center_ip_id, monomers, dimers, trimers=None, tetramers=None, trimer_dists=None, tetramer_dists=None, kjmol=True):
    """Energies as the radius is increased from the central frag."""

    if kjmol:
        conversion = 2625.4996
    else:
        conversion = 1
    os_coef = 1.752
    monomer = monomers[str(center_ip_id)]

    # write monomer energies
    if False:
        new_dict = {"mon": [], "hf": [], "os": [], "ss": []}
        for id_, dict_ in monomers.items():
            new_dict["mon"].append(id_)
            new_dict["hf"].append(dict_['hf'])
            new_dict["os"].append(dict_['os'])
            new_dict["ss"].append(dict_['ss'])
        pd.DataFrame(new_dict).to_csv("monomer_energies.csv", index=False)

    dists_list, ids_list, r1, r2, r3, r4, r5, r6, rmax = [0], [None], [None], [None], [None], [None], [None], [None], [None]

    def energy_list(dists_l, e_dict, tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, typ):
        """Merge distances and converted energies and change to lists."""

        if typ == "dimer":
            num_frags = 2
        elif typ == "trimer":
            num_frags = 3
        elif typ == "tetramer":
            num_frags = 4

        for d, r1_, r2_, r3_, r4_, r5_, r6_, rm, key in dists_l:
            key = str(key)
            dists_list.append(d)
            r1.append(r1_)
            r2.append(r2_)
            r3.append(r3_)
            r4.append(r4_)
            r5.append(r5_)
            r6.append(r6_)
            rmax.append(rm)
            ids_list.append(key)
            hf = e_dict[key]['hf'] / num_frags * conversion
            mp2 = (e_dict[key]['os'] + e_dict[key]['ss']) / num_frags * conversion
            srs = e_dict[key]['os'] / num_frags * os_coef * conversion
            if np.isnan(srs):
                tot = hf
            else:
                tot = hf + srs
            tot_frag.append(tot)
            mp2_frag.append(mp2)
            srs_frag.append(srs)
            hf_frag.append(hf)
            type_frag.append(typ)
            # tot_acc += tot
            # mp2_acc += mp2
            # srs_acc += srs
            # hf_acc  += hf
            # tot_acc_list.append(tot_acc)
            # mp2_acc_list.append(mp2_acc)
            # srs_acc_list.append(srs_acc)
            # hf_acc_list.append(hf_acc)

        return tot_frag, mp2_frag, srs_frag, hf_frag, type_frag

    # MONOMER ENERGIES
    hf = monomer['hf'] * conversion
    mp2 = (monomer['os'] + monomer['ss']) * conversion
    srs = monomer['os'] * os_coef * conversion
    tot = hf + srs
    tot_frag, mp2_frag, srs_frag, hf_frag, type_frag  = [tot], [mp2], [srs], [hf], ["monomer"]
    # tot_acc = tot
    # mp2_acc = mp2
    # srs_acc = srs
    # hf_acc  = hf
    # tot_acc_list, mp2_acc_list, srs_acc_list, hf_acc_list = [tot_acc], [mp2_acc], [srs_acc], [hf_acc]

    # convert energies, merge with distances and add to lists
    tot_frag, mp2_frag, srs_frag, hf_frag, type_frag = energy_list(dimer_dists, dimers, tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, typ="dimer")
    if trimers:
        tot_frag, mp2_frag, srs_frag, hf_frag, type_frag = energy_list(trimer_dists, trimers, tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, typ="trimer")
    if tetramers:
        tot_frag, mp2_frag, srs_frag, hf_frag, type_frag = energy_list(tetramer_dists, tetramers, tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, typ="tetramer")

    data = {
        'dave': dists_list,
        'ids'  : ids_list,
        'tot'  : tot_frag,
        'hf'   : hf_frag,
        'mp2'  : mp2_frag,
        'srs'  : srs_frag,
        'type' : type_frag,
        'r1'   : r1,
        'r2'   : r2,
        'r3'   : r3,
        'r4'   : r4,
        'r5'   : r5,
        'r6'   : r6,
        'rmax' : rmax,
        # 'acc_tot': tot_acc_list,
        # 'acc_hf': hf_acc_list,
        # 'acc_mp2': mp2_acc_list,
        # 'acc_srs': srs_acc_list,
        # 'acc_diff_tot': [x - tot_acc for x in tot_acc_list],
        # 'acc_diff_hf' : [x - hf_acc for x in hf_acc_list],
        # 'acc_diff_srs': [x - srs_acc for x in srs_acc_list],
        # 'acc_diff_mp2': [x - mp2_acc for x in mp2_acc_list],
    }
    # JUST DIMERS
    # dimer_data = {
    #     'dists': dists_list[1:],
    #     'ids'  : ids_list[1:],
    #     'tot'  : tot_frag[1:],
    #     'hf'   : hf_frag[1:],
    #     'mp2'  : mp2_frag[1:],
    #     'srs'  : srs_frag[1:],
    # }

    return data#, dimer_data, tot_acc, hf_acc, mp2_acc, srs_acc


def energies_corr_from_log_when_calculated(filename):
    """Correlation energies from log file where they are calculated."""
    fragments = {}
    # EXTRACT ENERGIES
    with open(filename, 'r') as r:
        for line in r:
            # QUEUE ID OF MOLECULES
            if 'Molecules' in line:
                line = line.replace(';','').replace('+', ' ')
                line = line.split()
                q_id = str(int(line[-1])-1)
                # DIMER
                if len(line) == 6:
                    fragments[q_id] = {'id1': line[1], 'id2': line[2]}
                # MONOMER
                elif len(line) == 5:
                    fragments[q_id] = {'id1': line[1]}
                else:
                    os.exit(f'Line had a different number of items than expected. exiting ...')

            # FRAGMENT ENERGIES
            elif 'Fragment' in line:
                _, q_id, _, _, _, _, os, ss, _ = line.replace(',','').split()
                fragments[q_id]['os'] = float(os)
                fragments[q_id]['ss'] = float(ss)
                # if float(os)*2625.5*1.752 > 2000:
                #     print('os', os)

    return fragments


def energies_dumped_hf(filename, fragments):
    """Read in HF energies from:
            h5dump -m "%.15f" -g "frag_energies" *.h5 > hf.h5dump"""

    monomers, dimers = {}, {}
    with open(filename, 'r') as r:
        for line in r:
            if "):" in line:
                cid, _, hf = line.replace(',','').replace('(','').replace(')',' ').split()
                fragments[cid]['hf'] = float(hf)

    # CONVERT FRAG DICT INTO MONOMER AND DIMER DICT
    for cid, dict_ in fragments.items():
        id1 = dict_['id1']
        id2 = dict_.get('id2')
        # DIMER
        if id2:
            if not dimers.get(id1):
                dimers[id1] = {}
            if not dimers.get(id2):
                dimers[id2] = {}
            dimers[id1][id2] = {'hf': dict_['hf'], 'os': dict_['os'], 'ss': dict_['ss'], 'cid': cid}
            dimers[id2][id1] = {'hf': dict_['hf'], 'os': dict_['os'], 'ss': dict_['ss'], 'cid': cid}
        else:
            monomers[id1] = {'hf': dict_['hf'], 'os': dict_['os'], 'ss': dict_['ss'], 'cid': cid}

    # REMOVE MONOMER CONTRIBUTIONS FROM DIMERS
    for id1, pair_dicts in dimers.items():
        for id2, e_dict in pair_dicts.items():
            e_dict['os'] = e_dict['os'] - monomers[id1]['os'] - monomers[id2]['os']
            e_dict['ss'] = e_dict['ss'] - monomers[id1]['ss'] - monomers[id2]['ss']
            e_dict['hf'] = e_dict['hf'] - monomers[id1]['hf'] - monomers[id2]['hf']

    return monomers, dimers


def energies_from_sep_mbe_dimers(logfiles, center_ip_id):
    """Read energies from each dimer log file and add together."""

    monomers, dimers = {}, {}
    for log in logfiles:
        typ, id1, id2 = log.split('/')[-1].split('.')[0].split('-')
        key = keyName(id1, id2)
        mon, dim = energies_from_mbe_log(log)

        if typ == "add" or id1 != str(center_ip_id):
            first  = mon['0']
            second = mon['1']
        else:
            first  = mon['1']
            second = mon['0']

        if not monomers.get(id1) or np.isnan(monomers[id1].get('os')):
            monomers[id1] = first
        if not monomers.get(id2) or np.isnan(monomers[id2].get('os')):
            monomers[id2] = second

        dimers[key] = dim['0-1']
        dimers[key]['type'] = typ  # add or cntr
    return monomers, dimers


def check_all_completed(logfiles):
    """Check all logfiles were successful."""

    for log in logfiles:
        success = False
        lines = eof('', log, 1)
        for line in lines:
            if "Thanks for using EXESS!" in line:  # MP2
                success = True
            if "CALCULATION IS NOT CONVERGED" in line:
                success = False
                break
        if not success:
            print("!!failed!!", log)


def energies_from_sep_calcs(logfiles, mers):
    """Read energies from each log files and add together."""

    dict_ = {}
    for log in logfiles:
        if mers == "trimers":
            typ, id1, id2, id3 = log.split('/')[-1].split('.')[0].split('-')
            key = keyName(id1, id2, id3)
        elif mers == "tetramers":
            typ, id1, id2, id3, id4 = log.split('/')[-1].split('.')[0].split('-')
            key = keyName(id1, id2, id3, id4)

        hf, os_, ss_ = energies_from_log(log)
        dict_[key] = {
            'hf': hf,
            'os': os_,
            'ss': ss_,
            'type': typ,
        }

    return dict_


def trimer_contributions(trimers, dimers, monomers):
    """Remove dimer and monomer energies from trimer energies."""

    for key, dict_ in trimers.items():
        id1, id2, id3 = key.split('-')
        dict_['hf'] = dict_['hf'] - \
            dimers[keyName(id1, id2)]['hf'] - dimers[keyName(id1, id3)]['hf'] - dimers[keyName(id2, id3)]['hf'] - \
            monomers[id1]['hf'] - monomers[id2]['hf'] - monomers[id3]['hf']
        # if key == "0-2-3":
            # print(1, "dict_['os']", dict_['os'])
            # print(2, "dimers[keyName(id1, id2)]['os']", dimers[keyName(id1, id2)]['os'])
            # print(3, "dimers[keyName(id1, id3)]['os']", dimers[keyName(id1, id3)]['os'])
            # print(4, "dimers[keyName(id2, id3)]['os']", dimers[keyName(id2, id3)]['os'])
            # print(5, "monomers[id1]['os']", monomers[id1]['os'])
            # print(6, "monomers[id2]['os']", monomers[id2]['os'])
            # print(7, "monomers[id3]['os']", monomers[id3]['os'])
        dict_['os'] = dict_['os'] - \
            dimers[keyName(id1, id2)]['os'] - dimers[keyName(id1, id3)]['os'] - dimers[keyName(id2, id3)]['os'] - \
            monomers[id1]['os'] - monomers[id2]['os'] - monomers[id3]['os']
        # if key == "0-2-3":
            # print(8, "dict_['os']", dict_['os'])
        dict_['ss'] = dict_['ss'] - \
            dimers[keyName(id1, id2)]['ss'] - dimers[keyName(id1, id3)]['ss'] - dimers[keyName(id2, id3)]['ss']- \
            monomers[id1]['ss'] - monomers[id2]['ss'] - monomers[id3]['ss']
        # except KeyError:
        #     dict_['os'], dict_['ss'] = np.nan, np.nan
    return trimers


def tetramer_contributions(tetramers, trimers, dimers, monomers):
    """Remove trimer, dimer and monomer energies from tetramer energies."""

    for key, dict_ in tetramers.items():
        id1, id2, id3, id4 = key.split('-')
        dict_['hf'] = dict_['hf'] - \
            trimers[keyName(id1, id2, id3)]['hf'] - trimers[keyName(id1, id2, id4)]['hf'] - \
            trimers[keyName(id1, id3, id4)]['hf'] - trimers[keyName(id2, id3, id4)]['hf'] - \
            dimers[keyName(id1, id2)]['hf'] - dimers[keyName(id1, id3)]['hf'] - dimers[keyName(id1, id4)]['hf'] - \
            dimers[keyName(id2, id3)]['hf']- dimers[keyName(id2, id4)]['hf'] - dimers[keyName(id3, id4)]['hf'] - \
            monomers[id1]['hf'] - monomers[id2]['hf'] - monomers[id3]['hf'] - monomers[id4]['hf']
        # try:
        dict_['os'] = dict_['os'] - \
            trimers[keyName(id1, id2, id3)]['os'] - trimers[keyName(id1, id2, id4)]['os'] - \
            trimers[keyName(id1, id3, id4)]['os'] - trimers[keyName(id2, id3, id4)]['os'] - \
            dimers[keyName(id1, id2)]['os'] - dimers[keyName(id1, id3)]['os'] - dimers[keyName(id1, id4)]['os'] - \
            dimers[keyName(id2, id3)]['os']- dimers[keyName(id2, id4)]['os'] - dimers[keyName(id3, id4)]['os'] - \
            monomers[id1]['os'] - monomers[id2]['os'] - monomers[id3]['os'] - monomers[id4]['os']
        dict_['ss'] = dict_['ss'] - \
            trimers[keyName(id1, id2, id3)]['ss'] - trimers[keyName(id1, id2, id4)]['ss'] - \
            trimers[keyName(id1, id3, id4)]['ss'] - trimers[keyName(id2, id3, id4)]['ss'] - \
            dimers[keyName(id1, id2)]['ss'] - dimers[keyName(id1, id3)]['ss'] - dimers[keyName(id1, id4)]['ss'] - \
            dimers[keyName(id2, id3)]['ss']- dimers[keyName(id2, id4)]['ss'] - dimers[keyName(id3, id4)]['ss'] - \
            monomers[id1]['ss'] - monomers[id2]['ss'] - monomers[id3]['ss'] - monomers[id4]['ss']
        # except KeyError: # IF NOT ENTERED INTO DICT
        #     dict_['os'], dict_['ss'] = np.nan, np.nan

    return tetramers


def remove_additional_mers(dict_):
    """Remove mers of type add which are not with central frag."""

    for key in list(dict_.keys()):
        if dict_[key]["type"] == "add":
            del dict_[key]
    return dict_


def distance_central_com_of_dimer(fragList, atmList, center_ip_id, cutoff):
    """Get distances for trimer extrapolation where R is the distance of the center of mass of the central ion pair
    with the center of mass of the remaining dimer."""

    dists_dict = {}
    central = fragList[center_ip_id]

    # FRAG INDICES THOSE BELOW CUTOFFS
    frags_cutoff = []
    for i in range(len(fragList)):
        if fragList[i]["dist"] < cutoff and i != center_ip_id:
            frags_cutoff.append(i)

    for i in range(len(frags_cutoff)):
        frag1 = fragList[frags_cutoff[i]]
        for j in range(i+1, len(frags_cutoff)):
            frag2 = fragList[frags_cutoff[j]]

            m, x, y, z = 0, 0, 0, 0
            for id in frag1['ids'] + frag2['ids']:
                atm = atmList[id]
                x += atm["x"] * atm['mas']
                y += atm["y"] * atm['mas']
                z += atm["z"] * atm['mas']
                m += atm['mas']
            comx = x / m
            comy = y / m
            comz = z / m

            r = distance(central['comx'], central['comy'], central['comz'],
                    comx, comy, comz)
            dists_dict[keyName(frag1["grp"], frag2["grp"], center_ip_id)] = r

    return dists_dict


### TOP LEVEL --------------------------------------------------------


def make_dimer_trimer_tetramer_calcs(jsonfile, method="RIMP2", num_json_dimers=50, num_json_trimers=30, num_json_tetramers=10, dimers=True, trimers=True, tetramers=True, debug=False, cutoff_dims=None, cutoff_trims=None, cutoff_tets=None):
    """Make dimer job files for each dimer with the central fragment."""

    p = Pprint(to_print=debug)

    p.print_("name", jsonfile)

    # READ JSON
    json_data = read_json(jsonfile)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe = json_to_frags(json_data)

    # ADD CENTER OF MASS (COM) - USED FOR CENTRAL IP
    fragList = add_center_of_mass(fragList, atmList)

    # GET MIDPOINT OF ALL COORDS
    mx, my, mz = coords_midpoint(atmList)
    p.print_("midpoint", (mx, my, mz))

    # GET CENTRAL IP
    center_ip_id = central_frag(fragList, mx, my, mz)
    p.print_("center_ip_id", center_ip_id)

    # ADD DIST FROM CENTRAL IP
    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    # WRITE XYZS, COMS, MIDPOINT AND CENTRAL IP TO XYZ
    if debug:
        write_central_ip(fragList, atmList, center_ip_id, mx, my, mz)

    cutoff_pents = 0
    if dimers:
        write_xxmers(fragList, atmList, center_ip_id,
                     num_json_per_job=num_json_dimers, method=method, typ="dimers", cutoff_central=cutoff_dims, cutoff_all=cutoff_trims)
    if trimers:
        write_xxmers(fragList, atmList, center_ip_id,
                     num_json_per_job=num_json_trimers, method=method, typ="trimers", cutoff_central=cutoff_trims, cutoff_all=cutoff_tets)
    if tetramers:
        write_xxmers(fragList, atmList, center_ip_id,
                     num_json_per_job=num_json_tetramers, method=method, typ="tetramers", cutoff_central=cutoff_tets, cutoff_all=cutoff_pents)


def df_from_logs(
        jsonfile,
        logfile=None,
        hf_dump_file=None,
        debug=False,
        get_energies="end"
    ):
    """Make pandas data frame from json and log files."""

    cutoff_dims = None
    cutoff_trims = 35
    cutoff_tets = 20
    cutoff_pents = 0

    if cutoff_dims == None:
        cutoff_dims = 10000
    if cutoff_trims == None:
        cutoff_trims = 10000

    p = Pprint(to_print=debug)

    # READ JSON
    json_data = read_json(jsonfile)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe = json_to_frags(json_data)

    # ADD CENTER OF MASS (COM) - USED FOR CENTRAL IP
    fragList = add_center_of_mass(fragList, atmList)

    # GET MIDPOINT OF XYZ
    mx, my, mz = coords_midpoint(atmList)
    p.print_("midpoint", (mx, my, mz))

    # GET CENTRAL IP
    center_ip_id = central_frag(fragList, mx, my, mz)
    p.print_("center_ip_id", center_ip_id)

    # ADD DISTANCE FROM CENTRAL FRAG
    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    # WRITE XYZS, COMS, MIDPOINT AND CENTRAL IP TO XYZ
    if debug:
        write_central_ip(fragList, atmList, center_ip_id, mx, my, mz)

    # GET ENERGIES
    trimers = {}
    tetramers = {}
    if get_energies == "end":
        name = logfile.split('.')
        monomers, dimers = energies_from_mbe_log(logfile)
    elif get_energies == "dump":
        name = logfile.split('.')
        fragments = energies_corr_from_log_when_calculated(logfile) # mp2 from start of files
        monomers, dimers = energies_dumped_hf(hf_dump_file, fragments) # hf from h5dump
    elif get_energies == "manual":
        name = "output"
        # print("check dimer logs")
        # check_all_completed(glob.glob("dimers/*log"))
        # print("check trimer logs")
        # check_all_completed(glob.glob("trimers/*log"))
        # print("check tetramer logs")
        # check_all_completed(glob.glob("tetramers/*log"))
        # print("done")
        monomers, dimers = energies_from_sep_mbe_dimers(glob.glob("dimers/*log"), center_ip_id)
        # print("monomers")
        # for i, j in monomers.items():
        #     print(i, j)
        # print("dimers")
        # for i, j in dimers.items():
        #     print(i, j)

        # get higher level energies
        if os.path.isdir("trimers"):
            trimers = energies_from_sep_calcs(glob.glob("trimers/*log"), mers="trimers")
            trimers = trimer_contributions(trimers, dimers, monomers)
        if os.path.isdir("tetramers"):
            tetramers = energies_from_sep_calcs(glob.glob("tetramers/*log"), mers="tetramers")
            tetramers = tetramer_contributions(tetramers, trimers, dimers, monomers)
        # remove higher level mers without central frag
        dimers = remove_additional_mers(dimers)
        if os.path.isdir("trimers"):
            trimers = remove_additional_mers(trimers)
    p.print_("name", name)
    p.print_(f"monomers['{center_ip_id}']", monomers[str(center_ip_id)])
    # p.print_(f"dimers['{center_ip_id}-1']", dimers[keyName(center_ip_id, 1)])

    # DISTANCE FROM EACH FRAG TO CENTRAL IP
    dimers_dists = distances_to_central_frag(fragList, atmList, center_ip_id, cutoff_dims, mers="dimers")
    trimers_dists = distances_to_central_frag(fragList, atmList, center_ip_id, cutoff_trims, mers="trimers")
    tetramer_dists = distances_to_central_frag(fragList, atmList, center_ip_id, cutoff_tets, mers="tetramers")
    p.print_("dimers_dists[0]", dimers_dists[0])

    # ENERGY PER DISTANCE - PANDAS
    df_data = distance_energy_df(dimers_dists, center_ip_id, monomers, dimers, trimers, tetramers, trimers_dists, tetramer_dists)
    df      = pd.DataFrame(df_data)
    p.print_("df_data.keys()", df_data.keys())
    p.print_("df_data", df_data, still_print=False)
    df.to_csv("df.csv", index=False)
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    # print(df.head())
    # print(dimer_df.head())

    # PRINT ENERGIES
    # if print_energies:
    #     print()
    #     # print("tot hf mp2_cor srs_cor")
    #     # print(total, hf, mp2_cor, srs_cor)
    #     os_h  = srs_cor / 1.752 / 2625.4996
    #     ss_h  = mp2_cor / 2625.4996 - os_h
    #     tot_h = total/2625.4996
    #     print(tot_h, os_h, ss_h)
    #     print(total, mp2_cor, srs_cor)
    #
    # # PLOT ENERGIES
    # if plot_graphs:
    #     e = graph_energy_cutoff
    #     d = 30
    #     plot_dist_v_acc_energy(df, f"{name}-acc.pdf")
    #     plot_dist_v_acc_energy(df[(df['acc_diff_hf'] < e) & (df['acc_diff_hf'] > -e)], f"{name}-acc-{e}kj.pdf")
    #     plot_dist_v_energy(dimer_df, f"{name}-e.pdf")
    #     plot_dist_v_energy(dimer_df[dimer_df['dists'] > d], f"{name}-e-{d}.pdf")
    #     # plot_dist_v_energy(df[(df['acc_diff_hf'] < e) & (df['acc_diff_hf'] > -e)], f"{name}-e-{e}kj.pdf")
    #     single_plot(df, 'acc_diff_hf', f"{name}-acc_diff_hf.pdf")
    #     single_plot(df[df['dists'] > d], 'acc_diff_hf', f"{name}-acc_diff_hf-{d}.pdf")
    #     single_plot(dimer_df, 'hf', f"{name}-hf.pdf")
    #     single_plot(dimer_df[dimer_df['dists'] > d], 'hf', f"{name}-hf-{d}.pdf")


def make_central_ip(jsonfile, debug=False):
    """Make central ion pair from the central anion and cation."""

    p = Pprint(to_print=debug)

    p.print_("name", jsonfile)

    # READ JSON
    json_data = read_json(jsonfile)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe = json_to_frags(json_data)

    # ADD CENTER OF MASS (COM) - USED FOR CENTRAL IP
    fragList = add_center_of_mass(fragList, atmList)

    # GET MIDPOINT OF ALL COORDS
    mx, my, mz = coords_midpoint(atmList)
    p.print_("midpoint", (mx, my, mz))

    # GET CENTRAL ANION AND CENTRAL CATION
    center_cat = central_frag_with_charge(fragList, atmList, mx, my, mz, 1)
    center_an = central_frag_with_charge(fragList, atmList, mx, my, mz, -1)
    p.print_("center_cat", center_cat)
    p.print_("center_an", center_an)

    # ADD CENTRAL ANION AND CATION TOGETHER
    fragList, center_frag = add_two_frags_together(fragList, atmList, center_cat, center_an)

    # print(center_frag)
    # print(fragList[center_frag])

    # WRITE XYZS, COMS, MIDPOINT AND CENTRAL IP TO XYZ
    if debug:
        write_central_ip(fragList, atmList, center_frag)

    # MAKE NEW JSON
    json_lines, xyz_lines = make_exess_from_frag_ids(list(range(len(fragList))), fragList, atmList)
    write_file(f"../input_files/c2mim-bf4-1159-single-ions-central-ip-logfiles/new-json-central-ip.json", json_lines)

    # write_dimers(fragList, atmList, center_frag, 120)


def make_job_from_jsons(num_json_per_job):
    """Make job with exess files from either a file called torun.txt or those in the dir."""

    # get files
    if os.path.exists("torun.txt"):
        print("Found:", "torun.txt")
        fs = open("torun.txt", 'r').readlines()
    else:
        print("Using all json in dir")
        fs = glob.glob("*json")

    # chunk files
    files = list(chunk(fs, num_json_per_job))
    for val, input_list in enumerate(files):
        create_tar("../json_sep_frag_calcs", val, input_list, True)
        write_job_from_list("../json_sep_frag_calcs", val, input_list)


def make_smaller_shell_from_json(json_, cutoff):
    """Takes a json mbe file and keeps only frags within the cutoff."""

    # READ JSON
    json_data = read_json(json_)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe = json_to_frags(json_data)

    # CHECK IS MBE
    if not mbe:
        sys.exit('Expected MBE input. exiting ...')

    # ADD CENTER OF MASS (COM) - USED FOR CENTRAL IP
    fragList = add_center_of_mass(fragList, atmList)

    # GET MIDPOINT OF ALL COORDS
    mx, my, mz = coords_midpoint(atmList)

    # GET CENTRAL IP
    center_ip_id = central_frag(fragList, mx, my, mz)

    # ADD DIST FROM CENTRAL IP
    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    # INDEXES WITHIN NEW CUTOFF
    indexes = frags_in_cutoff(fragList, cutoff, center_ip_id)
    indexes.insert(0, center_ip_id)
    print(f"Fragments within {cutoff}: {len(indexes)}")

    # NEW MBE
    json_lines, lines = make_exess_input_from_frag_ids(indexes, fragList, atmList, number_checkpoints=0, mbe=mbe)

    # WRITE FILES
    write_file(json_.replace('.js', f'-{cutoff}.js'), json_lines)
    write_xyz(json_.replace('.json', f'-{cutoff}.xyz'), lines)
    write_file("indexes.txt", [str(i) for i in indexes])


def xyz_to_json(filename, mbe, method):
    """Convert xyz file to exess json input."""

    from system import systemData

    dir, File = os.path.split(filename)
    fragList, atmList, totChrg, totMult = systemData(dir, File, True)

    # NEW MBE
    json_lines, lines = make_exess_input_from_frag_ids(list(range(0,len(fragList))), fragList, atmList, number_checkpoints=0, mbe=mbe, method=method)

    # WRITE FILES
    write_file(File.replace('.xyz', '_new.json'), json_lines)


def run(value, filename):
    """Call function depending on user input."""

    if value == "0" or value == "":
        pass

    # make sep dimer/trimer/tetramer calcs
    elif value == "1":

        def t_or_f(value):
            if value == 't':
                return True
            elif value == 'f':
                return False
            else:
                return None

        jsn = glob.glob("*.json")[0]
        print(f"Using: {jsn}")

        # which mers to create
        user_ = input("Perform Dimers Trimers Tetramers t/f [t t t]: ")
        if user_ == "":
            user_ = "t t t"
        dim, tri, tet = user_.split()
        dim = t_or_f(dim)
        tri = t_or_f(tri)
        tet = t_or_f(tet)

        # cutoffs to use
        user_ = input("Cutoffs Dimers Trimers Tetramers [None 35 20]: ")
        if user_ == "":
            user_ = "None 35 20"
        user_ = user_.replace("None", "10000")
        cutoff_dims, cutoff_trims, cutoff_tets = [float(i) for i in user_.split()]

        # method
        method = input("Method [RIMP2]: ")
        if method == "":
            method = "RIMP2"

        # number of inputs per job file
        parent_dir = os.getcwd().split('/')[-1]
        # if 'c1mpyr-ntf2' in parent_dir:
        #     num_json_dimers, num_json_trimers, num_json_tetramers = 64, 27, 16
        if 'c2mim-bf4' in parent_dir:
            num_json_dimers, num_json_trimers, num_json_tetramers = 130, 70, 40
        # elif 'gdm-cl' in parent_dir:
        #     num_json_dimers, num_json_trimers, num_json_tetramers = 292, 146, 78
        # elif 'gdm-etso3' in parent_dir:
        #     num_json_dimers, num_json_trimers, num_json_tetramers = 125, 61, 35
        else:
            user_ = input("Inputs per job Dimers Trimers Tetramers [130 70 40]: ")
            if user_ == "":
                user_ = "130 70 40"
            num_json_dimers, num_json_trimers, num_json_tetramers = [int(i) for i in user_.split()]

        make_dimer_trimer_tetramer_calcs(
            jsonfile=jsn,
            num_json_dimers=num_json_dimers,
            num_json_trimers=num_json_trimers,
            num_json_tetramers=num_json_tetramers,
            dimers=dim,
            trimers=tri,
            tetramers=tet,
            debug=True,
            cutoff_dims=cutoff_dims,
            cutoff_trims=cutoff_trims,
            cutoff_tets=cutoff_tets,
            method=method
        )

    # make job from jsons
    elif value == "2":
        user_ = input("Number of xxmers per job: ")
        make_job_from_jsons(int(user_))

    # dataframe from logs
    elif value == "3":
        jsn = glob.glob("*.json")[0]
        log = None
        df_from_logs(
            jsonfile=jsn,
            logfile=log,
            hf_dump_file=None,
            get_energies="manual",
            debug=True
        )

    # json to xyz
    elif value == "4":

        if not filename:
            filename = glob.glob("*json")[0]

        # READ JSON
        json_data = read_json(filename)

        # CONVERT JSON TO FRAG DATA
        _, atmList, _, _, _ = json_to_frags(json_data)

        # WRITE XYZ
        xyzfile = filename.replace(".json", ".xyz")
        write_xyz(xyzfile, lines=[], atmList=atmList)

    # xyz to json
    elif value == "5":
        from system import systemData

        mbe = input("MBE [y]: ")
        meth = input("Method [RHF]: ")

        if mbe == "n" or mbe == "N":
            mbe = False
        else:
            mbe = True

        if meth == "":
            meth = "RHF"

        if not filename:
            filename = glob.glob("*xyz")[0]

        xyz_to_json(filename, mbe, meth)

    # makes smaller shell from json
    elif value == "6":

        user_ = float(input("Cutoff distance of new shell: "))

        if not filename:
            filename = glob.glob("*json")[0]

        make_smaller_shell_from_json(filename, user_)



# CALL SCRIPT -------------------------------------------

print("What would you like to do?")
print("    1. Make separate dimer/trimer/tetramer calculations")
print("    2. Make job from json files")
print("    3. CSV from log files")
print("    4. Json to xyz")
print("    5. Xyz to json")
print("    6. Make smaller shell from json")
print("    0. Quit")

user = None
filename = None
while not user in ["1", "2", "3", "4", "5", "6", "0", '']:

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    user = input("Value: ")
    run(user, filename)
