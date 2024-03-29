#!/usr/bin/env python3
"""Working with exess files."""

import re
import os
import sys
import glob
import math
import json
import tqdm
import h5py
from itertools import permutations
import numpy as np
import pandas as pd

sys.path.append("/Users/zoes/apps/qcp-python-app/qcp")
sys.path.append("/g/data/k96/apps/qcp/qcp")


### GENERAL FUNCTIONS --------------------------------------------

class Pprint:
    """Print when to_print is set to True."""

    def __init__(self, to_print):
        self.to_print = to_print

    def print_(self, name, value=None, still_print=True):
        if self.to_print and still_print:
            if value:
                print('%-20s %s' % (name, value))
            else:
                print('%-20s' % name)


def chunk(list_, n):
    """Turn list into list of lists where inner list has length n."""

    for i in range(0, len(list_), n):
        yield list_[i:i + n]


def endOfFile(path, file, perc):
    """Return percentage end of file as list of lines."""

    # OPEN IN BYTES
    with open(path + file, "rb") as f:
        f.seek(0, 2)  # Seek @ EOF
        fsize = f.tell()  # Get size
        Dsize = int(perc * fsize)
        f.seek(max(fsize - Dsize, 0), 0)  # Set pos @ last n chars lines
        lines = f.readlines()  # Read to end
    # RETURN DECODED LINES
    for i in range(len(lines)):
        try:
            lines[i] = lines[i].decode("utf-8")
        except:
            lines[i] = "CORRUPTLINE"
            print("eof function passed a corrupt line in file ", file)
    return lines


def periodicTable():
    """Periodic table."""

    return {
        "H": [1.0, 1.007825, 0.430],
        "He": [2.0, 4.0026022, 0.741],
        "Li": [3.0, 6.938, 0.880],
        "Be": [4.0, 9.01218315, 0.550],
        "B": [5.0, 10.806, 1.030],
        "C": [6.0, 12.0096, 0.900],
        "N": [7.0, 14.00643, 0.880],
        "O": [8.0, 15.99491, 0.880],
        "F": [9.0, 18.99840316, 0.840],
        "Ne": [10.0, 20.17976, 0.815],
        "Na": [11.0, 22.98976928, 1.170],
        "Mg": [12.0, 24.304, 1.300],
        "Al": [13.0, 26.98153857, 1.550],
        "Si": [14.0, 28.084, 1.400],
        "P": [15.0, 30.973762, 1.250],
        "S": [16.0, 32.059, 1.220],
        "Cl": [17.0, 35.446, 1.190],
        "Ar": [18.0, 39.9481, 0.995],
        "K": [19.0, 39.09831, 1.530],
        "Ca": [20.0, 40.0784, 1.190],
        "Sc": [21.0, 44.9559085, 1.640],
        "Ti": [22.0, 47.8671, 1.670],
        "V": [23.0, 50.94151, 1.530],
        "Cr": [24.0, 51.99616, 1.550],
        "Mn": [25.0, 54.9380443, 1.555],
        "Fe": [26.0, 55.8452, 1.540],
        "Co": [27.0, 58.9331944, 1.530],
        "Ni": [28.0, 58.69344, 1.700],
        "Cu": [29.0, 63.5463, 1.720],
        "Zn": [30.0, 65.382, 1.650],
        "Ga": [31.0, 69.7231, 1.420],
        "Ge": [32.0, 72.6308, 1.370],
        "As": [33.0, 74.9215956, 1.410],
        "Se": [34.0, 78.9718, 1.420],
        "Br": [35.0, 79.901, 1.410],
        "Kr": [36.0, 83.7982, 1.069],
        "Rb": [37.0, 85.46783, 1.670],
        "Sr": [38.0, 87.621, 1.320],
        "Y": [39.0, 88.905842, 1.980],
        "Zr": [40.0, 91.2242, 1.760],
        "Nb": [41.0, 92.906372, 1.680],
        "Mo": [42.0, 95.951, 1.670],
        "Tc": [43.0, 98, 1.550],
        "Ru": [44.0, 101.072, 1.600],
        "Rh": [45.0, 102.905502, 1.650],
        "Pd": [46.0, 106.421, 1.700],
        "Ag": [47.0, 107.86822, 1.790],
        "Cd": [48.0, 112.4144, 1.890],
        "In": [49.0, 114.8181, 1.830],
        "Sn": [50.0, 118.7107, 1.660],
        "Sb": [51.0, 121.7601, 1.660],
        "Te": [52.0, 127.603, 1.670],
        "I": [53.0, 126.9045, 1.600],
        "Xe": [54.0, 131.2936, 1.750],
        "Cs": [55.0, 132.90545, 1.870],
        "Ba": [56.0, 137.3277, 1.540],
        "La": [57.0, 138.9055, 2.070],
        "Ce": [58.0, 140.1161, 2.030],
        "Pr": [59.0, 140.9077, 2.020],
        "Nd": [60.0, 144.242, 2.010],
        "Pm": [61.0, 145, 2.000],
        "Sm": [62.0, 150.362, 2.000],
        "Eu": [63.0, 151.9641, 2.190],
        "Gd": [64.0, 157.253, 1.990],
        "Tb": [65.0, 158.9253, 1.960],
        "Dy": [66.0, 162.5001, 1.950],
        "Ho": [67.0, 164.930, 1.940],
        "Er": [68.0, 167.2593, 1.930],
        "Tm": [69.0, 00.0000, 1.920],
        "Yb": [70.0, 00.0000, 2.140],
        "Lu": [71.0, 00.0000, 1.920],
        "Hf": [72.0, 00.0000, 1.770],
        "Ta": [73.0, 00.0000, 1.630],
        "W": [74.0, 00.0000, 1.570],
        "Re": [75.0, 00.0000, 1.550],
        "Os": [76.0, 00.0000, 1.570],
        "Ir": [77.0, 00.0000, 1.520],
        "Pt": [78.0, 00.0000, 1.700],
        "Au": [79.0, 00.0000, 1.700],
        "Hg": [80.0, 200.5923, 1.900],
        "Tl": [81.0, 00.0000, 1.750],
        "Pb": [82.0, 00.0000, 1.740],
        "Bi": [83.0, 00.0000, 1.740],
        "Po": [84.0, 00.0000, 1.880],
        "At": [85.0, 00.0000, 0.200],
        "Rn": [86.0, 00.0000, 0.200],
        "Fr": [87.0, 00.0000, 0.200],
        "Ra": [88.0, 00.0000, 2.100],
        "Ac": [89.0, 00.0000, 2.080],
        "Th": [90.0, 00.0000, 1.990],
        "Pa": [91.0, 00.0000, 1.810],
        "U": [92.0, 00.0000, 1.780],
        "Np": [93.0, 00.0000, 1.750],
        "Pu": [94.0, 00.0000, 0.200],
        "Am": [95.0, 00.0000, 1.710],
        "Cm": [96.0, 00.0000, 0.200],
        "Bk": [97.0, 00.0000, 0.200],
        "Cf": [98.0, 00.0000, 1.730],
        "Es": [99.0, 00.0000, 0.100],
        "Fm": [100.0, 00.0000, 0.200],
    }


def getAllKeys(*args):
    """For a key get all contrtibuting frags."""

    mons = [str(i) for i in [*args]]
    frags = [*mons]

    # dims
    if len(mons) > 1:
        for i in range(len(mons)-1):
            for j in range(i+1, len(mons)):
                frags.append(keyName(mons[i], mons[j]))

    # tris
    if len(mons) > 2:
        for i in range(len(mons)-2):
            for j in range(i+1, len(mons)-1):
                for k in range(j+1, len(mons)):
                    frags.append(keyName(mons[i], mons[j], mons[k]))

    # tet
    if len(mons) == 4:
        frags.append(keyName(*mons))

    return frags


def keyName(*args):
    """Sort values in order and separate by hyphen - used as a unique key."""

    a = [int(i) for i in [*args]]  # make sure all ints
    a = [str(i) for i in sorted(a)]  # sort ints and then return strings
    return '-'.join(a)


### COORD MANIPULATION ------------------------------------------

angstrom2bohr = 1.88973
hartree2kjmol = 2625.4996
osvdz2srs = 1.752
osvtz2srs = 1.64
bsse_cutoff = 100


def midpoint(list_):
    """Return midpoint between a list of values in 1D."""

    return np.min(list_) + (np.max(list_) - np.min(list_)) / 2


def distance(x1, y1, z1, x2, y2, z2):
    """Return distance between 2 points. Inputs are 6 floats."""

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def add_centroids(frag_list, atm_list):
    """Add centroid to each frag."""

    for frag in frag_list:
        i, x, y, z = 0, 0, 0, 0
        for id in frag['ids']:
            atm = atm_list[id]
            x += atm["x"]
            y += atm["y"]
            z += atm["z"]
            i += 1
        frag['cx'] = x / i
        frag['cy'] = y / i
        frag['cz'] = z / i

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
    distance to the centroid for each fragment."""

    min_dist = 10000
    min_ion = None
    for frag in frag_list:
        dist = distance(midpointx, midpointy, midpointz,
                        frag['cx'], frag['cy'], frag['cz'])
        if dist < min_dist:
            min_dist = dist
            min_ion = frag['grp']
    return min_ion


def distances_between_frags(fragList, cutoff, mers="dimers"):
    """Find the distance between each frag calc."""

    dists_list = []

    if mers == "dimers":
        for i in range(len(fragList) - 1):
            frag1 = fragList[i]
            for j in range(i + 1, len(fragList)):
                frag2 = fragList[j]
                r1 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag2['cx'], frag2['cy'], frag2['cz'])
                if r1 < cutoff:
                    dists_list.append([r1, None, None, None, None, None, None, r1, keyName(frag1["grp"], frag2["grp"])])

    elif mers == "trimers":
        for i in range(len(fragList) - 2):
            frag1 = fragList[i]
            for j in range(i + 1, len(fragList) - 1):
                frag2 = fragList[j]
                r1 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag2['cx'], frag2['cy'], frag2['cz'])
                if r1 < cutoff:
                    for k in range(j + 1, len(fragList)):
                        frag3 = fragList[k]
                        r2 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag3['cx'], frag3['cy'], frag3['cz'])
                        r3 = distance(frag2['cx'], frag2['cy'], frag2['cz'], frag3['cx'], frag3['cy'], frag3['cz'])
                        if r2 < cutoff and r3 < cutoff:
                            dmax = max(r1, r2, r3)
                            dist = (r1 + r2 + r3) / 3
                            dists_list.append([dist, r1, r2, r3, None, None, None, dmax,
                                               keyName(frag1["grp"], frag2["grp"], frag3["grp"])])

    elif mers == "tetramers":
        for i in range(len(fragList) - 3):
            frag1 = fragList[i]
            for j in range(i + 1, len(fragList) - 2):
                frag2 = fragList[j]
                r1 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag2['cx'], frag2['cy'], frag2['cz'])
                if r1 < cutoff:
                    for k in range(j + 1, len(fragList) - 1):
                        frag3 = fragList[k]
                        r2 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag3['cx'], frag3['cy'], frag3['cz'])
                        r3 = distance(frag2['cx'], frag2['cy'], frag2['cz'], frag3['cx'], frag3['cy'], frag3['cz'])
                        if r2 < cutoff and r3 < cutoff:
                            for l in range(k + 1, len(fragList)):
                                frag4 = fragList[l]
                                r4 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag4['cx'], frag4['cy'], frag4['cz'])
                                r5 = distance(frag2['cx'], frag2['cy'], frag2['cz'], frag4['cx'], frag4['cy'], frag4['cz'])
                                r6 = distance(frag3['cx'], frag3['cy'], frag3['cz'], frag4['cx'], frag4['cy'], frag4['cz'])
                                if (r4 < cutoff and r5 < cutoff) and r6 < cutoff:
                                    dmax = max(r1, r2, r3, r4, r5, r6)
                                    dist = (r1 + r2 + r3 + r4 + r5 + r6) / 6
                                    dists_list.append([dist, r1, r2, r3, r4, r5, r6, dmax,
                                                       keyName(frag1["grp"], frag2["grp"], frag3["grp"], frag4["grp"])])

    return sorted(dists_list, key=lambda x: x[0])


def distances_to_central_frag(fragList, center_ip_id, cutoff, mers="dimers"):
    """Find the distance between each frag and the central ion pair."""

    dists_list = []

    # FRAG INDICES THOSE BELOW CUTOFFS
    frags_cutoff = []
    for i in range(len(fragList)):
        if fragList[i]["dist"] < cutoff and i != center_ip_id:
            frags_cutoff.append(i)

    if mers == "dimers":
        for i in frags_cutoff:
            dist = fragList[i]['dist']

            # ADD TO LIST [dist, grp]
            dists_list.append(
                [dist, None, None, None, None, None, None, dist, keyName(fragList[i]["grp"], center_ip_id)])

    elif mers == "trimers":

        for i in range(len(frags_cutoff)):
            frag1 = fragList[frags_cutoff[i]]
            for j in range(i + 1, len(frags_cutoff)):
                frag2 = fragList[frags_cutoff[j]]
                r3 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag2['cx'], frag2['cy'], frag2['cz'])
                if r3 < cutoff:
                    r1 = frag1["dist"]
                    r2 = frag2["dist"]
                    dist = (r1 + r2 + r3) / 3
                    dmax = max(r1, r2)
                    dists_list.append(
                        [dist, r1, r2, r3, None, None, None, dmax, keyName(frag1["grp"], frag2["grp"], center_ip_id)])

    elif mers == "tetramers":

        for i in range(len(frags_cutoff)):
            frag1 = fragList[frags_cutoff[i]]
            for j in range(i + 1, len(frags_cutoff)):
                frag2 = fragList[frags_cutoff[j]]
                r4 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag2['cx'], frag2['cy'], frag2['cz'])
                if r4 < cutoff:
                    for k in range(j + 1, len(frags_cutoff)):
                        frag3 = fragList[frags_cutoff[k]]
                        r5 = distance(frag1['cx'], frag1['cy'], frag1['cz'], frag3['cx'], frag3['cy'], frag3['cz'])
                        r6 = distance(frag2['cx'], frag2['cy'], frag2['cz'], frag3['cx'], frag3['cy'], frag3['cz'])
                        if r5 < cutoff and r6 < cutoff:
                            r1 = frag1["dist"]
                            r2 = frag2["dist"]
                            r3 = frag3["dist"]
                            dist = (r1 + r2 + r3 + r4 + r5 + r6) / 6
                            dmax = max(r1, r2, r3)
                            dists_list.append([dist, r1, r2, r3, r4, r5, r6, dmax,
                                               keyName(frag1["grp"], frag2["grp"], frag3["grp"], center_ip_id)])

    return sorted(dists_list, key=lambda x: x[0])


def central_frag_with_charge(frag_list, atmList, midpointx, midpointy, midpointz, charge=0):
    """Returns the frag_id/grp of the central fragment with charge=charge by finding the average
    distance to the midpoint for each fragment."""

    min_dist = 10000
    min_ion = None
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
                min_dist = dist
                min_ion = frag['grp']
    return min_ion


def pair_ions_lowest_dist(fragList, atmList):
    """Get pairing of molecules that has lowest total distance."""

    # cation/anion lists
    cations, anions = [], []
    for i in range(len(fragList)):
        if fragList[i]['chrg'] == 1:
            cations.append(i)
        elif fragList[i]['chrg'] == -1:
            anions.append(i)
        else:
            sys.exit("Only written for singly charged species. exiting ...")

    anions = list(permutations(anions)) # perms of anions
    cations = [cations] * len(anions)   # make list of lists of cations

    # make combinations
    combinations = []
    for an_list, cat_list in zip(anions, cations):
        comb = []
        for an, cat in zip(an_list, cat_list):
            comb.append([cat, an])
        combinations.append(comb)

    # pair
    comb, min_dist = combination_smallest_distance(fragList, combinations)

    # sort combinations largest val to smallest so can combine frags safely
    comb_sorted = []
    starting_frags = len(fragList)
    for i in range(starting_frags-1, -1, -1):
        for _ in comb:
            if i in _:
                comb_sorted.append(_)
                comb.remove(_)
                break

    # combine frags
    for index1, index2 in comb_sorted:
        lines = []
        fragList, newid = add_two_frags_together(fragList, atmList, index1, index2)
        # for id in fragList[newid]['ids']:
        #     atm = atmList[id]
        #     lines.append(f"{atm['sym']} {atm['x']} {atm['y']} {atm['z']}\n")
        # write_xyz_zoe(f"{index1}-{index2}.xyz", lines)

    return fragList


def add_two_frags_together(fragList, atm_list, frag1_id, frag2_id):
    """Combine two fragments in fragList."""

    new_id = min(frag1_id, frag2_id)
    other_id = max(frag1_id, frag2_id)
    new_fragList = fragList[:new_id]  # copy up to the combined one

    new_frag = {  # combined frag
        'ids': fragList[frag1_id]['ids'] + fragList[frag2_id]['ids'],
        'syms': fragList[frag1_id]['syms'] + fragList[frag2_id]['syms'],
        'grp': new_id,
        'chrg': fragList[frag1_id]['chrg'] + fragList[frag2_id]['chrg'],
        'mult': fragList[frag1_id]['mult'] + fragList[frag2_id]['mult'] - 1,
        'name': f"frag{new_id}",
    }

    new_frag = add_centroids([new_frag], atm_list)

    new_fragList.extend(new_frag)  # add new frag

    # add up to removed frag
    new_fragList.extend(fragList[new_id + 1:other_id])

    # change rest of values
    for i in range(other_id + 1, len(fragList)):
        fragList[i]['grp'] = i - 1
        fragList[i]['name'] = f"frag{i - 1}"
        new_fragList.append(fragList[i])

    for i in range(len(new_fragList)):
        if i != new_fragList[i]["grp"]:
            print(i, "does not")

    return new_fragList, new_id


def combination_smallest_distance(fragList, combinations):
    """Return the list of anion-cation pairs that has the smallest distance."""

    comb_use = None
    min_dist = 1000
    for comb in combinations:

        tot_dist = 0

        # FOR EACH CATION, ANION PAIR
        for cat, an in comb:

            tot_dist += distance(
                fragList[cat]['cx'], fragList[cat]['cy'], fragList[cat]['cz'],
                fragList[an]['cx'], fragList[an]['cy'], fragList[an]['cz']
            )

        if tot_dist < min_dist:
            min_dist = tot_dist
            comb_use = comb

    return comb_use, min_dist


def add_dist_from_central_ip(fragList, center_ip_id):
    """Add distance from center_ip which should be first in list."""

    for frag in fragList:
        frag['dist'] = distance(fragList[center_ip_id]['cx'], fragList[center_ip_id]['cy'],
                                fragList[center_ip_id]['cz'],
                                frag['cx'], frag['cy'], frag['cz'])
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

    atmList = []
    fragList = []
    totChrg = 0
    totMult = 0
    pTable = periodicTable()
    mbe = False
    lattice = False
    level = None
    central_mon = None

    if json_data['model'].get('fragmentation'):
        mbe = True
        if json_data["keywords"]["frag"].get("lattice_energy_calc"):
            lattice = json_data["keywords"]["frag"].get("lattice_energy_calc")
        level = json_data["keywords"]["frag"].get("level")

    central_mon = json_data["keywords"]["frag"].get("reference_monomer")

    # FROM JSON
    symbols = json_data["molecule"]["symbols"]
    geometry = json_data["molecule"]["geometry"]

    if mbe:
        frag_ids = json_data["molecule"]["fragments"]["fragid"]
        nfrags = json_data["molecule"]["fragments"]["nfrag"]
        charges = json_data["molecule"]["fragments"]["fragment_charges"]
        # broken  = json_data["molecule"]["fragments"]["broken_bonds"]
    else:
        frag_ids = [1] * len(symbols)
        nfrags = 1
        charges = [0]

    # SPLIT GEOMETRY INTO LIST OF [X, Y, Z]
    coords = list(chunk(geometry, 3))

    # MAKE EMPTY fragList ORDERED BY GROUP
    for i in range(nfrags):
        fragList.append({
            'ids': [],
            'syms': [],
            'grp': i,
            'chrg': charges[i],
            'mult': 1,
            'name': "frag" + str(i),
        })
        totChrg += charges[i]

    # MAKE atmList ORDERED BY APPEARANCE
    for i in range(len(frag_ids)):
        grp = int(frag_ids[i]) - 1
        atmDict = {
            # 'id'  : i,
            'x': float(coords[i][0]),
            'y': float(coords[i][1]),
            'z': float(coords[i][2]),
            'sym': symbols[i],
            'grp': grp,
        }
        for sym, data in pTable.items():
            if atmDict["sym"] == sym:
                atmDict["nu"] = data[0]
                atmDict["mas"] = data[1]
                atmDict["vdw"] = data[2]
        atmList.append(atmDict)
        fragList[grp]['ids'].append(i)
        fragList[grp]['syms'].append(symbols[i])

    return fragList, atmList, totChrg, totMult, mbe, lattice, level, central_mon


def exess_mbe_template(frag_ids, frag_charges, symbols, geometry, method="RIMP2", nfrag_stop=None, basis="cc-pVDZ",
                       auxbasis="cc-pVDZ-RIFIT", number_checkpoints=3, level=4, ref_mon=0):
    """Json many body energy exess template."""

    # FRAGS
    mons = len(frag_charges)
    total_frags = int(mons + mons * (mons - 1) / 2)

    if not nfrag_stop:
        nfrag_stop = total_frags

    # CHECKPOINTING
    ncheck = number_checkpoints + 1
    ncheck = int((mons + ncheck) / ncheck)

    to_checkpoint = True
    if number_checkpoints == 0:
        to_checkpoint = False

    dict_ = {
        "driver": "energy",
        "model": {
            "method": method,
            "basis": basis,
            "aux_basis": auxbasis,
            "fragmentation": True
        },
        "keywords": {
            "scf": {
                "niter": 100,
                "ndiis": 10,
                "dele": 1E-8,
                "rmsd": 1E-8,
                "dynamic_threshold": 10,
                "debug": False,
            },
            "frag": {
                "method": "MBE",
                "level": level,
                "ngpus_per_group": 4,
                "lattice_energy_calc": True,
                "reference_monomer": ref_mon,
                "dimer_cutoff": 1000 * angstrom2bohr,
                "dimer_mp2_cutoff": 20 * angstrom2bohr,
                "trimer_cutoff": 40 * angstrom2bohr,
                "trimer_mp2_cutoff": 20 * angstrom2bohr,
                "tetramer_cutoff": 25 * angstrom2bohr,
                "tetramer_mp2_cutoff": 10 * angstrom2bohr
            },
            "FMO": {
                "fmo_type": "CPF",
                "mulliken_approx": False,
                "esp_cutoff": 100000,
                "esp_maxit": 50
            },
            "check_rst": {
                "checkpoint": to_checkpoint,
                "restart": False,
                "nfrag_check": min(ncheck, total_frags),
                "nfrag_stop": min(nfrag_stop, total_frags)
            }
        },
        "molecule": {
            "fragments": {
                "nfrag": len(frag_charges),
                "fragid": frag_ids,
                "fragment_charges": frag_charges,
                "broken_bonds": [],
            },
            "symbols": symbols,
            "geometry": geometry,
        },
    }

    return dict_


def exess_template(symbols, geometry, method="RIMP2", basis="cc-pVDZ", auxbasis="cc-pVDZ-RIFIT"):
    """Full ab initio template for exess."""

    dict_ = {
        "driver": "energy",
        "model": {
            "method": method,
            "basis": basis,
            "aux_basis": auxbasis,
            "fragmentation": False
        },
        "keywords": {
            "scf": {
                "niter": 100,
                "ndiis": 10,
                "dele": 1E-8,
                "rmsd": 1E-8,
                "dynamic_threshold": 10,
                "debug": False,
            },
        },
        "molecule": {
            "symbols": symbols,
            "geometry": geometry,
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
    list_lines = []
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


def make_exess_input_from_frag_ids(frag_indexs, fragList, atmList, method="RIMP2", nfrag_stop=None, basis="cc-pVDZ",
                                   auxbasis="cc-pVDZ-RIFIT", number_checkpoints=3, mbe=False, ref_mon=0, level=4):
    """Make exess input from frag indexes and fraglist."""

    symbols = []
    frag_ids = []
    frag_charges = []
    geometry = []
    xyz_lines = []
    num = 0

    # convert to integers
    frag_indexs = [int(x) for x in frag_indexs]

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
        json_dict = exess_mbe_template(frag_ids, frag_charges, symbols, geometry, method, nfrag_stop, basis, auxbasis,
                                       number_checkpoints, level, ref_mon=ref_mon)
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
    """WRITE XYZS, MIDPOINT AND CENTRAL IP TO XYZ"""

    lines = []

    for val, atm in enumerate(atmList):
        # WRITE
        lines.append(f"Cl {mx} {my} {mz}\n")
        if val in fragList[center_ip_id]['ids']:
            lines.append(f"N {atm['x']} {atm['y']} {atm['z']}\n")
        else:
            lines.append(f"H {atm['x']} {atm['y']} {atm['z']}\n")

    write_xyz("central.xyz", lines)


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


def write_xxmers(fragList, atmList, center_ip_id, method="RIMP2", typ="dimers", num_json_per_job=120,
                 cutoff_central=None, cutoff_all=None):
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
            json_lines, lines = make_exess_input_from_frag_ids(indexes, fragList, atmList, method=method,
                                                               number_checkpoints=0, mbe=mbe)
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
    print("Ion pairs in", cutoff_central, ":", len(frags_cutoff_central) + 1)
    print("Ion pairs in", cutoff_all, ":", len(frags_cutoff_all) + 1)
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
            for j in range(i + 1, len(frags_cutoff_all)):
                inputs = write_json([frags_cutoff_all[i], frags_cutoff_all[j]], False, fragList, atmList, typ, inputs,
                                    True)

    elif typ == "trimers":

        print("Trimer json files ...")

        # TRIMERS WITH CENTRAL IP
        for i in tqdm.tqdm(range(len(frags_cutoff_central))):
            for j in range(i + 1, len(frags_cutoff_central)):
                inputs = write_json([frags_cutoff_central[i], frags_cutoff_central[j], center_ip_id], True, fragList,
                                    atmList, typ, inputs, False)

        # TRIMERS WITH ALL
        for i in tqdm.tqdm(range(len(frags_cutoff_all))):
            for j in range(i + 1, len(frags_cutoff_all)):
                for k in range(j + 1, len(frags_cutoff_all)):
                    inputs = write_json([frags_cutoff_all[i], frags_cutoff_all[j], frags_cutoff_all[k]], False,
                                        fragList, atmList, typ, inputs, False)

    elif typ == "tetramers":

        print("Tetramer json files ...")

        # TETRAMERS WITH CENTRAL IP
        for i in tqdm.tqdm(range(len(frags_cutoff_central))):
            for j in range(i + 1, len(frags_cutoff_central)):
                for k in range(j + 1, len(frags_cutoff_central)):
                    inputs = write_json(
                        [frags_cutoff_central[i], frags_cutoff_central[j], frags_cutoff_central[k], center_ip_id], True,
                        fragList, atmList, typ, inputs, False)

        # TETRAMERS WITH ALL
        for i in tqdm.tqdm(range(len(frags_cutoff_all))):
            for j in range(i + 1, len(frags_cutoff_all)):
                for k in range(j + 1, len(frags_cutoff_all)):
                    for l in range(k + 1, len(frags_cutoff_all)):
                        inputs = write_json(
                            [frags_cutoff_all[i], frags_cutoff_all[j], frags_cutoff_all[k], frags_cutoff_all[l]], False,
                            fragList, atmList, typ, inputs, False)

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


### ENERGIES --------------------------------------------


def energies_from_mbe_log(filename):
    """Monomer dimer energies from log file."""

    monomers, dimers, trimers, tetramers = {}, {}, {}, {}
    hf, os_, ss_ = True, False, False
    mons, dims, tris, tets = True, False, False, False
    energies = False

    def storeEnergy(dict_, key, energy):
        """Store energy in given dict depending on whether HF, OS or SS."""

        energy = float(energy)
        if hf:
            dict_[key] = {'hf': energy, 'os': 0.0, 'ss': 0.0}
        elif os_:
            dict_[key]['os'] = energy
        elif ss_:
            dict_[key]['ss'] = energy

        return dict_

    dir, File = os.path.split(filename)
    dir = dir or "."
    lines = endOfFile(dir + '/', File, 0.25)
    print(lines[1])
    for line in lines:

        # if "1     9699" in line:
        #     print(mons, dims, tris, tets, len(line.split()))
        #     print(line)

        if not line.strip():
            continue

        elif '-----ENERGIES OF MONOMERS------' in line:
            energies = True
            tets = False
            tris = False
            dims = False
            mons = True

        elif not energies:
            continue

        elif 'Final E(HF) =' in line:
            break

        elif 'DIMER ENERGY CORRECTION' in line:
            dims = True
            mons = False

        elif 'TRIMER ENERGY CORRECTION' in line:
            dims = False
            tris = True

        elif 'TETRAMER ENERGY CORRECTION' in line:
            tris = False
            tets = True

        elif 'RI-MP2 OS energies***' in line:
            ss_ = False
            os_ = True
            hf = False

        elif 'RI-MP2 SS energies***' in line:
            ss_ = True
            os_ = False
            hf = False

        elif 'ID' in line:
            if 'RIJ' in line:
                rij = True
            else:
                rij = False

        # ENERGIES
        else:

            # IF ENERGIES IN LINE
            if re.search('^[0-9]', line) or line.startswith('('):
                if mons:
                    if rij:
                        spl_line = line.split()
                        if len(spl_line) == 3:
                            id, e, rij = spl_line
                        elif len(spl_line) == 2:
                            id, hold = spl_line
                            e = hold[:-1]
                            rij = hold[-1]
                        else:
                            sys.exit("Unexpected number of items in split line")
                    else:
                        id, e = line.split()
                    monomers = storeEnergy(monomers, id, e)

                elif dims:
                    if rij:
                        spl_line = line.split()
                        if len(spl_line) == 4:
                            id1, id2, e, rij = spl_line
                        elif len(spl_line) == 3:
                            id1, id2, hold = spl_line
                            e = hold[:-1]
                            rij = hold[-1]
                        else:
                            sys.exit("Unexpected number of items in split line")
                    else:
                        id1, id2, e = line.split()
                    key = keyName(id1, id2)
                    dimers = storeEnergy(dimers, key, e)

                elif tris:
                    if rij:
                        spl_line = line.split()
                        if len(spl_line) == 5:
                            id1, id2, id3, e, rij = spl_line
                        elif len(spl_line) == 4:
                            id1, id2, id3, hold = spl_line
                            e = hold[:-1]
                            rij = hold[-1]
                    else:
                        id1, id2, id3, e = line.split()
                    key = keyName(id1, id2, id3)
                    trimers = storeEnergy(trimers, key, e)

                elif tets:
                    if rij:
                        if len(spl_line) == 5:
                            id1, id2, id3, id4, e, rij = spl_line
                        elif len(spl_line) == 4:
                            id1, id2, id3, id4, hold = spl_line
                            e = hold[:-1]
                            rij = hold[-1]
                    else:
                        id1, id2, id3, id4, e = line.split()
                    key = keyName(id1, id2, id3, id4)
                    tetramers = storeEnergy(tetramers, key, e)

    return monomers, dimers, trimers, tetramers


def energies_from_log(filename):
    """Energies from non-mbe RIMP2 or RHF log file."""

    hf, hf2, os_, ss_ = None, None, None, None
    dir, File = os.path.split(filename)
    lines = endOfFile(dir + '/', File, 0.15)
    for line in lines:
        if "Final E(HF)" in line:  # MP2
            hf = line.split()[3]
        elif "Final energy is:" in line:  # RHF
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
            hf, os_, ss_ = float(hf2), 0.0, 0.0
        except TypeError:
            print('LOG NOT SUCCESSFUL:', filename)
            sys.exit()
    return hf, os_, ss_


def distance_energy_df(dimer_dists, center_ip_id, monomers, dimers, trimers=None, tetramers=None, trimer_dists=None,
                       tetramer_dists=None, kjmol=True, lattice=True):
    """Energies as the radius is increased from the central frag."""

    if kjmol:
        conversion = hartree2kjmol
    else:
        conversion = 1

    basis = "vdz"

    center_ip_id = str(center_ip_id)

    if basis == 'vdz':
        os_coef = osvdz2srs
    elif basis == 'vtz':
        os_coef = osvtz2srs

    dists_list, ids_list, r1, r2, r3, r4, r5, r6, rmax, bsse = [], [], [], [], [], [], [], [], [], []
    tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central = [], [], [], [], [], []

    # monomers
    for id_, dict_ in monomers.items():
        dists_list.append(np.nan)
        ids_list.append(id_)
        r1.append(np.nan)
        r2.append(np.nan)
        r3.append(np.nan)
        r4.append(np.nan)
        r5.append(np.nan)
        r6.append(np.nan)
        rmax.append(np.nan)
        bsse.append(np.nan)
        hf_frag.append(dict_['hf'] * conversion)
        mp2_frag.append((dict_['os'] + dict_['ss']) * conversion)
        srs_frag.append(dict_['os'] * os_coef * conversion)
        tot_frag.append(hf_frag[-1] + srs_frag[-1])
        type_frag.append("monomer")
        if id_ == center_ip_id:
            central.append(True)
        else:
            central.append(False)

    def energy_list(dists_l, central_ip_id, e_dict, tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central, typ):
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
            if os.path.exists("../bsse"):
                # only did trimers within 8Å radius
                if (typ == "trimer" and rm > bsse_cutoff) or typ == "tetramer":
                    bsse.append(np.nan)
                else:
                    bsse_ = getBSSEFrag(key, lattice, central_ip_id) / num_frags * conversion
                    bsse.append(bsse_)
            else:
                bsse.append(np.nan)
            # get hf
            try:
                hf = e_dict[key]['hf']
            except:
                # print(e_dict)
                # hf = np.nan
                print(f"!!!Frag {key} HF not found with ave distance {d}!!!")
                sys.exit()

            hf = hf / num_frags * conversion

            # get corr
            os_ = e_dict.get(key, {}).get('os')
            ss = e_dict.get(key, {}).get('ss')
            if not os_:
                os_ = np.nan
                ss = np.nan
                # print(f"!!!Frag {key} OS not found with ave distance {d}!!!")
                # sys.exit()
            mp2 = (os_ + ss) / num_frags * conversion
            srs = os_ / num_frags * os_coef * conversion

            # add hf to cor if calculated for this frag
            if np.isnan(srs):
                tot = hf
            else:
                tot = hf + srs

            tot_frag.append(tot)
            mp2_frag.append(mp2)
            srs_frag.append(srs)
            hf_frag.append(hf)
            type_frag.append(typ)
            if central_ip_id in key.split('-'):
                central.append(True)
            else:
                central.append(False)

        return tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central

    # convert energies, merge with distances and add to lists
    tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central = energy_list(dimer_dists, center_ip_id, dimers, tot_frag,
                mp2_frag, srs_frag, hf_frag, type_frag, central, typ="dimer")
    if trimers:
        tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central = energy_list(trimer_dists, center_ip_id, trimers,
                tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central, typ="trimer")
    if tetramers:
        tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central = energy_list(tetramer_dists, center_ip_id, tetramers,
                tot_frag, mp2_frag, srs_frag, hf_frag, type_frag, central, typ="tetramer")

    data = {
        'dave': dists_list,
        'ids': ids_list,
        'tot': tot_frag,
        'hf': hf_frag,
        'mp2': mp2_frag,
        'srs': srs_frag,
        'type': type_frag,
        'central_frag': central,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'r4': r4,
        'r5': r5,
        'r6': r6,
        'rmax': rmax,
        'hf_bsse': bsse,
    }

    return data


def energies_corr_from_log_when_calculated(filename):
    """Correlation energies from log file where they are calculated."""
    fragments = {}
    # EXTRACT ENERGIES
    with open(filename, 'r') as r:
        for line in r:
            # QUEUE ID OF MOLECULES
            if 'Molecules' in line:
                line = line.replace(';', '').replace('+', ' ')
                line = line.split()
                q_id = str(int(line[-1]) - 1)
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
                _, q_id, _, _, _, _, os, ss, _ = line.replace(',', '').split()
                fragments[q_id]['os'] = float(os)
                fragments[q_id]['ss'] = float(ss)

    return fragments


def energies_dumped_hf(filename, fragments):
    """Read in HF energies from:
            h5dump -m "%.15f" -g "frag_energies" *.h5 > hf.h5dump"""

    monomers, dimers = {}, {}
    with open(filename, 'r') as r:
        for line in r:
            if "):" in line:
                cid, _, hf = line.replace(',', '').replace('(', '').replace(')', ' ').split()
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
        mon, dim, _, _ = energies_from_mbe_log(log)

        if typ == "add" or id1 != str(center_ip_id):
            first = mon['0']
            second = mon['1']
        else:
            first = mon['1']
            second = mon['0']

        if not monomers.get(id1) or np.isnan(monomers[id1].get('os')):
            monomers[id1] = first
        if not monomers.get(id2) or np.isnan(monomers[id2].get('os')):
            monomers[id2] = second

        try:
            dimers[key] = dim['0-1']
        except Exception:
            print(log)
            print(Exception)
        dimers[key]['type'] = typ  # add or cntr
    return monomers, dimers


def check_all_completed(logfiles):
    """Check all logfiles were successful."""

    for log in logfiles:
        success = False
        lines = endOfFile('', log, 1)
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


def dimer_contributions(dimers, monomers, np_for_zero=False):
    """Remove monomer energies from dimer energies."""

    # 0.0 for energies not calculated will mess up contributions
    for key, dict_ in dimers.items():
        id1, id2 = key.split('-')
        if np_for_zero:
            if dict_['hf'] == 0.0:
                dict_['hf'] = np.nan
            if dict_['os'] == 0.0:
                dict_['os'] = np.nan
            if dict_['ss'] == 0.0:
                dict_['ss'] = np.nan
        dict_['hf'] = dict_['hf'] - monomers[id1]['hf'] - monomers[id2]['hf']
        dict_['os'] = dict_['os'] - monomers[id1]['os'] - monomers[id2]['os']
        dict_['ss'] = dict_['ss'] - monomers[id1]['ss'] - monomers[id2]['ss']
    return dimers


def trimer_contributions(trimers, dimers, monomers, np_for_zero=False):
    """Remove dimer and monomer energies from trimer energies."""

    for key, dict_ in trimers.items():

        # 0.0 for energies not calculated will mess up contributions
        if np_for_zero:
            if dict_['hf'] == 0.0:
                dict_['hf'] = np.nan
            if dict_['os'] == 0.0:
                dict_['os'] = np.nan
            if dict_['ss'] == 0.0:
                dict_['ss'] = np.nan

        id1, id2, id3 = key.split('-')
        dict_['hf'] = dict_['hf'] - \
                      dimers[keyName(id1, id2)]['hf'] - dimers[keyName(id1, id3)]['hf'] - dimers[keyName(id2, id3)][
                          'hf'] - \
                      monomers[id1]['hf'] - monomers[id2]['hf'] - monomers[id3]['hf']

        dict_['os'] = dict_['os'] - \
                      dimers[keyName(id1, id2)]['os'] - dimers[keyName(id1, id3)]['os'] - dimers[keyName(id2, id3)][
                          'os'] - \
                      monomers[id1]['os'] - monomers[id2]['os'] - monomers[id3]['os']

        dict_['ss'] = dict_['ss'] - \
                      dimers[keyName(id1, id2)]['ss'] - dimers[keyName(id1, id3)]['ss'] - dimers[keyName(id2, id3)][
                          'ss'] - \
                      monomers[id1]['ss'] - monomers[id2]['ss'] - monomers[id3]['ss']

    return trimers


def tetramer_contributions(tetramers, trimers, dimers, monomers, np_for_zero=False):
    """Remove trimer, dimer and monomer energies from tetramer energies."""
    print("-------tetramer_contributions--------")
    for key, dict_ in tetramers.items():

        # 0.0 for energies not calculated will mess up contributions
        if np_for_zero:
            if dict_['hf'] == 0.0:
                dict_['hf'] = np.nan
            if dict_['os'] == 0.0:
                dict_['os'] = np.nan
            if dict_['ss'] == 0.0:
                dict_['ss'] = np.nan

        id1, id2, id3, id4 = key.split('-')
        # hf_init = dict_['hf']
        dict_['hf'] = dict_['hf'] - \
                      trimers[keyName(id1, id2, id3)]['hf'] - trimers[keyName(id1, id2, id4)]['hf'] - \
                      trimers[keyName(id1, id3, id4)]['hf'] - trimers[keyName(id2, id3, id4)]['hf'] - \
                      dimers[keyName(id1, id2)]['hf'] - dimers[keyName(id1, id3)]['hf'] - dimers[keyName(id1, id4)][
                          'hf'] - \
                      dimers[keyName(id2, id3)]['hf'] - dimers[keyName(id2, id4)]['hf'] - dimers[keyName(id3, id4)][
                          'hf'] - \
                      monomers[id1]['hf'] - monomers[id2]['hf'] - monomers[id3]['hf'] - monomers[id4]['hf']
        dict_['os'] = dict_['os'] - \
                      trimers[keyName(id1, id2, id3)]['os'] - trimers[keyName(id1, id2, id4)]['os'] - \
                      trimers[keyName(id1, id3, id4)]['os'] - trimers[keyName(id2, id3, id4)]['os'] - \
                      dimers[keyName(id1, id2)]['os'] - dimers[keyName(id1, id3)]['os'] - dimers[keyName(id1, id4)][
                          'os'] - \
                      dimers[keyName(id2, id3)]['os'] - dimers[keyName(id2, id4)]['os'] - dimers[keyName(id3, id4)][
                          'os'] - \
                      monomers[id1]['os'] - monomers[id2]['os'] - monomers[id3]['os'] - monomers[id4]['os']
        dict_['ss'] = dict_['ss'] - \
                      trimers[keyName(id1, id2, id3)]['ss'] - trimers[keyName(id1, id2, id4)]['ss'] - \
                      trimers[keyName(id1, id3, id4)]['ss'] - trimers[keyName(id2, id3, id4)]['ss'] - \
                      dimers[keyName(id1, id2)]['ss'] - dimers[keyName(id1, id3)]['ss'] - dimers[keyName(id1, id4)][
                          'ss'] - \
                      dimers[keyName(id2, id3)]['ss'] - dimers[keyName(id2, id4)]['ss'] - dimers[keyName(id3, id4)][
                          'ss'] - \
                      monomers[id1]['ss'] - monomers[id2]['ss'] - monomers[id3]['ss'] - monomers[id4]['ss']
        # print(hf_init, dict_['hf'])

    return tetramers


def remove_additional_mers(dict_):
    """Remove mers of type add which are not with central frag."""

    for key in list(dict_.keys()):
        if dict_[key]["type"] == "add":
            del dict_[key]
    return dict_


def getCuttoffsFromJson(json_data):
    """Get HF and MP2 cutoffs from json file."""

    cutoff_dims = json_data["keywords"]["frag"].get("dimer_cutoff", 10000) / angstrom2bohr
    cutoff_trims = json_data["keywords"]["frag"].get("trimer_cutoff", 10000) / angstrom2bohr
    cutoff_tets = json_data["keywords"]["frag"].get("tetramer_cutoff", 10000) / angstrom2bohr

    return cutoff_dims, cutoff_trims, cutoff_tets


def combineEnergiesAndKeys(frags_dict, hf_list, os_list, ss_list):
    """Combine list of energies with dictionary of frag ids and keys (mine)."""

    # dicts for monomers, dimers, trimers
    dicts = {1: {}, 2: {}, 3: {}, 4: {}}

    for frag_id, key in frags_dict.items():
        dicts[len(key.split('-'))][key] = {
            'hf': hf_list[frag_id],
            'os': os_list[frag_id],
            'ss': ss_list[frag_id],
        }
    return dicts[1], dicts[2], dicts[3], dicts[4]


def readHdf5Energies(filename):
    """Open and get parts of HDF5 exess restart file."""

    with h5py.File(filename, "r") as f:
        hf = list(f["frag_energies"]["frag_energies"])
        os = list(f["frag_os_energies"]["frag_os_energies"])
        ss = list(f["frag_ss_energies"]["frag_ss_energies"])
    return hf, os, ss


def readFragIdsFromLog(filename):
    """Read monomers that make up each frag, from:
      Molecules 1+7463+7460+7385; 0 <- 20
    """

    d = {}
    with open(filename, 'r') as r:
        for line in r:
            if line.strip():
                if len(line.split()) != 5:
                    print("Error w line:", line)
                _, mons, _, _, frag_id = line.split()
                mons = mons.replace(';', '').split('+')
                frag_id = int(frag_id) - 1  # starts from 1 in log file
                keyname = keyName(*mons)

                if not d.get(frag_id):
                    d[frag_id] = keyname
                else:
                    # some are in logs twice check if already assigned is expected value
                    if d[frag_id] != keyname:
                        print(f"frag id was different than expected: is {d[frag_id]}, expected {keyname}")
    return d


def checkCorrelationNan(frags_dict, fragList, atmList, jsonfile):
    """Check cor energies are not error. Cor energies that were not calculated are 0.0 and nan if error."""

    lines = ""
    for key, dict_ in frags_dict.items():
        if np.isnan(dict_['os']):
            lines += f"{key}\n"
            geometryFromListIds(key.split('-'), fragList, atmList, jsonfile, newDir="nan-cor")

    if lines:
        with open('correlation-nan.txt', 'w') as w:
            w.write(lines)


def outlierEnergies(dimers, trimers, tetramers, fragList, atmList, json_, write_outliers=True):
    """Check for very large hf and cor energies."""

    def getZScore(energies, keys, typ_frag, typ_e, threshold=50):
        """Z score for each value returning keys of outliers."""

        if not energies:
            return []

        outliers  = []

        # for z score
        median = np.percentile(energies, 50)
        std = np.std(energies)

        # find z score for each value
        for i in range(len(energies)):
            z_score = (energies[i] - median)/std
            diff = abs(energies[i] - median)
            if np.abs(z_score) > threshold or diff > 1000:
                # print(keys[i])
                outliers.append([keys[i], z_score, energies[i] - median, median, typ_frag, typ_e])
        return outliers

    def getOutliers(mers, typ, threshold=50):
        """Performs Z score for each value for HF and OS. Returns outliers."""

        # dont include frags that weren't calculated - will mess with median
        keys_hf = []
        keys_os = []
        hf = []
        os_ = []
        # check_energies = getAllKeys(0,10256,12410,12447) + getAllKeys(0,13498,13499,14617)
        for key, dict_ in mers.items():
            # if key in check_energies:
            #     print(key, dict_["hf"])
            if dict_["hf"] != 0:
                keys_hf.append(key)
                hf.append(dict_["hf"]*2625.5)
                if dict_["os"] != 0:
                    keys_os.append(key)
                    os_.append(dict_["os"]*2625.5)

        # for z score
        hf_outliers = getZScore(hf, keys_hf, typ, "hf", threshold)
        os_outliers = getZScore(os_, keys_os, typ, "os", threshold)

        # remove os outliers if already in hf
        os_outliers = [x for x in os_outliers if x[0] not in [y[0] for y in hf_outliers]]

        return hf_outliers+os_outliers

    dimer_outliers = getOutliers(dimers, "dimers")
    trimer_outliers = getOutliers(trimers, "trimers")
    tetramer_outliers = getOutliers(tetramers, "tetramers")
    outliers = dimer_outliers+trimer_outliers+tetramer_outliers

    if outliers:
        with open('outlier-energies.txt', 'w') as w:
            w.write('typ_frag key typ_e ediff median z_score\n')
            for key, z_score, ediff, median, typ_frag, typ_e in outliers:
                w.write(f'{typ_frag} {key} {typ_e} {round(ediff,1)} {round(median,1)} {round(z_score,1)}\n')
                if write_outliers:
                    geometryFromListIds(key.split('-'), fragList, atmList, json_, newDir="outlier")


def addInRerunFrags(monomers, dimers, trimers, tetramers, contributions=False):
    """Find rerun frags and rewrite energies. If contributions=True then convert added energies to contributions."""

    logs = glob.glob('./nan-cor/*.log') + glob.glob('./lower-conv/*.log') + glob.glob('./outlier-done/*.log')

    for log in logs:
        ids = log.split('/')[-1].replace("sphere-", "").replace(".log", "").split('-')
        length = len(ids)
        key = keyName(*ids)

        # if non mbe
        hf, os_, ss_ = energies_from_log(log)
        dict_ = {'hf': hf, 'os': os_, 'ss': ss_}

        # if mbe
        mons, dims, tris, tets = energies_from_mbe_log(log)

        # monomer
        if length == 1:
            monomers[key] = dict_
        # dimer
        if length == 2:
            # not mbe and want contributions
            if contributions and not mons:
                new_dimers = dimer_contributions({key: dict_}, monomers, np_for_zero=True)
                dict_ = new_dimers[key]
            # mbe
            elif mons and contributions:
                dict_ = dims['0-1']
            dimers[key] = dict_
        # trimer
        if length == 3:
            if contributions and not mons:
                new_trimers = trimer_contributions({key: dict_}, dimers, monomers, np_for_zero=True)
                dict_ = new_trimers[key]
            elif mons and contributions:
                dict_ = tris['0-1-2']
            trimers[key] = dict_
        # tetramer
        if length == 4:
            if contributions and not mons:
                new_tetramers = tetramer_contributions({key: dict_}, trimers, dimers, monomers, np_for_zero=True)
                dict_ = new_tetramers[key]
            elif mons and contributions:
                dict_ = tets['0-1-2-3']
            tetramers[key] = dict_

    return monomers, dimers, trimers, tetramers


def getBSSEFrag(key, lattice, center_ip_id):
    """Read in separate PSI4 CP calcs and return the BSSE for a dimer or trimer."""

    mons = key.split('-')

    # dimer
    if len(mons) == 2:
        if lattice:
            mons.remove(str(center_ip_id))
            a_ = mons[0]
            b_ = center_ip_id
        else:
            a_ = mons[0]
            b_ = mons[1]

        a = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-cp.log")
        b = scf_energy_from_psi4(f"../bsse/sphere-NML-{b_}-cp.log")
        a_ab = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-GH-{b_}-cp.log")
        b_ab = scf_energy_from_psi4(f"../bsse/sphere-NML-{b_}-GH-{a_}-cp.log")
        return a + b - a_ab - b_ab

    # trimer
    elif len(mons) == 3:
        if lattice:
            mons.remove(str(center_ip_id))
            a_ = mons[0]
            b_ = mons[1]
            c_ = center_ip_id

        else:
            a_ = mons[0]
            b_ = mons[1]
            c_ = mons[2]

        a = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-cp.log")
        b = scf_energy_from_psi4(f"../bsse/sphere-NML-{b_}-cp.log")
        c = scf_energy_from_psi4(f"../bsse/sphere-NML-{c_}-cp.log")
        a_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-GH-{b_}-{c_}-cp.log")
        b_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{b_}-GH-{a_}-{c_}-cp.log")
        c_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{c_}-GH-{a_}-{b_}-cp.log")
        ab = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-{b_}-cp.log")
        ac = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-{c_}-cp.log")
        if lattice:
            bc = scf_energy_from_psi4(f"../bsse/sphere-NML-{c_}-{b_}-cp.log")
        else:
            bc = scf_energy_from_psi4(f"../bsse/sphere-NML-{b_}-{c_}-cp.log")
        ab_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-{b_}-GH-{c_}-cp.log")
        if lattice:
            ac_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{c_}-{a_}-GH-{b_}-cp.log")
        else:
            ac_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{a_}-{c_}-GH-{b_}-cp.log")
        bc_abc = scf_energy_from_psi4(f"../bsse/sphere-NML-{b_}-{c_}-GH-{a_}-cp.log")
        return ab + bc + ac - ab_abc - ac_abc - bc_abc - a - b - c + a_abc + b_abc + c_abc


def writeCentralMBE(center_frag_id, fragList, fragList_init, atmList, method, File):

    # get initial frag ids which contain atoms of central ip
    frag_ids_in_central = []
    for frg in fragList:
        if center_frag_id == frg["grp"]:
            for atm in frg['ids']:
                for val, frag_init in enumerate(fragList_init):
                    if atm in frag_init['ids']:
                        frag_ids_in_central.append(val)

    frag_ids_in_central = list(set(frag_ids_in_central))

    # mbe of central
    json_lines, lines = make_exess_input_from_frag_ids(frag_ids_in_central, fragList_init, atmList,
                                                       ref_mon=0, number_checkpoints=0,
                                                       mbe=True, method=method, level=2)

    # WRITE FILES
    write_file(File.replace('.xyz', '_central.json'), json_lines)


def psi4Template(name, chrg, mult, xyz_lines, mem=64, method="scf", bset="cc-pVDZ", ref="rhf"):
    """PSI4 template."""

    lines = f"""memory {mem} Gb
molecule complex {{
 {chrg} {mult}
{xyz_lines}}}
set globals {{
    basis {bset}
    scf_type df
    freeze_core True
    guess sad
    reference {ref}
    s_orthogonalization canonical
    basis_guess 6-31G
}}
set print 2
energy('{method}')
"""
    with open(name, 'w') as w:
        w.write(lines)


def psi4GadiJobHugememTemplate(inputname):

    lines = f"""#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l ncpus=48
#PBS -l mem=1470GB
#PBS -l jobfs=1400GB
#PBS -q hugemem
#PBS -P k96
#PBS -l storage=gdata/k96+scratch/k96
#PBS -l wd
#PBS -m a

module load psi4
psi4 -n $PBS_NCPUS {inputname}.inp {inputname}.log"""

    with open(f"{inputname}.job", 'w') as w:
        w.write(lines)


def psi4GadiJobNormalTemplate(inputname):

    lines = f"""#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l jobfs=350GB
#PBS -q normal
#PBS -P k96
#PBS -l storage=gdata/k96+scratch/k96
#PBS -l wd
#PBS -m a

module load psi4
psi4 -n $PBS_NCPUS {inputname}.inp {inputname}.log"""

    with open(f"{inputname}.job", 'w') as w:
        w.write(lines)


def scf_energy_from_psi4(filename):
    """SCF energy from psi4 log file."""

    e = None
    dir, File = os.path.split(filename)
    dir = dir or "."
    lines = endOfFile(dir + '/', File, 0.8)

    for line in lines:
        # use last occurence
        if "    Total Energy =" in line:
            e = float(line.split()[3])

    if not e:
        print(f"Could not get energy from {filename}")
        return np.nan
    return e


# def bsseCentralMon(jsonfile):
#     """Make central monomer BSSE calcs from json file."""
#
#     # READ JSON
#     json_data = read_json(jsonfile)
#
#     # CONVERT JSON TO FRAG DATA
#     fragList, atmList, totChrg, totMult, mbe, lattice, level, center_ip_id = json_to_frags(json_data)
#
#     if center_ip_id is None:
#         sys.exit("Expected ref monomer in json. exiting . . .")
#
#     atoms = []
#     ghost_atoms = []
#     for frag in fragList:
#         if frag["grp"] == center_ip_id:
#             chrg = frag["chrg"]
#             mult = frag["mult"]
#             for atm in frag["ids"]:
#                 atoms.append(atmList[atm])
#         else:
#             for atm in frag["ids"]:
#                 ghost_atoms.append(atmList[atm])
#
#     name = jsonfile.replace(".json", "-central-mon-cp")
#     psi4CalcFromAtoms(f"{name}.inp", chrg, mult, atoms, ghost_atoms, mem=1468)
#     psi4GadiJobHugememTemplate(name)
#
#     name = jsonfile.replace(".json", "-central-mon")
#     psi4CalcFromAtoms(f"{name}.inp", chrg, mult, atoms, mem=46)
#     psi4GadiJobNormalTemplate(name)


def bsseFromJson(jsonfile):
    """Make BSSE calcs from json lattice energy file."""

    # READ JSON
    json_data = read_json(jsonfile)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe, lattice, level, center_ip_id = json_to_frags(json_data)

    fragList = add_centroids(fragList, atmList)

    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    def psi4CalcFromIds(fragList, atmList, frag_ids=(), ghost_frag_ids=()):

        if isinstance(frag_ids, int):
            frag_ids = [frag_ids]
        if isinstance(ghost_frag_ids, int):
            ghost_frag_ids = [ghost_frag_ids]

        chrg = 0
        spin = 0
        atoms = []
        ghost_atoms = []
        for val in frag_ids:
            chrg += fragList[val]["chrg"]
            spin += fragList[val]["mult"] - 1
            for atm in fragList[val]["ids"]:
                atoms.append(atmList[atm])
        for val in ghost_frag_ids:
            for atm in fragList[val]["ids"]:
                ghost_atoms.append(atmList[atm])

        mult = spin + 1
        if ghost_frag_ids:
            name = jsonfile.replace(".json", f"-NML-{'-'.join(str(x) for x in frag_ids)}-GH-{'-'.join(str(x) for x in ghost_frag_ids)}-cp")
        else:
            name = jsonfile.replace(".json", f"-NML-{'-'.join(str(x) for x in frag_ids)}-cp")
        psi4CalcFromAtoms(f"{name}.inp", chrg, mult, atoms, ghost_atoms, mem=46)
        psi4GadiJobNormalTemplate(name)

    # dimer CP
    if lattice:
        for i in range(len(fragList)):
            if not i == center_ip_id:
                psi4CalcFromIds(fragList, atmList, i)
                psi4CalcFromIds(fragList, atmList, center_ip_id)
                psi4CalcFromIds(fragList, atmList, i, center_ip_id)
                psi4CalcFromIds(fragList, atmList, center_ip_id, i)
    else:
        for i in range(len(fragList)-1):
            for j in range(i+1, len(fragList)):
                psi4CalcFromIds(fragList, atmList, i)
                psi4CalcFromIds(fragList, atmList, j)
                psi4CalcFromIds(fragList, atmList, i, j)
                psi4CalcFromIds(fragList, atmList, j, i)

    # trimer CP w/ central mon and within 6Å
    if lattice:
        for i in range(len(fragList) - 1):
            if fragList[i]["dist"] < bsse_cutoff and not i == center_ip_id:
                for j in range(i + 1, len(fragList)):
                    if fragList[j]["dist"] < bsse_cutoff and not j == center_ip_id:
                        psi4CalcFromIds(fragList, atmList, [i, j])
                        psi4CalcFromIds(fragList, atmList, [center_ip_id, j])
                        psi4CalcFromIds(fragList, atmList, [i, center_ip_id])
                        psi4CalcFromIds(fragList, atmList, center_ip_id, [i, j])
                        psi4CalcFromIds(fragList, atmList, i, [j, center_ip_id])
                        psi4CalcFromIds(fragList, atmList, j, [i, center_ip_id])
                        psi4CalcFromIds(fragList, atmList, [center_ip_id, i], [j])
                        psi4CalcFromIds(fragList, atmList, [center_ip_id, j], [i])
                        psi4CalcFromIds(fragList, atmList, [i, j], [center_ip_id])

    # trimer CP full fmo within 6Å
    else:
        for i in range(len(fragList) - 2):
            if fragList[i]["dist"] < bsse_cutoff:
                for j in range(i + 1, len(fragList) - 1):
                    if fragList[j]["dist"] < bsse_cutoff:
                        for k in range(j + 1, len(fragList)):
                            if fragList[k]["dist"] < bsse_cutoff:
                                psi4CalcFromIds(fragList, atmList, [i, j])
                                psi4CalcFromIds(fragList, atmList, [i, k])
                                psi4CalcFromIds(fragList, atmList, [j, k])
                                psi4CalcFromIds(fragList, atmList, k, [i, j])
                                psi4CalcFromIds(fragList, atmList, i, [j, k])
                                psi4CalcFromIds(fragList, atmList, j, [i, k])
                                psi4CalcFromIds(fragList, atmList, [i, k], [j])
                                psi4CalcFromIds(fragList, atmList, [j, k], [i])
                                psi4CalcFromIds(fragList, atmList, [i, j], [k])


def psi4CalcFromAtoms(name, chrg, mult, atoms, ghost_atoms=(), mem=46):

    xyz_lines = ""
    for atm in atoms:
        xyz_lines += f" {atm['sym']} {atm['x']} {atm['y']} {atm['z']}\n"
    for atm in ghost_atoms:
        xyz_lines += f" @{atm['sym']} {atm['x']} {atm['y']} {atm['z']}\n"

    psi4Template(name, chrg, mult, xyz_lines, mem=mem)


### TOP LEVEL --------------------------------------------------------

def make_dimer_trimer_tetramer_calcs(jsonfile, method="RIMP2", num_json_dimers=50, num_json_trimers=30,
                                     num_json_tetramers=10, dimers=True, trimers=True, tetramers=True, debug=False,
                                     cutoff_dims=None, cutoff_trims=None, cutoff_tets=None):
    """Make dimer job files for each dimer with the central fragment."""

    p = Pprint(to_print=debug)

    p.print_("name", jsonfile)

    # READ JSON
    json_data = read_json(jsonfile)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe, lattice, level, central_ip_id = json_to_frags(json_data)

    # ADD CENTROID - USED FOR CENTRAL IP
    fragList = add_centroids(fragList, atmList)

    # GET MIDPOINT OF ALL COORDS
    mx, my, mz = coords_midpoint(atmList)
    p.print_("midpoint", (mx, my, mz))

    # GET CENTRAL IP
    if central_ip_id is None:
        center_ip_id = central_frag(fragList, mx, my, mz)
        p.print_("center_ip_id", center_ip_id)

    # ADD DIST FROM CENTRAL IP
    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    # WRITE XYZS, MIDPOINT AND CENTRAL IP TO XYZ
    if debug:
        write_central_ip(fragList, atmList, center_ip_id, mx, my, mz)

    cutoff_pents = 0
    if dimers:
        write_xxmers(fragList, atmList, center_ip_id,
                     num_json_per_job=num_json_dimers, method=method, typ="dimers", cutoff_central=cutoff_dims,
                     cutoff_all=cutoff_trims)
    if trimers:
        write_xxmers(fragList, atmList, center_ip_id,
                     num_json_per_job=num_json_trimers, method=method, typ="trimers", cutoff_central=cutoff_trims,
                     cutoff_all=cutoff_tets)
    if tetramers:
        write_xxmers(fragList, atmList, center_ip_id,
                     num_json_per_job=num_json_tetramers, method=method, typ="tetramers", cutoff_central=cutoff_tets,
                     cutoff_all=cutoff_pents)


def df_from_logs(
        jsonfile,
        logfile=None,
        hf_dump_file=None,
        debug=False,
        get_energies="end",
        cutoff_dims=10000,
        cutoff_trims=10000,
        cutoff_tets=10000,
):
    """Make pandas data frame from json and log files."""

    p = Pprint(to_print=debug)

    # READ JSON
    json_data = read_json(jsonfile)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe, lattice, level, center_ip_id = json_to_frags(json_data)
    p.print_("mbe", mbe)
    p.print_("latice", lattice)

    # CHECK IS MBE
    if not mbe:
        sys.exit('Expected MBE input. exiting ...')

    # ADD CENTROIDS - USED FOR CENTRAL IP
    fragList = add_centroids(fragList, atmList)

    # GET MIDPOINT OF XYZ
    mx, my, mz = coords_midpoint(atmList)
    p.print_("Midpoint", (mx, my, mz))

    # GET CENTRAL IP
    if center_ip_id is None:
        if os.path.exists("central_ip_id"):
            center_ip_id = int(open("central_ip_id", 'r').read().strip())
            print("Central ion pair from file", center_ip_id)
        else:
            center_ip_id = central_frag(fragList, mx, my, mz)
            print("Central ion pair in the middle of the cluster is", center_ip_id)

    # ADD DISTANCE FROM CENTRAL FRAG
    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    # WRITE XYZS, CENTROIDS, MIDPOINT AND CENTRAL IP TO XYZ
    if debug:
        write_central_ip(fragList, atmList, center_ip_id, mx, my, mz)

    # GET ENERGIES
    p.print_("Getting energies")
    trimers = {}
    tetramers = {}
    trimers_dists = []
    tetramer_dists = []

    if get_energies == "log":
        name = logfile.split('.')
        monomers, dimers, trimers, tetramers = energies_from_mbe_log(logfile)
        cutoff_dims, cutoff_trims, cutoff_tets = getCuttoffsFromJson(json_data)
        p.print_("name", name)

        # find any frags rerun and override energies
        p.print_("Adding rerun frags")
        monomers, dimers, trimers, tetramers = addInRerunFrags(monomers, dimers, trimers, tetramers, contributions=True)

        # find any correlation nan's which are an error
        checkCorrelationNan({**monomers, **dimers, **trimers, **tetramers}, fragList, atmList, jsonfile)

        # write info and files for large energies
        p.print_("Writing outliers")
        outlierEnergies(dimers, trimers, tetramers, fragList, atmList, jsonfile)

    elif get_energies == "restart":

        restart_file = glob.glob("*h5")[0]
        ids_from_log = "frag_ids.txt"

        # read energies
        hf_energies, os_energies, ss_energies = readHdf5Energies(restart_file)

        # read frag ids
        p.print_("Reading Hdf5")
        keys_of_frag_ids = readFragIdsFromLog(ids_from_log)

        # check length of frags and keys
        if len(keys_of_frag_ids.keys()) != len(hf_energies):
            print(f"Length of frags ({len(keys_of_frag_ids.keys())}) does not equal length of HF energies "
                  f"({len(hf_energies)})")

        # combine energies
        monomers, dimers, trimers, tetramers = combineEnergiesAndKeys(keys_of_frag_ids, hf_energies, os_energies,
                                                                      ss_energies)

        # find any frags rerun and override energies
        p.print_("Adding rerun frags")
        monomers, dimers, trimers, tetramers = addInRerunFrags(monomers, dimers, trimers, tetramers)

        # find any correlation nan's which are an error
        checkCorrelationNan({**monomers, **dimers, **trimers, **tetramers}, fragList, atmList, jsonfile)

        # write info and files for large energies
        p.print_("Writing outliers")
        outlierEnergies(dimers, trimers, tetramers, fragList, atmList, jsonfile)

        # contributions from total energies
        dimers = dimer_contributions(dimers, monomers, np_for_zero=True)
        trimers = trimer_contributions(trimers, dimers, monomers, np_for_zero=True)
        tetramers = tetramer_contributions(tetramers, trimers, dimers, monomers, np_for_zero=True)

        # cutoffs
        cutoff_dims, cutoff_trims, cutoff_tets = getCuttoffsFromJson(json_data)

    elif get_energies == "dump":
        name = logfile.split('.')
        fragments = energies_corr_from_log_when_calculated(logfile)  # mp2 from start of files
        monomers, dimers, trimers, tetramers = energies_dumped_hf(hf_dump_file, fragments)  # hf from h5dump
        p.print_("name", name)

    elif get_energies == "separate":
        monomers, dimers = energies_from_sep_mbe_dimers(glob.glob("dimers/*log"), center_ip_id)

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

    print("Dimer cutoff:", cutoff_dims)
    print("Trimer cutoff:", cutoff_trims)
    print("Tetramer cutoff:", cutoff_tets)

    p.print_("Getting distances")

    if lattice:

        p.print_(f"monomers['{center_ip_id}']", monomers[str(center_ip_id)])

        # DISTANCE FROM EACH FRAG TO CENTRAL IP
        dimers_dists = distances_to_central_frag(fragList, center_ip_id, cutoff_dims, mers="dimers")
        if level > 2:
            trimers_dists = distances_to_central_frag(fragList, center_ip_id, cutoff_trims, mers="trimers")
        if level > 3:
            tetramer_dists = distances_to_central_frag(fragList, center_ip_id, cutoff_tets, mers="tetramers")

        # only keep central frag
        monomers = {str(center_ip_id): monomers[str(center_ip_id)]}

    else:
        dimers_dists = distances_between_frags(fragList, cutoff_dims, mers="dimers")
        if level > 2:
            trimers_dists = distances_between_frags(fragList, cutoff_trims, mers="trimers")
        else:
            trimers_dists = []
        if level > 3:
            tetramer_dists = distances_between_frags(fragList, cutoff_tets, mers="tetramers")
        else:
            tetramer_dists = []

    # ENERGY PER DISTANCE - PANDAS
    p.print_("Compiling data and making dataframe")

    df_data = distance_energy_df(dimers_dists, center_ip_id, monomers, dimers, trimers, tetramers, trimers_dists,
                                 tetramer_dists, lattice=lattice)
    df = pd.DataFrame(df_data)
    p.print_("df_data.keys()", df_data.keys())
    p.print_("df_data", df_data, still_print=False)
    df.to_csv("df.csv", index=False)


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
        create_tar(".", val, input_list, True)
        write_job_from_list(".", val, input_list)


def make_smaller_shell_from_json(json_, cutoff):
    """Takes a json mbe file and keeps only frags within the cutoff."""

    # READ JSON
    json_data = read_json(json_)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe, lattice, level, center_ip_id = json_to_frags(json_data)

    # CHECK IS MBE
    if not mbe:
        sys.exit('Expected MBE input. exiting ...')

    # ADD CENTROIDS - USED FOR CENTRAL IP
    fragList = add_centroids(fragList, atmList)

    # GET MIDPOINT OF ALL COORDS
    mx, my, mz = coords_midpoint(atmList)

    # GET CENTRAL IP
    if center_ip_id is None:
        center_ip_id = central_frag(fragList, mx, my, mz)

    # ADD DIST FROM CENTRAL IP
    fragList = add_dist_from_central_ip(fragList, center_ip_id)

    # INDEXES WITHIN NEW CUTOFF
    indexes = frags_in_cutoff(fragList, cutoff, center_ip_id)
    indexes.insert(0, center_ip_id)
    print(f"Fragments within {cutoff}: {len(indexes)}")

    # NEW MBE
    json_lines, lines = make_exess_input_from_frag_ids(indexes, fragList, atmList, number_checkpoints=0, mbe=mbe,
                                                       ref_mon=0)

    # WRITE FILES
    write_file(json_.replace('.js', f'-{cutoff}.js'), json_lines)
    write_xyz(json_.replace('.json', f'-{cutoff}.xyz'), lines)
    write_file("indexes.txt", [str(i) for i in indexes])


def geometryFromFragIds(json_, id_list, newDir=None):
    """Takes a json mbe file and keeps only frags listed by user."""

    # READ JSON
    json_data = read_json(json_)

    # CONVERT JSON TO FRAG DATA
    fragList, atmList, totChrg, totMult, mbe, lattice, level, center_ip_id = json_to_frags(json_data)

    # CHECK IS MBE
    if not mbe:
        sys.exit('Expected MBE input. exiting ...')

    geometryFromListIds(id_list, fragList, atmList, json_, mbe, newDir)


def geometryFromListIds(id_list, fragList, atmList, jsonfile, mbe=False, newDir=None):
    """From a list of ids and fragList create new json."""

    # NEW MBE
    json_lines, lines = make_exess_input_from_frag_ids(id_list, fragList, atmList, number_checkpoints=0, mbe=mbe)

    # WRITE FILES
    ids = '-'.join([str(x) for x in id_list])

    if newDir:
        if not os.path.exists(newDir):
            os.mkdir(newDir)
        write_file(newDir + "/" + jsonfile.replace('.js', f'-{ids}.js'), json_lines)
        write_xyz(newDir + "/" + jsonfile.replace('.json', f'-{ids}.xyz'), lines)
    else:
        write_file(jsonfile.replace('.js', f'-{ids}.js'), json_lines)
        write_xyz(jsonfile.replace('.json', f'-{ids}.xyz'), lines)


def xyz_to_json(filename, mbe, method, pair_type=None):
    """Convert xyz file to exess json input."""

    from system import systemData

    dir, File = os.path.split(filename)
    fragList_init, atmList, totChrg, totMult = systemData(dir, File, True)

    # ADD CENTROIDS - USED FOR CENTRAL IP
    fragList_init = add_centroids(fragList_init, atmList)

    # Pair molecules by lowest total pairing distance
    if pair_type == "all":
            fragList = pair_ions_lowest_dist(fragList_init, atmList)
    else:
        fragList = fragList_init


    # GET MIDPOINT OF ALL COORDS
    mx, my, mz = coords_midpoint(atmList)

    # central frag
    center_frag_id = central_frag_with_charge(fragList, atmList, mx, my, mz, 0)

    # make dimer calc of central ion pair if paired
    if pair_type == "all":
        writeCentralMBE(center_frag_id, fragList, fragList_init, atmList, method, File)

    # NEW MBE
    json_lines, lines = make_exess_input_from_frag_ids(list(range(0, len(fragList))), fragList, atmList,
                            ref_mon=center_frag_id, number_checkpoints=0, mbe=mbe, method=method)

    # WRITE FILES
    write_file(File.replace('.xyz', '.json'), json_lines)


def run(value, filename):
    """Call function depending on user input."""

    if value == "0" or value == "":
        pass

    # dataframe from logs
    elif value == "1":

        if not filename:
            jsn = glob.glob("*json")[0]
            log = glob.glob("*.log")[0]
        else:
            log = filename
            jsn = log.replace('.json', '.log')

        if os.path.isdir("dimers"):
            log = None
            get_energies = "separate"

            # cutoffs
            user_ = input("Cutoffs Dimers Trimers Tetramers [None 35 20]: ")
            if user_ == "":
                user_ = "None 35 20"
            user_ = user_.replace("None", "10000")
            dim, tri, tet = [float(i) for i in user_.split()]

        elif os.path.exists("frag_ids.txt"):
            log = None
            get_energies = "restart"
            dim, tri, tet = "None", "None", "None"

        else:
            get_energies = "log"
            dim, tri, tet = "None", "None", "None"

        print(f"Getting energies: {get_energies}")
        print(f"File used: {jsn}")
        df_from_logs(
            jsonfile=jsn,
            logfile=log,
            hf_dump_file=None,
            get_energies=get_energies,
            debug=True,
            cutoff_dims=dim,
            cutoff_trims=tri,
            cutoff_tets=tet,
        )

    # json to xyz
    elif value == "2":

        if not filename:
            files = glob.glob("*json")
        else:
            files = [filename]

        print(f"Files used: {files}")
        for filename in files:
            # READ JSON
            json_data = read_json(filename)

            # CONVERT JSON TO FRAG DATA
            _, atmList, _, _, _, _, _, _ = json_to_frags(json_data)

            # WRITE XYZ
            xyzfile = filename.replace(".json", ".xyz")
            write_xyz(xyzfile, lines=[], atmList=atmList)

    # xyz to json
    elif value == "3":

        mbe = input("MBE [y]: ")
        meth = input("Method [RIMP2]: ")
        pair = input("Pair ions [y]: ")

        if mbe == "n" or mbe == "N":
            mbe = False
        else:
            mbe = True

        if meth == "":
            meth = "RIMP2"

        if pair == "n" or pair == "N":
            pair = "none"
        else:
            pair = "all"


        if not filename:
            files = glob.glob("*xyz")
            print(f"Files used: {files}")
            for filename in files:
                xyz_to_json(filename, mbe, meth, pair)
        else:
            print(f"File used: {filename}")
            xyz_to_json(filename, mbe, meth, pair)

    # make job from jsons
    elif value == "4":
        user_ = input("Number of xxmers per job: ")
        make_job_from_jsons(int(user_))

    # makes smaller shell from json
    elif value == "5":

        user_ = float(input("Cutoff distance of new shell: "))

        if not filename:
            filename = glob.glob("*json")[0]

        print(f"File used: {filename}")
        make_smaller_shell_from_json(filename, user_)

    # print json with only frags listed
    elif value == "6":

        user_ = input("Frags IDs separated with space: ")

        try:
            ids = [int(x) for x in user_.split()]
        except:
            sys.exit("Could not split input into integers.")

        if not filename:
            filename = glob.glob("*json")[0]

        print(f"File used: {filename}")
        geometryFromFragIds(filename, ids)

    # make sep dimer/trimer/tetramer calcs
    elif value == "7":

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

        print(f"File used: {jsn}")
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

    # bsse from json each mon w/ all other mons as ghost
    elif value == "8":
        if not filename:
            filename = glob.glob("*json")[0]
        print(f"File used: {filename}")
        bsseFromJson(filename)


# CALL SCRIPT -------------------------------------------

print("What would you like to do?")
print("    1. CSV from log files")
print("    2. Json to xyz")
print("    3. Xyz to json")
print("    4. Make job from json files")
print("    5. Make smaller shell from json")
print("    6. Get geometry from frag list")
print("    7. Make separate dimer/trimer/tetramer calculations")
print("    8. BSSE from json")
print("    0. Quit")

user = None
filename = None
while not user in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", '']:

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    user = input("Value: ")
    run(user, filename)
