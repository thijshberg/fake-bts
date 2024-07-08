#!/bin/python
# %%
import json
import os
import re
import subprocess
import sys
from time import sleep
from typing import List, Optional, Tuple

from amarisoft_gain import set_amarisoft_gain as _set_amarisoft_gain

#Change this to your own setup
AMARI_IP = "192.168.1.114"
AMARI_PCI = 1
SRS_PCI = 2

HUAWEI = False
DEBUG = False

#Change this to your own setup
SRS_CMD = "srsenb enb.conf"


def debug(*args, **kwargs):
    if DEBUG:
        print("[D]\t", *args, **kwargs)


def check_print(*args, **kwargs):
    print("[C]\t", *args, **kwargs)


class SRS_runner():
    proc: subprocess.Popen
    outfile = None

    def __init__(self):
        ...

    def __enter__(self):
        debug("Opening srs")
        self.outfile = open("exp.log", "a")
        self.proc = subprocess.Popen(
            SRS_CMD.split(), stdout=self.outfile, stderr=self.outfile)
        sleep(5)

    def __exit__(self, *args, **kwargs):
        debug("Closing srs")
        self.outfile.close()
        self.proc.kill()


def set_amarisoft_gain(gain: int):
    debug("Setting amarisoft gain to", gain)
    _set_amarisoft_gain(AMARI_IP, gain, AMARI_PCI, verbose=False)


def toggle_airplane_mode():
    with open("adb.log", 'a') as logfile:
        debug("Navigating to airplane settings page")
        subprocess.run(
            "adb shell am start -a android.settings.AIRPLANE_MODE_SETTINGS".split(), stdout=logfile, stderr=logfile)
        if HUAWEI:
            sleep(1)
            debug("Tapping")
            subprocess.run("adb shell input tap 950 345".split())
        else:
            debug("Key event 19")
            subprocess.run("adb shell input keyevent 19".split())
            debug("Key event 23")
            subprocess.run("adb shell input keyevent 23".split())
        sleep(3)


r_pci = re.compile("mPci=\d+")


def get_serving_pci():
    debug("Getting serving pci")
    p = subprocess.run(
        "adb shell dumpsys telephony.registry".split(), capture_output=True)
    output = str(p.stdout, encoding='utf-8')
    last_pci_line = ""
    first_pci_line = ""
    for line in output.splitlines():
        if 'Pci' in line:
            last_pci_line = line
            if first_pci_line == "":
                first_pci_line = line
    if HUAWEI:
        return int(r_pci.findall(first_pci_line)[0].split("=")[1])
    else:
        return int(r_pci.findall(last_pci_line)[0].split("=")[1])


r_rsrp = re.compile("rsrp=-\d+")


def get_serving_strength() -> Optional[int]:
    debug("Getting serving pci")
    p = subprocess.run(
        "adb shell dumpsys telephony.registry".split(), capture_output=True)
    output = str(p.stdout, encoding='utf-8')
    rsrp = None
    for line in output.splitlines():
        if 'rsrp' in line:
            if (maybe_rsrp := r_rsrp.findall(line)):
                rsrp = int(maybe_rsrp[0].split("=")[1])
                break
    return rsrp


def calibrate(filename: str):
    res = []
    for gain in range(0, -100, -1):
        set_amarisoft_gain(gain)
        sleep(5)
        rsrp = get_serving_strength()
        res.append((gain, rsrp))
        print(f"Calibrated: gain of {gain} = RSRP of {rsrp}")
    with open(filename, 'w') as f:
        json.dump(res, f)


def iteration(gain) -> bool:
    toggle_airplane_mode()
    set_amarisoft_gain(0)
    toggle_airplane_mode()
    if get_serving_pci() != AMARI_PCI:
        debug("Serving pci is incorrect, retrying...")
        sleep(10)
        if (pci := get_serving_pci()) != AMARI_PCI:
            raise Exception(
                f"Not connected to amari at the start, but to {pci}!")
    with SRS_runner() as srs:
        set_amarisoft_gain(gain)
        sleep(15)
        pci = get_serving_pci()
        debug(f"PCI={pci}")
        if pci == AMARI_PCI:
            return False
        elif pci == SRS_PCI:
            return True
        else:
            raise Exception(f"Unknown pci {pci}")


def rsrp_from_gain(gain: int, calib: List[Tuple[int, int]]) -> Optional[int]:
    for (_gain, rsrp) in calib:
        if _gain == gain:
            debug(f"RSRP={rsrp} when gain={gain}")
            return int(rsrp)
    else:
        return None


def run_experiment(calibration_file: str, /, do_check: bool):
    with open(calibration_file, 'r') as f:
        calib = json.load(f)
    lower_bound, upper_bound = -100, 0
    if do_check:
        toggle_airplane_mode()
        set_amarisoft_gain(0)
        toggle_airplane_mode()
        sleep(3)
        target_rsrp = rsrp_from_gain(0, calib)
        assert target_rsrp is not None, "gain 0 could not be translated to RSRP"
        actual_rsrp = get_serving_strength()
        assert actual_rsrp is not None, "Could not measure RSRP at gain 0"
        assert abs(
            actual_rsrp - target_rsrp) <= 1, f"RSRP at gain 0 was {actual_rsrp} and should be close to {target_rsrp}"

        assert iteration(
            lower_bound), "There was no reselection at the lower bound"
        check_print("Checked lower bound")
        assert not iteration(
            upper_bound), "There was a reselection at the upper bound"
        check_print("Checked upper bound")
    while upper_bound - lower_bound > 1:
        gain = lower_bound + (upper_bound-lower_bound)//2
        debug(f"Iteration with gain={gain}")
        if iteration(gain):
            lower_bound = gain
        else:
            upper_bound = gain
        print(f"The border is between {lower_bound} and {upper_bound}")
    upper_rsrp = rsrp_from_gain(upper_bound, calib)
    assert upper_rsrp is not None
    lower_rsrp = rsrp_from_gain(lower_bound, calib)
    assert lower_rsrp is not None
    print(f"The result is between {lower_rsrp} and {upper_rsrp}")
    if do_check:
        check_print("Double-checking...")
        assert iteration(
            lower_bound), "There was no reselection at the lower bound upon double-checking"
        check_print("Double-checked final lower bound")
        assert not iteration(
            upper_bound), "There was a reselection at the upper bound upon double-checking"
        check_print("Double-checked final upper bound")


def main() -> int:
    args = sys.argv
    do_calibrate = False
    do_check = False
    do_gain = False
    filename = None
    for arg in sys.argv:
        if arg == "-g":
            do_gain = True
        elif arg == "-d":
            global DEBUG
            DEBUG = True
        elif arg == "-h":
            global HUAWEI
            HUAWEI = True
        elif arg == "-C":
            do_calibrate = True
        elif arg == "--check":
            do_check = True
        else:
            filename = arg
    assert filename is not None
    if do_gain:
        set_amarisoft_gain(int(filename))
        return 0
    try:
        if do_calibrate:
            calibrate(filename)
        else:
            run_experiment(filename, do_check=do_check)
        return 0
    except Exception as e:
        print(e)
        return 1


"""
Usage:

practical_experiments.py (-g GAIN | CALIBRATION [--check] [-C] [-h]) [-d]

Options:
-g GAIN:    sets gain of the amarisoft enb to GAIN and exits
else, the experiment is run and the gain at which reselection occurs is found
--check:    enable all checks
-C:         writes a new calibration file to CALIBRATION
-h:         use specific touch commands to toggle airplane mode on the huawei phone
-d:         enable debug logging

Requires hooking up:
 - A phone running ADB
 - An amarisoft running the core and an eNB which can be reached through the websocket API
 - A confinguration of the amarisoft that ensures reselection occurs
 - A local srsenb installation that connects to the amarisoft core

It is possible to do this with the amarisoft and srs replaced by other core/enb implementations, although that would require hooking the relevant functions up to those interfaces
"""

# %%
if __name__ == "__main__":
    exit(main())
