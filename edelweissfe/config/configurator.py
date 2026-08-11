#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------
"""
Created on Mon Apr 10 20:26:22 2017

@author: Matthias Neuner
"""
from copy import deepcopy

from edelweissfe.config.phenomena import (
    fieldCorrectionTolerance,
    fluxResidualTolerance,
    fluxResidualToleranceAlternative,
)
from edelweissfe.utils.misc import convertLinesToStringDictionary


def loadConfiguration(jobInfo):
    """load the default (field) tolerance settings from phenomena.py

    The dictionaries are COPIED, not referenced. updateConfiguration writes into
    jobInfo[configurationType][key]; handing out the module-level dictionaries of
    phenomena.py would make every `*updateConfiguration` in an input file mutate the
    defaults for the rest of the Python process. A job that deliberately loosens a
    tolerance would then silently loosen it for every job run after it in the same
    process - e.g. a verification run following a production model, judged by the
    production model's tolerances without anything saying so.
    """
    jobInfo["fieldCorrectionTolerance"] = deepcopy(fieldCorrectionTolerance)
    jobInfo["fluxResidualTolerance"] = deepcopy(fluxResidualTolerance)
    jobInfo["fluxResidualToleranceAlternative"] = deepcopy(fluxResidualToleranceAlternative)
    return jobInfo


def updateConfiguration(newConfiguration, jobInfo, journal):
    """update configurations of the job, such as tolerances"""
    configurationType = newConfiguration["configuration"]
    if configurationType not in jobInfo:
        raise KeyError("configuration type {:} invalid".format(configurationType))

    settings = convertLinesToStringDictionary(newConfiguration.get("data", newConfiguration.get("datalines")))
    for key, val in settings.items():
        if key not in jobInfo[configurationType]:
            raise KeyError("configuration type {:}/{:} invalid".format(configurationType, key))

        configurationValueType = type(jobInfo[configurationType][key])
        val = configurationValueType(val)

        journal.message(
            "Setting {:}/{:} to {:}".format(configurationType, key, val),
            "configuration",
            level=1,
        )
        jobInfo[configurationType][key] = val
