import numpy as np

from hapi_comment import *

""""
This file is used to generate profiles of temperature, mole fraction for further generating spectra
"""


def beta_function(sub_column, variance):
    beta = np.zeros(sub_column)
    mid = (sub_column - 1) / 2
    wing = sub_column
    beta = [exp(-(i - mid) ** 2 * variance / wing ** 2) for i in range(sub_column)]
    return beta


def beta_function_2(sub_column, variance):
    beta = np.zeros(sub_column)
    mid1 = float((sub_column - 1)) / 4.
    mid2 = float((sub_column - 1)) * 3 / 4.
    wing = sub_column
    beta = [exp(-(i - mid1) ** 2 * variance / wing ** 2) + exp(-(i - mid2) ** 2 * variance / wing ** 2) for i in
            range(sub_column)]
    return beta


# set the mole number and temperature
def temp_mole_random(sub_column, beta, base_tem, limit_tem):
    rad1 = np.random.rand(sub_column)
    rad2 = np.random.rand(sub_column)
    rad3 = np.random.rand(sub_column)
    rad4 = np.random.rand(sub_column)
    rad5 = np.random.rand(sub_column)

    t0 = [base_tem + limit_tem * rad for rad in rad1]
    ts = zeros(sub_column)
    xco2 = zeros(sub_column)
    xh2o = zeros(sub_column)
    xco = zeros(sub_column)
    deltT = zeros(sub_column)
    for i in range(sub_column):
        deltT[i] = ((base_tem + limit_tem) - t0[i]) * rad2[i]
        ts[i] = deltT[i] * beta[i] + t0[i]
        xco2[i] = 0.1 * rad3[i] * beta[i] + 0.05
        xh2o[i] = 0.1 * rad4[i] * beta[i] + 0.05
        xco[i] = 0.1 * rad5[i] * beta[i] + 0.01

    return ts, xco2, xh2o, xco


def temp_mole_random_2(sub_column, beta, base_tem, limit_tem):
    rad1 = np.random.rand(sub_column)
    rad2 = np.random.rand(sub_column)
    rad3 = np.random.rand(sub_column)
    rad4 = np.random.rand(sub_column)
    rad5 = np.random.rand(sub_column)

    t0 = [base_tem + limit_tem * rad for rad in rad1]
    ts = zeros(sub_column)
    xco2 = zeros(sub_column)
    xh2o = zeros(sub_column)
    xco = zeros(sub_column)
    deltT = zeros(sub_column)
    for i in range(sub_column):
        deltT[i] = ((base_tem + limit_tem) - t0[i]) * rad2[i]
        ts[i] = limit_tem * beta[i] * rad1[i] + base_tem
        xco2[i] = 0.1 * rad3[i] * beta[i] + 0.05
        xh2o[i] = 0.1 * rad4[i] * beta[i] + 0.05
        xco[i] = 0.1 * rad5[i] * beta[i] + 0.01

    return ts, xco2, xh2o, xco


def temp_mole_random_3(sub_column, beta, base_tem, limit_tem):
    rad1 = 2 * np.random.rand(sub_column) - 1.
    rad2 = 2 * np.random.rand(sub_column) - 1
    rad3 = 2 * np.random.rand(sub_column) - 1
    rad4 = 2 * np.random.rand(sub_column) - 1
    t_rand = [rand * 300. for rand in rad1]
    t_ideal = [b * limit_tem + base_tem for b in beta]
    co2_rand = [rand * 0.015 for rand in rad2]
    co2_ideal = [b * 0.1 + 0.05 for b in beta]
    co_rand = [rand * 0.015 for rand in rad3]
    co_ideal = [b * 0.1 + 0.015 for b in beta]
    h2o_rand = [rand * 0.015 for rand in rad4]
    h2o_ideal = [b * 0.1 + 0.05 for b in beta]

    ts = zeros(sub_column)
    xco2 = zeros(sub_column)
    xh2o = zeros(sub_column)
    xco = zeros(sub_column)
    for i in range(sub_column):
        ts[i] = t_ideal[i] + t_rand[i]
        xco2[i] = co2_ideal[i] + co2_rand[i]
        xco[i] = co_ideal[i] + co_rand[i]
        xh2o[i] = h2o_ideal[i] + h2o_rand[i]
    return ts, xco2, xh2o, xco
