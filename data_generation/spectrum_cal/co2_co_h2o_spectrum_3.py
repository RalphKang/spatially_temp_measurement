import matplotlib.pyplot as plt

from profile_element import *


def main():
    # set seed
    plt.rcParams["figure.figsize"] = (5.0, 5.0)

    # set the basic input parameters
    nu_min = 1800;
    nu_max = 2500  # cm-1
    p = 1.;
    base_temp = 300.;
    limit_temp = 2700.
    total_length = 10.0  # cm
    sub_column = 11  # the number of sub_columns
    length_column = total_length / sub_column
    reso = 4.;
    af_wing = 10.;
    stepsize = 0.1  # resolution of instrument cm-1
    data_start = 1809
    data_number = 500
    nasity_check = False
    # choose the databank
    db_begin("data_set_use")
    select("co", DestinationTableName="CO")
    select("h2o", DestinationTableName="H2O")
    select("co2", DestinationTableName="CO2")

    beta = beta_function_2(sub_column=sub_column, variance=16)

    for data_id in range(data_start, data_start + data_number):
        t, list_co2, list_h2o, list_co = temp_mole_random_3(sub_column=sub_column, beta=beta, base_tem=base_temp,
                                                            limit_tem=limit_temp)
        environ_list = np.vstack((t, list_co2, list_h2o, list_co)).T
        Iv_before = 0.
        for i, (temp, mole_frac_co2, mole_frac_h2o, mole_frac_co) in enumerate(environ_list):
            intensity = 0
            intensity_filt = 0
            coef = 0
            coef_co = 0
            coef_co2 = 0
            coef_h2o = 0  # clear the coef

            nu, coef_co = absorptionCoefficient_Voigt(SourceTables="CO", WavenumberRange=(nu_min, nu_max),
                                                      HITRAN_units=False, Environment={'T': temp, 'p': 1},
                                                      mole_fraction=mole_frac_co, OmegaStep=stepsize)
            nu, coef_h2o = absorptionCoefficient_Voigt(SourceTables="H2O", WavenumberRange=(nu_min, nu_max),
                                                       HITRAN_units=False, Environment={'T': temp, 'p': 1},
                                                       mole_fraction=mole_frac_h2o, OmegaStep=stepsize)
            nu, coef_co2 = absorptionCoefficient_Voigt(SourceTables="CO2", WavenumberRange=(nu_min, nu_max),
                                                       HITRAN_units=False, Environment={'T': temp, 'p': 1},
                                                       mole_fraction=mole_frac_co2, OmegaStep=stepsize)
            coef = coef_co + coef_h2o + coef_co2
            nu, intensity, radi_black = radianceSpectrum(nu, coef, Environment={"l": length_column, "T": temp},
                                                         Iv0=Iv_before)
            Iv_before = intensity
            i += 1

        nu, intensity_filt, _, _, _ = convolveSpectrum(nu, intensity, Resolution=reso, SlitFunction=SLIT_TRIANGULAR,
                                                       AF_wing=af_wing)

        plt.plot(nu, intensity_filt, linewidth=0.5, color="red")
        plt.axis('off')
        out_put = './output/figure_type/' + str(data_id) + '.png'
        plt.savefig(out_put, bbox_inches='tight', pad_inches=0.0, dpi=100)
        plt.close()

        out_file_envi = './output/environment/' + str(data_id) + '.txt'
        out_file_spect = './output/spect/' + str(data_id) + '.txt'
        spc = np.vstack((nu, intensity_filt)).T
        np.savetxt(out_file_spect, spc)
        np.savetxt(out_file_envi, environ_list)
        print("finished")
        if nasity_check:
            if data_id >= 1:
                break


if __name__ == '__main__':
    main()
