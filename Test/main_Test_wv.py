import sys
print(sys.path)
import shutil
import numpy as np
import os
import csv
import time

print('current directory = ', os.getcwd())

def Test():

    t_start = time.process_time()

    method = ['./SSF_tests', './SSW_tests', './WWP_tests', './WWPH_tests']
    ground = ['/No_Ground', '/Plane', '/Triangle', '/Relief_sup']
    ground_type = ['/PEC_TE', '/PEC_TM', '/Dielectric']
    atmosphere = ['/Homogeneous', '/Linear', '/Bilinear', '/Trilinear', '/Evaporation']


    dest_propa = '../propagation/inputs/configuration.csv'
    dest_source = '../source/inputs/configuration.csv'
    dest_field = '../source/outputs/E_field.csv'
    dest_relief = '../terrain/outputs/z_relief.csv'

    i = 0
    for meth in method:
        for groun in ground:
            if groun == '/No_Ground' and meth != './WWPH_tests':
                for atm in atmosphere:

                    config_propa = meth + groun + atm + '/propagation/configuration.csv'
                    config_source = meth + groun + atm + '/source/configuration.csv'
                    config_relief = meth + groun + atm + '/terrain/z_relief.csv'
                    config_field = meth + groun + atm + '/E_field/E_field.csv'
                    config_final = meth + groun + atm + '/E_final/wv_total.npy'
                    print(config_final)
                    shutil.copy(config_propa, dest_propa)
                    shutil.copyfile(config_source, dest_source)
                    shutil.copyfile(config_field, dest_field)
                    shutil.copyfile(config_relief, dest_relief)
                    E_field = np.load(config_final, allow_pickle=True)


                    f_propagation_config = open(dest_propa, newline='')
                    file_tmp = csv.reader(f_propagation_config)
                    for row in file_tmp:
                        if row[0] == 'N_x':
                            N_x = np.int64(row[1])
                        elif row[0] == 'wavelet level':
                            wvl = np.int64(row[1])
                    i += 1
                    # propagation computation
                    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
                        cmd = 'cd ../propagation && python3 ./main_propagation.py > propa.log'
                    else:
                        cmd = 'cd ../propagation && python ./main_propagation.py'
                    os.system(cmd)

                    E_field_comp = np.load('../propagation/outputs/wv_total.npy', allow_pickle=True)

                    E1 = [[]] * (wvl + 1)
                    E2 = [[]] * (wvl + 1)

                    for ii_x in np.arange(0, N_x):
                        for ii_lvl in np.arange(0, wvl + 1):
                            E1[ii_lvl] = E_field[ii_x][ii_lvl].todense()
                            E2[ii_lvl] = E_field_comp[ii_x][ii_lvl].todense()
                            if (E1[ii_lvl] != E2[ii_lvl]).all():
                                print('Test invalide!')
                                print("La différence entre les deux champs est :", E_diff_db)
                                rep = str(input("Voulez-vous continuer ?(o/n)"))
                                if rep == "o":
                                    break
                                else:
                                    raise ValueError('test invalide!')
                    print('Test valide!')
                    print(i, "/100")


            elif meth == './WWP_tests' and groun != '/No_Ground':
                break
            elif meth == './WWPH_tests' and groun != '/Plane':
                pass
            else:
                for type in ground_type:
                    if meth == './SSF_tests' and type == '/Dielectric':
                        break
                    elif meth == './WWPH_tests' and type == '/Dielectric':
                        break
                    else:
                        for atm in atmosphere:
                            config_propa = meth + groun + type + atm + '/propagation/configuration.csv'
                            config_source = meth + groun + type + atm + '/source/configuration.csv'
                            config_relief = meth + groun + type + atm + '/terrain/z_relief.csv'
                            config_field = meth + groun + type + atm + '/E_field/E_field.csv'
                            config_final = meth + groun + type + atm + '/E_final/wv_total.npy'
                            print(config_final)
                            shutil.copy(config_propa, dest_propa)
                            shutil.copyfile(config_source, dest_source)
                            shutil.copyfile(config_field, dest_field)
                            shutil.copyfile(config_relief, dest_relief)
                            E_field = np.load(config_final, allow_pickle=True)

                            f_propagation_config = open(dest_propa, newline='')
                            file_tmp = csv.reader(f_propagation_config)
                            for row in file_tmp:
                                if row[0] == 'N_x':
                                    N_x = np.int64(row[1])
                                elif row[0] == 'wavelet level':
                                    wvl = np.int64(row[1])

                            i += 1
                            # propagation computation
                            if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
                                cmd = 'cd ../propagation && python3 ./main_propagation.py > propa_log'
                            else:
                                cmd = 'cd ../propagation && python ./main_propagation.py'
                            os.system(cmd)


                            E_field_comp = np.load('../propagation/outputs/wv_total.npy', allow_pickle=True)

                            E1 = [[]] * (wvl + 1)
                            E2 = [[]] * (wvl + 1)

                            for ii_x in np.arange(0, N_x):
                                for ii_lvl in np.arange(0, wvl + 1):
                                    E1[ii_lvl] = E_field[ii_x][ii_lvl].todense()
                                    E2[ii_lvl] = E_field_comp[ii_x][ii_lvl].todense()
                                    if (E1[ii_lvl] != E2[ii_lvl]).all():
                                        print('Test invalide!')
                                        print("La différence entre les deux champs est :", E_diff_db)
                                        rep = str(input("Voulez-vous continuer ?(o/n)"))
                                        if rep == "o":
                                            break
                                        else:
                                            raise ValueError('test invalide!')
                            print('Test valide!')
                            print(i, "/100")


    t_end = time.process_time()
    print('Total Time (s)', np.round((t_end - t_start)))

if __name__ == '__main__':
    Test()

