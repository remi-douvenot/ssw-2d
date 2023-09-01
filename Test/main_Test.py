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
                    config_final = meth + groun + atm + '/E_final/E_field.npy'
                    print(config_final)
                    shutil.copy(config_propa, dest_propa)
                    shutil.copyfile(config_source, dest_source)
                    shutil.copyfile(config_field, dest_field)
                    shutil.copyfile(config_relief, dest_relief)
                    E_field = np.load(config_final, allow_pickle=True)

                    #f_propagation_config = open(dest_propa, newline='')
                    #file_tmp = csv.reader(f_propagation_config)
                    #for row in file_tmp:
                     #   if row[0] == 'N_x':
                      #      N_x = np.int64(row[1])
                       # elif row[0] == 'N_z':
                        #    wvl = np.int64(row[1])

                    i += 1
                    print(i, "/100")

                    # propagation computation
                    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
                        cmd = 'cd ../propagation && python3 ./main_propagation.py'
                    else:
                        cmd = 'cd ../propagation && python ./main_propagation.py'
                    #sys.stdout = open("test.txt", 'a')
                    os.system(cmd)
                    #sys.stdout = sys.__stdout__

                    E_field_comp = np.load('../propagation/outputs/E_field.npy', allow_pickle=True)

                    E_field_comp_db = 20 * np.log10(np.abs(E_field_comp))
                    E_field_db = 20 * np.log10(np.abs(E_field))
                    #print(E_field_comp_db)
                    #print(E_field_db)

                    f_propa_config = open(config_propa, newline='')
                    file_tmp = csv.reader(f_propa_config)
                    for row in file_tmp:
                        if row[0] == 'apodisation size':
                            apod = np.float64(row[1])
                        elif row[0] == 'ground':
                            groundType = np.str_(row[1])
                        elif row[0] == 'N_z':
                            N_z = np.int32(row[1])



                    E_field_comp_db_apo = E_field_comp_db[:int(len(E_field_comp_db) - (apod * N_z))]
                    E_field_db_apo = E_field_db[:int(len(E_field_db) - (apod * N_z))]


                    if groundType == "No Ground":
                        E_field_comp_db_apo = E_field_comp_db_apo[int(apod * N_z):]
                        E_field_db_apo = E_field_db_apo[int(apod * N_z):]

                    #print(E_field_comp_db_apo,len(E_field_comp_db_apo))
                    #print(E_field_db_apo,len(E_field_db_apo))


                    E_diff = E_field_comp_db_apo - E_field_db_apo
                    if (E_diff != 0).all():
                        print('Test invalide!')
                        print("La différence entre les deux champs est :", E_diff)
                        rep=str(input("Voulez-vous continuer ?(o/n)"))
                        if rep == "o":
                            break
                        else:
                            raise ValueError('test invalide!')

                    print('Test valide!')


                   # E1 = [[]] * (wvl + 1)
                    #E2 = [[]] * (wvl + 1)

                    #for ii_x in np.arange(0, N_x):
                     #   for ii_lvl in np.arange(0, wvl + 1):
                      #      E1[ii_lvl] = E_field[ii_x][ii_lvl].todense()
                       #     E2[ii_lvl] = E_field_comp[ii_x][ii_lvl].todense()
                        #    if (E1[ii_lvl] != E2[ii_lvl]).all():
                         #       print('Test invalide!')
                          #      raise ValueError('test invalide!')


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
                            config_final = meth + groun + type + atm + '/E_final/E_field.npy'
                            print(config_final)
                            shutil.copy(config_propa, dest_propa)
                            shutil.copyfile(config_source, dest_source)
                            shutil.copyfile(config_field, dest_field)
                            shutil.copyfile(config_relief, dest_relief)
                            E_field = np.load(config_final, allow_pickle=True)

                            f_propagation_config = open(dest_propa, newline='')
                            file_tmp = csv.reader(f_propagation_config)
                            #for row in file_tmp:
                             #   if row[0] == 'N_x':
                              #      N_x = np.int64(row[1])
                               # elif row[0] == 'wavelet level':
                                #    wvl = np.int64(row[1])

                            i += 1
                            print(i, "/100")

                            # propagation computation
                            if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
                                cmd = 'cd ../propagation && python3 ./main_propagation.py'
                            else:
                                cmd = 'cd ../propagation && python ./main_propagation.py'
                            # sys.stdout = open("test.txt", 'a')
                            os.system(cmd)

                            E_field_comp = np.load('../propagation/outputs/E_field.npy', allow_pickle=True)

                            E_field_comp_db = 20 * np.log10(np.abs(E_field_comp))
                            E_field_db = 20 * np.log10(np.abs(E_field))
                            # print(E_field_comp_db)
                            # print(E_field_db)

                            f_propa_config = open(config_propa, newline='')
                            file_tmp = csv.reader(f_propa_config)
                            for row in file_tmp:
                                if row[0] == 'apodisation size':
                                    apod = np.float64(row[1])
                                elif row[0] == 'ground':
                                    groundType = np.str_(row[1])
                                elif row[0] == 'N_z':
                                    N_z = np.int32(row[1])

                            E_field_comp_db_apo = E_field_comp_db[:int(len(E_field_comp_db) - (apod * N_z))]
                            E_field_db_apo = E_field_db[:int(len(E_field_db) - (apod * N_z))]

                            if groundType == "No Ground":
                                E_field_comp_db_apo = E_field_comp_db_apo[int(apod * N_z):]
                                E_field_db_apo = E_field_db_apo[int(apod * N_z):]

                            #print(E_field_comp_db_apo, len(E_field_comp_db_apo))
                            #print(E_field_db_apo, len(E_field_db_apo))

                            #if (E_field_comp_db != E_field_db).all():
                             #   print('Test invalide!')
                              #  raise ValueError('test invalide!')

                            E_diff = E_field_comp_db - E_field_db
                            if (E_diff != 0).all():
                                print('Test invalide!')
                                print("La différence entre les deux champs est :", E_diff)
                                rep = str(input("Voulez-vous continuer ?(o/n)"))
                                if rep == "o":
                                    break
                                else:
                                    raise ValueError('test invalide!')
                            print('Test valide!')


                            #E1 = [[]] * (wvl + 1)
                            #E2 = [[]] * (wvl + 1)

                            #for ii_x in np.arange(0, N_x):
                             #   for ii_lvl in np.arange(0, wvl + 1):
                              #      E1[ii_lvl] = E_field[ii_x][ii_lvl].todense()
                               #     E2[ii_lvl] = E_field_comp[ii_x][ii_lvl].todense()
                                #    if (E1[ii_lvl] != E2[ii_lvl]).all():
                                 #       print('Test invalide!')
                                  #      raise ValueError('test invalide!')


    t_end = time.process_time()
    print('Total Time (s)', np.round((t_end - t_start)))

if __name__ == '__main__':
    Test()

