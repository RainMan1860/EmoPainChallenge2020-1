# Esplorando manualmente i dati, sono stati notati molte entry inutili
# Si procede quindi ad una pulizia di questi dati per migliorare le prestazioni di classificazione
# Il primo step è di rimuovere i dati che vengono considerati inutili
# il secondo step, invece, è quello di sostituire questi dati con una entry contenente la media su tutte le colonne
# Per facilità, si considerano i dati presenti nella cartella "Geometric Features" per la verifica dei dati
# in quanto la raccolta dei dati errata in un gruppo di features trova corrispondenza in tutti i gruppi di feature
# Si è ritenuto comunque necessario eliminare le features riguardanti il video C390-001_B_L4_2011-07-08T11-40-57_D_cam8
# in quanto si è notato che i loro valori sono tutti nulli

import cleaning as cln


# data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/"
label_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/train/"
geometric_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/Geometric Features/train/"
hog_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/HOG Features/train/"
res_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/Res_Net Features/train/"
vgg_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/VGG Features/train/"
path_dict = {}
# path_dict["base"] = data_path
path_dict["geometric"] = geometric_data_path
path_dict["hog"] = hog_data_path
path_dict["res"] = res_data_path
path_dict["vgg"] = vgg_data_path
path_dict["label"] = label_data_path

# data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/"
label_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/train/"
geometric_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/geometric/train/"
hog_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/hog/train/"
resnet_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/resnet/train/"
vgg_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/vgg/train/"
clean_path_dict = {}
clean_path_dict["geometric"] = geometric_data_clean_path
clean_path_dict["hog"] = hog_data_clean_path
clean_path_dict["res"] = resnet_data_clean_path
clean_path_dict["vgg"] = vgg_data_clean_path
clean_path_dict["label"] = label_data_clean_path

num_class = 11

# geometric_duplicates = cln.count_duplicate(geometric_data_path, has_header=True)
# hog_duplicates = cln.count_duplicate(hog_data_path, has_header=False)
# res_duplicates = cln.count_duplicate(res_data_path, has_header=False)
# vgg_duplicates = cln.count_duplicate(vgg_data_path, has_header=False)

# elimino le righe in tutti i gruppi di features per le quali il valore di success delle feature geometriche è 0
cln.drop_no_detection(path_dict, clean_path_dict)
# cln.match_rows(path_dict)