# stesso tipo di pulizia del training
# si ritiene non corretto considerare i frame nei quali non è avvenuto il riconoscimento del volto

import cleaning as cln


label_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Pain Labels/valid/"
geometric_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/Geometric Features/valid/"
hog_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/HOG Features/valid/"
res_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/Res_Net Features/valid/"
vgg_data_path = "D:/dataset/EmoPain Challenge 2020/Facedata/Face Features/VGG Features/valid/"
path_dict = {}
# path_dict["base"] = data_path
path_dict["geometric"] = geometric_data_path
path_dict["hog"] = hog_data_path
path_dict["res"] = res_data_path
path_dict["vgg"] = vgg_data_path
path_dict["label"] = label_data_path

data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/"
label_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/labels/valid/"
geometric_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/geometric/valid/"
hog_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/hog/valid/"
resnet_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/resnet/valid/"
vgg_data_clean_path = "D:/dataset/EmoPain Challenge 2020/Facedata/face_features_clean/vgg/valid/"
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
# cln.count_no_detection(path_dict)
cln.count_no_detection(clean_path_dict)