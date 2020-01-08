import pandas as pd
import numpy as np
import os


def clean(data_path):
    pass


def count_duplicate(data_path, has_header=False):
    count_duplicates = []
    for df_path in os.listdir(data_path):
        if has_header:
            df = pd.read_csv(data_path + df_path)
        else:
            df = pd.read_csv(data_path + df_path, header=None)
        dup_entry = {}
        dup_entry["file"] = df_path
        dup_entry["directory"] = data_path
        dup_entry["duplicates"] = df.groupby(df.columns.tolist(), as_index=False).size()
        print(dup_entry["duplicates"])
        count_duplicates.append(dup_entry)
    return count_duplicates


def drop_no_detection(data_path, data_clean_path):
    tot_frame = 0
    for df_path in os.listdir(data_path["geometric"]):
        try:
            geometric_df = pd.read_csv(data_path["geometric"] + df_path)
            geometric_df = geometric_df.rename(columns=lambda x: x.strip())
            no_success_df = geometric_df.loc[geometric_df["success"] == 0]
            if no_success_df.shape[0] > 0:
                tot_frame = tot_frame + no_success_df.shape[0]
                print(df_path + ": " + str(no_success_df.shape[0]))

                hog_df = pd.read_csv(data_path["hog"] + df_path, header=None)
                res_df = pd.read_csv(data_path["res"] + df_path.replace(".csv", "_res.csv"), header=None)
                vgg_df = pd.read_csv(data_path["vgg"] + df_path.replace(".csv", "_vgg.csv"), header=None)
                label_df = pd.read_csv(data_path["label"] + df_path, header=None)

                if geometric_df.shape[0] == hog_df.shape[0] and geometric_df.shape[0] == res_df.shape[0] and geometric_df.shape[0] == vgg_df.shape[0] and geometric_df.shape[0] == label_df.shape[0]:
                    drop_index = no_success_df.index.values.tolist()
                    geometric_df = geometric_df.drop(geometric_df.index[drop_index])
                    hog_df = hog_df.drop(hog_df.index[drop_index])
                    res_df = res_df.drop(res_df.index[drop_index])
                    vgg_df = vgg_df.drop(vgg_df.index[drop_index])
                    label_df = label_df.drop(label_df.index[drop_index])

                    geometric_df.to_csv(data_clean_path["geometric"] + df_path, index=False, float_format="%g")
                    hog_df.to_csv(data_clean_path["hog"] + df_path, index=False, header=False, float_format="%g")
                    res_df.to_csv(data_clean_path["res"] + df_path.replace(".csv", "_res.csv"), index=False, header=False, float_format="%g")
                    vgg_df.to_csv(data_clean_path["vgg"] + df_path.replace(".csv", "_vgg.csv"), index=False, header=False, float_format="%g")
                    label_df.to_csv(data_clean_path["label"] + df_path, index=False, header=False, float_format="%g")
                else:
                    print("Numero righe diverso per " + df_path)
            else:
                print("{}: non modificato".format(df_path))

                geometric_df.to_csv(data_clean_path["geometric"] + df_path, index=False, float_format="%g")
                hog_df.to_csv(data_clean_path["hog"] + df_path, index=False, header=False, float_format="%g")
                res_df.to_csv(data_clean_path["res"] + df_path.replace(".csv", "_res.csv"), index=False, header=False, float_format="%g")
                vgg_df.to_csv(data_clean_path["vgg"] + df_path.replace(".csv", "_vgg.csv"), index=False, header=False, float_format="%g")
                label_df.to_csv(data_clean_path["label"] + df_path, index=False, header=False, float_format="%g")

        except KeyError:
            print(df_path)
    print("Totale frame non validi: {}".format(tot_frame))


def count_no_detection(data_path):
    tot_frame = 0
    frame_class = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
        "10": 0,
    }
    for df_path in os.listdir(data_path["geometric"]):
        try:
            geometric_df = pd.read_csv(data_path["geometric"] + df_path)
            geometric_df = geometric_df.rename(columns=lambda x: x.strip())
            no_success_df = geometric_df.loc[geometric_df["success"] == 0]
            if no_success_df.shape[0] > 0:
                label_df = pd.read_csv(data_path["label"] + df_path, header=None)

                no_valid_index = no_success_df.index.values.tolist()
                sub_label_df = label_df.iloc[label_df.index[no_valid_index]]
                if sub_label_df.shape[0] != no_success_df.shape[0]:
                    print("Errore nel recupero delle righe. Righe label: {} e Righe data: {}".format(sub_label_df.shape[0], no_success_df.shape[0]))
                    continue
                else:
                    count_df = sub_label_df.groupby(sub_label_df[0]).size().reset_index(name="count")
                    count_df.columns = ["intensity", "count"]
                    for row in count_df.itertuples():
                        frame_class[str(row.intensity)] = frame_class[str(row.intensity)] + row.count
                    # print(count_df)

                tot_frame = tot_frame + no_success_df.shape[0]
                print(df_path + ": " + str(no_success_df.shape[0]))
            else:
                print("{}: non modificato".format(df_path))
        except KeyError:
            print(df_path)

    print("Totale frame non validi: {}".format(tot_frame))

    print("Frame non validi divisi per classi:")
    print(frame_class)


def match_shapes(data_path):
    pass


def match_rows(data_path):
    geometric = os.listdir(data_path["geometric"])
    hog = os.listdir(data_path["hog"])
    res = os.listdir(data_path["res"])
    vgg = os.listdir(data_path["vgg"])
    label = os.listdir(data_path["label"])

    geometric = [x.replace(".csv", "") for x in geometric]
    hog = [x.replace(".csv", "") for x in hog]
    res = [x.replace("_res.csv", "") for x in res]
    vgg = [x.replace("_vgg.csv", "") for x in vgg]
    label = [x.replace(".csv", "") for x in label]

    if set(geometric) != set(hog):
        print("Geometric e HOG non sono uguali")
    if set(geometric) != set(res):
        print("Geometric e RES non sono uguali")
    if set(geometric) != set(vgg):
        print("Geometric e VGG non sono uguali")
    if set(geometric) != set(label):
        print("Geometric e Label non sono uguali")

