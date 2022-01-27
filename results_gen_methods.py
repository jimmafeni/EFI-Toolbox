# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:05:37 2021
@author: Aayush Kumar
"""


#######################################################################################################################
# ------------------------- Methods used in generating final report --------------------------- #
#######################################################################################################################


def generate_eval_report(x):
    global ImgFile
    import os
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'results', x)
    # Importing Required Module
    from PIL import Image
    Img_List = list()
    # Creating Image File Object
    for i in os.listdir(final_directory):
        if i.endswith(".png"):
            ImgFile = Image.open(os.path.join(final_directory, i))
            # Checking if Image File is in 'RGBA' Mode
            if ImgFile.mode == "RGBA":
                # If in 'RGBA' Mode Convert to 'RGB' Mode
                ImgFile = ImgFile.convert("RGB")
                Img_List.append(ImgFile)

            else:
                Img_List.append(ImgFile)

    # Converting and Saving file in PDF format
    Img_List[0].save(os.path.join(generating_results('Reports'), f"{x}.pdf"), "PDF", resolution=300.0, save_all=True,
                     append_images=Img_List[1:])
    # Closing the Image File Object
    Img_List.clear()
    ImgFile.close()


def generating_results(x):
    import os
    # creating results folder, if it doesnt exist
    current_directory = os.getcwd()
    result_directory = os.path.join(current_directory, r'results')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    specific_directory = os.path.join(result_directory, x)
    if not os.path.exists(specific_directory):
        os.makedirs(specific_directory)
    return specific_directory


def generate_multi_fusion(model_selected):
    from PyPDF2 import PdfFileMerger, PdfFileReader
    import os
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r"results", r"Reports")
    add_on = ["Rank.pdf", "Multi-Fusion.pdf"]
    file_merger = PdfFileMerger()
    for model in model_selected:
        for file in os.listdir(final_directory):
            if file.endswith(".pdf") and file.startswith(model):
                file_merger.append(PdfFileReader(open(os.path.join(final_directory, file), 'rb')))
    for spec in add_on:
        for i in os.listdir(final_directory):
            if i.endswith(".pdf") and spec in i:
                file_merger.append(PdfFileReader(open(os.path.join(final_directory, i), 'rb')))
    file_merger.write(os.path.join(final_directory, "Multi_Fusion_Report") + ".pdf")
    file_merger.close()


# tracking report in the folder

def convert(lst):
    s = (lst[0].split())
    return s[0]


def generate_model_report(model_name):
    import os
    from PIL import Image
    global ImgFile
    current_directory = os.getcwd()
    directory_cm = os.path.join(current_directory, r'results', r'Confusion_Matrix')
    directory_lime = os.path.join(current_directory, r'results', r"Lime")
    directory_shap = os.path.join(current_directory, r'results', r"SHAP")
    directory_roc = os.path.join(current_directory, r'results', r'ROC')
    directory_Rank = os.path.join(current_directory, r'results', r'Rank')
    directory_PI = os.path.join(current_directory, r'results', r"Permutation Importances")
    directory_SF = os.path.join(current_directory, r'results', r'single_fusion')
    directory_ME = os.path.join(current_directory, r'results', r'perm')
    directory_MV = os.path.join(current_directory, r'results', r'Majority Voting')
    directory_MF = os.path.join(current_directory, r'results', r'Multi-Fusion')
    directory_FZ = os.path.join(current_directory, r'results', r'FUZZY')

    direc = [directory_ME, directory_roc, directory_cm, directory_shap, directory_lime, directory_PI, directory_MV,
             directory_Rank, directory_SF, directory_MF, directory_FZ]

    rsg = os.path.join(current_directory, r"results", r"Reports")
    Img_List = list()
    Img1 = []
    for d in direc:
        for i in os.listdir(d):
            if i.endswith(".png") and convert([model_name]) in i:
                ImgFile = Image.open(os.path.join(d, i))
                # Checking if Image File is in 'RGBA' Mode
                if ImgFile.mode == "RGBA":
                    # If in 'RGBA' Mode Convert to 'RGB' Mode
                    ImgFile = ImgFile.convert("RGB")
                    Img_List.append(ImgFile)
                else:
                    Img_List.append(ImgFile)
    Img = Img_List[0]
    Img1 = Img.resize((2800, 1080), Image.LANCZOS)
    # Converting and Saving file in PDF format
    Img1.save(os.path.join(rsg, f"{model_name}.pdf"), "PDF", resolution=300.0, save_all=True,
              append_images=Img_List[1:])


