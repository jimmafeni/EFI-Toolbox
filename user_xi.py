import data_preprocessing as dp
import pandas as pd

models_to_eval = ['LightGBM Classifier', 'Logistic Regressor classifier', 'Artificial Neural Network',
                  'Random Forest classifier', 'Support vector machines']


def restart_clean_direc():
    import os
    import shutil
    current_directory = os.getcwd()
    directory = os.path.join(current_directory, r'results')
    if os.path.exists(directory):
        shutil.rmtree(directory)
    else:
        pass


def read_config_file(x):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(x)
    param_dataset = [config['DATA']['Loc'], config['DATA']['TARGET'], float((config['DATA']['Test_Size'])),
                     float(config['DATA']['Interpretability_dataset_size']),
                     int(config['DATA']['cross_val']), int(config['DATA']['K-Fold-FUZZY'])]
    print(param_dataset)
    model_selected = config['MODELS']['MODEL_SELECTION']
    print("Your Selection:", model_selected)
    return param_dataset, eval(model_selected)


def read_gui_input(dict):
    model_selected = []
    data_loc = dict.get('load_Data')
    target = dict.get('-class-')
    data_size_for_testing = dict.get('data_size_for_testing')
    data_size_for_interpretability = dict.get('data_size_for_interpretability')
    cv = dict.get("cv")
    fcv = dict.get("fcv")
    param_dataset = [data_loc, target, float(data_size_for_testing), float(data_size_for_interpretability), int(cv),
                     int(fcv)]
    print(param_dataset)
    for key, value in dict.items():
        if value == True:
            model_selected.append(key)
    print("Your Selection:", model_selected)
    return param_dataset, model_selected


def exp_config_portal():
    import PySimpleGUI as sg
    sg.theme('DarkTeal9')
    font = ("Helvitica", 12)

    EXCEL_FILE = 'Config_History.xlsx'
    df = pd.read_excel(EXCEL_FILE)

    mylist = [1, 2, 3, 4, 5, 6, 7, 8]
    progressbar = [
        [sg.ProgressBar(len(mylist), orientation='h', size=(51, 10), key='progressbar',
                        bar_color=('lightgreen', 'grey'))]]

    layout = [
        [sg.Frame('Getting things ready', layout=progressbar)],
        [sg.Text('Experiment Configuration', font=('Helvetica', 12, 'bold'))],
        [sg.Text('' * 20)],
        [sg.Text('Experiment Name : ', size=(25, 1), font=font), sg.InputText(key='exp_name')],
        [sg.Text('' * 20)],
        [sg.Frame(layout=[
            [sg.Text('Load Data :', size=(25, 1), font=font), sg.Input("", key='load_Data'),
             sg.FileBrowse(auto_size_button=True)],
            [sg.Text('' * 20)],
            [sg.Text('Target Variable : ', size=(25, 1), font=font), sg.InputText(key="-class-",
                                                                                  default_text="click on -LOAD-CLASS- "
                                                                                               "to capture class "
                                                                                               "variable",
                                                                                  font=('Helvitica', 10)),
             sg.Button("LOAD-CLASS", button_color=("white", "grey"), auto_size_button=True, size=(20, 1))]],
            title='Dataset',
            title_color='lightgreen',
            tooltip='Last column of your dataset should be your TARGET variable, press LOAD CLASS, to capture it :)')],
        [sg.Text('' * 20)],
        [sg.Frame(layout=[
            [sg.Text('Classification Models : ', size=(25, 1), font=font)],
            [sg.Text('' * 5)],
            [sg.Checkbox("LightGBM Classifier",
                         key="LightGBM Classifier", font=font, text_color='lightblue'),
             sg.Checkbox("Logistic Regressor classifier",
                         key="Logistic Regressor classifier", font=font, text_color='lightblue'),
             sg.Checkbox("Artificial Neural Network",
                         key="Artificial Neural Network", font=font, text_color='lightblue'),
             sg.Checkbox("Random Forest classifier",
                         key="Random Forest classifier", font=font, text_color='lightblue'),
             sg.Checkbox('Support vector machines',
                         key='Support vector machines', font=font, text_color='lightblue')]], title='Machine-Learning'
                                                                                                    '-Models',
            title_color='lightgreen', tooltip='Use these to select models for evaluation')],
        [sg.Text('' * 20)],
        [sg.Text('Test-Data Size (%) : ', size=(25, 1), font=font),
         sg.InputText(key='data_size_for_testing', default_text=30, tooltip=('Train_data size is '
                                                                             'automatically calculated'))],
        [sg.Text('' * 20)],
        [sg.Text('Interpretability-Data Size (%) : ', size=(25, 1), font=font),
         sg.InputText(key='data_size_for_interpretability', default_text=50)],
        [sg.Text('' * 20)],
        [sg.Text('ML model cv (2-100) : ', size=(25, 1), font=font), sg.Slider(
            (2, 100), 2, orientation="h", size=(25, 25), key="cv")],
        [sg.Text('' * 20)],
        [sg.Text('K-Fold fuzzy logic (2-100) : ', size=(25, 1), font=font), sg.Slider(
            (2, 100), 4, orientation="h", size=(25, 25), key="fcv")],
        [sg.Text('' * 20)],
        [sg.Text('Use configuration file : ', size=(25, 1), font=font), sg.Input("", key="config"),
         sg.FileBrowse(auto_size_button=True),
         sg.Button("LOAD-CONFIG", button_color=("white",
                                                "green"), auto_size_button=True, size=(15, 1))],
        [sg.Text('' * 20)],
        [sg.Text('' * 20)],
        [sg.Submit(size=(15, 1)), sg.Exit(size=(15, 1))]
    ]

    window = sg.Window(' Feature Importance Fusion Framework', layout, element_justification='l',
                       resizable=True).Finalize()
    window.Maximize()

    progress_bar = window['progressbar']

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if event == "LOAD-CLASS":
            mv = values
            input_text = dp.get_class(mv.get('load_Data'))
            window["-class-"].update(input_text)
            mv["-class-"] = input_text
            window.refresh()

        if event == 'Submit':
            for i, item in enumerate(mylist):
                import time
                restart_clean_direc()
                time.sleep(1)
                progress_bar.UpdateBar(i + 1)
            mv = values
            df = df.append(values, ignore_index=True)
            df.to_excel(EXCEL_FILE, index=False)
            return read_gui_input(mv)

        if event == "LOAD-CONFIG":
            for i, item in enumerate(mylist):
                import time
                restart_clean_direc()
                time.sleep(1)
                progress_bar.UpdateBar(i + 1)
            mv = values
            df = df.append(values, ignore_index=True)
            df.to_excel(EXCEL_FILE, index=False)
            return read_config_file(mv.get("config"))

    window.close()


