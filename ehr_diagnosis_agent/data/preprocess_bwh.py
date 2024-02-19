import pandas as pd
import os
import re
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser


tables = {
    'Cardiology Reports': 'Car',
    'Discharge Summaries': 'Dis',
    'Endoscopy Reports': 'End',
    'Microbiology Data': 'Mic',
    'Operative Notes': 'Opn',
    'Pathology Reports': 'Pat',
    'Pulmonary Reports': 'Pul',
    'Radiology Reports': 'Rad',
    'Transfusion Data': 'Trn',
    'Contact Information': 'Con',
    'Demographic Data': 'Dem',
    'Diagnoses': 'Dia',
    'Encounters': 'Enc',
    'Medications': 'Med',
    'Procedures': 'Prc',
    'Providers': 'Prv',
    'Allergy': 'All',
    'Radiology Tests': 'Rdt',
    # 'Health History': '',
    # 'EPIC Reason for Visit': '',
    # 'EPIC Ambulatory Visit Notes': '',
    # 'LMR Ambulatory Visit Notes': '',
    # 'EPIC Progress Reports': '',
    # 'EPIC History & Physical': '',
    # Unidentified: Let, Phy, Lno, Mrn, Log
    # Previous unidentified:
    # 'Hpn': 'Hpn',
    # 'ProgressNote': 'Prg',
    # 'ReasonForVisit': 'Rfv',
    # 'VisitNote': 'Vis'
}


def preprocess_to_csv(folder, subfolder, filename, contains_report=False):
    with open(os.path.join(folder, filename), 'r') as infile:
        with open(os.path.join(folder, subfolder, filename), 'w') as outfile:
            for i,line in enumerate(infile):
                if (not contains_report) or i == 0:
                    line = line.replace('\n', '\r')
                else:
                    line = line.replace('[report_end]\n', '[report_end]\r')
                outfile.write(line)


report_files = {
    'Cardiology Reports',
    'Discharge Summaries',
    'Endoscopy Reports',
    'Operative Notes',
    'Pathology Reports',
    'Pulmonary Reports',
    'Radiology Reports',
}
code_files = {
    # 'Diagnoses'
}


def main(folder):
    subfolder1 = 'PreprocessedTextFiles'
    subfolder2 = 'preprocessed'
    files = os.listdir(folder)
    if not os.path.exists(os.path.join(folder, subfolder1)):
        os.mkdir(os.path.join(folder, subfolder1))
    table_files = {k: [f for f in files if f.endswith('%s.txt' % v)] for k, v in tables.items()}
    for k, v in tables.items():
        for file in table_files[k]:
            print('converting %s to csv' % file)
            if os.path.exists(os.path.join(folder, subfolder1, file)):
                continue
            preprocess_to_csv(folder, subfolder1, file, contains_report=k in report_files)
    print('reading csvs')
    if not os.path.exists(os.path.join(folder, subfolder2)):
        os.mkdir(os.path.join(folder, subfolder2))
    text_columns = ['report_id', 'patient_id', 'date', 'report_type', 'description', 'text', 'from_table', 'other_info']
    all_report_files = []
    for k in report_files:
        print(k)
        for num, f in enumerate(table_files[k]):
            filename = os.path.join(folder, subfolder2, 'reports_%s_%i.csv' % (k, num))
            if os.path.exists(filename):
                continue
            print('%i / %i' % (num + 1, len(table_files[k])))
            text_rows = []
            df = pd.read_csv(os.path.join(folder, subfolder1, f), delimiter='|', lineterminator='\r')
            for i, row in tqdm(df.iterrows(), total=len(df)):
                results = re.findall('\d+', str(row.EMPI))
                if len(results) == 1:
                    patient_id = int(results[0])
                    # this is the first one, so sometimes it has an extra newline at the beginning
                else:
                    continue
                # date = row.Report_Date_Time # TODO: make this into a number for sorting purposes
                date = pd.to_datetime(datetime.strptime(row.Report_Date_Time, '%m/%d/%Y %I:%M:%S %p'))
                report_type = from_table = k
                description = row.Report_Description
                text = row.Report_Text
                report_number = row.Report_Number
                other_info = {'report_status': row.Report_Status, 'epic_pmrn': row.EPIC_PMRN, 'mrn_type': row.MRN_Type, 'mrn': row.MRN}
                text_rows.append(
                    [report_number, patient_id, date, report_type, description, text, from_table, other_info])
            df = pd.DataFrame(text_rows, columns=text_columns)
            df.to_csv(filename, index=False)
            all_report_files.append(df)
    pd.concat(all_report_files).to_csv(os.path.join(folder, subfolder2, 'medical_reports.csv'), index=False)
    code_columns = ['patient_id', 'date', 'flag', 'name', 'code_type', 'code', 'from_table', 'other_info']
    for k in code_files:
        print(k)
        for num, f in enumerate(table_files[k]):
            filename = os.path.join(folder, subfolder2, 'codes_%s_%i.csv' % (k, num))
            if os.path.exists(filename):
                continue
            print('%i / %i' % (num + 1, len(table_files[k])))
            code_rows = []
            df = pd.read_csv(os.path.join(folder, subfolder1, f), delimiter='|', lineterminator='\r')
            for i,row in tqdm(df.iterrows(), total=len(df)):
                patient_id = row.EMPI
                # date = row.Date # TODO: make this into a number for sorting purposes
                date = pd.to_datetime(datetime.strptime(row.Date, '%m/%d/%Y'))
                # need to split to handle flag, name, code_type, and code for different tables
                if k == 'Diagnoses':
                    flag = row.Diagnosis_Flag
                    name = row.Diagnosis_Name
                    code_type = row.Code_Type
                    code = row.Code
                else:
                    raise NotImplementedError
                from_table = k
                code_rows.append([patient_id, date, flag, name, code_type, code, from_table,
                                  {'encounter_number': row.Encounter_number}])
            pd.DataFrame(code_rows, columns=code_columns).to_csv(filename, index=False)


def preprocess_bwh(data_folder):
    main(data_folder)
