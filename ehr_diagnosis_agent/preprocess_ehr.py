from argparse import ArgumentParser
from data.preprocess_mimic import preprocess_mimic
import pandas as pd
import os
import random
from tqdm import tqdm


def create_dataset(reports, codes, path):
    if os.path.exists(path):
        raise Exception
    os.mkdir(path)
    patient_ids = sorted(list(set(reports.patient_id)))
    random.seed(0)
    random.shuffle(patient_ids)
    div1 = int(len(patient_ids) * .7)
    div2 = int(len(patient_ids) * .85)
    splits = {
        'val1': patient_ids[div1:div1 + (div2 - div1) // 2],
        'val2': patient_ids[div1 + (div2 - div1) // 2:div2],
        'test': patient_ids[div2:],
        'train': patient_ids[:div1],
    }
    reports = reports.sort_values('date')
    codes = codes.sort_values('date')
    for split, split_patient_ids in splits.items():
        rows = []
        for patient_id in tqdm(split_patient_ids, total=len(split_patient_ids), desc=split):
            report_rows = reports[reports.patient_id == patient_id]
            code_rows = codes[codes.patient_id == patient_id]
            rows.append({
                'patient_id': patient_id,
                'reports': report_rows.to_csv(index=False),
                'codes': code_rows.to_csv(index=False),
                'start_date': report_rows.iloc[0].date,
                'end_date': report_rows.iloc[-1].date,
                'num_reports': len(report_rows),
            })
            if 'hadm_id' in report_rows.columns:
                rows[-1]['num_admissions'] = len(set(report_rows.hadm_id))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(path, f'{split}.data'), index=False, compression='gzip')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("data_type")
    parser.add_argument("data_path")
    parser.add_argument("dataset_name")
    args = parser.parse_args()
    processed_path = os.path.join(args.data_path, 'preprocessed')
    if not os.path.exists(os.path.join(processed_path, 'medical_reports.csv')):
        print(f'preprocessing {args.data_type} at {args.data_path}')
        if args.data_type == 'mimic':
            preprocess_mimic(args.data_path)
        else:
            raise Exception('Data type does not exist!')
    if os.path.exists(os.path.join(processed_path, args.dataset_name)):
        print('dataset \"{}\" already exists in the data path at:\n\t{}'.format(
            args.dataset_name, os.path.join(processed_path, args.dataset_name)))
    else:
        print(f'creating a {args.data_type} dataset called {args.dataset_name} at {args.data_path}')
        reports = pd.read_csv(os.path.join(processed_path, 'medical_reports.csv'), parse_dates=['date'])
        codes = pd.read_csv(os.path.join(processed_path, 'medical_codes.csv'), parse_dates=['date'])
        create_dataset(reports, codes, os.path.join(processed_path, args.dataset_name))
