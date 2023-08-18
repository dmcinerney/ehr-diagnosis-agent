import pandas as pd
import os
from utils import get_args
import pickle as pkl
import io
from tqdm import tqdm
from data.preprocess_icd_codes import icd9_file


if __name__ == '__main__':
    args = get_args('config.yaml')
    icd_code_descriptions = {
        'ICD9': pd.read_csv(
            icd9_file+'_processed.csv', names=['code', 'description'],
            delimiter='\t'),
    }
    # for split in ['val', 'test', 'train']:
    # for split in ['val', 'train']:
    for split in ['train']:
        data = pd.read_csv(os.path.join(
            args.data.path, args.data.dataset, f'{split}.data'),
            compression='gzip')
        cache_path = args.env[f'{split}_cache_path']
        dfs = []
        pbar = tqdm(data.iterrows(), total=len(data))
        num_truncated = 0
        num_gt_48 = 0
        num_with_complaint = 0
        for i, row in pbar:
            reports = pd.read_csv(
                io.StringIO(row.reports), parse_dates=['date'])
            codes = pd.read_csv(io.StringIO(row.codes), parse_dates=['date'])
            with open(os.path.join(
                    cache_path, f'cached_instance_{i}.pkl'), 'rb') as f:
                instance = pkl.load(f)
            confident_diagnoses = [
                set(diagnoses)
                for diagnoses in next(
                    iter(instance.values()))['confident diagnoses']]
            num_gt_48 += len(reports) > 48
            num_truncated += len(reports) > len(confident_diagnoses)
            num_with_complaint += 'presenting complaint' in next(iter(instance.values())).keys()
            pbar.set_postfix(
                {'num_gt_48': num_gt_48, 'num_truncated': num_truncated,
                 'num_with_complaint': num_with_complaint})
            if len(reports) > len(confident_diagnoses):
                assert len(confident_diagnoses) == 48
                reports = reports[:len(confident_diagnoses)]
                last_date = reports.iloc[-1].date.date()
                codes = codes[codes.date.apply(lambda x: x.date() < last_date)]
            code_tuples = []
            for j, code_row in codes.iterrows():
                code_descs = icd_code_descriptions[code_row.code_type]
                code_tuples.append(set([(
                    code_row.code_type, code_row.code,
                    ', '.join(eval(code_row['name'])),
                    ', '.join(code_descs[
                        code_descs.code.str.startswith('3488')
                    ].description.to_list())
                )]))
            df = pd.DataFrame({
                'date': [date.date() for date in reports.date.tolist()] + [
                    date.date() for date in codes.date.tolist()],
                'codes': [set()] * len(confident_diagnoses) + code_tuples,
                'extracted_diagnoses':
                    confident_diagnoses + [set()] * len(codes)
            })
            df = df.groupby('date').agg({
                'codes': lambda x: set().union(*x),
                'extracted_diagnoses': lambda x: set().union(*x),
            }).reset_index()
            df['instance_idx'] = [i] * len(df)
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(os.path.join(
            args.data.path, args.data.dataset, f'{split}_codes.csv'),
            index=False)
