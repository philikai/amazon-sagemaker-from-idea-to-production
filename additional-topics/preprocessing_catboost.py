
import pandas as pd
import numpy as np
import argparse
import os

def _parse_args():
    
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='bank-additional-full.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    
    return parser.parse_known_args()


if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    
    target_col = "y"
    
    # Load data
    df_data = pd.read_csv(os.path.join(args.filepath, args.filename), sep=";")

    # starting with the variables that have order and we want to map
    targetMapping = {'no':0, 'yes':1}
    educationMapping = {'illiterate':0, 'basic.4y':1, 'basic.6y':1, 'basic.9y':2,
                        'high.school':3,'professional.course':4, 'university.degree':5,
                        'unknown':-999}
    dayMapping = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5}
    monthMapping = {'jan': 1, 'feb': 2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    defaultMapping = {"no":0,"yes":1,"unknown":-999}
    housingMapping = {"no":0,"yes":1,"unknown":-999}
    loanMapping = {"no":0,"yes":1,"unknown":-999}
    poutcomeMapping = {"failure":0,"nonexistent":-999,"success":1}
    contactMapping = {'telephone':0, 'cellular':1}

    df = df.copy()
    df['y'] = df['y'].map(targetMapping)
    df['education'] = df['education'].map(educationMapping)
    df['day_of_week'] = df['day_of_week'].map(dayMapping)
    df['month'] = df['month'].map(monthMapping)
    df['default'] = df['default'].map(defaultMapping)
    df['housing'] = df['housing'].map(housingMapping)
    df['loan'] = df['loan'].map(loanMapping)
    df['poutcome'] = df['poutcome'].map(poutcomeMapping)
    df['contact'] = df['contact'].map(contactMapping)
    df.drop('duration', axis = 1, inplace = True)

    # Shuffle and splitting dataset
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=1729),
        [int(0.7 * len(df)), int(0.9 * len(df))],
    )

    print(f"Data split > train:{train_data.shape} | validation:{validation_data.shape} | test:{test_data.shape}")
    
    # Save datasets locally
    train_data.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False, header=False)
    validation_data.to_csv(os.path.join(args.outputpath, 'validation/validation.csv'), index=False, header=False)
    test_data[target_col].to_csv(os.path.join(args.outputpath, 'test/test_y.csv'), index=False, header=False)
    test_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'test/test_x.csv'), index=False, header=False)
    
    # Save the baseline dataset for model monitoring
    df_model_data.drop([target_col], axis=1).to_csv(os.path.join(args.outputpath, 'baseline/baseline.csv'), index=False, header=False)
    
    print("## Processing complete. Exiting.")
