import os

import io
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import fdrcorrection

from scipy.stats import pearsonr, spearmanr, norm
from itertools import combinations
import pingouin as pg

# import rpy2.robjects as ro
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import default_converter


def PANSS_estimator(df, dataset,):
    """Estimate PANSS score cited by Converting Positive and Negative Symptom Scores BetweenPANSS and SAPS/SANS (2014)"""

    if dataset == "MCIC":
        sans_global_ratings = ["Global Rating of Affective Flattening",
                               "Global Rating of Alogia",
                               "Global Rating of Avolition - Apathy",
                               "Global Rating of Anhedonia - Asociality",
                               "Global Rating of Attention"
                               ]
        
        saps_global_ratings = [ "Global Rating of Severity of Delusions",
                               "Global Rating of Severity of Hallucinations",
                               "Global Rating of Severity of Bizarre Behavior",
                               "Global Rating of Positive Formal Thought Disorder"]
        
        for score in sans_global_ratings+saps_global_ratings:
            df.dropna(subset=score, axis=0, inplace=True)
            df[score].replace(['DK', "MD"], np.nan, inplace=True)
            df.dropna(subset=[score], axis=0, inplace=True)
            df[score] = df[score].astype(float)
        
    else:
        ValueError("Invalid dataset")



    df['Positive Scale'] = 9.3264 + (1.1072*df[saps_global_ratings].sum(axis=1))
    df['Negative Scale'] = 6.7515 + (1.0287*df[sans_global_ratings].sum(axis=1))

    return df


def MeanAbsError(df, real_col, pred_col):
    real_age = df[real_col]
    pred_age = df[pred_col]
    
    mae = np.mean(np.abs(real_age - pred_age))
    
    return mae


def Correlation(df, real_col, pred_col, method='r2'):
    
    real_age = df[real_col]
    pred_age = df[pred_col]
    
    if method == 'r':
        corr = pearsonr(real_age, pred_age)[0]
        
    elif method == 'r2':
        corr = pearsonr(real_age, pred_age)[0]**2
        
    else:
        raise ValueError("Invalid method for Correlation")
    
    return corr


def MeanError(df, real_col, pred_col):
    real_age = df[real_col]
    pred_age = df[pred_col]
    
    me = np.mean(pred_age - real_age)
    return me


def mMAE(df, real_col, pred_col, interval_list, verbose=False): # Robustness
    
    MAE_list = []
    
    for i in range(len(interval_list)):
        
        if i == 0: # First Interval 
            temp_df = df[df[real_col] < interval_list[i]]
            if verbose:
                print(f"Interval  < {interval_list[i]} : {len(temp_df)}")
            
        else:
            temp_df = df[(df[real_col] >= interval_list[i-1]) & (df[real_col] < interval_list[i])]
            if verbose:
                print(f"Interval {interval_list[i-1]} ~ {interval_list[i]} : {len(temp_df)}")
            
        temp_MAE = np.mean(np.abs(temp_df[real_col] - temp_df[pred_col]))
        
        MAE_list.append(temp_MAE)
    
    MAE_list = np.array(MAE_list)    
    mMAE = np.max(MAE_list)
    
    return mMAE
    
    
def BiasCorrection(df, real_col, pred_col, method='linear', base_df=None):
    
    if ("Control" in df.columns) and (base_df is None):
        base_df = df[df["Control"] == "HC"]

    real_age = df[real_col].values
    pred_age = df[pred_col].values
    
    temp_df = df.copy()
    
    if method == 'linear':
        if base_df is not None:
            base_real_age = base_df[real_col].values
            base_pred_age = base_df[pred_col].values
            z = np.polyfit(base_real_age, base_pred_age, 1)
            
        else:
            z = np.polyfit(real_age, pred_age, 1)
        corr_pred_age = pred_age + (real_age - (z[0]*real_age + z[1]))
        temp_df["PAD"] = pred_age - real_age
        temp_df["corr_PAD"] = corr_pred_age - temp_df[real_col].values
        
    elif method == 'cole':
        if base_df is not None:
            base_real_age = base_df[real_col].values
            base_pred_age = base_df[pred_col].values
            z = np.polyfit(base_real_age, base_pred_age, 1)
            
        else:
            z = np.polyfit(real_age, pred_age, 1)

        corr_pred_age = (pred_age - z[1]) / z[0] # z[0] is slope, z[1] is intercept
        temp_df["PAD"] = pred_age - real_age
        temp_df["corr_PAD"] = corr_pred_age - temp_df[real_col].values
        
    elif method == 'offset':
        ME = MeanError(df, real_col, pred_col)    
        corr_pred_age = pred_age - ME
        temp_df['corr_pred_age'] = corr_pred_age
        temp_df["PAD"] = pred_age - real_age
        temp_df["corr_PAD"] = corr_pred_age - temp_df[real_col].values

    elif 'age_level' in method:
        if base_df is not None:
            base_real_age = base_df[real_col].values
            base_pred_age = base_df[pred_col].values
            z = np.polyfit(base_real_age, base_pred_age, 1)
        
        else:
            z = np.polyfit(real_age, pred_age, 1)

        if method == 'age_level_wBeheshti': # correction with Beheshti's method
            corr_pred_age = pred_age + (real_age - (z[0]*real_age + z[1]))
            temp_df['corr_pred_age'] = corr_pred_age
            temp_df["PAD"] = pred_age - real_age
            temp_df["corr_PAD"] = corr_pred_age - temp_df[real_col].values

        elif method == 'age_level_wCole':
            corr_pred_age = (pred_age - z[1]) / z[0]
            temp_df['corr_pred_age'] = corr_pred_age
            temp_df["PAD"] = pred_age - real_age
            temp_df["corr_PAD"] = corr_pred_age - temp_df[real_col].values

        elif method == 'age_level_wOffset':
            ME = MeanError(df, real_col, pred_col)
            corr_pred_age = pred_age - ME
            temp_df['corr_pred_age'] = corr_pred_age
            temp_df["PAD"] = pred_age - real_age
            temp_df["corr_PAD"] = corr_pred_age - temp_df[real_col].values

        else: # Without any correction
            temp_df["PAD"] = temp_df[pred_col] - temp_df[real_col]
            corr_pred_age = None
            temp_df["corr_PAD"] = corr_pred_age

        temp_df["Age_int"] = np.round(temp_df[real_col]).astype(int)
        pad_mean = temp_df.groupby("Age_int")["PAD"].mean().reset_index()
        pad_std = temp_df.groupby("Age_int")["PAD"].std().reset_index()
        pad_std.fillna(1, inplace=True)
        pad_mean.rename({"PAD": "PAD_mean"}, axis=1, inplace=True)
        pad_std.rename({"PAD": "PAD_std"}, axis=1, inplace=True)
        temp_df = pd.merge(temp_df, pad_mean, on="Age_int")
        temp_df = pd.merge(temp_df, pad_std, on="Age_int")

        corr_PAD = (temp_df["PAD"].values - temp_df["PAD_mean"].values) / temp_df["PAD_std"].values
        temp_df["corr_PAD"] = corr_PAD

    else:
        raise ValueError("Invalid method for Bias Correction")
        
    return temp_df


def cohen_d(x, y, reverse=False): # Effect Size
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    
    if reverse:
        d = -((np.mean(x) - np.mean(y)) / pooled_std)
        
    else:
        d = (np.mean(y) - np.mean(x)) / pooled_std
    
    return d


def LMEM(df, fixed_cols, random_cols, dependent_col, intercept, verbose=False):
    condition = f'{dependent_col} ~ C({fixed_cols}, Treatment(reference="{intercept}"))'
    model = mixedlm(condition,
                    df, 
                    groups=df[random_cols], )
                    # re_formula=fixed_cols)
    result = model.fit()
    
    table = result.summary().tables[1]

    modality_cols = list(result.summary().tables[1].index)
    names = []

    for col in modality_cols:
        if col not in ['Intercept', 'Group Var']:
            names.append(col.split('[')[1][2:-1])
            
        else:
            if col == 'Intercept':
                names.append(intercept)
            else:
                names.append(col)
    # table
    values = table[['Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]']].values
    r_df = pd.DataFrame(values, columns=['Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]'])
    r_df["modality"] = names
    new_df = pd.concat([r_df.iloc[:,-1],r_df.iloc[:,0:-1]], axis=1)
    
    if verbose:
        print(result.summary())
    
    return result, new_df


def TukeyHSD(df, fixed_col, dependent_col, verbose=False, alpha=0.05, effect_size_reverse=False):
    result = []
    tukey_result = pairwise_tukeyhsd(df[dependent_col], df[fixed_col], alpha=alpha)
    
    if verbose:
        print(tukey_result.summary())
        
    meandiffs = tukey_result.meandiffs
    groups = tukey_result.groupsunique
    
    combs = list(combinations(groups, 2))
    
    for i, comb in enumerate(combs):
        
        effect_size = cohen_d(df[df[fixed_col] == comb[0]][dependent_col], 
                              df[df[fixed_col] == comb[1]][dependent_col],
                              reverse=effect_size_reverse)
        
        result.append({'group1': comb[0],
                       'group2': comb[1],
                       'meandiff': meandiffs[i],
                       'p-adj': tukey_result.pvalues[i],
                       'lower': tukey_result.confint[i][0],
                       'upper': tukey_result.confint[i][1],
                       'reject': tukey_result.reject[i],
                       'effect_size': effect_size,
                       })
        
    result_df = pd.DataFrame(result)
    
    return result_df

def Fisher_z_transformation(r):
    return 0.5 * np.log((1 + r) / (1 - r - 1e-10))


def compare_Fisher_z(r1, r2, n1, n2):
    z_diff = Fisher_z_transformation(r1) - Fisher_z_transformation(r2)
    SE = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_score = z_diff / SE
    p_val = 2 * (1 - norm.cdf(abs(z_score)))
    
    return z_score, p_val


# Association between cognitive score and age
def cognitive_association(df, real_col, pred_col, cognitive_cols, correction_method=None, use_gap=False):
    total_df = pd.DataFrame()
    corrected_pred_age, corr_PAD = BiasCorrection(df, real_col, pred_col, method=correction_method)
    if use_gap:
        df['corr_pred_age'] = corrected_pred_age
        df['corr_delta'] = corr_PAD
        
    else:
        df['corr_pred_age'] = corrected_pred_age
    
    real_age_association = {}
    pred_age_association = {}
    
    real_p_val_list = []
    pred_p_val_list = []
    
    for cognitive_col in cognitive_cols:
        
        if use_gap:
            use_col = [real_col, pred_col, 'corr_delta', cognitive_col]
            
        else:
            use_col = [real_col, pred_col, 'corr_pred_age', cognitive_col]
            
        temp_df = df[use_col].copy()
        if use_gap:
            temp_df = temp_df.dropna(subset=[cognitive_col, real_col, 'corr_delta'])
        else:
            temp_df = temp_df.dropna(subset=[cognitive_col, real_col, 'corr_pred_age'])
        
        real_age = temp_df[real_col]
        
        if use_gap:
            pred_age = temp_df['corr_delta']
        else:
            pred_age = temp_df['corr_pred_age']
            
        cognitive_score = temp_df[cognitive_col]

        real_corr, real_p = spearmanr(cognitive_score, real_age)
        pred_corr, pred_p = spearmanr(cognitive_score, pred_age)
        
        real_p_val_list.append(real_p)
        pred_p_val_list.append(pred_p)
        
        fisher_z, _ = compare_Fisher_z(real_corr, pred_corr, len(temp_df), len(temp_df))
        
        real_age_association[cognitive_col] = {'r_s': round(real_corr ,3), 'p': round(real_p ,3), 'N': int(len(temp_df))}
        
        pred_age_association[cognitive_col] = {'r_s': round(pred_corr ,3), 'p': round(pred_p ,3), 'N': int(len(temp_df))}
        
        temp_df = pd.DataFrame({'Measure': 'CA', 
                                'r_s': round(real_corr ,3),
                                'p': round(real_p ,3),
                                "Fisher's z": round(fisher_z, 4),
                                'N': int(len(temp_df))}, index=[cognitive_col])
        
        total_df = pd.concat([total_df, temp_df], axis=0)
        
        temp_df = pd.DataFrame({'Measure': 'BA',
                                'r_s': round(pred_corr ,3),
                                'p': round(pred_p ,3),
                                "Fisher's z": None,
                                "N": None}, index=[cognitive_col])
        
        total_df = pd.concat([total_df, temp_df], axis=0)
    
    real_fdr = fdrcorrection(real_p_val_list, alpha=0.05)
    pred_fdr = fdrcorrection(pred_p_val_list, alpha=0.05)
    
    real_df = pd.DataFrame(real_age_association).T
    pred_df = pd.DataFrame(pred_age_association).T
    
    real_df['FDR'] = real_fdr[1].round(4)
    pred_df['FDR'] = pred_fdr[1].round(4)
    
    fdr_total_list = []
    for real, pred in zip(real_fdr[1], pred_fdr[1]):
        fdr_total_list.append(real)
        fdr_total_list.append(pred)
    
    total_df['Corr. p'] = fdr_total_list
        
    return df, real_df, pred_df, total_df
        
        
def association_with_covariate(corrected_df, covariate_cols, cognitive_cols,verbose=False,):
    
    result_df = pd.DataFrame()
    rest_df = pd.DataFrame()
    
    total_df = corrected_df.copy()
    total_df.rename({"Sex(1=m,2=f)": "Sex"}, inplace=True, axis=1)
    
    total_df['Age2'] = total_df['Age']**2
    total_df['Sex*Age'] = total_df['Sex'] * total_df['Age']
    total_df['PAD*Sex'] = total_df['corr_PAD'] * total_df['Sex']
    total_df['PAD*Age'] = total_df['corr_PAD'] * total_df['Age']
    
    cov_df = total_df[covariate_cols]
    
    for cog_col in cognitive_cols:
        
        filter_col = [cog_col]
        temp_df = cov_df.copy()
        temp_df = pd.concat([temp_df, total_df[filter_col]], axis=1)
        drop_na_list = covariate_cols + [cog_col] + ['corr_PAD']
        
        temp_df = temp_df.dropna(subset=drop_na_list, axis=0)
        temp_y_df = temp_df[cog_col]
        temp_df.drop([cog_col], axis=1, inplace=True)
        
        temp_df['intercept'] = 1
        
        model = sm.OLS(temp_y_df, temp_df).fit()

        if verbose:
            print(model.summary())
        
        summary_frame = model.summary2().tables[1]
        summary_df = summary_frame[['Coef.', 't', 'P>|t|', '[0.025', '0.975]']]
        
        save_col = ['Coef.', 't', 'p_val', '[0.025', '0.975]']
        save_value = list(summary_df.loc['corr_PAD'].values)
        
        if save_value[2] < 0.05:
            print(model.summary())
        
        save_col.insert(0, 'Measure')
        save_value.insert(0, cog_col)
        
        save_row = pd.DataFrame([save_value], columns=save_col)
        save_row['r'] = r_from_t(save_row['t'].values[0], len(temp_df) - 2)
        save_row['r2'] = r2_from_t(save_row['t'].values[0], len(temp_df) - 2)

        result_df = pd.concat([result_df, save_row], axis=0)

        summary_frame['Variable'] = summary_frame.index
        summary_frame = pd.concat([summary_frame.iloc[:, -1], summary_frame.iloc[:,:-1]], axis=1)

        dummy_col = summary_frame.columns
        dummy_row = pd.DataFrame(np.zeros((1, len(dummy_col)), dtype=np.float32), columns=dummy_col)
        summary_frame = pd.concat([summary_frame, dummy_row, dummy_row], axis=0)

        rest_df = pd.concat([rest_df, summary_frame], axis=0)
        
    result_df.reset_index(drop=True, inplace=True)
    rest_df.reset_index(drop=True, inplace=True)
    rest_df.replace(0, np.nan, inplace=True)
    
    p_val_list = result_df['p_val'].values
    fdr = fdrcorrection(p_val_list, alpha=0.05)
    result_df['FDR'] = fdr[1].round(4)
        
    return result_df, total_df, rest_df


def r2_from_t(t, df):
    return t**2 / (t**2 + df)


def r_from_t(t,df):
    return t / np.sqrt(t**2 + df)


def reproduciability(total_df): # Reference: https://github.com/AralRalud/BASE/blob/main/src/analysis/03_test_retest.py    
    
    # Difference between predicted age at visit 2 and visit 1
    # add visit number
    total_df['visit'] = total_df.groupby(['modality', 'repeat_idx', 'Subject'])['scan'].transform(lambda x: x.astype('category').cat.codes + 1)

    results_wide = total_df.pivot_table(index=['Subject', 'modality', 'repeat_idx',], columns='visit',
                                    values=['pred_age', 'scan'], aggfunc='first').reset_index()

    results_wide.columns = [f'{col}_{lvl}' if lvl else col for col, lvl in results_wide.columns.values]

    results_wide['diff'] = results_wide['pred_age_1'] - results_wide['pred_age_2']
    mean_diff = results_wide.groupby('modality')['diff'].mean()
    avg_std_diff = results_wide.groupby(['modality'])['diff'].std()

    # average standard deviation of predicted age per scan
    avg_sd_y_pred = total_df.groupby(['modality', 'scan', 'Subject', 'visit'])[['real_age','pred_age']].std().\
    reset_index().groupby(['modality'])['pred_age'].mean()

    paper_table = pd.concat((avg_sd_y_pred, mean_diff, avg_std_diff),  axis=1)
    paper_table.columns = ['avg_std_y_pred', 'mean_diff', 'std_diff']


    icc_diff_df = results_wide[['Subject', 'modality', 'repeat_idx', 'diff']]
    icc_diff_result = pg.intraclass_corr(icc_diff_df, targets='Subject', 
                                         ratings='diff', raters='modality')
    
    icc_pred_df = total_df[['Subject', 'modality', 'repeat_idx', 'pred_age']]
    icc_pred_y = pg.intraclass_corr(icc_pred_df, targets='Subject', 
                                    ratings='pred_age', raters='modality')
    
    return paper_table, results_wide, icc_diff_result, icc_pred_y


def longitudinal_integrate_repeat_df(BASEPATH, data, num_repeat, modalities=['sMRI', "dMRI_fa", "GsLd_fa"]):

    final_df = pd.DataFrame()

    use_subject_df = pd.read_csv(os.path.join(BASEPATH, 'slim_longitudinal_phenotype.csv'))
    use_subject_df['Subject'] = [f"sub-{sub}" for sub in use_subject_df['Subject'].values]
    use_subject_long1 = use_subject_df[use_subject_df['scan_order_1']==1]
    use_subject_long2 = use_subject_df[use_subject_df['scan_order_2']==2]
    use_subject_long3 = use_subject_df[use_subject_df['scan_order_2']==3]

    for modal in modalities:
        for rep in range(num_repeat):
            total_test_df = pd.DataFrame()

            seed = 142 + 10*rep
            long1 = pd.read_csv(os.path.join(BASEPATH, f'{data}_long1_{modal}_axial_{seed}.csv'))
            long2 = pd.read_csv(os.path.join(BASEPATH, f'{data}_long2_{modal}_axial_{seed}.csv'))
            long3 = pd.read_csv(os.path.join(BASEPATH, f'{data}_long3_{modal}_axial_{seed}.csv'))

            long1 = long1[long1['Subject'].isin(use_subject_long1['Subject'])]
            long2 = long2[long2['Subject'].isin(use_subject_long2['Subject'])]
            long3 = long3[long3['Subject'].isin(use_subject_long3['Subject'])]

            long2.rename({'pred_age': 'pred_age_2', 'real_age':'real_age_2'}, axis=1, inplace=True)
            long3.rename({'pred_age': 'pred_age_3', 'real_age':'real_age_3'}, axis=1, inplace=True)

            total_test_df = pd.concat([total_test_df, long1], axis=0)
            total_test_df.rename({'pred_age': 'pred_age_1', 'real_age':'real_age_1'}, axis=1, inplace=True)

            total_test_df = pd.merge(total_test_df, long2[['Subject', 'real_age_2', 'pred_age_2']], on='Subject', how='left')
            total_test_df = pd.merge(total_test_df, long3[['Subject', 'real_age_3', 'pred_age_3']], on='Subject', how='left')

            # total_test_df.rename({'pred_age': 'pred_age_2'}, axis=1, inplace=True)
            total_test_df.fillna(0, inplace=True)

            total_test_df['real_age_2'] = total_test_df['real_age_2'] + total_test_df['real_age_3']
            total_test_df['pred_age_2'] = total_test_df['pred_age_2'] + total_test_df['pred_age_3']

            # total_test_df.rename({'real_age_2_x': 'real_age_2', 'pred_age_2_x':'pred_age_2'}, axis=1, inplace=True)
            total_test_df.drop(['real_age_3', 'pred_age_3'], axis=1, inplace=True) 

            total_test_df.replace(0, np.nan, inplace=True)
            total_test_df.dropna(axis=0, inplace=True)

            total_test_df['repeat_idx'] = [rep] * len(total_test_df)
            total_test_df['modality'] = [modal] * len(total_test_df)
            try:
                total_test_df.drop(['Unnamed: 0'], axis=1, inplace=True)
            except:
                pass

            final_df = pd.concat([final_df, total_test_df], axis=0)

    return final_df


def Longitudinal_performance(total_df, interval_list, verbose=False):

    result_df = total_df.copy()
    result_df = result_df.groupby(['Subject', 'model']).mean()
    result_df.reset_index(inplace=True)

    real_gap = result_df['real_age_2'] - result_df['real_age_1']
    pred_gap = result_df['pred_age_2'] - result_df['pred_age_1']

    result_df['real_gap'] = real_gap
    result_df['pred_gap'] = pred_gap

    result_df['gap_diff'] = result_df['real_gap'] - result_df['pred_gap']
    result_df['abs_gap_diff'] = abs(result_df['real_gap'] - result_df['pred_gap'])

    MdE = result_df.groupby('model')['gap_diff'].mean()
    MAdE = result_df.groupby('model')['abs_gap_diff'].mean()
    MdE_std = result_df.groupby('model')['gap_diff'].std()
    MAdE_std = result_df.groupby('model')['abs_gap_diff'].std()

    Center_mMAE = []
    FedAvg_mMAE = []
    FedProx_mMAE = []
    MOON_mMAE = []

    for i in range(len(interval_list)):
        
        if i == 0:
            temp_df = result_df[result_df['real_age_1'] < interval_list[i]]
            if verbose:
                print(f"Interval  < {interval_list[i]} : {len(temp_df)}")
        else:
            temp_df = result_df[(result_df['real_age_1'] >= interval_list[i-1]) & (result_df['real_age_1'] < interval_list[i])]
            if verbose:
                print(f"Interval {interval_list[i-1]} ~ {interval_list[i]} : {len(temp_df)}")

        interval_mae_df = temp_df.groupby('model')['abs_gap_diff'].mean()
        Center_mMAE.append(interval_mae_df['Center'])
        FedAvg_mMAE.append(interval_mae_df['FedAvg'])
        FedProx_mMAE.append(interval_mae_df['FedProx'])
        MOON_mMAE.append(interval_mae_df['MOON'])

    Center_mMAE = np.array(Center_mMAE)
    FedAvg_mMAE = np.array(FedAvg_mMAE)
    FedProx_mMAE = np.array(FedProx_mMAE)
    MOON_mMAE = np.array(MOON_mMAE)

    Center_mMAE = np.max(Center_mMAE)
    FedAvg_mMAE = np.max(FedAvg_mMAE)
    FedProx_mMAE = np.max(FedProx_mMAE)
    MOON_mMAE = np.max(MOON_mMAE)

    return  pd.DataFrame({"MdE": [MdE['Center'], MdE['FedAvg'], MdE['FedProx'], MdE['MOON']],
                          "MdE_std": [MdE_std['Center'], MdE_std['FedAvg'], MdE_std['FedProx'], MdE_std['MOON']],
                          "MAdE": [MAdE['Center'], MAdE['FedAvg'], MAdE['FedProx'], MAdE['MOON']],
                          "MAdE_std": [MAdE_std['Center'], MAdE_std['FedAvg'], MAdE_std['FedProx'], MAdE_std['MOON']],
                          "mMAdE": [Center_mMAE, FedAvg_mMAE, FedProx_mMAE, MOON_mMAE]}, 
                          index=['Center', 'FedAvg', 'FedProx', "MOON"]), result_df
    


def mMAE(df, real_col, pred_col, interval_list, verbose=False): # Robustness
    
    MAE_list = []
    
    for i in range(len(interval_list)):
        
        if i == 0: # First Interval 
            temp_df = df[df[real_col] < interval_list[i]]
            if verbose:
                print(f"Interval  < {interval_list[i]} : {len(temp_df)}")
            
        else:
            temp_df = df[(df[real_col] >= interval_list[i-1]) & (df[real_col] < interval_list[i])]
            if verbose:
                print(f"Interval {interval_list[i-1]} ~ {interval_list[i]} : {len(temp_df)}")
            
        temp_MAE = np.mean(np.abs(temp_df[real_col] - temp_df[pred_col]))
        
        MAE_list.append(temp_MAE)
    
    MAE_list = np.array(MAE_list)    
    mMAE = np.max(MAE_list)
    
    return mMAE