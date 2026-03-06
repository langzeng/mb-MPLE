import numpy as np
import pandas as pd

pheno_c1 = pd.read_csv('phs000001.v3.pht000375.v2.p1.c1.fundus.EDO.txt',skiprows=12,sep='\t')
pheno_c2 = pd.read_csv('phs000001.v3.pht000375.v2.p1.c2.fundus.GRU.txt',skiprows=12,sep='\t')

pheno_c2.columns.to_list() == pheno_c1.columns.to_list()  #good!

fundus = pd.concat([pheno_c1,pheno_c2])
fundus = fundus.drop_duplicates()
fundus = fundus.applymap(str)

eye_column = []
base_column = []
for i in fundus.columns.to_list():
    if np.logical_and(('LE' in i),(i != "SCALE")):  # consider "REDRRETI" and "LEDRRETI"
        new_i = i.replace("LE", "", 1)
        eye_column.append(new_i)
    elif ('RE' in i):
        new_i = i.replace("RE","",1)
        eye_column.append(new_i)
    else:
        base_column.append(i)

eye_column = list(set(eye_column)) # drop duplicates in list

df = fundus[base_column]
df = pd.concat([df,df])#.applymap(str)
df = df.sort_values(by=['ID2','VISNO'])
df.reset_index(drop=True,inplace=True)
df['eye'] = pd.DataFrame(['LE','RE']*39797)


for i in eye_column:
    df[i] = ""
    print(i)
    for j in range(df.shape[0]):
        print(j)
        eye = df['eye'].iloc[j]
        if i != 'AMDSEV':
            ph = eye + i
        else:
            ph = i + eye
        visno = df['VISNO'].iloc[j]
        id2 = df['ID2'].iloc[j]
        df[i].iloc[j] = fundus.loc[np.logical_and(fundus['VISNO']==visno,fundus['ID2']==id2),ph].values[0]

df.to_csv('orig_master_pheno.txt',index=False,sep="\t",header=True)


enrollment_c1 = pd.read_csv('phs000001.v3.pht000373.v2.p1.c1.enrollment_randomization.EDO.txt',skiprows=10,sep='\t')
enrollment_c2 = pd.read_csv('phs000001.v3.pht000373.v2.p1.c2.enrollment_randomization.GRU.txt',skiprows=10,sep='\t')
enrollment_c2.columns.to_list() == enrollment_c1.columns.to_list()

enrollment = pd.concat([enrollment_c1,enrollment_c2])
enrollment = enrollment.drop_duplicates()
enrollment = enrollment.applymap(str)

new_enrollment = enrollment[['ID2','SEX','RACE','ENROLLAGE']]

new_df = pd.merge(df,new_enrollment,how='left',left_on='ID2',right_on='ID2')
new_df.to_csv('longitudinal_master_pheno.txt',index=False,sep="\t",header=True)
