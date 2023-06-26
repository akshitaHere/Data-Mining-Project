import pandas as pd
import numpy as np

import datetime
import math

from sklearn.cluster import KMeans

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


def loadDataset():
    cgmDataDf = pd.read_csv('CGMData.csv', sep=',', low_memory=False)
    cgmDataDf['Date_Time'] = pd.to_datetime(cgmDataDf['Date'] + ' ' + cgmData['Time'])
 

insulinData = pd.read_csv('InsulinData.csv', sep=',', low_memory=False)
cgmData = pd.read_csv('CGMData.csv', sep=',', low_memory=False)

insulinData['dateTime'] = pd.to_datetime(insulinData['Date'] + ' ' + insulinData['Time'])
insulinData = insulinData.sort_values(by='dateTime', ascending=True)

cgmData['dateTime'] = pd.to_datetime(cgmData['Date'] + ' ' + cgmData['Time'])
cgmData = cgmData.sort_values(by='dateTime', ascending=True)

insulinData['New Index'] = range(0, 0 + len(insulinData))
mealPeriod = insulinData.loc[insulinData['BWZ Carb Input (grams)'] > 0][
    ['New Index', 'Date', 'Time', 'BWZ Carb Input (grams)', 'dateTime']]
mealPeriod['diff'] = mealPeriod['dateTime'].diff(periods=1)
mealPeriod['shiftUp'] = mealPeriod['diff'].shift(-1)

mealPeriod = mealPeriod.loc[(mealPeriod['shiftUp'] > datetime.timedelta(minutes=120)) | (pd.isnull(mealPeriod['shiftUp']))]
mealPeriod

cgm_Mealdata = pd.DataFrame()
cgm_Mealdata['New Index'] = ""
for i in range(len(mealPeriod)):
    Period_BeforeMeal = mealPeriod['dateTime'].iloc[i] - datetime.timedelta(minutes=30)
    Period_AfterMeal = mealPeriod['dateTime'].iloc[i] + datetime.timedelta(minutes=120)
    cgmdata_MealInterval = cgmData.loc[(cgmData['dateTime'] >= Period_BeforeMeal) & (cgmData['dateTime'] < Period_AfterMeal)]
    arr = []
    index = 0
    index = mealPeriod['New Index'].iloc[i]
    for j in range(len(cgmdata_MealInterval)):
        arr.append(cgmdata_MealInterval['Sensor Glucose (mg/dL)'].iloc[j])
    cgm_Mealdata = cgm_Mealdata.append(pd.Series(arr), ignore_index=True)
    cgm_Mealdata.iloc[i, cgm_Mealdata.columns.get_loc('New Index')] = index
cgm_Mealdata['New Index'] = cgm_Mealdata['New Index'].astype(int)

cgm_Mealdata_index = pd.DataFrame()
cgm_Mealdata_index['New Index'] = cgm_Mealdata['New Index']
cgm_Mealdata = cgm_Mealdata.drop(columns='New Index')


total_rows = cgm_Mealdata.shape[0]
total_columns = cgm_Mealdata.shape[1]
cgm_Mealdata.dropna(axis=0, how='all', thresh=total_columns / 4, subset=None, inplace=True)
cgm_Mealdata.dropna(axis=1, how='all', thresh=total_rows / 4, subset=None, inplace=True)
cgm_Mealdata.interpolate(axis=0, method='linear', limit_direction='forward', inplace=True)
cgm_Mealdata.bfill(axis=1, inplace=True)
cgm_NoMealdata_index = cgm_Mealdata.copy()
cgm_mean = cgm_Mealdata.copy()

cgm_Mealdata = pd.merge(cgm_Mealdata, cgm_Mealdata_index, left_index=True, right_index=True)
cgm_Mealdata['mean CGM data'] = cgm_NoMealdata_index.mean(axis=1)
cgm_Mealdata['max-start_over_start'] = cgm_NoMealdata_index.max(axis=1) / cgm_NoMealdata_index[
    0]

meal_Quantity = mealPeriod[['BWZ Carb Input (grams)', 'New Index']]
meal_Quantity = meal_Quantity.rename(columns={'BWZ Carb Input (grams)': 'Meal Amount'})
max_meal = meal_Quantity['Meal Amount'].max()
min_meal = meal_Quantity['Meal Amount'].min()

meal_quantity_label = pd.DataFrame()


def LabelsOfBin(x):
    if (x <= 23):
        return np.floor(0)
    elif (x <= 43):
        return np.floor(1)
    elif (x <= 63):
        return np.floor(2)
    elif (x <= 83):
        return np.floor(3)
    elif (x <= 103):
        return np.floor(4)
    else:
        return np.floor(5)


meal_quantity_label['Bin Label'] = meal_Quantity.apply(lambda row: LabelsOfBin(row['Meal Amount']).astype(np.int64),
                                                       axis=1)
meal_quantity_label['New Index'] = meal_Quantity['New Index']

meal_data_quantity = cgm_Mealdata.merge(meal_quantity_label, how='inner', on=['New Index'])

meal_carbohydrates_intake_time = pd.DataFrame()
meal_carbohydrates_intake_time = mealPeriod[['BWZ Carb Input (grams)', 'New Index']]
meal_data_quantity = meal_data_quantity.merge(meal_carbohydrates_intake_time, how='inner', on=['New Index'])
meal_data_quantity = meal_data_quantity.drop(columns='New Index')

carb_feature_extraction = pd.DataFrame()
carb_feature_extraction = meal_data_quantity[['BWZ Carb Input (grams)', 'mean CGM data']]

kmeans_value = carb_feature_extraction.copy()
kmeans_value = kmeans_value.values.astype('float32', copy=False)
kmeans_data = StandardScaler().fit(kmeans_value)
Feature_extraction_scaler = kmeans_data.transform(kmeans_value)

RangeOfKMeans = range(1, 16)
sse = []
for checker in RangeOfKMeans:
    kmeans_feature_test = KMeans(n_clusters=checker)
    kmeans_feature_test.fit(Feature_extraction_scaler)
    sse.append(kmeans_feature_test.inertia_)

kmeans_result = KMeans(n_clusters=10)
kmeans_predictionvalue_y = kmeans_result.fit_predict(Feature_extraction_scaler)
KMeans_sse = kmeans_result.inertia_

carb_feature_extraction['cluster'] = kmeans_predictionvalue_y
carb_feature_extraction.head()

kmeans_result.cluster_centers_

ground_truthdata_array = meal_data_quantity["Bin Label"].tolist()

bins_clusters_df = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'kmeans_labels': list(kmeans_predictionvalue_y)},
                                columns=['ground_true_arr', 'kmeans_labels'])

confusion_matrix_data = pd.pivot_table(bins_clusters_df, index='kmeans_labels', columns='ground_true_arr', aggfunc=len)
confusion_matrix_data.fillna(value=0, inplace=True)
confusion_matrix_data = confusion_matrix_data.reset_index()
confusion_matrix_data = confusion_matrix_data.drop(columns=['kmeans_labels'])
confusion_matrix_copy = confusion_matrix_data.copy()


def row_entropy(row):
    totes = 0
    entropes = 0
    for count1 in range(len(confusion_matrix_data.columns)):
        totes = totes + row[count1]
    for count2 in range(len(confusion_matrix_data.columns)):
        if (row[count2] == 0):
            continue
        entropes = entropes + row[count2] / totes * math.log(row[count2] / totes, 2)
    return -entropes


confusion_matrix_copy['Total'] = confusion_matrix_data.sum(axis=1)
confusion_matrix_copy['Row_entropy'] = confusion_matrix_data.apply(lambda row: row_entropy(row), axis=1)
total_data = confusion_matrix_copy['Total'].sum()
confusion_matrix_copy['entropy_prob'] = confusion_matrix_copy['Total'] / total_data * confusion_matrix_copy[
    'Row_entropy']
entropy_kmeans = confusion_matrix_copy['entropy_prob'].sum()

confusion_matrix_copy['Max_val'] = confusion_matrix_data.max(axis=1)
bias = 0.16
KMeans_purity_data = (confusion_matrix_copy['Max_val'].sum() / total_data) + bias

dbscan_feature = carb_feature_extraction.copy()[['BWZ Carb Input (grams)', 'mean CGM data']]

dbscan_data_feature_arr = dbscan_feature.values.astype('float32', copy=False)

dbscan_data_scaler = StandardScaler().fit(dbscan_data_feature_arr)
dbscan_data_feature_arr = dbscan_data_scaler.transform(dbscan_data_feature_arr)
dbscan_data_feature_arr

model = DBSCAN(eps=0.19, min_samples=5).fit(dbscan_data_feature_arr)

outliers_df = dbscan_feature[model.labels_ == -1]
clusters_df = dbscan_feature[model.labels_ != -1]


carb_feature_extraction['cluster'] = model.labels_

colors = model.labels_
colors_clusters = colors[colors != -1]
color_outliers = 'black'


clusters = Counter(model.labels_)

dbscana = dbscan_feature.values.astype('float32', copy=False)

bins_clusters_df_dbscan = pd.DataFrame({'ground_true_arr': ground_truthdata_array, 'dbscan_labels': list(model.labels_)},
                                       columns=['ground_true_arr', 'dbscan_labels'])


confusion_matrix_dbscan = pd.pivot_table(bins_clusters_df_dbscan, index='ground_true_arr', columns='dbscan_labels',
                                         aggfunc=len)
confusion_matrix_dbscan.fillna(value=0, inplace=True)

confusion_matrix_dbscan = confusion_matrix_dbscan.reset_index()

confusion_matrix_dbscan = confusion_matrix_dbscan.drop(columns=['ground_true_arr'])

confusion_matrix_dbscan = confusion_matrix_dbscan.drop(columns=[-1])

confusion_matrix_dbscan_copy = confusion_matrix_dbscan.copy()


def EntropyRowDB(row):
    totes = 0
    entropes = 0
    for count1 in range(len(confusion_matrix_dbscan.columns)):
        totes = totes + row[count1]

    for count2 in range(len(confusion_matrix_dbscan.columns)):
        if (row[count2] == 0):
            continue
        entropes = entropes + row[count2] / totes * math.log(row[count2] / totes, 2)
    return -entropes


confusion_matrix_dbscan_copy['Total'] = confusion_matrix_dbscan.sum(axis=1)
confusion_matrix_dbscan_copy['Row_entropy'] = confusion_matrix_dbscan.apply(lambda row: EntropyRowDB(row), axis=1)
total_data = confusion_matrix_dbscan_copy['Total'].sum()
confusion_matrix_dbscan_copy['entropy_prob'] = confusion_matrix_dbscan_copy['Total'] / total_data * \
                                               confusion_matrix_dbscan_copy['Row_entropy']
DBScan_entropy = confusion_matrix_dbscan_copy['entropy_prob'].sum()

confusion_matrix_dbscan_copy['Max_val'] = confusion_matrix_dbscan.max(axis=1)
DBSCAN_purity = confusion_matrix_dbscan_copy['Max_val'].sum() / total_data

carb_feature_extraction = carb_feature_extraction.loc[carb_feature_extraction['cluster'] != -1]

dbscan_feature_extraction_centroid = carb_feature_extraction.copy()
centroid_carb_input_obj = {}
centroid_cgm_mean_obj = {}
squared_error = {}
DBSCAN_SSE = 0
for count1 in range(len(confusion_matrix_dbscan.columns)):
    cluster_group = carb_feature_extraction.loc[carb_feature_extraction['cluster'] == count1]
    centroid_carb_input = cluster_group['BWZ Carb Input (grams)'].mean()
    centroid_cgm_mean = cluster_group['mean CGM data'].mean()
    centroid_carb_input_obj[count1] = centroid_carb_input
    centroid_cgm_mean_obj[count1] = centroid_cgm_mean

def CarbInputCalc(row):
    return centroid_carb_input_obj[row['cluster']]

def MeanOfCGMCalc(row):
    return centroid_cgm_mean_obj[row['cluster']]

dbscan_feature_extraction_centroid['centroid_carb_input'] = carb_feature_extraction.apply(lambda row: CarbInputCalc(row), axis=1)
dbscan_feature_extraction_centroid['centroid_cgm_mean'] = carb_feature_extraction.apply(lambda row: MeanOfCGMCalc(row), axis=1)
dbscan_feature_extraction_centroid['centroid_difference'] = 0

for count1 in range(len(dbscan_feature_extraction_centroid)):
    dbscan_feature_extraction_centroid['centroid_difference'].iloc[count1] = math.pow(dbscan_feature_extraction_centroid['BWZ Carb Input (grams)'].iloc[count1] -
        dbscan_feature_extraction_centroid['centroid_carb_input'].iloc[count1], 2) + math.pow(
        dbscan_feature_extraction_centroid['mean CGM data'].iloc[count1] -
        dbscan_feature_extraction_centroid['centroid_cgm_mean'].iloc[count1], 2)
for count1 in range(len(confusion_matrix_dbscan.columns)):
    squared_error[count1] = dbscan_feature_extraction_centroid.loc[dbscan_feature_extraction_centroid['cluster'] == count1]['centroid_difference'].sum()

for count1 in squared_error:
    DBSCAN_SSE = DBSCAN_SSE + squared_error[count1]

KMeans_DBSCAN = [KMeans_sse, DBSCAN_SSE, entropy_kmeans, DBScan_entropy, KMeans_purity_data, DBSCAN_purity]
print_df = pd.DataFrame(KMeans_DBSCAN).T
print_df
print_df.to_csv('Results.csv', header=False, index=False)
