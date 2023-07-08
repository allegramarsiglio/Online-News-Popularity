#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:08:03 2023

@author: allegramarsiglio
"""

from __future__ import print_function

import re
import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
#from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, BooleanType, DoubleType

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.ml.feature import StringIndexer

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor

from pyspark.ml import Pipeline

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator




# If needed, use this helper function
# You can implement your own version if you find it more appropriate 

if __name__ == "__main__":

    sc = SparkContext(appName="Assignment-5")
    spark = SparkSession.builder.getOrCreate()
    
    
    ### READ THE DATA
    
    file_path_train = "/Users/allegramarsiglio/Downloads/OnlineNewsPopularity/OnlineNewsPopularity.csv"
    
    df = spark.read.format('csv').options(header='true', 
                                          inferSchema='true',  
                                          sep =",").load(file_path_train)
    
    # Select only the features I am interested in
    df = df.select('url', 
                   ' n_tokens_title', 
                   ' n_tokens_content',
                   ' n_unique_tokens', 
                   ' n_non_stop_words', 
                   ' n_non_stop_unique_tokens',
                   ' num_hrefs', 
                   ' num_imgs', 
                   ' num_videos',
                   ' average_token_length', 
                   ' num_keywords', 
                   ' data_channel_is_lifestyle',
                   ' data_channel_is_entertainment', 
                   ' data_channel_is_bus',
                   ' data_channel_is_socmed', 
                   ' data_channel_is_tech',
                   ' data_channel_is_world', 
                   ' kw_min_min', 
                   ' kw_max_min', 
                   ' kw_avg_min',
                   ' kw_min_max', 
                   ' kw_max_max', 
                   ' kw_avg_max', 
                   ' kw_min_avg',
                   ' kw_max_avg', 
                   ' kw_avg_avg', 
                   ' weekday_is_monday', 
                   ' weekday_is_tuesday', 
                   ' weekday_is_wednesday',
                   ' weekday_is_thursday', 
                   ' weekday_is_friday', 
                   ' weekday_is_saturday',
                   ' weekday_is_sunday', 
                   ' global_subjectivity',
                   ' global_sentiment_polarity', 
                   ' title_subjectivity',
                   ' title_sentiment_polarity', 
                   ' shares')
    
    
    print(df.printSchema())
    
    # Check number of rows and features
    print('\nBEFORE PREPROCESSING:')
    n_rows = df.count()
    print('- Number of articles:', n_rows)
    n_col = len(df.columns)
    print('- Number of features:', n_col, '\n')
    
    df.describe(' shares').show()
    
    # Change datatypes
    df = df.withColumn(' n_tokens_title', df[' n_tokens_title'].cast('Integer'))
    df = df.withColumn(' n_tokens_content', df[' n_tokens_content'].cast('Integer'))
    df = df.withColumn(' num_hrefs', df[' num_hrefs'].cast('Integer'))
    df = df.withColumn(' num_imgs', df[' num_imgs'].cast('Integer'))
    df = df.withColumn(' num_videos', df[' num_videos'].cast('Integer'))
    df = df.withColumn(' num_keywords', df[' num_keywords'].cast('Integer'))
    df = df.withColumn(' data_channel_is_lifestyle', df[' data_channel_is_lifestyle'].cast('Integer'))
    df = df.withColumn(' data_channel_is_entertainment', df[' data_channel_is_entertainment'].cast('Integer'))
    df = df.withColumn(' data_channel_is_bus', df[' data_channel_is_bus'].cast('Integer'))
    df = df.withColumn(' data_channel_is_socmed', df[' data_channel_is_socmed'].cast('Integer'))
    df = df.withColumn(' data_channel_is_tech', df[' data_channel_is_tech'].cast('Integer'))
    df = df.withColumn(' data_channel_is_world', df[' data_channel_is_world'].cast('Integer'))
    df = df.withColumn(' kw_min_min', df[' kw_min_min'].cast('Integer'))
    df = df.withColumn(' kw_max_min', df[' kw_max_min'].cast('Integer'))
    df = df.withColumn(' kw_avg_min', df[' kw_avg_min'].cast('Integer'))
    df = df.withColumn(' kw_min_max', df[' kw_min_max'].cast('Integer'))
    df = df.withColumn(' kw_max_max', df[' kw_max_max'].cast('Integer'))
    df = df.withColumn(' kw_avg_max', df[' kw_avg_max'].cast('Integer'))
    df = df.withColumn(' kw_min_avg', df[' kw_min_avg'].cast('Integer'))
    df = df.withColumn(' kw_max_avg', df[' kw_max_avg'].cast('Integer'))
    df = df.withColumn(' kw_avg_avg', df[' kw_avg_avg'].cast('Integer'))
    df = df.withColumn(' weekday_is_monday', df[' weekday_is_monday'].cast('Integer'))
    df = df.withColumn(' weekday_is_tuesday', df[' weekday_is_tuesday'].cast('Integer'))
    df = df.withColumn(' weekday_is_wednesday', df[' weekday_is_wednesday'].cast('Integer'))
    df = df.withColumn(' weekday_is_thursday', df[' weekday_is_thursday'].cast('Integer'))
    df = df.withColumn(' weekday_is_friday', df[' weekday_is_friday'].cast('Integer'))
    df = df.withColumn(' weekday_is_saturday', df[' weekday_is_saturday'].cast('Integer'))
    df = df.withColumn(' weekday_is_sunday', df[' weekday_is_sunday'].cast('Integer'))
    
    # Check format
    print(df.select(' n_tokens_title', ' data_channel_is_lifestyle', ' data_channel_is_entertainment').show())
    
    # Count of null values for all columns
    print('\nNULL-ENTRIES CHECK:')
    for c in df.columns:
        print(f"Number of null entries in {c}:", df.where(F.col(c).isNull()).count())
    # 0 null in all comuns --> OK
    
    
    # Check if zero entries where there should not be
    print('\nZERO-ENTRIES CHECK:')
    print("n_tokens_title == 0:", df.where(F.col(' n_tokens_title') == 0).count()) # 0 --> OK
    print("n_tokens_content == 0:", df.where(F.col(' n_tokens_content') == 0).count()) # 1181
    print("n_non_stop_words == 0:", df.where(F.col(' n_non_stop_words') == 0).count()) # 1181 (assuming is because of no content)
    print("num_keywords == 0:", df.where(F.col(' num_keywords') == 0).count()) # 0 --> OK
    print("shares == 0:", df.where(F.col(' shares') == 0).count()) # 0 --> OK
    
    print("kw_min_min == 0:", df.where(F.col(' kw_min_min') == 0).count()) # 79
    print("kw_max_min == 0:", df.where(F.col(' kw_max_min') == 0).count()) # 819
    print("kw_avg_min == 0:", df.where(F.col(' kw_avg_min') == 0).count()) # 268
    print("kw_min_max == 0:", df.where(F.col(' kw_min_max') == 0).count()) # 17108
    print("kw_max_max == 0:", df.where(F.col(' kw_max_max') == 0).count()) # 79
    print("kw_avg_max == 0:", df.where(F.col(' kw_avg_max') == 0).count()) # 79 
    print("kw_min_avg == 0:", df.where(F.col(' kw_min_avg') == 0).count()) # 17102
    print("kw_max_avg == 0:", df.where(F.col(' kw_max_avg') == 0).count()) # 79
    print("kw_avg_avg == 0:", df.where(F.col(' kw_avg_avg') == 0).count()) # 79
    
    print('----')
    print("n_tokens_content < 10:", df.where(F.col(' n_tokens_content') < 10 ).count()) # 1181
    
    if df.where(F.col(' n_tokens_content') == 0).count() > 0:
        print('--> eliminate articles with no content')
        df = df.filter(F.col(' n_tokens_content') > 10)
        print("--> n_tokens_title == 0:", df.where(F.col(' n_tokens_title') == 0).count())
    
    print('----')
    
    print('\nAFTER PREPROCESSING:')
    n_rows = df.count()
    print('- Number of articles:', n_rows)
    n_col = len(df.columns)
    print('- Number of features:', n_col, '\n')
    
    
    ### CREATE FEATURES COLUMNS
    
    # Define columns by kind
    numeric_cols = [' n_tokens_title', 
                        ' n_tokens_content',
                        ' n_unique_tokens', 
                        ' n_non_stop_words', 
                        ' n_non_stop_unique_tokens',
                        ' num_hrefs', ' num_imgs', 
                        ' num_videos',
                        ' average_token_length', 
                        ' num_keywords', 
                        ' kw_min_min', ' kw_max_min', 
                        ' kw_avg_min',
                        ' kw_min_max', ' kw_max_max', 
                        ' kw_avg_max', 
                        ' kw_min_avg',
                        ' kw_max_avg', ' kw_avg_avg', 
                        ' global_subjectivity',
                        ' global_sentiment_polarity', 
                        ' title_subjectivity',
                        ' title_sentiment_polarity']
    
    categorical_cols_dummies = [' data_channel_is_lifestyle',
                                ' data_channel_is_entertainment', 
                                ' data_channel_is_bus',
                                ' data_channel_is_socmed', 
                                ' data_channel_is_tech',
                                ' data_channel_is_world', 
                                ' weekday_is_monday', 
                                ' weekday_is_tuesday', 
                                ' weekday_is_wednesday',
                                ' weekday_is_thursday', 
                                ' weekday_is_friday', 
                                ' weekday_is_saturday',
                                ' weekday_is_sunday']
    
    target_col = [' shares']
    
    # Get category column from dummies
    df = df.withColumn('data_channel', 
                       F.when(F.col(" data_channel_is_lifestyle") == 1, 'lifestyle').
                         when(F.col(" data_channel_is_entertainment") == 1, 'entertainment').
                         when(F.col(" data_channel_is_bus") == 1, 'bus').
                         when(F.col(" data_channel_is_socmed") == 1, 'socmed').
                         when(F.col(" data_channel_is_tech") == 1, 'tech').
                         when(F.col(" data_channel_is_world") == 1, 'world'))
    
    df = df.withColumn('weekday', 
                       F.when(df[" weekday_is_monday"] == 1, 'monday').
                         when(df[" weekday_is_tuesday"] == 1, 'tuesday').
                         when(df[" weekday_is_wednesday"] == 1, 'wednesday').
                         when(df[" weekday_is_thursday"] == 1, 'thursday').
                         when(df[" weekday_is_friday"] == 1, 'friday').
                         when(df[" weekday_is_saturday"] == 1, 'saturday').
                         when(df[" weekday_is_sunday"] == 1, 'sunday'))
    
    print(df.select(' n_tokens_title', ' data_channel_is_lifestyle', 'data_channel', 'weekday').show())
    
    categorical_cols = ['data_channel', 'weekday']
    
    
    # Define Scaler for numeric features
    scaler = StandardScaler(inputCol="numeric_features",
                            outputCol="scaled_numeric_features",
                            withMean=True,
                            withStd=True)
    
    # Define Indexers
    # - Index 1: 'data_channel' into 'data_channel_index'
    index1 = StringIndexer().setInputCol("data_channel").setOutputCol("data_channelIndex").setHandleInvalid("keep")
    
    # - Index 2: 'weekday' into 'weekday_index'
    index2 = StringIndexer().setInputCol("weekday").setOutputCol("weekdayIndex").setHandleInvalid("keep")
    
    # Define Assemblers 
    # - Assembler 1: numeric_cols into numeric_features (needed for scaling)
    assembler1 = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    
    # - Assembler 2: numeric_features_scaled + 'data_channel_index' + 'weekday_index'
    assembler2 = VectorAssembler(inputCols=('scaled_numeric_features', 'data_channelIndex', 'weekdayIndex'), 
                                outputCol=('scaled_features'))
    
    # - Assembler 3: categorical_cols_dummies into categorical_features_dummies
    assembler3 = VectorAssembler(inputCols=(categorical_cols_dummies), 
                                outputCol=('categorical_features_dummies'))
    
    # - Assembler 4: numeric_features_scaled + categorical_features_dummies
    #assembler4 = VectorAssembler(inputCols=('scaled_numeric_features', 'categorical_features_dummies'), 
    #                            outputCol=('scaled_features'))
    assembler4 = VectorAssembler(inputCols=('scaled_numeric_features', 'categorical_features_dummies'), 
                                outputCol=('scaled_features'))
    
    
    # PIPELINES
    #
    # - Pipeline Type 1:
    #    1. Assembler 1 (numeric_features)
    #    2. Scaler (numeric_features_scaled)
    #    3. Index 1 (data_channel_index)
    #    4. Index 2 (weekday_index)
    #    5. Assembler 2 (all_features_scaled)
    #    6. Model
    #    --> Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, model])
    #
    # - Pipeline Type 2:
    #    1. Assembler 1 (numeric_features)
    #    2. Scaler (numeric_features_scaled)
    #    3. Assembler 3 (categorical_features_dummies)
    #    4. Assembler 4 (scaled_features_wDummies)
    #    5. Model
    #    --> Pipeline(stages=[assembler1, scaler, assembler3, assembler4, model])
    
    
    ### MODELLING
    
    # LINEAR REGRESSION
    print('\n\nLINEAR REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=1)
    
    lr = LinearRegression(featuresCol='scaled_features', labelCol=' shares', maxIter=100, regParam=0)
    
    pipeline_lr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, lr])

    model_lr = pipeline_lr.fit(train_df)
    predictions_lr = model_lr.transform(test_df)
    
    print("Coefficients: " + str(model_lr.stages[-1].coefficients))
    print("Intercept: " + str(model_lr.stages[-1].intercept))
    
    evaluator_rmse_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_lr = evaluator_rmse_lr.evaluate(predictions_lr)
    r2_lr = evaluator_r2_lr.evaluate(predictions_lr)
    
    print(f"\nRoot Mean Squared Error (RMSE) = {rmse_lr}")
    print(f"R2 = {r2_lr}")
    
    
    
    # GENERALIZED LINEAR REGRESSION
    print('\n\nGENERALIZED LINEAR REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=2)
    
    glr = GeneralizedLinearRegression(featuresCol='scaled_features', labelCol=' shares', 
                                      family="gaussian", link="identity", 
                                      maxIter=10, regParam=0)

    pipeline_glr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, glr])
    
    model_glr = pipeline_glr.fit(train_df)
    predictions_glr = model_glr.transform(test_df)
    
    print("Coefficients: " + str(model_glr.stages[-1].coefficients))
    print("Intercept: " + str(model_glr.stages[-1].intercept))
    
    
    evaluator_rmse_glr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_glr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_glr = evaluator_rmse_lr.evaluate(predictions_glr)
    r2_glr = evaluator_r2_lr.evaluate(predictions_glr)
    
    print(f"\nRoot Mean Squared Error (RMSE) = {rmse_glr}")
    print(f"R2 = {r2_glr}")
    
    
    
    # DECISION TREE REGRESSION
    print('\n\nDECISION TREE REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=4)
    
    dt = DecisionTreeRegressor(featuresCol="scaled_features", labelCol=' shares')

    pipeline_dt = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, dt])

    model_dt = pipeline_dt.fit(train_df)
    predictions_dt = model_dt.transform(test_df)
    
    evaluator_rmse_dt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_dt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_dt = evaluator_rmse_dt.evaluate(predictions_dt)
    r2_dt = evaluator_r2_dt.evaluate(predictions_dt)
    
    print(f"Root Mean Squared Error (RMSE) = {rmse_dt}")
    print(f"R2 = {r2_dt}")
    
    
    
    # RANDOM FOREST REGRESSION
    print('\n\nRANDOM FOREST REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=5)
    
    rf = RandomForestRegressor(featuresCol="scaled_features", labelCol=' shares')
    
    pipeline_rf = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, rf])

    model_rf = pipeline_rf.fit(train_df)
    predictions_rf = model_rf.transform(test_df)
    
    evaluator_rmse_rf = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_rf = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_rf = evaluator_rmse_rf.evaluate(predictions_rf)
    r2_rf = evaluator_r2_rf.evaluate(predictions_rf)
    
    print(f"Root Mean Squared Error (RMSE) = {rmse_rf}")
    print(f"R2 = {r2_rf}")
    
    
    
    # GRADIENT BOOSTED TREE REGRESSION
    print('\n\nGRADIENT BOOSTED TREE REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=6)
    
    gbt = GBTRegressor(featuresCol="scaled_features", labelCol=' shares', maxIter=10)
    
    pipeline_gbt = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, gbt])

    model_gbt = pipeline_gbt.fit(train_df)
    predictions_gbt = model_gbt.transform(test_df)
    
    evaluator_rmse_gbt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_gbt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_gbt = evaluator_rmse_rf.evaluate(predictions_gbt)
    r2_gbt = evaluator_r2_rf.evaluate(predictions_gbt)
    
    print(f"Root Mean Squared Error (RMSE) = {rmse_gbt}")
    print(f"R2 = {r2_gbt}")
    
    
    
    
    ### FEATURES SELECTION
    print('\n\nMODELLING AFTER FEATURE SELECTION')
    
    # UnivariateFeatureSelector
    selector = UnivariateFeatureSelector(featuresCol="scaled_features", 
                                         outputCol="selectedFeatures",
                                         labelCol=" shares", selectionMode="fpr")
    selector.setFeatureType("continuous").setLabelType("continuous")

    
    # LINEAR REGRESSION
    print('\n\nLINEAR REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=1)
    
    lr = LinearRegression(featuresCol='selectedFeatures', labelCol=' shares', maxIter=100, regParam=0)
    
    pipeline_lr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, selector, lr])

    model_lr = pipeline_lr.fit(train_df)
    predictions_lr = model_lr.transform(test_df)
    
    
    print("Coefficients: " + str(model_lr.stages[-1].coefficients))
    print("Intercept: " + str(model_lr.stages[-1].intercept))
    
    evaluator_rmse_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_lr = evaluator_rmse_lr.evaluate(predictions_lr)
    r2_lr = evaluator_r2_lr.evaluate(predictions_lr)
    
    print(f"\nRoot Mean Squared Error (RMSE) = {rmse_lr}")
    print(f"R2 = {r2_lr}")
    
    
    
    # GENERALIZED LINEAR REGRESSION
    print('\n\nGENERALIZED LINEAR REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=2)
    
    glr = GeneralizedLinearRegression(featuresCol='selectedFeatures', labelCol=' shares', 
                                      family="gaussian", link="identity", 
                                      maxIter=10, regParam=0)

    pipeline_glr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, selector, glr])
    
    model_glr = pipeline_glr.fit(train_df)
    predictions_glr = model_glr.transform(test_df)
    
    print("Coefficients: " + str(model_glr.stages[-1].coefficients))
    print("Intercept: " + str(model_glr.stages[-1].intercept))
    
    
    evaluator_rmse_glr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_glr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_glr = evaluator_rmse_lr.evaluate(predictions_glr)
    r2_glr = evaluator_r2_lr.evaluate(predictions_glr)
    
    print(f"\nRoot Mean Squared Error (RMSE) = {rmse_glr}")
    print(f"R2 = {r2_glr}")
    
    
    
    # DECISION TREE REGRESSION
    print('\n\nDECISION TREE REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=4)
    
    dt = DecisionTreeRegressor(featuresCol="selectedFeatures", labelCol=' shares')

    pipeline_dt = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, selector, dt])

    model_dt = pipeline_dt.fit(train_df)
    predictions_dt = model_dt.transform(test_df)
    
    evaluator_rmse_dt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_dt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_dt = evaluator_rmse_dt.evaluate(predictions_dt)
    r2_dt = evaluator_r2_dt.evaluate(predictions_dt)
    
    print(f"Root Mean Squared Error (RMSE) = {rmse_dt}")
    print(f"R2 = {r2_dt}")
    
    
    
    # RANDOM FOREST REGRESSION
    print('\n\nRANDOM FOREST REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=5)
    
    rf = RandomForestRegressor(featuresCol="selectedFeatures", labelCol=' shares')
    
    pipeline_rf = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, selector, rf])

    model_rf = pipeline_rf.fit(train_df)
    predictions_rf = model_rf.transform(test_df)
    
    evaluator_rmse_rf = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_rf = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_rf = evaluator_rmse_rf.evaluate(predictions_rf)
    r2_rf = evaluator_r2_rf.evaluate(predictions_rf)
    
    print(f"Root Mean Squared Error (RMSE) = {rmse_rf}")
    print(f"R2 = {r2_rf}")
    
    
    
    # GRADIENT BOOSTED TREE REGRESSION
    print('\n\nGRADIENT BOOSTED TREE REGRESSION\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=6)
    
    gbt = GBTRegressor(featuresCol="selectedFeatures", labelCol=' shares', maxIter=10)
    
    pipeline_gbt = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, selector, gbt])

    model_gbt = pipeline_gbt.fit(train_df)
    predictions_gbt = model_gbt.transform(test_df)
    
    evaluator_rmse_gbt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_gbt = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_gbt = evaluator_rmse_gbt.evaluate(predictions_gbt)
    r2_gbt = evaluator_r2_gbt.evaluate(predictions_gbt)
    
    print(f"Root Mean Squared Error (RMSE) = {rmse_gbt}")
    print(f"R2 = {r2_gbt}")
    
    
    
    ### ADDING REGULARIZATION
    
    ## RIDGE
    
    # LINEAR REGRESSION
    print('\n\nLINEAR REGRESSION WITH RIDGE\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=1)
    
    lr = LinearRegression(featuresCol='selectedFeatures', labelCol=' shares', 
                          maxIter=100, regParam=0.5)
    
    pipeline_lr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, selector, lr])

    model_lr = pipeline_lr.fit(train_df)
    predictions_lr = model_lr.transform(test_df)
    
    
    print("Coefficients: " + str(model_lr.stages[-1].coefficients))
    print("Intercept: " + str(model_lr.stages[-1].intercept))
    
    evaluator_rmse_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_lr = evaluator_rmse_lr.evaluate(predictions_lr)
    r2_lr = evaluator_r2_lr.evaluate(predictions_lr)
    
    print(f"\nRoot Mean Squared Error (RMSE) = {rmse_lr}")
    print(f"R2 = {r2_lr}")
    
    
    ## LASSO
    
    # LINEAR REGRESSION
    print('\n\nLINEAR REGRESSION WITH LASSO\n')
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=1)
    
    lr = LinearRegression(featuresCol='selectedFeatures', labelCol=' shares', 
                          maxIter=100, regParam=0.5, elasticNetParam = 1)
    
    pipeline_lr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, selector, lr])

    model_lr = pipeline_lr.fit(train_df)
    predictions_lr = model_lr.transform(test_df)
    
    
    print("Coefficients: " + str(model_lr.stages[-1].coefficients))
    print("Intercept: " + str(model_lr.stages[-1].intercept))
    
    evaluator_rmse_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='rmse')
    evaluator_r2_lr = RegressionEvaluator(labelCol=' shares', predictionCol='prediction', metricName='r2')
    rmse_lr = evaluator_rmse_lr.evaluate(predictions_lr)
    r2_lr = evaluator_r2_lr.evaluate(predictions_lr)
    
    print(f"\nRoot Mean Squared Error (RMSE) = {rmse_lr}")
    print(f"R2 = {r2_lr}")
    
    
    
    # CLASSIFICATION MODELS
    
    df.select(' shares').summary("count", "mean", "min", "25%", "50%", "75%", 
                                 "85%", "95%", "max").show()
    
    # Define classes
    # - very poor (<25%)
    # - poor (<50%)
    # - average (<75%)
    # - good (<85%)
    # - very good (<90%)
    # - exceptionally good (<100%)
    
    
    df = df.withColumn('shares_label', 
                  F.when(F.col(" shares") < 945, 'very poor').
                    when((F.col(" shares") >= 945) & (F.col(" shares") < 1400), 'poor').
                    when((F.col(" shares") >= 1400) & (F.col(" shares") < 2700), 'average').
                    when((F.col(" shares") >= 2700) & (F.col(" shares") < 4300), 'good').
                    when((F.col(" shares") >= 4300) & (F.col(" shares") < 10700), 'very good').
                    when(F.col(" shares") >= 10700, 'exceptionally good'))
    
    df.select(' shares', 'shares_label').show()
    
    index3 = StringIndexer().setInputCol("shares_label").setOutputCol("shares_labelIndex").setHandleInvalid("keep")
    
    
    
    # BASELINE MODEL
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=1)
    
    # Find most popular class
    max_count = train_df.groupby('shares_label').count().select(F.max('count')).collect()[0][0]
    popular_class = train_df.groupby('shares_label').count().where(F.col('count')==max_count).select('shares_label').collect()[0][0]
    
    # Create a new shares label where popular class = 1, else = 0
    predictions_popular = test_df.withColumn('shares_label_01', 
                                                         F.when(F.col('shares_label') == popular_class, 
                                                                1.0).otherwise(0.0))
    
    # Predict everything = most popular class (=1)
    predictions_popular = predictions_popular.withColumn('prediction_01', F.lit(1.0))
    
    # Evaluate baseline model
    preds_and_labels = predictions_popular.select(['prediction_01','shares_label_01'])
    
    eval_accuracy = MulticlassClassificationEvaluator(labelCol="shares_label_01", predictionCol="prediction_01", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="shares_label_01", predictionCol="prediction_01", metricName="f1")
    
    accuracy = eval_accuracy.evaluate(preds_and_labels)
    f1score = eval_f1.evaluate(preds_and_labels)
    
    print('Accuracy:', accuracy)
    print('F1score:', f1score)
    
    
    
    # MULTIPLE LOGISTIC REGRESSION
    print('\n\nMULTIPLE LOGISTIC REGRESSION\n')
    
    from pyspark.ml.classification import LogisticRegression
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=89)
    
    lr = LogisticRegression(featuresCol='selectedFeatures', labelCol='shares_labelIndex', maxIter=10)
    pipeline_lr = Pipeline(stages=[assembler1, scaler, assembler3, assembler4, selector, index3, lr])
    
    model_lr = pipeline_lr.fit(train_df)
    predictions_lr = model_lr.transform(test_df)
    
    
    print("Coefficients: " + str(model_lr.stages[-1].coefficientMatrix))
    print("Intercept: " + str(model_lr.stages[-1].interceptVector))
    
    trainingSummary = model_lr.stages[-1].summary
    

    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
    
    
    
    # RANDOM FOREST
    print('\n\nRANDOM FOREST\n')
    
    from pyspark.ml.classification import RandomForestClassifier
    
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=67)
    
    rf = RandomForestClassifier(labelCol="shares_labelIndex", featuresCol="selectedFeatures", numTrees=10)
    
    pipeline_rf = Pipeline(stages=[assembler1, scaler, index1, index2, assembler2, selector, index3, rf])

    model_rf = pipeline_rf.fit(train_df)
    predictions_rf = model_rf.transform(test_df)
    
    trainingSummary = model_rf.stages[-1].summary
    
    accuracy = trainingSummary.accuracy
    falsePositiveRate = trainingSummary.weightedFalsePositiveRate
    truePositiveRate = trainingSummary.weightedTruePositiveRate
    fMeasure = trainingSummary.weightedFMeasure()
    precision = trainingSummary.weightedPrecision
    recall = trainingSummary.weightedRecall
    print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
          % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
    
    
    
    sc.stop()
    spark.stop()


    
    
    
    
    
    
    
    
    
    