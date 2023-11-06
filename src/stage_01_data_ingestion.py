import pandas as pd
import sys


class DataIngestion:

    def __init__(self) -> None:
        pass

    def load_data(self, path='data/Airlines.csv'):
        df = pd.read_csv(path)
        df.drop(['id'], axis=1, inplace=True)
        df.columns = df.columns.str.lower()

        return df
    
    def load_complementrry_data(self, path='data/GlobalAirportDatabase.txt'):
        #'data\GlobalAirportDatabase.txt'

        df_airpot_code = pd.read_csv(path, delimiter=':', header=None )

        df_airpot_code.columns = ['ICAO_code','IATA_code','airport_name','city_town','country','latitude_degrees',
                                'Latitude_Minutes','Latitude_Seconds', 'Latitude Direction', 'Longitude Degrees', 
                                'Longitude Minutes', 'Longitude Seconds', 'Longitude Direction', 'Altitude', 
                                'Latitude Decimal Degrees', 'Longitude Decimal Degrees']

        df_airpot_code = df_airpot_code[['IATA_code','country']].dropna()

        # Remove duplicates and display the DataFrame without duplicates
        df_airpot_code_without_duplicates = df_airpot_code.drop_duplicates(subset='IATA_code')

        return df_airpot_code_without_duplicates
    
    def merge_IATA_codes(self, df, df_airpot_code_without_duplicates):

        merged_df = pd.merge(df, df_airpot_code_without_duplicates, 
                             left_on='airportfrom',
                             right_on='IATA_code',
                             how='left')\
                            .rename(columns={'country': 'country_from'})

        merged_df = merged_df.drop('IATA_code', axis=1)

        merged_df = pd.merge(merged_df, df_airpot_code_without_duplicates, 
                             left_on='airportto',
                             right_on='IATA_code', 
                             how='left')\
                            .rename(columns={'country': 'country_to'})
        
        merged_df = merged_df.drop('IATA_code', axis=1)

        return merged_df
    
    def create_international_feature(self, merged_df):

        df = merged_df.copy()
        df['international'] = (df.country_from == df.country_to).astype('int')
        merged_df_final = df.dropna()

        return merged_df_final


    def convert_days_of_week(self, df):

        day_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
        df['dayofweek'] = df['dayofweek'].map(day_mapping)

        return df
    
    def converting_data_types(self, df):

        df['dayofweek'] = df['dayofweek'].astype('str')
        df['international'] = df['international'].astype('int8')
        df['delay'] = df['delay'].astype('int8') 

        return df
    
    def excluding_features(self, df):
        
        df = df.drop(['flight','airline', 'airportfrom', 'airportto',  
                      'country_from', 'country_to'], 
                     axis=1)
        return df

    
    def save_data_for_model(self, df):

        df.to_parquet('data/data_to_model/data_model.parquet', 
                      engine='fastparquet',
                      compression='snappy')
        


if __name__ == '__main__':
    data = DataIngestion()

    df = data.load_data()
    df_airpot_code_without_duplicates = data.load_complementrry_data()
    
    merged_df = data.merge_IATA_codes(df, df_airpot_code_without_duplicates)

    df = data.create_international_feature(merged_df)
    df = data.convert_days_of_week(df)
    df = data.converting_data_types(df)
    df = data.excluding_features(df)

    data.save_data_for_model(df)
    print('** STAGE 01 - end **')




