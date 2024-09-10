import os
import sys
sys.path.append(os.path.abspath('../scripts'))
sys.path.append(os.path.abspath('../src'))
import pandas as pd

from load_data import load_data_from_postgres, check_duplicates, check_numeric_anomalies, get_numeric_columns
from utils import bytes_to_gigabytes, kilobytes_per_second_to_megabytes_per_second, milliseconds_to_hours, milliseconds_to_minutes, bytes_to_megabytes, milliseconds_to_seconds

def transform_and_save_data(query, output_file='transformed_data.csv'):
    # Load data from PostgreSQL
    df = load_data_from_postgres(query)
    
    # Customizing some columns and doing some unit conversions
    df['Social Media (GB)'] = df['Social Media DL (Bytes)'].apply(bytes_to_megabytes) + df['Social Media UL (Bytes)'].apply(bytes_to_megabytes)
    df['Youtube (GB)'] = df['Youtube DL (Bytes)'].apply(bytes_to_megabytes) + df['Youtube UL (Bytes)'].apply(bytes_to_megabytes)
    df['Google (GB)'] = (df['Google DL (Bytes)'] + df['Google UL (Bytes)']).apply(bytes_to_megabytes)
    df['Email (GB)'] = (df['Email DL (Bytes)'] + df['Email UL (Bytes)']).apply(bytes_to_megabytes)
    df['Netflix (GB)'] = (df['Netflix DL (Bytes)'] + df['Netflix UL (Bytes)']).apply(bytes_to_megabytes)
    df['Gaming (GB)'] = (df['Gaming DL (Bytes)'] + df['Gaming UL (Bytes)']).apply(bytes_to_megabytes)
    df['Other (GB)'] = (df['Other DL (Bytes)'] + df['Other UL (Bytes)']).apply(bytes_to_megabytes)
    df['Total Data (GB)'] = df['Total DL (Bytes)'].apply(bytes_to_megabytes) + df['Total UL (Bytes)'].apply(bytes_to_megabytes)
    df['Dur. (hr)'] = df['Dur. (ms).1'].apply(milliseconds_to_hours)
    df['Avg RTT DL (sec)'] = df['Avg RTT DL (ms)'].apply(milliseconds_to_seconds)
    df['Avg RTT UL (sec)'] = df['Avg RTT UL (ms)'].apply(milliseconds_to_seconds)
    df['Avg Bearer TP DL (Mbps)'] = df['Avg Bearer TP DL (kbps)'].apply(kilobytes_per_second_to_megabytes_per_second)
    df['Avg Bearer TP UL (Mbps)'] = df['Avg Bearer TP UL (kbps)'].apply(kilobytes_per_second_to_megabytes_per_second)

    # Apply conversion functions to columns and store results in new columns
    df['Total DL (Mb)'] = df['Total DL (Bytes)'].apply(bytes_to_megabytes)
    df['Total UL (Mb)'] = df['Total UL (Bytes)'].apply(bytes_to_megabytes)
    df['Social Media DL (Mb)'] = df['Social Media DL (Bytes)'].apply(bytes_to_megabytes)
    df['Social Media UL (Mb)'] = df['Social Media UL (Bytes)'].apply(bytes_to_megabytes)
    df['Google DL (Mb)'] = df['Google DL (Bytes)'].apply(bytes_to_megabytes)
    df['Google UL (Mb)'] = df['Google UL (Bytes)'].apply(bytes_to_megabytes)
    df['Email DL (Mb)'] = df['Email DL (Bytes)'].apply(bytes_to_megabytes)
    df['Email UL (Mb)'] = df['Email UL (Bytes)'].apply(bytes_to_megabytes)
    df['Youtube DL (Mb)'] = df['Youtube DL (Bytes)'].apply(bytes_to_megabytes)
    df['Youtube UL (Mb)'] = df['Youtube UL (Bytes)'].apply(bytes_to_megabytes)
    df['Netflix DL (Mb)'] = df['Netflix DL (Bytes)'].apply(bytes_to_megabytes)
    df['Netflix UL (Mb)'] = df['Netflix UL (Bytes)'].apply(bytes_to_megabytes)
    df['Gaming DL (Mb)'] = df['Gaming DL (Bytes)'].apply(bytes_to_megabytes)
    df['Gaming UL (Mb)'] = df['Gaming UL (Bytes)'].apply(bytes_to_megabytes)
    df['Other DL (Mb)'] = df['Other DL (Bytes)'].apply(bytes_to_megabytes)
    df['Other UL (Mb)'] = df['Other UL (Bytes)'].apply(bytes_to_megabytes)
    df['Dur. (hr)'] = df['Dur. (ms).1'].apply(milliseconds_to_hours)
    df['Dur. (sec)'] = df['Dur. (ms).1'].apply(milliseconds_to_seconds)
    df['Dur. (min)'] = df['Dur. (ms).1'].apply(milliseconds_to_minutes)

    # Calculate total data volume (DL+UL) for each application
    df['Social Media Data (Mb)'] = df['Social Media DL (Mb)'] + df['Social Media UL (Mb)']
    df['Youtube Data (Mb)'] = df['Youtube DL (Mb)'] + df['Youtube UL (Mb)']
    df['Email Data (Mb)'] = df['Email DL (Mb)'] + df['Email UL (Mb)']
    df['Gaming Data (Mb)'] = df['Gaming DL (Mb)'] + df['Gaming UL (Mb)']
    df['Netflix Data (Mb)'] = df['Netflix DL (Mb)'] + df['Netflix UL (Mb)']
    df['Google Data (Mb)'] = df['Google DL (Mb)'] + df['Google UL (Mb)']
    df['Other Data (Mb)'] = df['Other DL (Mb)'] + df['Other UL (Mb)']
    df['Total Data (Mb)'] = df['Total DL (Mb)'] + df['Total UL (Mb)']
    df['Avg Bearer TP (Mbps)'] = df['Avg Bearer TP DL (Mbps)'] + df['Avg Bearer TP UL (Mbps)']
    df['Avg RTT (sec)'] = df['Avg RTT DL (sec)'] + df['Avg RTT UL (sec)']
    df['TCP Retrans. Vol (Bytes)'] = df['TCP DL Retrans. Vol (Bytes)'] + df['TCP UL Retrans. Vol (Bytes)']
    df['TCP Retrans. Vol (Mb)'] = df['TCP Retrans. Vol (Bytes)'].apply(bytes_to_megabytes)

    # Convert 'Start' to datetime format
    df['Start'] = pd.to_datetime(df['Start'])

    # Extract date part only and save it to a new column 'Date'
    df['Date'] = df['Start'].dt.date

    # Included this for engagement analysis 
    df['Handset Manufacturer'] = df['Handset Manufacturer']
    df['Handset Type'] = df['Handset Type']
    df['IMSI'] = df['IMSI']
    df['MSISDN/Number'] = df['MSISDN/Number']

    # Additional columns for analysis
    df['DL TP < 50 Kbps (%)'] = df['DL TP < 50 Kbps (%)']
    df['50 Kbps < DL TP < 250 Kbps (%)'] = df['50 Kbps < DL TP < 250 Kbps (%)']
    df['250 Kbps < DL TP < 1 Mbps (%)'] = df['250 Kbps < DL TP < 1 Mbps (%)']
    df['DL TP > 1 Mbps (%)'] = df['DL TP > 1 Mbps (%)']
    df['UL TP < 10 Kbps (%)'] = df['UL TP < 10 Kbps (%)']
    df['10 Kbps < UL TP < 50 Kbps (%)'] = df['10 Kbps < UL TP < 50 Kbps (%)']
    df['50 Kbps < UL TP < 300 Kbps (%)'] = df['50 Kbps < UL TP < 300 Kbps (%)']
    df['UL TP > 300 Kbps (%)'] = df['UL TP > 300 Kbps (%)']
    df['HTTP DL (Mb)'] = df['HTTP DL (Bytes)'].apply(bytes_to_megabytes)
    df['HTTP UL (Mb)'] = df['HTTP UL (Bytes)'].apply(bytes_to_megabytes)
    df['Activity Duration DL (sec)'] = df['Activity Duration DL (ms)'].apply(milliseconds_to_seconds)
    df['Activity Duration UL (sec)'] = df['Activity Duration UL (ms)'].apply(milliseconds_to_seconds)

    # Save the transformed data to a new CSV file
    df.to_csv(output_file, index=False)

    return df



