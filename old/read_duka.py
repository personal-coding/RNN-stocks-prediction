import lzma
import struct
import pandas as pd


def bi5_to_df(filename, fmt):
    chunk_size = struct.calcsize(fmt)
    data = []
    with lzma.open(filename) as f:
        while True:
            chunk = f.read(chunk_size)
            if chunk:
                data.append(struct.unpack(fmt, chunk))
            else:
                break
    df = pd.DataFrame(data)
    return df

df = bi5_to_df(r'C:\Users\russi\Downloads\BID_candles_min_1.bi5', '>3i2f')
print(df.head())