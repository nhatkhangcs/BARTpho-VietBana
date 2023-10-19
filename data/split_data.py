import pandas as pd
import re

def clean_da_shit(string):
    string = re.sub(r',\s*,', ',', string)
    string = re.sub(r'\s+,\s+', ',', string)
    string = re.sub(r'\s+', ' ', string)
    string = re.sub(r'\.', '', string)
    words = list(map(lambda x: x.strip(), string.split(',')))
    return words


def main():
    data_all = pd.read_csv('Anh_xa_phuong_ngu_v2.csv', encoding='utf-8')
    data_all = pd.DataFrame(data_all, columns=['Tiếng Việt', 'Kon Tum', 'Gia Lai', 'Bình Định']).drop_duplicates()

    kontum_indexes = data_all['Kon Tum'].notnull()
    gialai_indexes = data_all['Gia Lai'].notnull()
    binhdinh_indexes = data_all['Bình Định'].notnull()

    kontum_df = pd.DataFrame(data_all, columns=['Tiếng Việt', 'Kon Tum'])
    kontum_df = kontum_df[kontum_indexes]
    kontum_df['Kon Tum'] = kontum_df['Kon Tum'].apply(clean_da_shit)

    gialai_df = pd.DataFrame(data_all, columns=['Tiếng Việt', 'Gia Lai'])
    gialai_df = gialai_df[gialai_indexes]
    gialai_df['Gia Lai'] = gialai_df['Gia Lai'].apply(clean_da_shit)

    binhdinh_df = pd.DataFrame(data_all, columns=['Tiếng Việt', 'Bình Định'])
    binhdinh_df = binhdinh_df[binhdinh_indexes]
    binhdinh_df['Bình Định'] = binhdinh_df['Bình Định'].apply(clean_da_shit)

    kontum_df = kontum_df.explode('Kon Tum')
    gialai_df = gialai_df.explode('Gia Lai')
    binhdinh_df = binhdinh_df.explode('Bình Định')

    ba_data_path = 'dictionary/dict.ba'
    vi_data_path = 'dictionary/dict.vi'

    binhdinh_df['Bình Định'].to_csv('BinhDinh/' + ba_data_path, index=False, header=False)
    binhdinh_df['Tiếng Việt'].to_csv('BinhDinh/' + vi_data_path, index=False, header=False)

    kontum_df['Kon Tum'].to_csv('KonTum/' + ba_data_path, index=False, header=False)
    kontum_df['Tiếng Việt'].to_csv('KonTum/' + vi_data_path, index=False, header=False)

    gialai_df['Gia Lai'].to_csv('GiaLai/' + ba_data_path, index=False, header=False)
    gialai_df['Tiếng Việt'].to_csv('GiaLai/' + vi_data_path, index=False, header=False)

    print(kontum_df.head(10))

if __name__ == '__main__':
    main()